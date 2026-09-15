#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include <algorithm>
#include <limits>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/resource_var.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"

namespace tensorflow {

using GPUDevice = Eigen::GpuDevice;

template <typename T>
__device__ inline float ToFloat(T value) {
  return static_cast<float>(value);
}

template <typename T>
__device__ inline T FromFloat(float value) {
  return static_cast<T>(value);
}

template <typename T>
__global__ void SetZeroKernel(int64_t count, T* values) {
  for (int64_t index : GpuGridRangeX(count)) {
    values[index] = FromFloat<T>(0.0f);
  }
}

template <typename T, typename Index>
__global__ void CsrReorderKernel(
    int64_t count, const T* values, const Index* edge_ids, T* reordered) {
  for (int64_t index : GpuGridRangeX(count)) {
    reordered[index] = values[edge_ids[index]];
  }
}

inline int BlockCountFor(int64_t count, int threads, const GPUDevice& device) {
  const int64_t requested = (count + threads - 1) / threads;
  const int maximum = device.getNumGpuMultiProcessors() * 8;
  return static_cast<int>(
      std::max<int64_t>(1, std::min<int64_t>(requested, maximum)));
}

template <typename T>
__device__ inline void FastAtomicAdd(T* address, T value) {
  GpuAtomicAdd(address, value);
}

template <>
__device__ inline void FastAtomicAdd<Eigen::half>(
    Eigen::half* address, Eigen::half value) {
  atomicAdd(
      reinterpret_cast<__half*>(address),
      __float2half(ToFloat(value)));
}

template <typename T, typename Index>
__global__ void CsrSpikeForwardKernel(
    int count, int n_pre, int n_post, int n_basis, const T* spikes,
    const Index* post_ids, const T* weights,
    const Index* synapse_types, const T* basis,
    const Index* row_splits, T* currents) {
  const int index = blockIdx.x;
  if (index >= count) {
    return;
  }
  const float spike = ToFloat(spikes[index]);
  if (spike <= 0.0f) {
    return;
  }
  const int batch = index / n_pre;
  const int pre = index - batch * n_pre;
  const Index start = row_splits[pre];
  const int64_t work_items =
      static_cast<int64_t>(row_splits[pre + 1] - start) * n_basis;
  for (int64_t item = threadIdx.x; item < work_items; item += blockDim.x) {
    const Index edge = start + static_cast<Index>(item / n_basis);
    const int receptor = item % n_basis;
    const int post = static_cast<int>(post_ids[edge]);
    const int synapse_type = static_cast<int>(synapse_types[edge]);
    const T weighted_spike = FromFloat<T>(spike) * weights[edge];
    const T value =
        weighted_spike * basis[synapse_type * n_basis + receptor];
    const int64_t output_index =
        (static_cast<int64_t>(batch) * n_post + post) * n_basis + receptor;
    FastAtomicAdd(currents + output_index, value);
  }
}

template <typename T, typename Index>
__global__ void CsrSpikeGradKernel(
    int count, int n_pre, int n_post, int n_basis, const T* spikes,
    const T* current_grad, const Index* post_ids, const T* weights,
    const Index* synapse_types, const T* basis,
    const Index* row_splits, const Index* edge_ids, T* spike_grad,
    float* weight_grad, const T* spike_gradient_scale) {
  __shared__ float partial_gradients[128];
  const int index = blockIdx.x;
  if (index >= count) {
    return;
  }
  const int batch = index / n_pre;
  const int pre = index - batch * n_pre;
  const float spike = ToFloat(spikes[index]);
  float pre_gradient = 0.0f;
  for (int64_t edge =
           static_cast<int64_t>(row_splits[pre]) + threadIdx.x;
       edge < static_cast<int64_t>(row_splits[pre + 1]);
       edge += blockDim.x) {
    const int post = static_cast<int>(post_ids[edge]);
    const int synapse_type = static_cast<int>(synapse_types[edge]);
    const int64_t gradient_base =
        (static_cast<int64_t>(batch) * n_post + post) * n_basis;
    const int basis_base = synapse_type * n_basis;
    float edge_gradient = 0.0f;
    for (int receptor = 0; receptor < n_basis; ++receptor) {
      edge_gradient +=
          ToFloat(current_grad[gradient_base + receptor]) *
          ToFloat(basis[basis_base + receptor]);
    }
    pre_gradient += edge_gradient * ToFloat(weights[edge]);
    if (spike > 0.0f) {
      GpuAtomicAdd(
          weight_grad + static_cast<int64_t>(edge_ids[edge]),
          edge_gradient * spike);
    }
  }
  partial_gradients[threadIdx.x] = pre_gradient;
  __syncthreads();
  for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
    if (threadIdx.x < offset) {
      partial_gradients[threadIdx.x] +=
          partial_gradients[threadIdx.x + offset];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    spike_grad[index] = FromFloat<T>(
        partial_gradients[0] * ToFloat(*spike_gradient_scale));
  }
}

template <typename T, typename Index>
__global__ void PairProjectionBatch32Kernel(
    int64_t count, int n_post, const T* current_grad, const T* basis,
    const Index* pair_posts, const Index* pair_types, float* projected) {
  for (int64_t index : GpuGridRangeX(count)) {
    const int batch = static_cast<int>(index % 32);
    const int64_t pair = index / 32;
    const int post = static_cast<int>(pair_posts[pair]);
    const int synapse_type = static_cast<int>(pair_types[pair]);
    const int64_t gradient_base =
        (static_cast<int64_t>(batch) * n_post + post) * 4;
    const int basis_base = synapse_type * 4;
    float value = 0.0f;
#pragma unroll
    for (int receptor = 0; receptor < 4; ++receptor) {
      value += ToFloat(current_grad[gradient_base + receptor]) *
               ToFloat(basis[basis_base + receptor]);
    }
    projected[index] = value;
  }
}

template <typename T, typename Index>
__global__ void CsrSpikeGradPairBatch32Kernel(
    int n_pre, const T* spikes, const Index* row_splits,
    const Index* edge_ids, const Index* pair_ids, const T* weights,
    const float* projected, T* spike_grad, float* weight_grad,
    const T* spike_gradient_scale) {
  constexpr int kBatch = 32;
  constexpr int kWarpsPerBlock = 4;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int pre = blockIdx.x * kWarpsPerBlock + warp;
  if (pre >= n_pre) {
    return;
  }

  const float spike =
      ToFloat(spikes[static_cast<int64_t>(lane) * n_pre + pre]);
  float pre_gradient = 0.0f;
  const Index start = row_splits[pre];
  const Index end = row_splits[pre + 1];
  for (Index edge = start; edge < end; ++edge) {
    const float edge_gradient =
        projected[static_cast<int64_t>(pair_ids[edge]) * kBatch + lane];
    pre_gradient += edge_gradient * ToFloat(weights[edge]);
    float edge_weight_gradient = edge_gradient * spike;
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      edge_weight_gradient +=
          __shfl_down_sync(0xffffffff, edge_weight_gradient, offset);
    }
    if (lane == 0) {
      weight_grad[static_cast<int64_t>(edge_ids[edge])] =
          edge_weight_gradient;
    }
  }
  spike_grad[static_cast<int64_t>(lane) * n_pre + pre] =
      FromFloat<T>(pre_gradient * ToFloat(*spike_gradient_scale));
}

template <typename T, typename Index>
__global__ void CsrWeightGradKernel(
    int count, int n_pre, int n_post, int n_basis, const T* spikes,
    const T* current_grad, const Index* post_ids,
    const Index* synapse_types, const T* basis,
    const Index* row_splits, const Index* edge_ids,
    float* weight_grad) {
  const int index = blockIdx.x;
  if (index >= count) {
    return;
  }
  const float spike = ToFloat(spikes[index]);
  if (spike <= 0.0f) {
    return;
  }
  const int batch = index / n_pre;
  const int pre = index - batch * n_pre;
  for (int64_t edge =
           static_cast<int64_t>(row_splits[pre]) + threadIdx.x;
       edge < static_cast<int64_t>(row_splits[pre + 1]);
       edge += blockDim.x) {
    const int post = static_cast<int>(post_ids[edge]);
    const int synapse_type = static_cast<int>(synapse_types[edge]);
    const int64_t gradient_base =
        (static_cast<int64_t>(batch) * n_post + post) * n_basis;
    const int basis_base = synapse_type * n_basis;
    float edge_gradient = 0.0f;
    for (int receptor = 0; receptor < n_basis; ++receptor) {
      edge_gradient +=
          ToFloat(current_grad[gradient_base + receptor]) *
          ToFloat(basis[basis_base + receptor]);
    }
    GpuAtomicAdd(
        weight_grad + static_cast<int64_t>(edge_ids[edge]),
        edge_gradient * spike);
  }
}

inline void RequireVector(
    OpKernelContext* context, const Tensor& tensor, const char* name) {
  OP_REQUIRES(
      context, TensorShapeUtils::IsVector(tensor.shape()),
      errors::InvalidArgument(name, " must be rank 1, got ", tensor.shape()));
}

inline void RequireMatrix(
    OpKernelContext* context, const Tensor& tensor, const char* name) {
  OP_REQUIRES(
      context, TensorShapeUtils::IsMatrix(tensor.shape()),
      errors::InvalidArgument(name, " must be rank 2, got ", tensor.shape()));
}

inline void RequireScalar(
    OpKernelContext* context, const Tensor& tensor, const char* name) {
  OP_REQUIRES(
      context, TensorShapeUtils::IsScalar(tensor.shape()),
      errors::InvalidArgument(name, " must be rank 0, got ", tensor.shape()));
}

template <typename T>
bool LookupVariable(
    OpKernelContext* context, int input, const char* name,
    core::RefCountPtr<Var>* variable) {
  const absl::Status status =
      LookupResource(context, HandleFromInput(context, input), variable);
  if (!status.ok()) {
    context->SetStatus(status);
    return false;
  }
  if (!(*variable)->is_initialized) {
    context->SetStatus(
        errors::FailedPrecondition(name, " is uninitialized."));
    return false;
  }
  if ((*variable)->tensor()->dtype() != DataTypeToEnum<T>::v()) {
    context->SetStatus(errors::InvalidArgument(
        name, " has dtype ",
        DataTypeString((*variable)->tensor()->dtype()), ", expected ",
        DataTypeString(DataTypeToEnum<T>::v()), "."));
    return false;
  }
  return true;
}

template <typename T, typename Index>
class DpointnetCsrReorderOp : public OpKernel {
 public:
  explicit DpointnetCsrReorderOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("n_edges", &n_edges_));
    OP_REQUIRES_OK(context, context->GetAttr("n_sources", &n_sources_));
    OP_REQUIRES_OK(context, context->GetAttr("n_pairs", &n_pairs_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& values = context->input(0);
    core::RefCountPtr<Var> metadata_variable;
    if (!LookupVariable<Index>(
            context, 1, "metadata", &metadata_variable)) {
      return;
    }
    const Tensor& metadata = *metadata_variable->tensor();
    RequireVector(context, values, "values");
    RequireVector(context, metadata, "metadata");
    if (!context->status().ok()) {
      return;
    }
    OP_REQUIRES(
        context, values.NumElements() == n_edges_,
        errors::InvalidArgument("values length must equal n_edges."));
    OP_REQUIRES(
      context,
      metadata.NumElements() ==
        3 * n_edges_ + n_sources_ + 1 +
          (n_pairs_ > 0 ? n_edges_ + 2 * n_pairs_ : 0),
      errors::InvalidArgument("metadata size does not match connectivity."));
    const Index* edge_ids =
      metadata.flat<Index>().data() + 2 * n_edges_ + n_sources_ + 1;

    Tensor* reordered = nullptr;
    OP_REQUIRES_OK(
        context,
        context->allocate_output(0, values.shape(), &reordered));
    const int64_t count = values.NumElements();
    if (count == 0) {
      return;
    }
    const GPUDevice& device = context->eigen_device<GPUDevice>();
    constexpr int threads = 256;
    OP_REQUIRES_OK(
        context,
        GpuLaunchKernel(
            CsrReorderKernel<T, Index>,
            BlockCountFor(count, threads, device), threads, 0,
            device.stream(), count, values.flat<T>().data(),
            edge_ids, reordered->flat<T>().data()));
  }

 private:
  int64_t n_edges_;
  int64_t n_sources_;
  int64_t n_pairs_;
};

template <typename T, typename Index>
class DpointnetCsrSpikeForwardOp : public OpKernel {
 public:
  explicit DpointnetCsrSpikeForwardOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("n_post", &n_post_));
    OP_REQUIRES_OK(context, context->GetAttr("n_edges", &n_edges_));
    OP_REQUIRES_OK(context, context->GetAttr("n_pairs", &n_pairs_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& spikes = context->input(0);
    const Tensor& master_weights = context->input(1);
    const Tensor& weights = context->input(3);
    const Tensor& basis = context->input(4);
    const Tensor& spike_gradient_scale = context->input(5);
    core::RefCountPtr<Var> metadata_variable;
    if (!LookupVariable<Index>(
            context, 2, "metadata", &metadata_variable)) {
      return;
    }
    const Tensor& metadata = *metadata_variable->tensor();
    RequireMatrix(context, spikes, "spikes");
    RequireVector(context, master_weights, "master_weights");
    RequireVector(context, weights, "weights");
    RequireMatrix(context, basis, "basis");
    RequireScalar(context, spike_gradient_scale, "spike_gradient_scale");
    RequireVector(context, metadata, "metadata");
    if (!context->status().ok()) {
      return;
    }

    const int64_t batch = spikes.dim_size(0);
    const int64_t n_pre = spikes.dim_size(1);
    const int64_t n_basis = basis.dim_size(1);
    OP_REQUIRES(
        context, weights.NumElements() == n_edges_,
        errors::InvalidArgument("weights length must equal n_edges."));
    OP_REQUIRES(
        context, master_weights.NumElements() == n_edges_,
        errors::InvalidArgument(
            "master_weights length must equal n_edges."));
    OP_REQUIRES(
        context,
      metadata.NumElements() ==
        3 * n_edges_ + n_pre + 1 +
          (n_pairs_ > 0 ? n_edges_ + 2 * n_pairs_ : 0),
        errors::InvalidArgument(
            "metadata length does not match n_edges and n_pre."));
    OP_REQUIRES(
        context, basis.dim_size(0) > 0 && n_basis > 0,
        errors::InvalidArgument("basis must have non-zero dimensions."));
    OP_REQUIRES(
        context,
        batch * n_pre <= std::numeric_limits<int>::max(),
        errors::InvalidArgument("Tensor size exceeds CUDA kernel index range."));

    const Index* metadata_values = metadata.flat<Index>().data();
    const Index* post_ids = metadata_values;
    const Index* synapse_types = metadata_values + n_edges_;
    const Index* row_splits = metadata_values + 2 * n_edges_;
    Tensor* currents = nullptr;
    OP_REQUIRES_OK(
        context,
        context->allocate_output(
            0, TensorShape({batch * n_post_, n_basis}), &currents));
    const GPUDevice& device = context->eigen_device<GPUDevice>();
    const int64_t output_count = currents->NumElements();
    constexpr int zero_threads = 256;
    OP_REQUIRES_OK(
        context,
        GpuLaunchKernel(
            SetZeroKernel<T>,
            BlockCountFor(output_count, zero_threads, device),
            zero_threads, 0, device.stream(), output_count,
            currents->flat<T>().data()));

    const int work_count = static_cast<int>(batch * n_pre);
    OP_REQUIRES_OK(
        context,
        GpuLaunchKernel(
            CsrSpikeForwardKernel<T, Index>, work_count, 128, 0,
            device.stream(),
            work_count,
            static_cast<int>(n_pre), n_post_, static_cast<int>(n_basis),
            spikes.flat<T>().data(), post_ids,
            weights.flat<T>().data(), synapse_types,
            basis.flat<T>().data(), row_splits,
            currents->flat<T>().data()));
  }

 private:
  int n_post_;
  int64_t n_edges_;
  int64_t n_pairs_;
};

template <typename T, typename Index>
class DpointnetCsrSpikeGradOp : public OpKernel {
 public:
  explicit DpointnetCsrSpikeGradOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("n_post", &n_post_));
    OP_REQUIRES_OK(context, context->GetAttr("n_edges", &n_edges_));
    OP_REQUIRES_OK(context, context->GetAttr("n_pairs", &n_pairs_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& spikes = context->input(0);
    const Tensor& current_grad = context->input(1);
    const Tensor& weights = context->input(3);
    const Tensor& basis = context->input(4);
    const Tensor& spike_gradient_scale = context->input(5);
    core::RefCountPtr<Var> metadata_variable;
    if (!LookupVariable<Index>(
            context, 2, "metadata", &metadata_variable)) {
      return;
    }
    const Tensor& metadata = *metadata_variable->tensor();
    RequireMatrix(context, spikes, "spikes");
    RequireMatrix(context, current_grad, "current_grad");
    RequireVector(context, weights, "weights");
    RequireMatrix(context, basis, "basis");
    RequireScalar(context, spike_gradient_scale, "spike_gradient_scale");
    RequireVector(context, metadata, "metadata");
    if (!context->status().ok()) {
      return;
    }

    const int64_t batch = spikes.dim_size(0);
    const int64_t n_pre = spikes.dim_size(1);
    const int64_t n_basis = basis.dim_size(1);
    OP_REQUIRES(
        context, weights.NumElements() == n_edges_,
        errors::InvalidArgument("weights length must equal n_edges."));
    OP_REQUIRES(
        context,
      metadata.NumElements() ==
        3 * n_edges_ + n_pre + 1 +
          (n_pairs_ > 0 ? n_edges_ + 2 * n_pairs_ : 0),
        errors::InvalidArgument(
            "metadata length does not match n_edges and n_pre."));
    OP_REQUIRES(
        context,
        current_grad.dim_size(0) == batch * n_post_ &&
            current_grad.dim_size(1) == n_basis,
        errors::InvalidArgument(
            "current_grad shape does not match the forward output."));
    OP_REQUIRES(
        context, batch * n_pre <= std::numeric_limits<int>::max(),
        errors::InvalidArgument("Tensor size exceeds CUDA kernel index range."));

    const Index* metadata_values = metadata.flat<Index>().data();
    const Index* post_ids = metadata_values;
    const Index* synapse_types = metadata_values + n_edges_;
    const Index* row_splits = metadata_values + 2 * n_edges_;
    const Index* edge_ids = row_splits + n_pre + 1;
    const Index* pair_ids = edge_ids + n_edges_;
    const Index* pair_posts = pair_ids + n_edges_;
    const Index* pair_types = pair_posts + n_pairs_;
    Tensor* spike_grad = nullptr;
    Tensor* weight_grad = nullptr;
    OP_REQUIRES_OK(
        context,
        context->allocate_output(0, spikes.shape(), &spike_grad));
    OP_REQUIRES_OK(
        context,
        context->allocate_output(1, weights.shape(), &weight_grad));
    const GPUDevice& device = context->eigen_device<GPUDevice>();
    constexpr int zero_threads = 256;
    OP_REQUIRES_OK(
        context,
        GpuLaunchKernel(
            SetZeroKernel<float>,
            BlockCountFor(n_edges_, zero_threads, device),
            zero_threads, 0, device.stream(), n_edges_,
            weight_grad->flat<float>().data()));

    if (batch == 32 && n_basis == 4 && n_pairs_ > 0) {
      Tensor projected;
      const int64_t projected_count = n_pairs_ * 32;
      OP_REQUIRES_OK(
          context,
        context->allocate_temp(
          DT_FLOAT, TensorShape({projected_count}), &projected));
      constexpr int projection_threads = 256;
      OP_REQUIRES_OK(
        context,
        GpuLaunchKernel(
          PairProjectionBatch32Kernel<T, Index>,
          BlockCountFor(projected_count, projection_threads, device),
          projection_threads, 0, device.stream(), projected_count,
          n_post_, current_grad.flat<T>().data(), basis.flat<T>().data(),
          pair_posts, pair_types, projected.flat<float>().data()));
      constexpr int gradient_threads = 128;
      constexpr int warps_per_block = gradient_threads / 32;
      const int gradient_blocks = static_cast<int>(
        (n_pre + warps_per_block - 1) / warps_per_block);
      OP_REQUIRES_OK(
        context,
        GpuLaunchKernel(
          CsrSpikeGradPairBatch32Kernel<T, Index>, gradient_blocks,
          gradient_threads, 0, device.stream(), static_cast<int>(n_pre),
          spikes.flat<T>().data(), row_splits, edge_ids, pair_ids,
          weights.flat<T>().data(), projected.flat<float>().data(),
          spike_grad->flat<T>().data(),
          weight_grad->flat<float>().data(),
          spike_gradient_scale.flat<T>().data()));
    } else {
      const int work_count = static_cast<int>(batch * n_pre);
      OP_REQUIRES_OK(
        context,
        GpuLaunchKernel(
          CsrSpikeGradKernel<T, Index>, work_count, 128, 0,
          device.stream(), work_count, static_cast<int>(n_pre), n_post_,
          static_cast<int>(n_basis), spikes.flat<T>().data(),
          current_grad.flat<T>().data(), post_ids,
          weights.flat<T>().data(), synapse_types,
          basis.flat<T>().data(), row_splits, edge_ids,
          spike_grad->flat<T>().data(),
          weight_grad->flat<float>().data(),
          spike_gradient_scale.flat<T>().data()));
    }
  }

 private:
  int n_post_;
  int64_t n_edges_;
  int64_t n_pairs_;
};

template <typename T, typename Index>
class DpointnetCsrWeightGradOp : public OpKernel {
 public:
  explicit DpointnetCsrWeightGradOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("n_post", &n_post_));
    OP_REQUIRES_OK(context, context->GetAttr("n_edges", &n_edges_));
    OP_REQUIRES_OK(context, context->GetAttr("n_pairs", &n_pairs_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& spikes = context->input(0);
    const Tensor& current_grad = context->input(1);
    const Tensor& basis = context->input(3);
    core::RefCountPtr<Var> metadata_variable;
    if (!LookupVariable<Index>(
            context, 2, "metadata", &metadata_variable)) {
      return;
    }
    const Tensor& metadata = *metadata_variable->tensor();
    RequireMatrix(context, spikes, "spikes");
    RequireMatrix(context, current_grad, "current_grad");
    RequireMatrix(context, basis, "basis");
    RequireVector(context, metadata, "metadata");
    if (!context->status().ok()) {
      return;
    }

    const int64_t batch = spikes.dim_size(0);
    const int64_t n_pre = spikes.dim_size(1);
    const int64_t n_basis = basis.dim_size(1);
    OP_REQUIRES(
        context,
      metadata.NumElements() ==
        3 * n_edges_ + n_pre + 1 +
          (n_pairs_ > 0 ? n_edges_ + 2 * n_pairs_ : 0),
        errors::InvalidArgument(
            "metadata length does not match n_edges and n_pre."));
    OP_REQUIRES(
        context,
        current_grad.dim_size(0) == batch * n_post_ &&
            current_grad.dim_size(1) == n_basis,
        errors::InvalidArgument(
            "current_grad shape does not match the forward output."));
    OP_REQUIRES(
        context,
        batch * n_pre <= std::numeric_limits<int>::max(),
        errors::InvalidArgument("Tensor size exceeds CUDA kernel index range."));

    const Index* metadata_values = metadata.flat<Index>().data();
    const Index* post_ids = metadata_values;
    const Index* synapse_types = metadata_values + n_edges_;
    const Index* row_splits = metadata_values + 2 * n_edges_;
    const Index* edge_ids = row_splits + n_pre + 1;
    Tensor* weight_grad = nullptr;
    OP_REQUIRES_OK(
        context,
        context->allocate_output(
            0, TensorShape({n_edges_}), &weight_grad));
    const GPUDevice& device = context->eigen_device<GPUDevice>();
    constexpr int zero_threads = 256;
    OP_REQUIRES_OK(
        context,
        GpuLaunchKernel(
            SetZeroKernel<float>,
            BlockCountFor(n_edges_, zero_threads, device),
            zero_threads, 0, device.stream(), n_edges_,
            weight_grad->flat<float>().data()));

    const int work_count = static_cast<int>(batch * n_pre);
    OP_REQUIRES_OK(
        context,
        GpuLaunchKernel(
            CsrWeightGradKernel<T, Index>, work_count, 128, 0,
            device.stream(),
            work_count,
            static_cast<int>(n_pre), n_post_, static_cast<int>(n_basis),
            spikes.flat<T>().data(), current_grad.flat<T>().data(),
            post_ids, synapse_types, basis.flat<T>().data(),
            row_splits, edge_ids,
            weight_grad->flat<float>().data()));
  }

 private:
  int n_post_;
  int64_t n_edges_;
  int64_t n_pairs_;
};

#define REGISTER_GPU_KERNELS(T, Index)                                    \
  REGISTER_KERNEL_BUILDER(                                               \
      Name("DpointnetCsrReorder")                                        \
          .Device(DEVICE_GPU)                                            \
          .TypeConstraint<T>("T")                                        \
          .TypeConstraint<Index>("Tindex"),                              \
      DpointnetCsrReorderOp<T, Index>);                                  \
  REGISTER_KERNEL_BUILDER(                                               \
      Name("DpointnetCsrSpikeForward")                                   \
          .Device(DEVICE_GPU)                                            \
          .TypeConstraint<T>("T")                                        \
          .TypeConstraint<Index>("Tindex"),                              \
      DpointnetCsrSpikeForwardOp<T, Index>);                             \
  REGISTER_KERNEL_BUILDER(                                               \
      Name("DpointnetCsrSpikeGrad")                                      \
          .Device(DEVICE_GPU)                                            \
          .TypeConstraint<T>("T")                                        \
          .TypeConstraint<Index>("Tindex"),                              \
      DpointnetCsrSpikeGradOp<T, Index>);                                \
  REGISTER_KERNEL_BUILDER(                                               \
      Name("DpointnetCsrWeightGrad")                                     \
          .Device(DEVICE_GPU)                                            \
          .TypeConstraint<T>("T")                                        \
          .TypeConstraint<Index>("Tindex"),                              \
      DpointnetCsrWeightGradOp<T, Index>);

#define REGISTER_GPU_KERNELS_FOR_TYPE(T) \
  REGISTER_GPU_KERNELS(T, uint32);       \
  REGISTER_GPU_KERNELS(T, int64_t);

TF_CALL_half(REGISTER_GPU_KERNELS_FOR_TYPE);
TF_CALL_float(REGISTER_GPU_KERNELS_FOR_TYPE);

#undef REGISTER_GPU_KERNELS_FOR_TYPE
#undef REGISTER_GPU_KERNELS

}  // namespace tensorflow

#endif  // GOOGLE_CUDA
