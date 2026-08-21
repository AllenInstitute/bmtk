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
    float* weight_grad) {
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
    spike_grad[index] = FromFloat<T>(partial_gradients[0]);
  }
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
        context, metadata.NumElements() >= 3 * n_edges_ + 1,
        errors::InvalidArgument("metadata is too short for n_edges."));
    const int64_t row_splits_size =
        metadata.NumElements() - 3 * n_edges_;
    const Index* edge_ids =
        metadata.flat<Index>().data() + 2 * n_edges_ + row_splits_size;

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
};

template <typename T, typename Index>
class DpointnetCsrSpikeForwardOp : public OpKernel {
 public:
  explicit DpointnetCsrSpikeForwardOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("n_post", &n_post_));
    OP_REQUIRES_OK(context, context->GetAttr("n_edges", &n_edges_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& spikes = context->input(0);
    const Tensor& master_weights = context->input(1);
    const Tensor& weights = context->input(3);
    const Tensor& basis = context->input(4);
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
        metadata.NumElements() == 3 * n_edges_ + n_pre + 1,
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
};

template <typename T, typename Index>
class DpointnetCsrSpikeGradOp : public OpKernel {
 public:
  explicit DpointnetCsrSpikeGradOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("n_post", &n_post_));
    OP_REQUIRES_OK(context, context->GetAttr("n_edges", &n_edges_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& spikes = context->input(0);
    const Tensor& current_grad = context->input(1);
    const Tensor& weights = context->input(3);
    const Tensor& basis = context->input(4);
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
        metadata.NumElements() == 3 * n_edges_ + n_pre + 1,
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

    const int work_count = static_cast<int>(batch * n_pre);
    OP_REQUIRES_OK(
        context,
        GpuLaunchKernel(
            CsrSpikeGradKernel<T, Index>, work_count, 128, 0,
            device.stream(),
            work_count,
            static_cast<int>(n_pre), n_post_, static_cast<int>(n_basis),
            spikes.flat<T>().data(), current_grad.flat<T>().data(),
            post_ids, weights.flat<T>().data(),
            synapse_types, basis.flat<T>().data(),
            row_splits, edge_ids,
            spike_grad->flat<T>().data(), weight_grad->flat<float>().data()));
  }

 private:
  int n_post_;
  int64_t n_edges_;
};

template <typename T, typename Index>
class DpointnetCsrWeightGradOp : public OpKernel {
 public:
  explicit DpointnetCsrWeightGradOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("n_post", &n_post_));
    OP_REQUIRES_OK(context, context->GetAttr("n_edges", &n_edges_));
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
        metadata.NumElements() == 3 * n_edges_ + n_pre + 1,
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
