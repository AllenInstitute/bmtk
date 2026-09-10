#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include <algorithm>
#include <limits>
#include <type_traits>

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

template <typename T, typename Index>
__global__ void CsrRestoreKernel(
    int64_t count, const T* values, const Index* edge_ids, T* restored) {
  for (int64_t index : GpuGridRangeX(count)) {
    restored[edge_ids[index]] = values[index];
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
__global__ void CsrSpikeForwardGroupedBatch32Kernel(
    int64_t n_active_rows, int64_t n_pre, int n_post,
    const T* spikes, const int64_t* active_rows, const Index* post_ids,
    const T* weights, const Index* synapse_types, const T* basis,
    const Index* row_splits, T* currents) {
  const int64_t active_id = blockIdx.x;
  if (active_id >= n_active_rows) {
    return;
  }
  const int64_t pre = active_rows[active_id];
  __shared__ uint32 batch_mask;
  if (threadIdx.x < 32) {
    const bool active =
      ToFloat(spikes[static_cast<int64_t>(threadIdx.x) * n_pre + pre]) >
      0.0f;
    const uint32 mask = __ballot_sync(0xffffffff, active);
    if (threadIdx.x == 0) {
      batch_mask = mask;
    }
  }
  __syncthreads();

  for (Index edge = row_splits[pre] + threadIdx.x;
       edge < row_splits[pre + 1]; edge += blockDim.x) {
    const int post = static_cast<int>(post_ids[edge]);
    const int synapse_type = static_cast<int>(synapse_types[edge]);
    const float weight = ToFloat(weights[edge]);
    uint32 remaining = batch_mask;
    while (remaining != 0) {
      const int batch = __ffs(remaining) - 1;
      remaining &= remaining - 1;
      const float weighted_spike =
          ToFloat(spikes[static_cast<int64_t>(batch) * n_pre + pre]) * weight;
      T* output = currents +
          (static_cast<int64_t>(batch) * n_post + post) * 4;
      if constexpr (std::is_same<T, Eigen::half>::value) {
        const Eigen::half* type_basis = basis + synapse_type * 4;
        const float2 basis01 = __half22float2(
            *reinterpret_cast<const __half2*>(type_basis));
        const float2 basis23 = __half22float2(
            *reinterpret_cast<const __half2*>(type_basis + 2));
        atomicAdd(
            reinterpret_cast<__half2*>(output),
            __floats2half2_rn(
                weighted_spike * basis01.x, weighted_spike * basis01.y));
        atomicAdd(
            reinterpret_cast<__half2*>(output + 2),
            __floats2half2_rn(
                weighted_spike * basis23.x, weighted_spike * basis23.y));
      } else {
#pragma unroll
        for (int receptor = 0; receptor < 4; ++receptor) {
          FastAtomicAdd(
              output + receptor,
              FromFloat<T>(
                  weighted_spike * ToFloat(basis[synapse_type * 4 + receptor])));
        }
      }
    }
  }
}

template <typename T, typename Index>
__global__ void CsrSpikeForwardFixed4Kernel(
  int64_t count, int n_pre, int n_post, int64_t n_edges, int n_types,
  const T* spikes, const T* weights,
    const Index* incoming_pre_ids, const Index* incoming_edge_ids,
  const Index* incoming_types, const T* basis, const T* initial,
  T* currents) {
  for (int64_t index : GpuGridRangeX(count)) {
    const int batch = static_cast<int>(index / n_post);
    const int post = static_cast<int>(
        index - static_cast<int64_t>(batch) * n_post);
    float sums[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    if (initial != nullptr) {
#pragma unroll
      for (int receptor = 0; receptor < 4; ++receptor) {
        sums[receptor] = ToFloat(initial[index * 4 + receptor]);
      }
    }
#pragma unroll
    for (int offset = 0; offset < 4; ++offset) {
      const int64_t incoming = static_cast<int64_t>(post) * 4 + offset;
      const int64_t pre = static_cast<int64_t>(incoming_pre_ids[incoming]);
      const int64_t edge =
          static_cast<int64_t>(incoming_edge_ids[incoming]);
      const int64_t type = static_cast<int64_t>(incoming_types[incoming]);
      if (pre < 0 || pre >= n_pre || edge < 0 || edge >= n_edges ||
          type < 0 || type >= n_types) {
        continue;
      }
      const float spike =
          ToFloat(spikes[static_cast<int64_t>(batch) * n_pre + pre]);
      if (spike <= 0.0f) {
        continue;
      }
      const float weighted = spike * ToFloat(weights[edge]);
#pragma unroll
      for (int receptor = 0; receptor < 4; ++receptor) {
        sums[receptor] +=
            weighted * ToFloat(basis[type * 4 + receptor]);
      }
    }
#pragma unroll
    for (int receptor = 0; receptor < 4; ++receptor) {
      currents[index * 4 + receptor] = FromFloat<T>(sums[receptor]);
    }
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

template <int kHalf, int kMask>
__device__ __forceinline__ void ButterflyReduce(float* partial, int lane) {
  const bool upper = (lane & kMask) != 0;
#pragma unroll
  for (int index = 0; index < kHalf; ++index) {
    const float keep = upper ? partial[kHalf + index] : partial[index];
    const float send = upper ? partial[index] : partial[kHalf + index];
    partial[index] = keep + __shfl_xor_sync(0xffffffff, send, kMask);
  }
  if constexpr (kHalf > 1) {
    ButterflyReduce<kHalf / 2, kMask * 2>(partial, lane);
  }
}

template <int kBits>
__device__ __forceinline__ int ReverseBits(int value) {
  int result = 0;
#pragma unroll
  for (int bit = 0; bit < kBits; ++bit) {
    result |= ((value >> bit) & 1) << (kBits - 1 - bit);
  }
  return result;
}

template <typename Index, bool kWriteCsrGradient>
__global__ __launch_bounds__(64) void CsrSpikeGradPairPackedBatch32Kernel(
    int n_pre, const Eigen::half* spikes, const Index* row_splits,
    const Index* edge_ids, const Index* pair_ids, const Eigen::half* weights,
    const float* projected, Eigen::half* spike_grad, float* weight_grad,
    const Eigen::half* spike_gradient_scale) {
  constexpr int kBatch = 32;
  constexpr int kPack = 4;
  constexpr int kWarps = 2;
  constexpr int kTile = 32;
  constexpr int kLanesPerEdge = kBatch / kPack;
  constexpr int kSlots = 32 / kLanesPerEdge;
  constexpr int kPerSlot = kTile / kSlots;
  constexpr int kIndexBits = 3;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int slot = lane / kLanesPerEdge;
  const int sub = lane % kLanesPerEdge;
  const int pre = blockIdx.x;
  if (pre >= n_pre) {
    return;
  }

  __shared__ float spike_partials[kWarps][32];
  const Index start = row_splits[pre];
  const Index end = row_splits[pre + 1];
  float spike[kPack];
  float pre_gradient[kPack] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
  for (int sample = 0; sample < kPack; ++sample) {
    spike[sample] = ToFloat(
        spikes[static_cast<int64_t>(sub * kPack + sample) * n_pre + pre]);
  }
  const int target = ReverseBits<kIndexBits>(sub & (kPerSlot - 1));

  for (Index base = start + static_cast<Index>(warp * kTile); base < end;
       base += static_cast<Index>(kWarps * kTile)) {
    const Index edge_lane = base + static_cast<Index>(lane);
    const bool own = lane < kTile && edge_lane < end;
    const Index my_pair = own ? pair_ids[edge_lane] : static_cast<Index>(0);
    const float my_weight =
        own ? ToFloat(weights[edge_lane]) : 0.0f;
    float partial[kPerSlot];
#pragma unroll
    for (int step = 0; step < kPerSlot; ++step) {
      const int column = kSlots * step + slot;
      const Index pair = static_cast<Index>(
          __shfl_sync(0xffffffff, static_cast<unsigned>(my_pair), column));
      const float weight = __shfl_sync(0xffffffff, my_weight, column);
      float values[kPack] = {0.0f, 0.0f, 0.0f, 0.0f};
      if (base + static_cast<Index>(column) < end) {
        const ::float4 raw = *reinterpret_cast<const ::float4*>(
            projected + static_cast<int64_t>(pair) * kBatch + sub * kPack);
        values[0] = raw.x;
        values[1] = raw.y;
        values[2] = raw.z;
        values[3] = raw.w;
      }
      float sum = 0.0f;
#pragma unroll
      for (int sample = 0; sample < kPack; ++sample) {
        pre_gradient[sample] += values[sample] * weight;
        sum += values[sample] * spike[sample];
      }
      partial[step] = sum;
    }
    ButterflyReduce<kPerSlot / 2, 1>(partial, lane);
#pragma unroll
    for (int mask = kPerSlot; mask < kLanesPerEdge; mask <<= 1) {
      partial[0] += __shfl_xor_sync(0xffffffff, partial[0], mask);
    }
    const Index edge =
        base + static_cast<Index>(kSlots * target + slot);
    if (sub < kPerSlot && edge < end) {
      const int64_t target_edge =
          kWriteCsrGradient ? static_cast<int64_t>(edge)
                            : static_cast<int64_t>(edge_ids[edge]);
      weight_grad[target_edge] = partial[0];
    }
  }

#pragma unroll
  for (int mask = kLanesPerEdge; mask < 32; mask <<= 1) {
#pragma unroll
    for (int sample = 0; sample < kPack; ++sample) {
      pre_gradient[sample] +=
          __shfl_xor_sync(0xffffffff, pre_gradient[sample], mask);
    }
  }
  if (lane < kLanesPerEdge) {
#pragma unroll
    for (int sample = 0; sample < kPack; ++sample) {
      spike_partials[warp][sub * kPack + sample] = pre_gradient[sample];
    }
  }
  __syncthreads();
  if (warp == 0) {
    float total = 0.0f;
#pragma unroll
    for (int source = 0; source < kWarps; ++source) {
      total += spike_partials[source][lane];
    }
    spike_grad[static_cast<int64_t>(lane) * n_pre + pre] =
        FromFloat<Eigen::half>(total * ToFloat(*spike_gradient_scale));
  }
}

inline bool SupportsPackedBatch32Backward() {
  int device = 0;
  int major = 0;
  int minor = 0;
  return cudaGetDevice(&device) == cudaSuccess &&
         cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor,
                                device) == cudaSuccess &&
         cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor,
                                device) == cudaSuccess &&
         major * 10 + minor >= 86;
}

inline uint32 PackedRowSplitCount(int64_t n_rows) {
  constexpr int64_t kTargetBlocks = 4512;
  if (n_rows <= 0) {
    return 1;
  }
  const int64_t splits = (kTargetBlocks + n_rows - 1) / n_rows;
  return static_cast<uint32>(
      std::min<int64_t>(64, std::max<int64_t>(1, splits)));
}

__global__ __launch_bounds__(64) void CsrWeightGradPairPackedBatch32Kernel(
    int64_t n_pre, const Eigen::half* spikes, const float* projected,
    const uint32* pair_ids, const uint32* edge_ids,
    const uint32* row_splits, uint32 splits, float* weight_grad) {
  constexpr int kBatch = 32;
  constexpr int kPack = 4;
  constexpr int kWarps = 2;
  constexpr int kTile = 32;
  constexpr int kLanesPerEdge = kBatch / kPack;
  constexpr int kSlots = 32 / kLanesPerEdge;
  constexpr int kPerSlot = kTile / kSlots;
  constexpr int kIndexBits = 3;
  constexpr uint32 kGrain = kWarps * kTile;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int slot = lane / kLanesPerEdge;
  const int sub = lane % kLanesPerEdge;
  const int64_t pre = blockIdx.x;
  if (pre >= n_pre) {
    return;
  }

  const uint64_t row_start = row_splits[pre];
  const uint64_t row_end = row_splits[pre + 1];
  const uint64_t chunks = (row_end - row_start + kGrain - 1) / kGrain;
  const uint64_t chunks_per_split = (chunks + splits - 1) / splits;
  const uint64_t start =
      row_start + static_cast<uint64_t>(blockIdx.y) * chunks_per_split * kGrain;
  if (start >= row_end) {
    return;
  }
  const uint64_t end = min(row_end, start + chunks_per_split * kGrain);

  float spike[kPack];
#pragma unroll
  for (int sample = 0; sample < kPack; ++sample) {
    spike[sample] = ToFloat(
        spikes[static_cast<int64_t>(sub * kPack + sample) * n_pre + pre]);
  }
  const int target = ReverseBits<kIndexBits>(sub & (kPerSlot - 1));

  for (uint64_t base = start + warp * kTile; base < end; base += kGrain) {
    const uint64_t edge_lane = base + lane;
    const bool own = edge_lane < end;
    const uint32 my_pair = own ? pair_ids[edge_lane] : 0;
    float partial[kPerSlot];
#pragma unroll
    for (int step = 0; step < kPerSlot; ++step) {
      const int column = kSlots * step + slot;
      const uint32 pair = __shfl_sync(0xffffffff, my_pair, column);
      float values[kPack] = {0.0f, 0.0f, 0.0f, 0.0f};
      if (base + column < end) {
        const ::float4 raw = *reinterpret_cast<const ::float4*>(
            projected + static_cast<int64_t>(pair) * kBatch + sub * kPack);
        values[0] = raw.x;
        values[1] = raw.y;
        values[2] = raw.z;
        values[3] = raw.w;
      }
      float sum = 0.0f;
#pragma unroll
      for (int sample = 0; sample < kPack; ++sample) {
        sum += values[sample] * spike[sample];
      }
      partial[step] = sum;
    }
    ButterflyReduce<kPerSlot / 2, 1>(partial, lane);
#pragma unroll
    for (int mask = kPerSlot; mask < kLanesPerEdge; mask <<= 1) {
      partial[0] += __shfl_xor_sync(0xffffffff, partial[0], mask);
    }
    const uint64_t edge = base + kSlots * target + slot;
    if (sub < kPerSlot && edge < end) {
      weight_grad[static_cast<int64_t>(edge_ids[edge])] = partial[0];
    }
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
class DpointnetCsrRestoreOp : public OpKernel {
 public:
  explicit DpointnetCsrRestoreOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("n_edges", &n_edges_));
    OP_REQUIRES_OK(context, context->GetAttr("n_sources", &n_sources_));
    OP_REQUIRES_OK(context, context->GetAttr("n_pairs", &n_pairs_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& values = context->input(0);
    core::RefCountPtr<Var> metadata_variable;
    if (!LookupVariable<Index>(context, 1, "metadata", &metadata_variable)) {
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

    Tensor* restored = nullptr;
    OP_REQUIRES_OK(
        context, context->allocate_output(0, values.shape(), &restored));
    const int64_t count = values.NumElements();
    if (count == 0) {
      return;
    }
    const GPUDevice& device = context->eigen_device<GPUDevice>();
    constexpr int threads = 256;
    OP_REQUIRES_OK(
        context,
        GpuLaunchKernel(
            CsrRestoreKernel<T, Index>, BlockCountFor(count, threads, device),
            threads, 0, device.stream(), count, values.flat<T>().data(),
            edge_ids, restored->flat<T>().data()));
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
    OP_REQUIRES_OK(
      context,
      context->GetAttr(
        "use_grouped_batch32_forward", &use_grouped_batch32_forward_));
    OP_REQUIRES_OK(
        context,
        context->GetAttr("use_fixed4_forward", &use_fixed4_forward_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& spikes = context->input(0);
    const Tensor& master_weights = context->input(1);
    const Tensor& weights = context->input(3);
    const Tensor& basis = context->input(4);
    const Tensor& spike_gradient_scale = context->input(5);
    const Tensor& active_rows = context->input(6);
    const Tensor& incoming_pre_ids = context->input(7);
    const Tensor& incoming_edge_ids = context->input(8);
    const Tensor& incoming_types = context->input(9);
    const Tensor& initial = context->input(10);
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
    RequireVector(context, active_rows, "active_rows");
    RequireVector(context, incoming_pre_ids, "incoming_pre_ids");
    RequireVector(context, incoming_edge_ids, "incoming_edge_ids");
    RequireVector(context, incoming_types, "incoming_types");
    RequireVector(context, metadata, "metadata");
    if (!context->status().ok()) {
      return;
    }

    const int64_t batch = spikes.dim_size(0);
    const int64_t n_pre = spikes.dim_size(1);
    const int64_t n_basis = basis.dim_size(1);
    const TensorShape output_shape({batch * n_post_, n_basis});
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
    OP_REQUIRES(
      context, initial.NumElements() == 0 || initial.shape() == output_shape,
      errors::InvalidArgument(
        "initial currents must be empty or match the forward output."));
    if (use_fixed4_forward_) {
      OP_REQUIRES(
        context, n_basis == 4,
        errors::InvalidArgument(
          "Fixed-four forward requires four synaptic bases."));
      OP_REQUIRES(
        context,
          incoming_pre_ids.NumElements() == static_cast<int64_t>(4) * n_post_ &&
            incoming_edge_ids.NumElements() ==
              static_cast<int64_t>(4) * n_post_ &&
            incoming_types.NumElements() ==
              static_cast<int64_t>(4) * n_post_,
        errors::InvalidArgument(
          "Fixed-four forward requires exactly four incoming edges per "
          "postsynaptic neuron."));
    }

    const Index* metadata_values = metadata.flat<Index>().data();
    const Index* post_ids = metadata_values;
    const Index* synapse_types = metadata_values + n_edges_;
    const Index* row_splits = metadata_values + 2 * n_edges_;
    Tensor* currents = nullptr;
    OP_REQUIRES_OK(
        context,
      context->forward_input_or_allocate_output(
        {10}, 0, output_shape, &currents));
    const GPUDevice& device = context->eigen_device<GPUDevice>();
    const int64_t output_count = currents->NumElements();
    if (use_fixed4_forward_) {
      const int64_t work_count = batch * n_post_;
      constexpr int threads = 256;
      OP_REQUIRES_OK(
          context,
          GpuLaunchKernel(
              CsrSpikeForwardFixed4Kernel<T, Index>,
              BlockCountFor(work_count, threads, device), threads, 0,
              device.stream(), work_count, static_cast<int>(n_pre), n_post_,
              n_edges_, static_cast<int>(basis.dim_size(0)),
              spikes.flat<T>().data(), weights.flat<T>().data(),
              incoming_pre_ids.flat<Index>().data(),
              incoming_edge_ids.flat<Index>().data(),
              incoming_types.flat<Index>().data(), basis.flat<T>().data(),
              initial.NumElements() > 0 ? initial.flat<T>().data() : nullptr,
              currents->flat<T>().data()));
      return;
    }
    if (initial.NumElements() > 0 &&
        currents->flat<T>().data() != initial.flat<T>().data()) {
      OP_REQUIRES(
        context,
        cudaMemcpyAsync(
          currents->flat<T>().data(), initial.flat<T>().data(),
          output_count * sizeof(T), cudaMemcpyDeviceToDevice,
          device.stream()) == cudaSuccess,
        errors::Internal("Failed to copy initial current buffer."));
    }
    if (initial.NumElements() == 0) {
      constexpr int zero_threads = 256;
      OP_REQUIRES_OK(
        context,
        GpuLaunchKernel(
          SetZeroKernel<T>,
          BlockCountFor(output_count, zero_threads, device),
          zero_threads, 0, device.stream(), output_count,
          currents->flat<T>().data()));
    }

    if (use_grouped_batch32_forward_) {
      OP_REQUIRES(
          context, batch == 32 && n_basis == 4,
          errors::InvalidArgument(
              "Grouped forward requires runtime batch 32 and four "
              "synaptic bases."));
      const int64_t n_active_rows = active_rows.NumElements();
      if (n_active_rows > 0) {
        OP_REQUIRES_OK(
            context,
            GpuLaunchKernel(
                CsrSpikeForwardGroupedBatch32Kernel<T, Index>,
                static_cast<int>(n_active_rows), 128, 0, device.stream(),
                n_active_rows, n_pre, n_post_, spikes.flat<T>().data(),
                active_rows.flat<int64_t>().data(), post_ids,
                weights.flat<T>().data(), synapse_types, basis.flat<T>().data(),
                row_splits, currents->flat<T>().data()));
      }
    } else {
      const int work_count = static_cast<int>(batch * n_pre);
      OP_REQUIRES_OK(
          context,
          GpuLaunchKernel(
              CsrSpikeForwardKernel<T, Index>, work_count, 128, 0,
              device.stream(), work_count, static_cast<int>(n_pre), n_post_,
              static_cast<int>(n_basis), spikes.flat<T>().data(), post_ids,
              weights.flat<T>().data(), synapse_types, basis.flat<T>().data(),
              row_splits, currents->flat<T>().data()));
    }
  }

 private:
  int n_post_;
  int64_t n_edges_;
  int64_t n_pairs_;
  bool use_grouped_batch32_forward_;
  bool use_fixed4_forward_;
};

template <typename T, typename Index>
class DpointnetCsrSpikeGradOp : public OpKernel {
 public:
  explicit DpointnetCsrSpikeGradOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("n_post", &n_post_));
    OP_REQUIRES_OK(context, context->GetAttr("n_edges", &n_edges_));
    OP_REQUIRES_OK(context, context->GetAttr("n_pairs", &n_pairs_));
    OP_REQUIRES_OK(
        context,
        context->GetAttr(
            "use_packed_sm120_backward", &use_packed_sm120_backward_));
    OP_REQUIRES_OK(
      context,
      context->GetAttr(
        "write_csr_weight_gradient", &write_csr_weight_gradient_));
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
    if (use_packed_sm120_backward_) {
      OP_REQUIRES(
        context, batch == 32 && n_basis == 4 && n_pairs_ > 0,
        errors::InvalidArgument(
          "Packed SM120 recurrent backward requires batch 32, four "
          "synaptic bases, and compact pair metadata."));
      if constexpr (
        !std::is_same<T, Eigen::half>::value ||
        !std::is_same<Index, uint32>::value) {
      OP_REQUIRES(
        context, false,
        errors::InvalidArgument(
          "Packed SM120 recurrent backward requires float16 compute "
          "values and uint32 CSR metadata."));
      } else {
      OP_REQUIRES(
        context, SupportsPackedBatch32Backward(),
        errors::InvalidArgument(
          "Packed recurrent backward requires GPU compute capability "
          "SM86 or newer."));
      }
    }
      OP_REQUIRES(
        context,
        !write_csr_weight_gradient_ || use_packed_sm120_backward_,
        errors::InvalidArgument(
          "CSR-ordered gradients require packed recurrent backward."));

    const Index* metadata_values = metadata.flat<Index>().data();
    const Index* post_ids = metadata_values;
    const Index* synapse_types = metadata_values + n_edges_;
    const Index* row_splits = metadata_values + 2 * n_edges_;
    const Index* edge_ids = row_splits + n_pre + 1;
    const Index* pair_ids = n_pairs_ > 0 ? edge_ids + n_edges_ : nullptr;
    const Index* pair_posts = n_pairs_ > 0 ? pair_ids + n_edges_ : nullptr;
    const Index* pair_types = n_pairs_ > 0 ? pair_posts + n_pairs_ : nullptr;
    Tensor* spike_grad = nullptr;
    Tensor* weight_grad = nullptr;
    OP_REQUIRES_OK(
        context,
        context->allocate_output(0, spikes.shape(), &spike_grad));
    OP_REQUIRES_OK(
        context,
        context->allocate_output(1, weights.shape(), &weight_grad));
    const GPUDevice& device = context->eigen_device<GPUDevice>();
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
      if constexpr (
          std::is_same<T, Eigen::half>::value &&
          std::is_same<Index, uint32>::value) {
        if (use_packed_sm120_backward_) {
          if (write_csr_weight_gradient_) {
            OP_REQUIRES_OK(
                context,
                GpuLaunchKernel(
                    CsrSpikeGradPairPackedBatch32Kernel<Index, true>,
                    static_cast<int>(n_pre), 64, 0, device.stream(),
                    static_cast<int>(n_pre), spikes.flat<T>().data(), row_splits,
                    edge_ids, pair_ids, weights.flat<T>().data(),
                    projected.flat<float>().data(), spike_grad->flat<T>().data(),
                    weight_grad->flat<float>().data(),
                    spike_gradient_scale.flat<T>().data()));
          } else {
            OP_REQUIRES_OK(
                context,
                GpuLaunchKernel(
                    CsrSpikeGradPairPackedBatch32Kernel<Index, false>,
                    static_cast<int>(n_pre), 64, 0, device.stream(),
                    static_cast<int>(n_pre), spikes.flat<T>().data(), row_splits,
                    edge_ids, pair_ids, weights.flat<T>().data(),
                    projected.flat<float>().data(), spike_grad->flat<T>().data(),
                    weight_grad->flat<float>().data(),
                    spike_gradient_scale.flat<T>().data()));
          }
        } else {
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
        }
      } else {
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
      }
    } else {
      constexpr int zero_threads = 256;
      OP_REQUIRES_OK(
          context,
          GpuLaunchKernel(
              SetZeroKernel<float>,
              BlockCountFor(n_edges_, zero_threads, device), zero_threads, 0,
              device.stream(), n_edges_, weight_grad->flat<float>().data()));
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
  bool use_packed_sm120_backward_;
  bool write_csr_weight_gradient_;
};

template <typename T, typename Index>
class DpointnetCsrWeightGradOp : public OpKernel {
 public:
  explicit DpointnetCsrWeightGradOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("n_post", &n_post_));
    OP_REQUIRES_OK(context, context->GetAttr("n_edges", &n_edges_));
    OP_REQUIRES_OK(context, context->GetAttr("n_pairs", &n_pairs_));
    OP_REQUIRES_OK(
      context,
      context->GetAttr(
        "use_packed_sm120_backward", &use_packed_sm120_backward_));
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
    if (use_packed_sm120_backward_) {
      OP_REQUIRES(
          context, batch == 32 && n_basis == 4 && n_pairs_ > 0,
          errors::InvalidArgument(
              "Packed SM120 external backward requires batch 32, four "
              "synaptic bases, and compact pair metadata."));
      if constexpr (
          !std::is_same<T, Eigen::half>::value ||
          !std::is_same<Index, uint32>::value) {
        OP_REQUIRES(
            context, false,
            errors::InvalidArgument(
                "Packed SM120 external backward requires float16 compute "
                "values and uint32 CSR metadata."));
      } else {
        OP_REQUIRES(
            context, SupportsPackedBatch32Backward(),
            errors::InvalidArgument(
                "Packed external backward requires GPU compute capability "
              "SM86 or newer."));
      }
    }

    const Index* metadata_values = metadata.flat<Index>().data();
    const Index* post_ids = metadata_values;
    const Index* synapse_types = metadata_values + n_edges_;
    const Index* row_splits = metadata_values + 2 * n_edges_;
    const Index* edge_ids = row_splits + n_pre + 1;
    const Index* pair_ids = n_pairs_ > 0 ? edge_ids + n_edges_ : nullptr;
    const Index* pair_posts = n_pairs_ > 0 ? pair_ids + n_edges_ : nullptr;
    const Index* pair_types = n_pairs_ > 0 ? pair_posts + n_pairs_ : nullptr;
    Tensor* weight_grad = nullptr;
    OP_REQUIRES_OK(
        context,
        context->allocate_output(
            0, TensorShape({n_edges_}), &weight_grad));
    const GPUDevice& device = context->eigen_device<GPUDevice>();
    if constexpr (
      std::is_same<T, Eigen::half>::value &&
      std::is_same<Index, uint32>::value) {
      if (use_packed_sm120_backward_) {
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
        const uint32 splits = PackedRowSplitCount(n_pre);
        OP_REQUIRES_OK(
            context,
            GpuLaunchKernel(
                CsrWeightGradPairPackedBatch32Kernel,
                dim3(static_cast<unsigned>(n_pre), splits), 64, 0,
                device.stream(), n_pre, spikes.flat<T>().data(),
                projected.flat<float>().data(), pair_ids, edge_ids, row_splits,
                splits, weight_grad->flat<float>().data()));
        return;
      }
    }
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
  bool use_packed_sm120_backward_;
};

#define REGISTER_GPU_KERNELS(T, Index)                                    \
  REGISTER_KERNEL_BUILDER(                                               \
      Name("DpointnetCsrReorder")                                        \
          .Device(DEVICE_GPU)                                            \
          .TypeConstraint<T>("T")                                        \
          .TypeConstraint<Index>("Tindex"),                              \
      DpointnetCsrReorderOp<T, Index>);                                  \
        REGISTER_KERNEL_BUILDER(                                               \
          Name("DpointnetCsrRestore")                                        \
            .Device(DEVICE_GPU)                                            \
            .TypeConstraint<T>("T")                                        \
            .TypeConstraint<Index>("Tindex"),                              \
          DpointnetCsrRestoreOp<T, Index>);                                  \
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
