#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include <algorithm>
#include <limits>
#include <type_traits>

#include <cub/cub.cuh>
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/resource_var.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"

namespace tensorflow {

using GPUDevice = Eigen::GpuDevice;

constexpr int kActiveQueueRowBits = 24;
constexpr int64_t kActiveQueueMaxRows = int64_t{1} << kActiveQueueRowBits;
constexpr int64_t kActiveQueueMaxBatch = int64_t{1} << (32 - kActiveQueueRowBits);
constexpr int kDpointnetForwardThreads = 256;
constexpr int kDpointnetForwardBlocks = 4512;

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

__global__ void CopyFloatKernel(int64_t count, const float* source, float* target) {
  for (int64_t index : GpuGridRangeX(count)) {
    target[index] = source[index];
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
#if __CUDA_ARCH__ >= 700
  atomicAdd(
      reinterpret_cast<__half*>(address),
      __float2half(ToFloat(value)));
#else
  GpuAtomicAdd(address, value);
#endif
}

template <typename T>
__device__ inline void FastAtomicAddPair(T* address, float first, float second) {
  FastAtomicAdd(address, FromFloat<T>(first));
  FastAtomicAdd(address + 1, FromFloat<T>(second));
}

template <>
__device__ inline void FastAtomicAddPair<Eigen::half>(
    Eigen::half* address, float first, float second) {
#if __CUDA_ARCH__ >= 600
  atomicAdd(
      reinterpret_cast<__half2*>(address),
      __floats2half2_rn(first, second));
#else
  GpuAtomicAdd(address, FromFloat<Eigen::half>(first));
  GpuAtomicAdd(address + 1, FromFloat<Eigen::half>(second));
#endif
}

__device__ inline unsigned int MatchWarpValue(unsigned int value) {
#if __CUDA_ARCH__ >= 700
  return __match_any_sync(0xffffffffu, value);
#else
  const int lane = threadIdx.x & 31;
  unsigned int group = 0;
#pragma unroll
  for (int source_lane = 0; source_lane < 32; ++source_lane) {
    const unsigned int candidate = __shfl_sync(0xffffffffu, value, source_lane);
    const unsigned int matches = __ballot_sync(0xffffffffu, value == candidate);
    if (lane == source_lane) group = matches;
  }
  return group;
#endif
}

struct PackActiveSlot {
  int64_t n_pre;
  __host__ __device__ unsigned int operator()(int64_t index) const {
    return static_cast<unsigned int>(
        ((index / n_pre) << kActiveQueueRowBits) | (index % n_pre));
  }
};

template <typename T>
struct SlotIsActive {
  const T* spikes;
  __host__ __device__ bool operator()(int64_t index) const {
    return static_cast<float>(spikes[index]) > 0.0f;
  }
};

template <typename T>
Status BuildActiveSlotQueue(
    OpKernelContext* context, const T* spikes, int64_t slots, int64_t n_pre,
    unsigned int* queue, unsigned int* queue_count) {
  const cub::CountingInputIterator<int64_t> indices(0);
  const cub::TransformInputIterator<
      unsigned int, PackActiveSlot, cub::CountingInputIterator<int64_t>>
      packed(indices, PackActiveSlot{n_pre});
  const cub::TransformInputIterator<
      bool, SlotIsActive<T>, cub::CountingInputIterator<int64_t>>
      active(indices, SlotIsActive<T>{spikes});
  auto stream = context->eigen_device<GPUDevice>().stream();
  size_t scratch_bytes = 0;
  cub::DeviceSelect::Flagged(
      nullptr, scratch_bytes, packed, active, queue, queue_count, slots, stream);
  Tensor scratch;
  TF_RETURN_IF_ERROR(context->allocate_temp(
      DT_INT8, TensorShape({static_cast<int64_t>(scratch_bytes)}), &scratch));
  if (cub::DeviceSelect::Flagged(
          scratch.flat<int8>().data(), scratch_bytes, packed, active, queue,
          queue_count, slots, stream) != cudaSuccess) {
    return errors::Internal("building the active slot queue failed");
  }
  return OkStatus();
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
  int64_t n_active_rows, int64_t n_pre, int n_post, int batch_size,
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
      threadIdx.x < batch_size &&
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
__global__ void CsrSpikeForwardGroupedBatch32AggregateRunsKernel(
  int64_t n_active_rows, int64_t n_pre, int n_post, int batch_size,
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
      threadIdx.x < batch_size &&
      ToFloat(spikes[static_cast<int64_t>(threadIdx.x) * n_pre + pre]) >
      0.0f;
    const uint32 mask = __ballot_sync(0xffffffff, active);
    if (threadIdx.x == 0) {
      batch_mask = mask;
    }
  }
  __syncthreads();

  const Index start = row_splits[pre];
  const Index end = row_splits[pre + 1];
  for (Index edge = start + threadIdx.x; edge < end; edge += blockDim.x) {
    const int post = static_cast<int>(post_ids[edge]);
    if (edge != start && post_ids[edge - 1] == post_ids[edge]) {
      continue;
    }
    Index run_end = edge + 1;
    while (run_end < end && post_ids[run_end] == post_ids[edge]) {
      ++run_end;
    }
    uint32 remaining = batch_mask;
    while (remaining != 0) {
      const int batch = __ffs(remaining) - 1;
      remaining &= remaining - 1;
      const float spike =
          ToFloat(spikes[static_cast<int64_t>(batch) * n_pre + pre]);
      float sums[4] = {0.0f, 0.0f, 0.0f, 0.0f};
      for (Index item = edge; item < run_end; ++item) {
        const float weighted = spike * ToFloat(weights[item]);
        const int synapse_type = static_cast<int>(synapse_types[item]);
#pragma unroll
        for (int receptor = 0; receptor < 4; ++receptor) {
          sums[receptor] +=
              weighted * ToFloat(basis[synapse_type * 4 + receptor]);
        }
      }
      T* output = currents +
          (static_cast<int64_t>(batch) * n_post + post) * 4;
      if constexpr (std::is_same<T, Eigen::half>::value) {
        atomicAdd(
            reinterpret_cast<__half2*>(output),
            __floats2half2_rn(sums[0], sums[1]));
        atomicAdd(
            reinterpret_cast<__half2*>(output + 2),
            __floats2half2_rn(sums[2], sums[3]));
      } else {
#pragma unroll
        for (int receptor = 0; receptor < 4; ++receptor) {
          FastAtomicAdd(output + receptor, FromFloat<T>(sums[receptor]));
        }
      }
    }
  }
}

// Javier forward-path aggregation, adapted from
// v1_model_utils/cuda_csr_recurrent/csr_recurrent_ops.cu.cc at commit
// 2c52ec10. Credit: Javier. DPointNet keeps its canonical/CSR mapping and
// opt-in dispatch, but uses the same warp run matching for repeated LGN
// targets instead of the earlier serial run-leader loop.
template <int kValues>
__device__ __forceinline__ void DpointnetForwardRunSuffixSums(
    float (&values)[kValues], unsigned int group, int lane) {
#pragma unroll
  for (int offset = 1; offset < 32; offset <<= 1) {
    const int source = lane + offset;
    const bool take = source < 32 && ((group >> source) & 1u);
#pragma unroll
    for (int index = 0; index < kValues; ++index) {
      const float other = __shfl_down_sync(0xffffffffu, values[index], offset);
      if (take) values[index] += other;
    }
  }
}

template <typename T, typename Index, bool kAggregateRuns>
__global__ void CsrSpikeForwardDeviceQueueKernel(
    int64_t n_pre, int n_post, const T* spikes, const unsigned int* queue,
    const unsigned int* queue_count, unsigned int* ticket,
    unsigned int slots_per_ticket, const Index* post_ids, const T* weights,
    const Index* synapse_types, const T* basis, const Index* row_splits,
    T* currents) {
  __shared__ unsigned int next_slot;
  const unsigned int total = *queue_count;

  for (;;) {
    if (threadIdx.x == 0) {
      next_slot = atomicAdd(ticket, slots_per_ticket);
    }
    __syncthreads();
    const unsigned int first_slot = next_slot;
    __syncthreads();
    if (first_slot >= total) {
      break;
    }
    const unsigned int last_slot = min(first_slot + slots_per_ticket, total);
    for (unsigned int active_id = first_slot; active_id < last_slot; ++active_id) {
      const unsigned int packed = queue[active_id];
      const int64_t batch = packed >> kActiveQueueRowBits;
      const int64_t pre = packed & ((1u << kActiveQueueRowBits) - 1);
      const float spike = ToFloat(spikes[batch * n_pre + pre]);
      const Index start = row_splits[pre];
      const Index end = row_splits[pre + 1];

      if constexpr (!kAggregateRuns) {
        for (Index edge = start + threadIdx.x; edge < end; edge += blockDim.x) {
          const float weighted = spike * ToFloat(weights[edge]);
          const int post = static_cast<int>(post_ids[edge]);
          const int synapse_type = static_cast<int>(synapse_types[edge]);
          T* output = currents +
              (batch * static_cast<int64_t>(n_post) + post) * 4;
          FastAtomicAddPair(
              output,
              weighted * ToFloat(basis[synapse_type * 4]),
              weighted * ToFloat(basis[synapse_type * 4 + 1]));
          FastAtomicAddPair(
              output + 2,
              weighted * ToFloat(basis[synapse_type * 4 + 2]),
              weighted * ToFloat(basis[synapse_type * 4 + 3]));
        }
      } else {
        const int lane = threadIdx.x & 31;
        const int warp = threadIdx.x >> 5;
        const int warps = blockDim.x >> 5;
        for (Index base = start + static_cast<Index>(warp * 32);
             base < end; base += static_cast<Index>(warps * 32)) {
          const Index edge = base + static_cast<Index>(lane);
          const bool valid = edge < end;
          const unsigned int post =
              valid ? static_cast<unsigned int>(post_ids[edge]) : 0xffffffffu;
          const float weighted =
              valid ? spike * ToFloat(weights[edge]) : 0.0f;
          const int synapse_type =
              valid ? static_cast<int>(synapse_types[edge]) : 0;
          const unsigned int group = MatchWarpValue(post);
          const bool any_run =
              !__all_sync(0xffffffffu, __popc(group) == 1);
          const bool leader = valid && lane == __ffs(group) - 1;
          const int basis_base = synapse_type * 4;
          float values[4] = {
              weighted * ToFloat(basis[basis_base]),
              weighted * ToFloat(basis[basis_base + 1]),
              weighted * ToFloat(basis[basis_base + 2]),
              weighted * ToFloat(basis[basis_base + 3])};
          if (any_run) {
            DpointnetForwardRunSuffixSums(values, group, lane);
          }
          if (leader) {
            T* output = currents +
                (batch * static_cast<int64_t>(n_post) + post) * 4;
            FastAtomicAddPair(output, values[0], values[1]);
            FastAtomicAddPair(output + 2, values[2], values[3]);
          }
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
    float* weight_grad, const T* spike_gradient_scale,
    bool write_csr_weight_gradient, const float* projected,
    const Index* pair_ids, int batch_size) {
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
    if (projected != nullptr) {
      edge_gradient = projected[static_cast<int64_t>(pair_ids[edge]) * batch_size + batch];
    } else {
      for (int receptor = 0; receptor < n_basis; ++receptor) {
        edge_gradient +=
            ToFloat(current_grad[gradient_base + receptor]) *
            ToFloat(basis[basis_base + receptor]);
      }
    }
    pre_gradient += edge_gradient * ToFloat(weights[edge]);
    if (spike > 0.0f) {
      GpuAtomicAdd(
          weight_grad + (write_csr_weight_gradient
                             ? edge : static_cast<int64_t>(edge_ids[edge])),
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
__global__ void CsrSpikeGradSmallBatchKernel(
    int batch_size, int n_pre, int n_post, int n_basis, const T* spikes,
    const T* current_grad, const Index* post_ids, const T* weights,
    const Index* synapse_types, const T* basis, const Index* row_splits,
    const Index* edge_ids, T* spike_grad, float* weight_grad,
    const T* spike_gradient_scale, bool write_csr_weight_gradient,
    const float* projected, const Index* pair_ids) {
  const int pre = blockIdx.x;
  if (pre >= n_pre) return;
  float pre_gradients[8] = {};
  float spike_values[8];
  for (int sample = 0; sample < batch_size; ++sample) {
    spike_values[sample] = ToFloat(spikes[static_cast<int64_t>(sample) * n_pre + pre]);
  }
  for (int64_t edge = static_cast<int64_t>(row_splits[pre]) + threadIdx.x;
       edge < static_cast<int64_t>(row_splits[pre + 1]); edge += blockDim.x) {
    const int post = static_cast<int>(post_ids[edge]);
    const int type = static_cast<int>(synapse_types[edge]);
    const float weight = ToFloat(weights[edge]);
    float weight_gradient = 0.0f;
    for (int sample = 0; sample < batch_size; ++sample) {
      const int64_t base = (static_cast<int64_t>(sample) * n_post + post) * n_basis;
      float projection = 0.0f;
      if (projected != nullptr) {
        projection = projected[static_cast<int64_t>(pair_ids[edge]) * batch_size + sample];
      } else {
        for (int receptor = 0; receptor < n_basis; ++receptor) {
          projection += ToFloat(current_grad[base + receptor]) * ToFloat(basis[type * n_basis + receptor]);
        }
      }
      pre_gradients[sample] += projection * weight;
      if (spike_values[sample] > 0.0f) weight_gradient += projection * spike_values[sample];
    }
    weight_grad[write_csr_weight_gradient ? edge : static_cast<int64_t>(edge_ids[edge])] = weight_gradient;
  }
  __shared__ float partial[128];
  for (int sample = 0; sample < batch_size; ++sample) {
    partial[threadIdx.x] = pre_gradients[sample];
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
      if (threadIdx.x < offset) partial[threadIdx.x] += partial[threadIdx.x + offset];
      __syncthreads();
    }
    if (threadIdx.x == 0) spike_grad[static_cast<int64_t>(sample) * n_pre + pre] = FromFloat<T>(partial[0] * ToFloat(*spike_gradient_scale));
    __syncthreads();
  }
}

template <typename T, typename Index>
__global__ void PairProjectionGeneralKernel(
    int64_t count, int batch_size, int n_post, int n_basis,
    const T* current_grad, const T* basis, const Index* pair_posts,
    const Index* pair_types, float* projected) {
  for (int64_t index : GpuGridRangeX(count)) {
    const int sample = static_cast<int>(index % batch_size);
    const int64_t pair = index / batch_size;
    const int64_t base = (static_cast<int64_t>(sample) * n_post + pair_posts[pair]) * n_basis;
    const int64_t basis_base = static_cast<int64_t>(pair_types[pair]) * n_basis;
    float value = 0.0f;
    for (int receptor = 0; receptor < n_basis; ++receptor) {
      value += ToFloat(current_grad[base + receptor]) * ToFloat(basis[basis_base + receptor]);
    }
    projected[index] = value;
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

template <typename T>
__global__ void DpointnetAbsMaxFiniteKernel(int64_t count, const T* values,
                                            unsigned int* result) {
  float local = 0.0f;
  for (int64_t index : GpuGridRangeX(count)) {
    const float value = fabsf(ToFloat(values[index]));
    if (isfinite(value) && value > local) local = value;
  }
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1) {
    local = fmaxf(local, __shfl_xor_sync(0xffffffffu, local, mask));
  }
  if ((threadIdx.x & 31) == 0 && local > 0.0f) {
    atomicMax(result, __float_as_uint(local));
  }
}

template <typename T>
__global__ void DpointnetProjectionScaleKernel(
    const unsigned int* max_bits, const T* basis, int n_types, int n_basis,
    float* scale_out) {
  float basis_l1 = 0.0f;
  for (int type = threadIdx.x; type < n_types; type += 32) {
    float row = 0.0f;
    for (int receptor = 0; receptor < n_basis; ++receptor) {
      row += fabsf(ToFloat(basis[type * n_basis + receptor]));
    }
    basis_l1 = fmaxf(basis_l1, row);
  }
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    basis_l1 = fmaxf(basis_l1, __shfl_xor_sync(0xffffffffu, basis_l1, offset));
  }
  if (threadIdx.x == 0) {
    const float bound = __uint_as_float(*max_bits) * basis_l1;
    float scale = 1.0f;
    if (isfinite(bound) && bound > 0.0f) {
      scale = exp2f(floorf(log2f(8192.0f / bound)));
      if (!isfinite(scale) || scale <= 0.0f) scale = 1.0f;
    }
    scale_out[0] = scale;
    scale_out[1] = 1.0f / scale;
  }
}

template <typename T, typename Index, int kSlice = 32>
__global__ void DpointnetPreprojectPairsHalfBatch32Kernel(
  int64_t n_pairs, int n_post, int batch, const T* current_grad, const T* basis,
    const Index* pair_posts, const Index* pair_types, Eigen::half* projected,
    const float* scale) {
  constexpr int kPairsPerTile = 32;
  __shared__ float tile[kSlice][kPairsPerTile + 2];
  const int64_t pair_base = static_cast<int64_t>(blockIdx.x) * kPairsPerTile;
  for (int index = threadIdx.x; index < kSlice * kPairsPerTile; index += blockDim.x) {
    const int pair_offset = index % kPairsPerTile;
    const int sample = index / kPairsPerTile;
    const int64_t pair = pair_base + pair_offset;
    float value = 0.0f;
    if (pair < n_pairs && sample < batch) {
      const int post = static_cast<int>(pair_posts[pair]);
      const int type = static_cast<int>(pair_types[pair]);
      const int64_t gradient_base = (static_cast<int64_t>(sample) * n_post + post) * 4;
      const int basis_base = type * 4;
#pragma unroll
      for (int receptor = 0; receptor < 4; ++receptor) {
        value += ToFloat(current_grad[gradient_base + receptor]) *
                 ToFloat(basis[basis_base + receptor]);
      }
    }
    tile[sample][pair_offset] = value;
  }
  __syncthreads();
  const float factor = scale[0];
  for (int index = threadIdx.x; index < kSlice * kPairsPerTile; index += blockDim.x) {
    const int sample = index % kSlice;
    const int pair_offset = index / kSlice;
    const int64_t pair = pair_base + pair_offset;
    if (pair < n_pairs) {
      projected[pair * kSlice + sample] = FromFloat<Eigen::half>(
          tile[sample][pair_offset] * factor);
    }
  }
}

template <typename T, typename Index, int kSlice = 32>
__global__ __launch_bounds__(128) void DpointnetSpikeGradRowPerWarpHalfBatch32Kernel(
  int n_pre, int batch, const Index* row_splits, const Index* pair_ids, const T* weights,
    const Eigen::half* projected, T* spike_grad, const T* spike_gradient_scale,
    const float* inverse_scale, Index sentinel_pair) {
  constexpr int kPack = kSlice < 8 ? kSlice : 8;
  constexpr int kLanesPerEdge = kSlice / kPack;
  constexpr int kSlots = 32 / kLanesPerEdge;
  constexpr int kRows = 4;
  const int lane = threadIdx.x & 31;
  const int row_lane = threadIdx.x >> 5;
  const int slot = lane / kLanesPerEdge;
  const int sub = lane % kLanesPerEdge;
  const int pre = blockIdx.x * kRows + row_lane;
  if (pre >= n_pre) return;
  const int sample_base = sub * kPack;
  float grad[kPack] = {};
  const Index start = row_splits[pre];
  const Index end = row_splits[pre + 1];
  for (Index base = start; base < end; base += static_cast<Index>(32)) {
    const Index edge_lane = base + static_cast<Index>(lane);
    const bool own = edge_lane < end;
    const Index my_pair = own ? pair_ids[edge_lane] : sentinel_pair;
    const float my_weight = own ? ToFloat(weights[edge_lane]) : 0.0f;
#pragma unroll
    for (int step = 0; step < kLanesPerEdge; ++step) {
      const int column = kSlots * step + slot;
      const Index pair = static_cast<Index>(
          __shfl_sync(0xffffffffu, static_cast<unsigned>(my_pair), column));
      const float weight = __shfl_sync(0xffffffffu, my_weight, column);
        const Eigen::half* packed =
          projected + static_cast<int64_t>(pair) * kSlice + sample_base;
      float values[kPack];
      if constexpr (kPack == 8) {
        const ::uint4 raw = *reinterpret_cast<const ::uint4*>(packed);
        const Eigen::half* vector_values =
            reinterpret_cast<const Eigen::half*>(&raw);
#pragma unroll
        for (int sample = 0; sample < kPack; ++sample) {
          values[sample] = ToFloat(vector_values[sample]);
        }
      } else {
#pragma unroll
        for (int sample = 0; sample < kPack; ++sample) {
          values[sample] = ToFloat(packed[sample]);
        }
      }
#pragma unroll
      for (int sample = 0; sample < kPack; ++sample) {
        grad[sample] += values[sample] * weight;
      }
    }
  }
#pragma unroll
  for (int mask = kLanesPerEdge; mask < 32; mask <<= 1) {
#pragma unroll
    for (int sample = 0; sample < kPack; ++sample) {
      grad[sample] += __shfl_xor_sync(0xffffffffu, grad[sample], mask);
    }
  }
  if (lane < kLanesPerEdge) {
    const float factor = inverse_scale[1] * ToFloat(*spike_gradient_scale);
#pragma unroll
    for (int sample = 0; sample < kPack; ++sample) {
      if (sample_base + sample < batch) {
        spike_grad[static_cast<int64_t>(sample_base + sample) * n_pre + pre] =
            FromFloat<T>(grad[sample] * factor);
      }
    }
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

template <typename T, typename Index, int kSlice>
__global__ void CsrSpikeGradPairVariableBatchKernel(
    int batch, int n_pre, const Index* row_splits,
    const Index* pair_ids, const T* weights, const float* projected,
    T* spike_grad, const T* spike_gradient_scale) {
  constexpr int kPack = 4;
  constexpr int kLanesPerEdge = kSlice / kPack;
  constexpr int kSlots = 32 / kLanesPerEdge;
  const int lane = threadIdx.x & 31;
  const int pre = blockIdx.x * 4 + (threadIdx.x >> 5);
  if (pre >= n_pre) return;
  const int slot = lane / kLanesPerEdge;
  const int sample_base = (lane % kLanesPerEdge) * kPack;
  float gradients[kPack] = {};
  for (Index edge = row_splits[pre] + slot; edge < row_splits[pre + 1];
       edge += kSlots) {
    const float weight = ToFloat(weights[edge]);
    const int64_t base = static_cast<int64_t>(pair_ids[edge]) * batch;
#pragma unroll
    for (int sample = 0; sample < kPack; ++sample) {
      if (sample_base + sample < batch) {
        gradients[sample] += projected[base + sample_base + sample] * weight;
      }
    }
  }
#pragma unroll
  for (int mask = kLanesPerEdge; mask < 32; mask <<= 1) {
#pragma unroll
    for (int sample = 0; sample < kPack; ++sample) {
      gradients[sample] += __shfl_xor_sync(0xffffffffu, gradients[sample], mask);
    }
  }
  if (lane < kLanesPerEdge) {
#pragma unroll
    for (int sample = 0; sample < kPack; ++sample) {
      if (sample_base + sample < batch) {
        spike_grad[static_cast<int64_t>(sample_base + sample) * n_pre + pre] =
            FromFloat<T>(gradients[sample] * ToFloat(*spike_gradient_scale));
      }
    }
  }
}

// Event-sparse recurrent weight-gradient path adapted from Javier's
// `v1_model_utils/cuda_csr_recurrent/event_weight_grad.cuh` at commit
// 2c52ec10. Credit: Javier. The DPointNet port keeps DPointNet's CSR metadata
// layout and compute-dtype basis, and adds into the already-initialized FP32
// CSR accumulator/output.
constexpr int kDpointnetEventThreads = 256;
constexpr int kDpointnetEventChunk = 4 * kDpointnetEventThreads;
constexpr int kDpointnetEventBlocks = 4512;

template <typename T, typename Index, bool kPositiveOnly = false>
__global__ __launch_bounds__(kDpointnetEventThreads)
void DpointnetEventRowQueueKernel(
    int64_t n_pre, int64_t batch, const T* activity, const Index* row_splits,
    unsigned int* queue, unsigned int* queue_count) {
  const int lane = threadIdx.x & 31;
  const int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  unsigned int chunks = 0;
  unsigned int mask = 0;
  if (row < n_pre) {
#pragma unroll
    for (int sample = 0; sample < 32; ++sample) {
      if (sample < batch) {
        const float activity_value =
            ToFloat(activity[static_cast<int64_t>(sample) * n_pre + row]);
        mask |= static_cast<unsigned int>(
              kPositiveOnly ? activity_value > 0.0f : activity_value != 0.0f)
                << sample;
      }
    }
    if (mask != 0) {
      const unsigned int edges = static_cast<unsigned int>(
          row_splits[row + 1] - row_splits[row]);
      chunks = (edges + kDpointnetEventChunk - 1) / kDpointnetEventChunk;
    }
  }
  unsigned int inclusive = chunks;
#pragma unroll
  for (int offset = 1; offset < 32; offset <<= 1) {
    const unsigned int other = __shfl_up_sync(0xffffffffu, inclusive, offset);
    if (lane >= offset) inclusive += other;
  }
  const unsigned int total = __shfl_sync(0xffffffffu, inclusive, 31);
  if (total == 0) return;
  unsigned int base = 0;
  if (lane == 31) base = atomicAdd(queue_count, total);
  base = __shfl_sync(0xffffffffu, base, 31) + inclusive - chunks;
  for (unsigned int chunk = 0; chunk < chunks; ++chunk) {
    reinterpret_cast<::uint4*>(queue)[base + chunk] =
        ::make_uint4(static_cast<unsigned int>(row), chunk, mask, 0u);
  }
}

template <typename T, int kBasis>
__device__ __forceinline__ float DpointnetEventProjection(
    const T* grad, const T* type_basis, int n_basis) {
  float result = 0.0f;
#pragma unroll
  for (int receptor = 0; receptor < (kBasis == 0 ? n_basis : kBasis); ++receptor) {
    result += ToFloat(type_basis[receptor]) * ToFloat(grad[receptor]);
  }
  return result;
}

template <typename T, typename Index, int kBasis, bool kWriteCsrGradient>
__global__ __launch_bounds__(kDpointnetEventThreads)
void DpointnetEventWeightGradKernel(
    int64_t n_pre, int n_post, int n_basis, int64_t batch, const T* activity,
    const T* current_grad, const T* basis, const Index* post_ids,
    const Index* synapse_types, const Index* row_splits, const Index* edge_ids,
    const unsigned int* queue, const unsigned int* queue_count, float* weight_grad) {
  constexpr int kEdgesPerThread = kDpointnetEventChunk / kDpointnetEventThreads;
  const int64_t sample_stride = static_cast<int64_t>(n_post) * n_basis;
  const unsigned int total = *queue_count;
  for (unsigned int item = blockIdx.x; item < total; item += gridDim.x) {
    const ::uint4 entry = reinterpret_cast<const ::uint4*>(queue)[item];
    const int64_t pre = entry.x;
    const Index start =
        row_splits[pre] + static_cast<Index>(entry.y * kDpointnetEventChunk);
    const Index end = min(
        static_cast<Index>(start + kDpointnetEventChunk), row_splits[pre + 1]);
    float sum[kEdgesPerThread] = {};
#pragma unroll
    for (int k = 0; k < kEdgesPerThread; ++k) {
      const Index csr = start + static_cast<Index>(threadIdx.x + k * kDpointnetEventThreads);
      if (csr >= end) continue;
      const int post = static_cast<int>(post_ids[csr]);
      const int type = static_cast<int>(synapse_types[csr]);
      const T* type_basis = basis + static_cast<int64_t>(type) * n_basis;
      for (unsigned int bits = entry.z; bits; bits &= bits - 1) {
        const int sample = __ffs(bits) - 1;
        const T spike = activity[static_cast<int64_t>(sample) * n_pre + pre];
        const T* grad_row =
            current_grad + static_cast<int64_t>(sample) * sample_stride +
            static_cast<int64_t>(post) * n_basis;
        sum[k] += ToFloat(spike) *
                  DpointnetEventProjection<T, kBasis>(grad_row, type_basis, n_basis);
      }
    }
#pragma unroll
    for (int k = 0; k < kEdgesPerThread; ++k) {
      const Index csr = start + static_cast<Index>(threadIdx.x + k * kDpointnetEventThreads);
      if (csr < end) {
        const int64_t target = kWriteCsrGradient
            ? static_cast<int64_t>(csr)
            : static_cast<int64_t>(edge_ids[csr]);
        weight_grad[target] = __fadd_rn(weight_grad[target], sum[k]);
      }
    }
  }
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

template <typename T, typename Index, bool kWriteCsrGradient, bool kAccumulate = false>
__global__ __launch_bounds__(64) void CsrSpikeGradPairPackedBatch32Kernel(
    int n_pre, const T* spikes, const Index* row_splits,
    const Index* edge_ids, const Index* pair_ids, const T* weights,
    const float* projected, T* spike_grad, float* weight_grad,
    const T* spike_gradient_scale, const float* accumulator) {
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
  const bool compute_weight_gradient = weight_grad != nullptr;
  float spike[kPack];
  float pre_gradient[kPack] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
  for (int sample = 0; sample < kPack; ++sample) {
    spike[sample] = ToFloat(
        spikes[static_cast<int64_t>(sub * kPack + sample) * n_pre + pre]);
  }
  // Reuse the FP32 register packing, not half arithmetic. Silent source rows
  // still need spike adjoints, but need no batch reduction for weight adjoints.
  bool active = true;
  if constexpr (std::is_same<T, float>::value && kWriteCsrGradient) {
    if (compute_weight_gradient) {
      active = __any_sync(0xffffffff,
          spike[0] > 0.0f || spike[1] > 0.0f ||
          spike[2] > 0.0f || spike[3] > 0.0f);
    }
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
        if (compute_weight_gradient) {
          if constexpr (std::is_same<T, float>::value && kWriteCsrGradient) {
            // Match the generic direct-CSR FP32 backward's positive-spike gate.
            // The older canonical pair path multiplies spikes without that gate.
            if (spike[sample] > 0.0f) {
              sum += values[sample] * spike[sample];
            }
          } else {
            sum += values[sample] * spike[sample];
          }
        }
      }
      if (compute_weight_gradient) partial[step] = sum;
    }
    if (compute_weight_gradient) {
      if (!active) {
        if (own) {
          const int64_t target_edge = kWriteCsrGradient
              ? static_cast<int64_t>(edge_lane)
              : static_cast<int64_t>(edge_ids[edge_lane]);
          weight_grad[target_edge] =
              kAccumulate ? __fadd_rn(accumulator[target_edge], 0.0f) : 0.0f;
        }
        continue;
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
        weight_grad[target_edge] = kAccumulate
            ? __fadd_rn(accumulator[target_edge], partial[0]) : partial[0];
      }
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
        FromFloat<T>(total * ToFloat(*spike_gradient_scale));
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
         major * 10 + minor >= 61;
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
    OP_REQUIRES_OK(
        context,
        context->GetAttr(
            "use_forward_run_aggregation", &use_forward_run_aggregation_));
    OP_REQUIRES_OK(
        context,
        context->GetAttr(
            "use_device_active_queue_forward", &use_device_active_queue_forward_));
    OP_REQUIRES_OK(context, context->GetAttr("vjp_only", &vjp_only_));
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
    if (vjp_only_) {
      OP_REQUIRES(context, (std::is_same<T, float>::value && initial.NumElements() == 0),
                  errors::InvalidArgument("VJP-only projection requires FP32 operands and no initial currents."));
      constexpr int threads = 256;
      OP_REQUIRES_OK(context, GpuLaunchKernel(
          SetZeroKernel<T>, BlockCountFor(output_count, threads, device),
          threads, 0, device.stream(), output_count, currents->flat<T>().data()));
      return;
    }
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
          context, batch >= 1 && batch <= 32 && n_basis == 4,
          errors::InvalidArgument(
              "Grouped forward requires runtime batch 1..32 and four "
              "synaptic bases."));
      if (use_device_active_queue_forward_) {
        OP_REQUIRES(
            context, n_pre > 0 && n_pre <= kActiveQueueMaxRows &&
              batch < kActiveQueueMaxBatch,
            errors::InvalidArgument(
                "Device active-queue forward cannot pack this batch/source "
                "shape."));
        const int64_t slots = batch * n_pre;
        Tensor queue_tensor;
        OP_REQUIRES_OK(
            context,
            context->allocate_temp(
                DT_UINT32, TensorShape({slots + 2}), &queue_tensor));
        unsigned int* queue = queue_tensor.flat<uint32>().data();
        unsigned int* queue_count = queue + slots;
        OP_REQUIRES(
            context,
            cudaMemsetAsync(
                queue_count, 0, 2 * sizeof(unsigned int),
                device.stream()) == cudaSuccess,
            errors::Internal("Failed to initialize active queue counters."));
        OP_REQUIRES_OK(
            context,
            BuildActiveSlotQueue<T>(
                context, spikes.flat<T>().data(), slots, n_pre, queue,
                queue_count));
        const int64_t mean_edges = std::max<int64_t>(1, n_edges_ / n_pre);
        const unsigned int slots_per_ticket =
            static_cast<unsigned int>(
                std::max<int64_t>(1, std::min<int64_t>(8, 1024 / mean_edges)));
        if (use_forward_run_aggregation_) {
          OP_REQUIRES_OK(
              context,
              GpuLaunchKernel(
                  CsrSpikeForwardDeviceQueueKernel<T, Index, true>,
                  kDpointnetForwardBlocks, kDpointnetForwardThreads, 0,
                  device.stream(), n_pre, n_post_,
                  spikes.flat<T>().data(), queue, queue_count, queue_count + 1,
                  slots_per_ticket, post_ids, weights.flat<T>().data(),
                  synapse_types, basis.flat<T>().data(), row_splits,
                  currents->flat<T>().data()));
        } else {
          OP_REQUIRES_OK(
              context,
              GpuLaunchKernel(
                  CsrSpikeForwardDeviceQueueKernel<T, Index, false>,
                  kDpointnetForwardBlocks, kDpointnetForwardThreads, 0,
                  device.stream(), n_pre, n_post_,
                  spikes.flat<T>().data(), queue, queue_count, queue_count + 1,
                  slots_per_ticket, post_ids, weights.flat<T>().data(),
                  synapse_types, basis.flat<T>().data(), row_splits,
                  currents->flat<T>().data()));
        }
      } else {
        const int64_t n_active_rows = active_rows.NumElements();
        if (n_active_rows > 0) {
        if (use_forward_run_aggregation_) {
          OP_REQUIRES_OK(
              context,
              GpuLaunchKernel(
                  CsrSpikeForwardGroupedBatch32AggregateRunsKernel<T, Index>,
                  static_cast<int>(n_active_rows), 128, 0, device.stream(),
                  n_active_rows, n_pre, n_post_, static_cast<int>(batch),
                  spikes.flat<T>().data(), active_rows.flat<int64_t>().data(),
                  post_ids, weights.flat<T>().data(), synapse_types,
                  basis.flat<T>().data(), row_splits,
                  currents->flat<T>().data()));
        } else {
          OP_REQUIRES_OK(
              context,
              GpuLaunchKernel(
                  CsrSpikeForwardGroupedBatch32Kernel<T, Index>,
                  static_cast<int>(n_active_rows), 128, 0, device.stream(),
                  n_active_rows, n_pre, n_post_, static_cast<int>(batch), spikes.flat<T>().data(),
                  active_rows.flat<int64_t>().data(), post_ids,
                  weights.flat<T>().data(), synapse_types, basis.flat<T>().data(),
                  row_splits, currents->flat<T>().data()));
        }
        }
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
  bool use_forward_run_aggregation_;
  bool use_device_active_queue_forward_;
  bool vjp_only_;
};

template <typename T, typename Index, bool kAccumulate = false>
class DpointnetCsrSpikeGradOp : public OpKernel {
 public:
  explicit DpointnetCsrSpikeGradOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("n_post", &n_post_));
    OP_REQUIRES_OK(context, context->GetAttr("n_edges", &n_edges_));
    OP_REQUIRES_OK(context, context->GetAttr("n_pairs", &n_pairs_));
    OP_REQUIRES_OK(context, context->GetAttr("use_small_batch_backward", &use_small_batch_backward_));
    OP_REQUIRES_OK(
        context,
        context->GetAttr(
            "use_packed_sm120_backward", &use_packed_sm120_backward_));
    OP_REQUIRES_OK(
      context,
      context->GetAttr(
        "write_csr_weight_gradient", &write_csr_weight_gradient_));
    use_javier_batch32_backward_ = false;
    (void)context->GetAttr(
        "use_javier_batch32_backward", &use_javier_batch32_backward_);
    compute_spike_gradient_ = true;
    if constexpr (kAccumulate) {
      OP_REQUIRES_OK(
          context, context->GetAttr("compute_spike_gradient", &compute_spike_gradient_));
    }
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
    if constexpr (kAccumulate) {
      OP_REQUIRES(
          context,
          compute_spike_gradient_ ||
              (use_javier_batch32_backward_ && std::is_same<T, Eigen::half>::value),
          errors::InvalidArgument(
              "Weight-only accumulation requires FP16 Javier event-weight backward."));
      const Tensor& accumulator = context->input(6);
      OP_REQUIRES(
          context, TensorShapeUtils::IsVector(accumulator.shape()) &&
              accumulator.NumElements() == n_edges_,
          errors::InvalidArgument("Accumulator must have one FP32 value per CSR edge."));
      OP_REQUIRES(
          context, batch >= 1 && batch <= 32 && n_basis == 4 && n_pairs_ > 0 &&
              write_csr_weight_gradient_ && !use_small_batch_backward_ &&
              !use_packed_sm120_backward_ && SupportsPackedBatch32Backward(),
          errors::InvalidArgument(
              "Fused accumulation requires SM61+, batch1..32/four bases, pairs, "
              "direct CSR and generic packed backward flags."));
    }
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
          "SM61 or newer."));
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
    Tensor* spike_grad = nullptr;
    Tensor* weight_grad = nullptr;
    OP_REQUIRES_OK(
        context,
        context->allocate_output(0, spikes.shape(), &spike_grad));
    const float* accumulator = nullptr;
    if constexpr (kAccumulate) {
      accumulator = context->input(6).flat<float>().data();
      OP_REQUIRES_OK(context, context->forward_input_or_allocate_output(
          {6}, 1, weights.shape(), &weight_grad));
    } else {
      OP_REQUIRES_OK(
          context, context->allocate_output(1, weights.shape(), &weight_grad));
    }
    const GPUDevice& device = context->eigen_device<GPUDevice>();
    if constexpr ((kAccumulate || std::is_same<T, float>::value) &&
                  std::is_same<Index, uint32>::value) {
        if (!use_small_batch_backward_ &&
          (batch == 32 || (use_javier_batch32_backward_ && batch > 0 && batch < 32)) &&
          n_basis == 4 && n_pairs_ > 0) {
        if (use_javier_batch32_backward_ && write_csr_weight_gradient_) {
          OP_REQUIRES(
              context, kAccumulate,
              errors::InvalidArgument(
                  "Javier batch32 recurrent backward is currently supported "
                  "only for FP16 compute with the FP32 accumulator/direct-CSR path."));
          Tensor projected_half;
          int slice = 1;
          while (slice < batch) slice *= 2;
          Tensor scale_tensor;
          float* scale = nullptr;
          if (compute_spike_gradient_) {
          const int64_t projected_count = (n_pairs_ + 1) * slice;
          OP_REQUIRES_OK(context, context->allocate_temp(
              DT_HALF, TensorShape({projected_count}), &projected_half));
          OP_REQUIRES_OK(context, context->allocate_temp(
              DT_FLOAT, TensorShape({3}), &scale_tensor));
          scale = scale_tensor.flat<float>().data();
          unsigned int* max_bits = reinterpret_cast<unsigned int*>(scale + 2);
          OP_REQUIRES_OK(context, GpuLaunchKernel(
              SetZeroKernel<Eigen::half>, 1, 32, 0, device.stream(), slice,
              projected_half.flat<Eigen::half>().data() + n_pairs_ * slice));
          OP_REQUIRES_OK(context, GpuLaunchKernel(
              SetZeroKernel<uint32>, 1, 1, 0, device.stream(), 1,
              reinterpret_cast<uint32*>(max_bits)));
          OP_REQUIRES_OK(context, GpuLaunchKernel(
              DpointnetAbsMaxFiniteKernel<T>, 1024, 256, 0, device.stream(),
              current_grad.NumElements(), current_grad.flat<T>().data(), max_bits));
          OP_REQUIRES_OK(context, GpuLaunchKernel(
              DpointnetProjectionScaleKernel<T>, 1, 32, 0, device.stream(),
              max_bits, basis.flat<T>().data(), static_cast<int>(basis.dim_size(0)),
              static_cast<int>(n_basis), scale));
#define LAUNCH_HALF_PROJECTION(SLICE) \
          OP_REQUIRES_OK(context, GpuLaunchKernel( \
              DpointnetPreprojectPairsHalfBatch32Kernel<T, Index, SLICE>, \
              static_cast<int>((n_pairs_ + 31) / 32), 128, 0, device.stream(), \
              n_pairs_, n_post_, static_cast<int>(batch), current_grad.flat<T>().data(), \
              basis.flat<T>().data(), pair_posts, pair_types, \
              projected_half.flat<Eigen::half>().data(), scale))
          switch (slice) {
            case 1: LAUNCH_HALF_PROJECTION(1); break;
            case 2: LAUNCH_HALF_PROJECTION(2); break;
            case 4: LAUNCH_HALF_PROJECTION(4); break;
            case 8: LAUNCH_HALF_PROJECTION(8); break;
            case 16: LAUNCH_HALF_PROJECTION(16); break;
            default: LAUNCH_HALF_PROJECTION(32); break;
          }
#undef LAUNCH_HALF_PROJECTION
          } else {
            OP_REQUIRES_OK(context, GpuLaunchKernel(
                SetZeroKernel<T>,
                BlockCountFor(spikes.NumElements(), 256, device), 256, 0,
                device.stream(), spikes.NumElements(), spike_grad->flat<T>().data()));
          }
          if (weight_grad->flat<float>().data() != accumulator) {
            OP_REQUIRES(
                context,
                cudaMemcpyAsync(
                    weight_grad->flat<float>().data(), accumulator,
                    n_edges_ * sizeof(float), cudaMemcpyDeviceToDevice,
                    device.stream()) == cudaSuccess,
                errors::Internal(
                    "Failed to seed recurrent weight accumulator."));
          }
          if (n_pre > 0) {
            if (compute_spike_gradient_) {
#define LAUNCH_HALF_ROWS(SLICE) \
            OP_REQUIRES_OK(context, GpuLaunchKernel( \
                DpointnetSpikeGradRowPerWarpHalfBatch32Kernel<T, Index, SLICE>, \
                static_cast<int>((n_pre + 3) / 4), 128, 0, device.stream(), \
                static_cast<int>(n_pre), static_cast<int>(batch), row_splits, pair_ids, \
                weights.flat<T>().data(), projected_half.flat<Eigen::half>().data(), \
                spike_grad->flat<T>().data(), spike_gradient_scale.flat<T>().data(), \
                scale, static_cast<Index>(n_pairs_)))
            switch (slice) {
              case 1: LAUNCH_HALF_ROWS(1); break;
              case 2: LAUNCH_HALF_ROWS(2); break;
              case 4: LAUNCH_HALF_ROWS(4); break;
              case 8: LAUNCH_HALF_ROWS(8); break;
              case 16: LAUNCH_HALF_ROWS(16); break;
              default: LAUNCH_HALF_ROWS(32); break;
            }
#undef LAUNCH_HALF_ROWS
            }
            const int64_t capacity =
                n_pre + n_edges_ / kDpointnetEventChunk + 1;
            Tensor queue_tensor;
            OP_REQUIRES_OK(context, context->allocate_temp(
                DT_UINT32, TensorShape({4 * capacity + 1}), &queue_tensor));
            unsigned int* queue =
                reinterpret_cast<unsigned int*>(queue_tensor.flat<uint32>().data());
            unsigned int* queue_count = queue + 4 * capacity;
            OP_REQUIRES_OK(context, GpuLaunchKernel(
                SetZeroKernel<uint32>, 1, 1, 0, device.stream(), 1,
                reinterpret_cast<uint32*>(queue_count)));
            OP_REQUIRES_OK(context, GpuLaunchKernel(
                DpointnetEventRowQueueKernel<T, Index, false>,
                static_cast<int>((n_pre + kDpointnetEventThreads - 1) /
                                 kDpointnetEventThreads),
                kDpointnetEventThreads, 0, device.stream(), n_pre, batch,
                spikes.flat<T>().data(), row_splits, queue, queue_count));
            OP_REQUIRES_OK(context, GpuLaunchKernel(
                DpointnetEventWeightGradKernel<T, Index, 4, true>,
                kDpointnetEventBlocks, kDpointnetEventThreads, 0,
                device.stream(), n_pre, n_post_, static_cast<int>(n_basis),
                batch, spikes.flat<T>().data(), current_grad.flat<T>().data(),
                basis.flat<T>().data(), post_ids, synapse_types, row_splits,
                edge_ids, queue, queue_count,
                weight_grad->flat<float>().data()));
          }
          return;
        }
        Tensor projected;
        const int64_t count = n_pairs_ * 32;
        OP_REQUIRES_OK(context, context->allocate_temp(
            DT_FLOAT, TensorShape({count}), &projected));
        OP_REQUIRES_OK(context, GpuLaunchKernel(
            PairProjectionBatch32Kernel<T, Index>,
            BlockCountFor(count, 256, device), 256, 0, device.stream(),
            count, n_post_, current_grad.flat<T>().data(),
            basis.flat<T>().data(), pair_posts, pair_types,
            projected.flat<float>().data()));
        if (n_pre > 0) {
          if (write_csr_weight_gradient_) {
            OP_REQUIRES_OK(context, GpuLaunchKernel(
                CsrSpikeGradPairPackedBatch32Kernel<T, Index, true, kAccumulate>,
                static_cast<int>(n_pre), 64, 0, device.stream(),
                static_cast<int>(n_pre), spikes.flat<T>().data(), row_splits,
                edge_ids, pair_ids, weights.flat<T>().data(),
                projected.flat<float>().data(), spike_grad->flat<T>().data(),
                weight_grad->flat<float>().data(),
                spike_gradient_scale.flat<T>().data(), accumulator));
          } else {
            OP_REQUIRES_OK(context, GpuLaunchKernel(
                CsrSpikeGradPairPackedBatch32Kernel<T, Index, false>,
                static_cast<int>(n_pre), 64, 0, device.stream(),
                static_cast<int>(n_pre), spikes.flat<T>().data(), row_splits,
                edge_ids, pair_ids, weights.flat<T>().data(),
                projected.flat<float>().data(), spike_grad->flat<T>().data(),
                weight_grad->flat<float>().data(),
                spike_gradient_scale.flat<T>().data(), nullptr));
          }
        }
        return;
      }
    }
    Tensor general_projected;
    const float* general_projection = nullptr;
    if (n_pairs_ > 0 && (use_small_batch_backward_ || batch != 32 || n_basis != 4 ||
              (write_csr_weight_gradient_ && !use_packed_sm120_backward_))) {
      const int64_t count = n_pairs_ * batch;
      OP_REQUIRES_OK(context, context->allocate_temp(DT_FLOAT, TensorShape({count}), &general_projected));
      OP_REQUIRES_OK(context, GpuLaunchKernel(
        PairProjectionGeneralKernel<T, Index>, BlockCountFor(count, 256, device), 256, 0,
        device.stream(), count, static_cast<int>(batch), n_post_, static_cast<int>(n_basis),
        current_grad.flat<T>().data(), basis.flat<T>().data(), pair_posts, pair_types,
        general_projected.flat<float>().data()));
      general_projection = general_projected.flat<float>().data();
    }
    if (!use_small_batch_backward_ && batch > 0 && batch < 32 &&
        general_projection != nullptr) {
      if constexpr (kAccumulate) {
        if (weight_grad->flat<float>().data() != accumulator) {
          OP_REQUIRES(context, cudaMemcpyAsync(
              weight_grad->flat<float>().data(), accumulator,
              n_edges_ * sizeof(float), cudaMemcpyDeviceToDevice,
              device.stream()) == cudaSuccess,
              errors::Internal("Failed to seed recurrent weight accumulator."));
        }
      } else {
        OP_REQUIRES_OK(context, GpuLaunchKernel(
            SetZeroKernel<float>, BlockCountFor(n_edges_, 256, device), 256, 0,
            device.stream(), n_edges_, weight_grad->flat<float>().data()));
      }
      if (n_pre > 0) {
  #define LAUNCH_VARIABLE_BATCH(SLICE) \
      OP_REQUIRES_OK(context, GpuLaunchKernel( \
        CsrSpikeGradPairVariableBatchKernel<T, Index, SLICE>, \
        static_cast<int>((n_pre + 3) / 4), 128, 0, device.stream(), \
        static_cast<int>(batch), static_cast<int>(n_pre), row_splits, pair_ids, \
        weights.flat<T>().data(), general_projection, \
        spike_grad->flat<T>().data(), spike_gradient_scale.flat<T>().data()))
      if (batch <= 4) { LAUNCH_VARIABLE_BATCH(4); }
      else if (batch <= 8) { LAUNCH_VARIABLE_BATCH(8); }
      else if (batch <= 16) { LAUNCH_VARIABLE_BATCH(16); }
      else { LAUNCH_VARIABLE_BATCH(32); }
  #undef LAUNCH_VARIABLE_BATCH
      const int64_t capacity = n_pre + n_edges_ / kDpointnetEventChunk + 1;
      Tensor queue_tensor;
      OP_REQUIRES_OK(context, context->allocate_temp(
        DT_UINT32, TensorShape({4 * capacity + 1}), &queue_tensor));
      unsigned int* queue = reinterpret_cast<unsigned int*>(
        queue_tensor.flat<uint32>().data());
      unsigned int* queue_count = queue + 4 * capacity;
      OP_REQUIRES_OK(context, GpuLaunchKernel(
        SetZeroKernel<uint32>, 1, 1, 0, device.stream(), 1,
        reinterpret_cast<uint32*>(queue_count)));
      OP_REQUIRES_OK(context, GpuLaunchKernel(
        DpointnetEventRowQueueKernel<T, Index, true>,
        static_cast<int>((n_pre + kDpointnetEventThreads - 1) /
                 kDpointnetEventThreads),
        kDpointnetEventThreads, 0, device.stream(), n_pre, batch,
        spikes.flat<T>().data(), row_splits, queue, queue_count));
  #define LAUNCH_VARIABLE_WEIGHTS(CSR) \
      OP_REQUIRES_OK(context, GpuLaunchKernel( \
        DpointnetEventWeightGradKernel<T, Index, 0, CSR>, \
        kDpointnetEventBlocks, kDpointnetEventThreads, 0, device.stream(), \
        n_pre, n_post_, static_cast<int>(n_basis), batch, \
        spikes.flat<T>().data(), current_grad.flat<T>().data(), \
        basis.flat<T>().data(), post_ids, synapse_types, row_splits, edge_ids, \
        queue, queue_count, weight_grad->flat<float>().data()))
      if (write_csr_weight_gradient_) { LAUNCH_VARIABLE_WEIGHTS(true); }
      else { LAUNCH_VARIABLE_WEIGHTS(false); }
  #undef LAUNCH_VARIABLE_WEIGHTS
      }
      return;
    }
    if (use_small_batch_backward_) {
      OP_REQUIRES(context, batch >= 1 && batch <= 8,
                  errors::InvalidArgument("Small-batch backward requires batch size 1..8."));
      if (n_pre > 0) {
        OP_REQUIRES_OK(context, GpuLaunchKernel(
            CsrSpikeGradSmallBatchKernel<T, Index>, static_cast<int>(n_pre), 128, 0,
            device.stream(), static_cast<int>(batch), static_cast<int>(n_pre), n_post_,
            static_cast<int>(n_basis), spikes.flat<T>().data(), current_grad.flat<T>().data(),
            post_ids, weights.flat<T>().data(), synapse_types, basis.flat<T>().data(),
            row_splits, edge_ids, spike_grad->flat<T>().data(), weight_grad->flat<float>().data(),
            spike_gradient_scale.flat<T>().data(), write_csr_weight_gradient_,
            general_projection, pair_ids));
      }
    } else if (batch == 32 && n_basis == 4 && n_pairs_ > 0 &&
      (!write_csr_weight_gradient_ || use_packed_sm120_backward_)) {
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
                    CsrSpikeGradPairPackedBatch32Kernel<T, Index, true>,
                    static_cast<int>(n_pre), 64, 0, device.stream(),
                    static_cast<int>(n_pre), spikes.flat<T>().data(), row_splits,
                    edge_ids, pair_ids, weights.flat<T>().data(),
                    projected.flat<float>().data(), spike_grad->flat<T>().data(),
                    weight_grad->flat<float>().data(),
                    spike_gradient_scale.flat<T>().data(), nullptr));
          } else {
            OP_REQUIRES_OK(
                context,
                GpuLaunchKernel(
                    CsrSpikeGradPairPackedBatch32Kernel<T, Index, false>,
                    static_cast<int>(n_pre), 64, 0, device.stream(),
                    static_cast<int>(n_pre), spikes.flat<T>().data(), row_splits,
                    edge_ids, pair_ids, weights.flat<T>().data(),
                    projected.flat<float>().data(), spike_grad->flat<T>().data(),
                    weight_grad->flat<float>().data(),
                    spike_gradient_scale.flat<T>().data(), nullptr));
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
          spike_gradient_scale.flat<T>().data(), write_csr_weight_gradient_,
          general_projection, pair_ids, static_cast<int>(batch)));
    }
  }

 private:
  int n_post_;
  int64_t n_edges_;
  int64_t n_pairs_;
  bool use_packed_sm120_backward_;
  bool write_csr_weight_gradient_;
  bool use_javier_batch32_backward_;
  bool use_small_batch_backward_;
  bool compute_spike_gradient_;
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
              "SM61 or newer."));
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

REGISTER_KERNEL_BUILDER(
    Name("DpointnetCsrSpikeGradAccumulate")
        .Device(DEVICE_GPU).TypeConstraint<float>("T")
        .TypeConstraint<uint32>("Tindex"),
    DpointnetCsrSpikeGradOp<float, uint32, true>);
REGISTER_KERNEL_BUILDER(
    Name("DpointnetCsrSpikeGradAccumulate")
        .Device(DEVICE_GPU).TypeConstraint<Eigen::half>("T")
        .TypeConstraint<uint32>("Tindex"),
    DpointnetCsrSpikeGradOp<Eigen::half, uint32, true>);

#undef REGISTER_GPU_KERNELS_FOR_TYPE
#undef REGISTER_GPU_KERNELS

}  // namespace tensorflow

#endif  // GOOGLE_CUDA
