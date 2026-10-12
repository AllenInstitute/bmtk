#if GOOGLE_CUDA
#define EIGEN_USE_GPU

#include <algorithm>
#include <type_traits>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"

using namespace tensorflow;
using GPUDevice = Eigen::GpuDevice;

template <typename T>
__device__ __forceinline__ float ToFloat(T value) {
  return static_cast<float>(value);
}

template <typename T>
__device__ __forceinline__ T FromFloat(float value) {
  return static_cast<T>(value);
}

template <typename T>
__device__ __forceinline__ T NestAdd(T left, T right) {
  return FromFloat<T>(__fadd_rn(ToFloat(left), ToFloat(right)));
}

template <typename T>
__device__ __forceinline__ T NestMul(T left, T right) {
  return FromFloat<T>(__fmul_rn(ToFloat(left), ToFloat(right)));
}

template <>
__device__ __forceinline__ Eigen::half NestAdd(Eigen::half left, Eigen::half right) {
  return Eigen::half(__hadd_rn(static_cast<__half>(left), static_cast<__half>(right)));
}

template <>
__device__ __forceinline__ Eigen::half NestMul(Eigen::half left, Eigen::half right) {
  return Eigen::half(__hmul_rn(static_cast<__half>(left), static_cast<__half>(right)));
}

// Only coefficient addresses change; the state arithmetic below is shared.
template <typename T, bool SoA>
struct NestCoefficients {
  const T* values;
  int neurons;
  int neuron;

  __device__ __forceinline__ T operator[](int column) const {
    return values[SoA ? static_cast<int64>(column) * neurons + neuron
                      : static_cast<int64>(neuron) * 28 + column];
  }
};

template <typename T>
struct NestTypeCoefficients {
  const T* values;
  int type;

  __device__ __forceinline__ T operator[](int column) const {
    return values[static_cast<int64>(type) * 28 + column];
  }
};

// State/history fusion structure adapted from Javier's
// v1_model_utils/cuda_glif_state/glif_state_ops.cu.cc at commit 2c52ec10
// (credit: Javier): keep the 1-D batch*neuron launch, derive the spike while
// threshold voltage is still in registers, and shift delay history in the same
// pass. DPointNet keeps its NEST coefficients, event map, dtype boundaries and
// pre-reset voltage output contract.
template <typename T, typename R, typename S, bool SoA = false,
          bool EmitPreResetVoltage = false, bool History = false>
__global__ void NestForwardKernel(
    int64 count, int neurons, const T* voltage, const R* refractory,
    const T* asc, const S* rise, const S* psc, const S* currents,
    const T* coefficients, const R* t_ref, const T* dt, const T* v_th,
    bool hard_reset, T* threshold, T* new_v, R* new_r, T* new_asc,
    S* new_rise, S* new_psc, int64 history_width, const S* history,
    S* spikes, S* new_history) {
  GPU_1D_KERNEL_LOOP(index, count) {
    const int neuron = index % neurons;
    const NestCoefficients<T, SoA> params{coefficients, neurons, neuron};
    const bool active = refractory[index] <= 0;
    const T mean_asc = NestAdd(NestMul(asc[index * 2], params[14]),
                               NestMul(asc[index * 2 + 1], params[15]));
    T contributions[4];
    #pragma unroll
    for (int basis = 0; basis < 4; ++basis) {
      const int64 offset = index * 4 + basis;
      contributions[basis] = NestAdd(NestMul(static_cast<T>(psc[offset]), params[18 + basis]),
                                      NestMul(static_cast<T>(rise[offset]), params[22 + basis]));
      new_rise[offset] = static_cast<S>(NestAdd(NestMul(static_cast<T>(rise[offset]), params[basis]),
                                  NestMul(static_cast<T>(currents[offset]), params[4 + basis])));
      new_psc[offset] = static_cast<S>(NestAdd(NestMul(static_cast<T>(psc[offset]), params[basis]),
                                 NestMul(NestMul(*dt, params[basis]), static_cast<T>(rise[offset]))));
    }
    const T integrated = NestAdd(NestAdd(contributions[0], contributions[1]),
                                   NestAdd(contributions[2], contributions[3]));
    const T retained = NestAdd(FromFloat<T>(1.0f), FromFloat<T>(-ToFloat(params[27])));
    const T dampened_voltage = NestAdd(NestMul(voltage[index], retained),
                       NestMul(voltage[index], params[27]));
    const T candidate = NestAdd(
        NestMul(params[12], dampened_voltage),
        NestAdd(NestMul(params[13], mean_asc), integrated));
    const T before_reset = hard_reset && !active ? params[26] : candidate;
    const T threshold_value = NestAdd(before_reset, FromFloat<T>(-ToFloat(*v_th)));
    const bool fired = active && ToFloat(threshold_value) > 0.0f;
    if constexpr (History) {
      const int64 base = (index / neurons) * history_width + neuron;
      const S spike = FromFloat<S>(fired ? 1.0f : 0.0f);
      spikes[index] = spike;
      new_history[base] = spike;
      for (int64 delay = neurons; delay < history_width; delay += neurons) {
        new_history[base + delay] = history[base + delay - neurons];
      }
    }
    if constexpr (EmitPreResetVoltage) {
      threshold[index] = before_reset;
    } else {
      threshold[index] = threshold_value;
    }
    new_v[index] = hard_reset
        ? (fired ? params[26] : before_reset)
        : NestAdd(before_reset, FromFloat<T>(-ToFloat(NestMul(
            FromFloat<T>(fired ? 1.0f : 0.0f),
            NestAdd(FromFloat<T>(1.0f), FromFloat<T>(-ToFloat(params[26])))))));
    new_r[index] = fired ? t_ref[neuron]
                        : static_cast<R>(max(static_cast<int>(refractory[index]) - 1, 0));
    #pragma unroll
    for (int component = 0; component < 2; ++component) {
      T adaptation = asc[index * 2 + component];
      if (active) adaptation = NestMul(adaptation, params[8 + component]);
      new_asc[index * 2 + component] = fired
          ? NestAdd(params[10 + component], NestMul(adaptation, params[16 + component]))
          : adaptation;
    }
  }
}

template <typename T, typename R, typename S, bool EmitPreResetVoltage = false,
          bool History = false>
__global__ void NestForwardTypeIndexedKernel(
    int64 count, int neurons, int type_count, const T* voltage, const R* refractory,
    const T* asc, const S* rise, const S* psc, const S* currents,
    const T* type_coefficients, const int64* type_indices, const R* t_ref,
    const T* dt, const T* v_th, bool hard_reset, T* threshold, T* new_v,
    R* new_r, T* new_asc, S* new_rise, S* new_psc,
    int64 history_width, const S* history, S* spikes, S* new_history) {
  extern __shared__ __align__(16) unsigned char shared_bytes[];
  T* block_coefficients = reinterpret_cast<T*>(shared_bytes);
  const int coefficient_entries = type_count * 28;
  for (int entry = threadIdx.x; entry < coefficient_entries; entry += blockDim.x) {
    block_coefficients[entry] = type_coefficients[entry];
  }
  __syncthreads();
  GPU_1D_KERNEL_LOOP(index, count) {
    const int neuron = index % neurons;
    const NestTypeCoefficients<T> params{block_coefficients, static_cast<int>(type_indices[neuron])};
    const bool active = refractory[index] <= 0;
    const T mean_asc = NestAdd(NestMul(asc[index * 2], params[14]),
                               NestMul(asc[index * 2 + 1], params[15]));
    T contributions[4];
    #pragma unroll
    for (int basis = 0; basis < 4; ++basis) {
      const int64 offset = index * 4 + basis;
      contributions[basis] = NestAdd(NestMul(static_cast<T>(psc[offset]), params[18 + basis]),
                                      NestMul(static_cast<T>(rise[offset]), params[22 + basis]));
      new_rise[offset] = static_cast<S>(NestAdd(NestMul(static_cast<T>(rise[offset]), params[basis]),
                                  NestMul(static_cast<T>(currents[offset]), params[4 + basis])));
      new_psc[offset] = static_cast<S>(NestAdd(NestMul(static_cast<T>(psc[offset]), params[basis]),
                                 NestMul(NestMul(*dt, params[basis]), static_cast<T>(rise[offset]))));
    }
    const T integrated = NestAdd(NestAdd(contributions[0], contributions[1]),
                                   NestAdd(contributions[2], contributions[3]));
    const T retained = NestAdd(FromFloat<T>(1.0f), FromFloat<T>(-ToFloat(params[27])));
    const T dampened_voltage = NestAdd(NestMul(voltage[index], retained),
                       NestMul(voltage[index], params[27]));
    const T candidate = NestAdd(
        NestMul(params[12], dampened_voltage),
        NestAdd(NestMul(params[13], mean_asc), integrated));
    const T before_reset = hard_reset && !active ? params[26] : candidate;
    const T threshold_value = NestAdd(before_reset, FromFloat<T>(-ToFloat(*v_th)));
    const bool fired = active && ToFloat(threshold_value) > 0.0f;
    if constexpr (History) {
      const int64 base = (index / neurons) * history_width + neuron;
      const S spike = FromFloat<S>(fired ? 1.0f : 0.0f);
      spikes[index] = spike;
      new_history[base] = spike;
      for (int64 delay = neurons; delay < history_width; delay += neurons) {
        new_history[base + delay] = history[base + delay - neurons];
      }
    }
    if constexpr (EmitPreResetVoltage) {
      threshold[index] = before_reset;
    } else {
      threshold[index] = threshold_value;
    }
    new_v[index] = hard_reset
        ? (fired ? params[26] : before_reset)
        : NestAdd(before_reset, FromFloat<T>(-ToFloat(NestMul(
            FromFloat<T>(fired ? 1.0f : 0.0f),
            NestAdd(FromFloat<T>(1.0f), FromFloat<T>(-ToFloat(params[26])))))));
    new_r[index] = fired ? t_ref[neuron]
                        : static_cast<R>(max(static_cast<int>(refractory[index]) - 1, 0));
    #pragma unroll
    for (int component = 0; component < 2; ++component) {
      T adaptation = asc[index * 2 + component];
      if (active) adaptation = NestMul(adaptation, params[8 + component]);
      new_asc[index * 2 + component] = fired
          ? NestAdd(params[10 + component], NestMul(adaptation, params[16 + component]))
          : adaptation;
    }
  }
}

template <typename T, typename R, typename S, bool SoA = false, bool Events = false>
__global__ void NestBackwardKernel(
    int64 count, int neurons, const T* threshold, const R* refractory,
    const T* coefficients, const T* dt, const T* retention,
    const T* grad_threshold, const T* grad_v, const T* grad_asc,
    const S* grad_rise, const S* grad_psc, bool hard_reset,
    const T* previous_asc, const T* v_th, const T* dampening, const T* gauss_std,
    bool detach_reset, bool detach_asc_reset, bool pseudo_gauss,
    T* voltage_grad, T* asc_grad, S* rise_grad, S* psc_grad, S* currents_grad) {
  GPU_1D_KERNEL_LOOP(index, count) {
    const NestCoefficients<T, SoA> params{coefficients, neurons, static_cast<int>(index % neurons)};
    const bool active = refractory[index] <= 0;
    const bool fired = active && ToFloat(threshold[index]) > 0.0f;
    T threshold_grad = grad_threshold[index];
    if constexpr (Events) {
      if (active && (!detach_reset || !detach_asc_reset)) {
        const T one = FromFloat<T>(1.0f);
        T event_grad = FromFloat<T>(0.0f);
        if (!detach_reset) {
          const T sensitivity = hard_reset
              ? NestAdd(params[26], FromFloat<T>(-ToFloat(NestAdd(threshold[index], *v_th))))
              : FromFloat<T>(-ToFloat(NestAdd(one, FromFloat<T>(-ToFloat(params[26])))));
          event_grad = NestAdd(event_grad, NestMul(grad_v[index], sensitivity));
        }
        if (!detach_asc_reset) {
          T terms[2];
          #pragma unroll
          for (int component = 0; component < 2; ++component) {
            const T adaptation = NestMul(previous_asc[index * 2 + component],
                                         params[8 + component]);
            const T sensitivity = NestAdd(params[10 + component], NestMul(
                NestAdd(params[16 + component], FromFloat<T>(-1.0f)), adaptation));
            terms[component] = NestMul(grad_asc[index * 2 + component], sensitivity);
          }
          event_grad = NestAdd(event_grad, NestAdd(terms[0], terms[1]));
        }
        T shape;
        if (pseudo_gauss) {
          // Preserve the separate TensorFlow dtype-rounding boundaries, not FMA.
          const T square = NestMul(threshold[index], threshold[index]);
          const T width_square = NestMul(*gauss_std, *gauss_std);
          const T exponent = FromFloat<T>(-ToFloat(square) / ToFloat(width_square));
          shape = FromFloat<T>(expf(ToFloat(exponent)));
        } else {
          const T distance = NestAdd(one, FromFloat<T>(-fabsf(ToFloat(threshold[index]))));
          shape = FromFloat<T>(fmaxf(ToFloat(distance), 0.0f));
        }
        const T derivative = NestMul(*dampening, shape);
        threshold_grad = NestAdd(threshold_grad, NestMul(event_grad, derivative));
      }
    }
    T candidate_grad = NestAdd(threshold_grad,
        hard_reset && fired ? FromFloat<T>(0.0f) : grad_v[index]);
    if (hard_reset && !active) candidate_grad = FromFloat<T>(0.0f);
    voltage_grad[index] = NestMul(NestMul(candidate_grad, params[12]), *retention);
    const T mean_grad = NestMul(candidate_grad, params[13]);
    #pragma unroll
    for (int component = 0; component < 2; ++component) {
      T adaptation_grad = grad_asc[index * 2 + component];
      if (fired) adaptation_grad = NestMul(adaptation_grad, params[16 + component]);
      if (active) adaptation_grad = NestMul(adaptation_grad, params[8 + component]);
      asc_grad[index * 2 + component] = NestAdd(
          adaptation_grad, NestMul(mean_grad, params[14 + component]));
    }
    #pragma unroll
    for (int basis = 0; basis < 4; ++basis) {
      const int64 offset = index * 4 + basis;
      rise_grad[offset] = static_cast<S>(NestAdd(
          NestAdd(NestMul(static_cast<T>(grad_rise[offset]), params[basis]),
                  NestMul(static_cast<T>(grad_psc[offset]), NestMul(*dt, params[basis]))),
          NestMul(candidate_grad, params[22 + basis])));
      psc_grad[offset] = static_cast<S>(NestAdd(NestMul(static_cast<T>(grad_psc[offset]), params[basis]),
                                  NestMul(candidate_grad, params[18 + basis])));
      currents_grad[offset] = static_cast<S>(NestMul(static_cast<T>(grad_rise[offset]), params[4 + basis]));
    }
  }
}

template <typename T, typename R, typename S, bool Events = false>
__global__ void NestBackwardTypeIndexedKernel(
    int64 count, int neurons, int type_count, const T* threshold, const R* refractory,
    const T* type_coefficients, const int64* type_indices, const T* dt,
    const T* retention, const T* grad_threshold, const T* grad_v,
    const T* grad_asc, const S* grad_rise, const S* grad_psc, bool hard_reset,
    const T* previous_asc, const T* v_th, const T* dampening, const T* gauss_std,
    bool detach_reset, bool detach_asc_reset, bool pseudo_gauss,
    T* voltage_grad, T* asc_grad, S* rise_grad, S* psc_grad, S* currents_grad) {
  extern __shared__ __align__(16) unsigned char shared_bytes[];
  T* block_coefficients = reinterpret_cast<T*>(shared_bytes);
  const int coefficient_entries = type_count * 28;
  for (int entry = threadIdx.x; entry < coefficient_entries; entry += blockDim.x) {
    block_coefficients[entry] = type_coefficients[entry];
  }
  __syncthreads();
  GPU_1D_KERNEL_LOOP(index, count) {
    const int neuron = static_cast<int>(index % neurons);
    const NestTypeCoefficients<T> params{block_coefficients, static_cast<int>(type_indices[neuron])};
    const bool active = refractory[index] <= 0;
    const bool fired = active && ToFloat(threshold[index]) > 0.0f;
    T threshold_grad = grad_threshold[index];
    if constexpr (Events) {
      if (active && (!detach_reset || !detach_asc_reset)) {
        const T one = FromFloat<T>(1.0f);
        T event_grad = FromFloat<T>(0.0f);
        if (!detach_reset) {
          const T sensitivity = hard_reset
              ? NestAdd(params[26], FromFloat<T>(-ToFloat(NestAdd(threshold[index], *v_th))))
              : FromFloat<T>(-ToFloat(NestAdd(one, FromFloat<T>(-ToFloat(params[26])))));
          event_grad = NestAdd(event_grad, NestMul(grad_v[index], sensitivity));
        }
        if (!detach_asc_reset) {
          T terms[2];
          #pragma unroll
          for (int component = 0; component < 2; ++component) {
            const T adaptation = NestMul(previous_asc[index * 2 + component],
                                         params[8 + component]);
            const T sensitivity = NestAdd(params[10 + component], NestMul(
                NestAdd(params[16 + component], FromFloat<T>(-1.0f)), adaptation));
            terms[component] = NestMul(grad_asc[index * 2 + component], sensitivity);
          }
          event_grad = NestAdd(event_grad, NestAdd(terms[0], terms[1]));
        }
        T shape;
        if (pseudo_gauss) {
          const T square = NestMul(threshold[index], threshold[index]);
          const T width_square = NestMul(*gauss_std, *gauss_std);
          const T exponent = FromFloat<T>(-ToFloat(square) / ToFloat(width_square));
          shape = FromFloat<T>(expf(ToFloat(exponent)));
        } else {
          const T distance = NestAdd(one, FromFloat<T>(-fabsf(ToFloat(threshold[index]))));
          shape = FromFloat<T>(fmaxf(ToFloat(distance), 0.0f));
        }
        const T derivative = NestMul(*dampening, shape);
        threshold_grad = NestAdd(threshold_grad, NestMul(event_grad, derivative));
      }
    }
    T candidate_grad = NestAdd(threshold_grad,
        hard_reset && fired ? FromFloat<T>(0.0f) : grad_v[index]);
    if (hard_reset && !active) candidate_grad = FromFloat<T>(0.0f);
    voltage_grad[index] = NestMul(NestMul(candidate_grad, params[12]), *retention);
    const T mean_grad = NestMul(candidate_grad, params[13]);
    #pragma unroll
    for (int component = 0; component < 2; ++component) {
      T adaptation_grad = grad_asc[index * 2 + component];
      if (fired) adaptation_grad = NestMul(adaptation_grad, params[16 + component]);
      if (active) adaptation_grad = NestMul(adaptation_grad, params[8 + component]);
      asc_grad[index * 2 + component] = NestAdd(
          adaptation_grad, NestMul(mean_grad, params[14 + component]));
    }
    #pragma unroll
    for (int basis = 0; basis < 4; ++basis) {
      const int64 offset = index * 4 + basis;
      rise_grad[offset] = static_cast<S>(NestAdd(
          NestAdd(NestMul(static_cast<T>(grad_rise[offset]), params[basis]),
                  NestMul(static_cast<T>(grad_psc[offset]), NestMul(*dt, params[basis]))),
          NestMul(candidate_grad, params[22 + basis])));
      psc_grad[offset] = static_cast<S>(NestAdd(NestMul(static_cast<T>(grad_psc[offset]), params[basis]),
                                  NestMul(candidate_grad, params[18 + basis])));
      currents_grad[offset] = static_cast<S>(NestMul(static_cast<T>(grad_rise[offset]), params[4 + basis]));
    }
  }
}

template <typename T, typename R, typename S, bool SoA = false, bool Events = false>
__global__ void NestHistoryBackwardKernel(
    int64 count, int neurons, int64 history_width, const T* threshold,
    const R* refractory, const T* coefficients, const T* dt, const T* retention,
    const T* grad_threshold, const T* grad_v, const T* grad_asc,
    const S* grad_rise, const S* grad_psc, bool hard_reset,
    const T* previous_asc, const T* v_th, const T* dampening, const T* gauss_std,
    const S* grad_spikes, const S* grad_history,
    bool detach_reset, bool detach_asc_reset, bool pseudo_gauss,
    T* voltage_grad, T* asc_grad, S* rise_grad, S* psc_grad, S* currents_grad,
    S* old_history_grad) {
  GPU_1D_KERNEL_LOOP(index, count) {
    const int neuron = static_cast<int>(index % neurons);
    const int64 history_base = (index / neurons) * history_width + neuron;
    const NestCoefficients<T, SoA> params{coefficients, neurons, neuron};
    const bool active = refractory[index] <= 0;
    const bool fired = active && ToFloat(threshold[index]) > 0.0f;
    const T square = FromFloat<T>(
        ToFloat(NestMul(threshold[index], threshold[index])) /
        ToFloat(NestMul(*gauss_std, *gauss_std)));
    const T triangle = pseudo_gauss
        ? FromFloat<T>(expf(-ToFloat(square)))
        : FromFloat<T>(fmaxf(1.0f - fabsf(ToFloat(threshold[index])), 0.0f));
    const T derivative = FromFloat<T>(ToFloat(*dampening) * ToFloat(triangle));
    const T spike_upstream = FromFloat<T>(
        ToFloat(grad_spikes[index]) + ToFloat(grad_history[history_base]));
    T threshold_grad = NestAdd(
        grad_threshold[index],
        active ? FromFloat<T>(ToFloat(spike_upstream) * ToFloat(derivative))
               : FromFloat<T>(0.0f));
    for (int64 delay = 0; delay < history_width; delay += neurons) {
      const int64 history_index = history_base + delay;
      old_history_grad[history_index] =
          delay + neurons < history_width
              ? grad_history[history_index + neurons]
              : FromFloat<S>(0.0f);
    }
    if constexpr (Events) {
      if (active && (!detach_reset || !detach_asc_reset)) {
        const T one = FromFloat<T>(1.0f);
        T event_grad = FromFloat<T>(0.0f);
        if (!detach_reset) {
          const T sensitivity = hard_reset
              ? NestAdd(params[26], FromFloat<T>(-ToFloat(NestAdd(threshold[index], *v_th))))
              : FromFloat<T>(-ToFloat(NestAdd(one, FromFloat<T>(-ToFloat(params[26])))));
          event_grad = NestAdd(event_grad, NestMul(grad_v[index], sensitivity));
        }
        if (!detach_asc_reset) {
          T terms[2];
          #pragma unroll
          for (int component = 0; component < 2; ++component) {
            const T adaptation = NestMul(previous_asc[index * 2 + component],
                                         params[8 + component]);
            const T sensitivity = NestAdd(params[10 + component], NestMul(
                NestAdd(params[16 + component], FromFloat<T>(-1.0f)), adaptation));
            terms[component] = NestMul(grad_asc[index * 2 + component], sensitivity);
          }
          event_grad = NestAdd(event_grad, NestAdd(terms[0], terms[1]));
        }
        threshold_grad = NestAdd(threshold_grad, NestMul(event_grad, derivative));
      }
    }
    T candidate_grad = NestAdd(threshold_grad,
        hard_reset && fired ? FromFloat<T>(0.0f) : grad_v[index]);
    if (hard_reset && !active) candidate_grad = FromFloat<T>(0.0f);
    voltage_grad[index] = NestMul(NestMul(candidate_grad, params[12]), *retention);
    const T mean_grad = NestMul(candidate_grad, params[13]);
    #pragma unroll
    for (int component = 0; component < 2; ++component) {
      T adaptation_grad = grad_asc[index * 2 + component];
      if (fired) adaptation_grad = NestMul(adaptation_grad, params[16 + component]);
      if (active) adaptation_grad = NestMul(adaptation_grad, params[8 + component]);
      asc_grad[index * 2 + component] = NestAdd(
          adaptation_grad, NestMul(mean_grad, params[14 + component]));
    }
    #pragma unroll
    for (int basis = 0; basis < 4; ++basis) {
      const int64 offset = index * 4 + basis;
      rise_grad[offset] = static_cast<S>(NestAdd(
          NestAdd(NestMul(static_cast<T>(grad_rise[offset]), params[basis]),
                  NestMul(static_cast<T>(grad_psc[offset]), NestMul(*dt, params[basis]))),
          NestMul(candidate_grad, params[22 + basis])));
      psc_grad[offset] = static_cast<S>(NestAdd(NestMul(static_cast<T>(grad_psc[offset]), params[basis]),
                                  NestMul(candidate_grad, params[18 + basis])));
      currents_grad[offset] = static_cast<S>(NestMul(static_cast<T>(grad_rise[offset]), params[4 + basis]));
    }
  }
}

template <typename T, typename R, typename S, bool Events = false>
__global__ void NestHistoryBackwardTypeIndexedKernel(
    int64 count, int neurons, int type_count, int64 history_width, const T* threshold,
    const R* refractory, const T* type_coefficients, const int64* type_indices,
    const T* dt, const T* retention,
    const T* grad_threshold, const T* grad_v, const T* grad_asc,
    const S* grad_rise, const S* grad_psc, bool hard_reset,
    const T* previous_asc, const T* v_th, const T* dampening, const T* gauss_std,
    const S* grad_spikes, const S* grad_history,
    bool detach_reset, bool detach_asc_reset, bool pseudo_gauss,
    T* voltage_grad, T* asc_grad, S* rise_grad, S* psc_grad, S* currents_grad,
    S* old_history_grad) {
  extern __shared__ __align__(16) unsigned char shared_bytes[];
  T* block_coefficients = reinterpret_cast<T*>(shared_bytes);
  const int coefficient_entries = type_count * 28;
  for (int entry = threadIdx.x; entry < coefficient_entries; entry += blockDim.x) {
    block_coefficients[entry] = type_coefficients[entry];
  }
  __syncthreads();
  GPU_1D_KERNEL_LOOP(index, count) {
    const int neuron = static_cast<int>(index % neurons);
    const int64 history_base = (index / neurons) * history_width + neuron;
    const NestTypeCoefficients<T> params{block_coefficients, static_cast<int>(type_indices[neuron])};
    const bool active = refractory[index] <= 0;
    const bool fired = active && ToFloat(threshold[index]) > 0.0f;
    const T square = FromFloat<T>(
        ToFloat(NestMul(threshold[index], threshold[index])) /
        ToFloat(NestMul(*gauss_std, *gauss_std)));
    const T triangle = pseudo_gauss
        ? FromFloat<T>(expf(-ToFloat(square)))
        : FromFloat<T>(fmaxf(1.0f - fabsf(ToFloat(threshold[index])), 0.0f));
    const T derivative = FromFloat<T>(ToFloat(*dampening) * ToFloat(triangle));
    const T spike_upstream = FromFloat<T>(
        ToFloat(grad_spikes[index]) + ToFloat(grad_history[history_base]));
    T threshold_grad = NestAdd(
        grad_threshold[index],
        active ? FromFloat<T>(ToFloat(spike_upstream) * ToFloat(derivative))
               : FromFloat<T>(0.0f));
    for (int64 delay = 0; delay < history_width; delay += neurons) {
      const int64 history_index = history_base + delay;
      old_history_grad[history_index] =
          delay + neurons < history_width
              ? grad_history[history_index + neurons]
              : FromFloat<S>(0.0f);
    }
    if constexpr (Events) {
      if (active && (!detach_reset || !detach_asc_reset)) {
        const T one = FromFloat<T>(1.0f);
        T event_grad = FromFloat<T>(0.0f);
        if (!detach_reset) {
          const T sensitivity = hard_reset
              ? NestAdd(params[26], FromFloat<T>(-ToFloat(NestAdd(threshold[index], *v_th))))
              : FromFloat<T>(-ToFloat(NestAdd(one, FromFloat<T>(-ToFloat(params[26])))));
          event_grad = NestAdd(event_grad, NestMul(grad_v[index], sensitivity));
        }
        if (!detach_asc_reset) {
          T terms[2];
          #pragma unroll
          for (int component = 0; component < 2; ++component) {
            const T adaptation = NestMul(previous_asc[index * 2 + component],
                                         params[8 + component]);
            const T sensitivity = NestAdd(params[10 + component], NestMul(
                NestAdd(params[16 + component], FromFloat<T>(-1.0f)), adaptation));
            terms[component] = NestMul(grad_asc[index * 2 + component], sensitivity);
          }
          event_grad = NestAdd(event_grad, NestAdd(terms[0], terms[1]));
        }
        threshold_grad = NestAdd(threshold_grad, NestMul(event_grad, derivative));
      }
    }
    T candidate_grad = NestAdd(threshold_grad,
        hard_reset && fired ? FromFloat<T>(0.0f) : grad_v[index]);
    if (hard_reset && !active) candidate_grad = FromFloat<T>(0.0f);
    voltage_grad[index] = NestMul(NestMul(candidate_grad, params[12]), *retention);
    const T mean_grad = NestMul(candidate_grad, params[13]);
    #pragma unroll
    for (int component = 0; component < 2; ++component) {
      T adaptation_grad = grad_asc[index * 2 + component];
      if (fired) adaptation_grad = NestMul(adaptation_grad, params[16 + component]);
      if (active) adaptation_grad = NestMul(adaptation_grad, params[8 + component]);
      asc_grad[index * 2 + component] = NestAdd(
          adaptation_grad, NestMul(mean_grad, params[14 + component]));
    }
    #pragma unroll
    for (int basis = 0; basis < 4; ++basis) {
      const int64 offset = index * 4 + basis;
      rise_grad[offset] = static_cast<S>(NestAdd(
          NestAdd(NestMul(static_cast<T>(grad_rise[offset]), params[basis]),
                  NestMul(static_cast<T>(grad_psc[offset]), NestMul(*dt, params[basis]))),
          NestMul(candidate_grad, params[22 + basis])));
      psc_grad[offset] = static_cast<S>(NestAdd(NestMul(static_cast<T>(grad_psc[offset]), params[basis]),
                                  NestMul(candidate_grad, params[18 + basis])));
      currents_grad[offset] = static_cast<S>(NestMul(static_cast<T>(grad_rise[offset]), params[4 + basis]));
    }
  }
}

template <typename T, typename R, typename S, bool Backward, bool Events = false,
          bool History = false>
class NestStateOp : public OpKernel {
 public:
  explicit NestStateOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("hard_reset", &hard_reset_));
    if (!Backward) {
      OP_REQUIRES_OK(context, context->GetAttr("emit_pre_reset_voltage", &emit_pre_reset_voltage_));
    }
    if constexpr (Events) {
      OP_REQUIRES_OK(context, context->GetAttr("detach_reset", &detach_reset_));
      OP_REQUIRES_OK(context, context->GetAttr("detach_asc_reset", &detach_asc_reset_));
      OP_REQUIRES_OK(context, context->GetAttr("pseudo_gauss", &pseudo_gauss_));
    }
    string layout;
    OP_REQUIRES_OK(context, context->GetAttr("coefficients_layout", &layout));
    soa_ = layout == "soa";
    OP_REQUIRES(context, !soa_ || (std::is_same<T, float>::value),
                errors::InvalidArgument("NEST SoA coefficients require FP32 voltage/ASC"));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& voltage = context->input(0);
    OP_REQUIRES(context, voltage.dims() == 2 && voltage.dim_size(1) > 0,
                errors::InvalidArgument("NEST voltage must have shape [batch, neurons]"));
    const int64 batch = voltage.dim_size(0);
    const int neurons = voltage.dim_size(1);
    const TensorShape asc_shape({batch, neurons * 2LL});
    const TensorShape psc_shape({batch, neurons * 4LL});
    const int coefficient_index = Backward ? 2 : 6;
    OP_REQUIRES(context, context->input(1).shape() == voltage.shape(),
                errors::InvalidArgument("NEST refractory shape differs from voltage"));
    const TensorShape coefficient_shape =
        soa_ ? TensorShape({28, neurons}) : TensorShape({neurons, 28});
    OP_REQUIRES(context, context->input(coefficient_index).shape() == coefficient_shape,
          errors::InvalidArgument("NEST coefficients must have shape ",
                                  soa_ ? "[28, neurons]" : "[neurons, 28]"));
    if (Backward) {
      OP_REQUIRES(context,
          context->input(3).NumElements() == 1 && context->input(4).NumElements() == 1 &&
          context->input(5).shape() == voltage.shape() && context->input(6).shape() == voltage.shape() &&
          context->input(7).shape() == asc_shape && context->input(8).shape() == psc_shape &&
          context->input(9).shape() == psc_shape,
          errors::InvalidArgument("Invalid NEST backward tensor shapes"));
      if constexpr (Events) {
        OP_REQUIRES(context,
            context->input(10).shape() == asc_shape &&
            TensorShapeUtils::IsScalar(context->input(11).shape()) &&
            TensorShapeUtils::IsScalar(context->input(12).shape()) &&
            TensorShapeUtils::IsScalar(context->input(13).shape()),
            errors::InvalidArgument("Invalid NEST event VJP tensor shapes"));
      }
    } else {
      OP_REQUIRES(context,
          context->input(2).shape() == asc_shape && context->input(3).shape() == psc_shape &&
          context->input(4).shape() == psc_shape && context->input(5).shape() == psc_shape &&
          context->input(7).shape() == TensorShape({neurons}) &&
          context->input(8).NumElements() == 1 && context->input(9).NumElements() == 1,
          errors::InvalidArgument("Invalid NEST forward tensor shapes; four bases are required"));
    }
    Tensor* outputs[8];
    const TensorShape shapes[6] = {
        voltage.shape(), Backward ? asc_shape : voltage.shape(),
        Backward ? psc_shape : voltage.shape(),
        Backward ? psc_shape : asc_shape, psc_shape, psc_shape};
    for (int output = 0; output < (Backward ? 5 : 6); ++output) {
      OP_REQUIRES_OK(context, context->allocate_output(output, shapes[output], &outputs[output]));
    }
    if constexpr (History) {
      const Tensor& history = context->input(10);
      OP_REQUIRES(context,
          history.dims() == 2 && history.dim_size(0) == batch &&
          history.dim_size(1) > 0 && history.dim_size(1) % neurons == 0,
          errors::InvalidArgument("NEST history must have shape [batch, positive delays * neurons]"));
      OP_REQUIRES_OK(context, context->allocate_output(6, voltage.shape(), &outputs[6]));
      OP_REQUIRES_OK(context, context->allocate_output(7, history.shape(), &outputs[7]));
    }
    if (voltage.NumElements() == 0) return;
    if (soa_) {
      if (!Backward && emit_pre_reset_voltage_) {
        Launch<true, true>(context, voltage, neurons, outputs);
      } else {
        Launch<true, false>(context, voltage, neurons, outputs);
      }
    } else {
      if (!Backward && emit_pre_reset_voltage_) {
        Launch<false, true>(context, voltage, neurons, outputs);
      } else {
        Launch<false, false>(context, voltage, neurons, outputs);
      }
    }
  }

 private:
  template <bool SoA, bool EmitPreResetVoltage>
  void Launch(OpKernelContext* context, const Tensor& voltage, int neurons,
              Tensor** outputs) {
    auto& device = context->eigen_device<GPUDevice>();
    if constexpr (Backward) {
      auto config = GetGpuLaunchConfig(
          voltage.NumElements(), device, NestBackwardKernel<T, R, S, SoA, Events>, 0, 0);
      OP_REQUIRES_OK(context, GpuLaunchKernel(
          NestBackwardKernel<T, R, S, SoA, Events>, config.block_count, config.thread_per_block, 0,
          device.stream(), voltage.NumElements(), neurons, voltage.flat<T>().data(),
          context->input(1).flat<R>().data(), context->input(2).flat<T>().data(),
          context->input(3).flat<T>().data(), context->input(4).flat<T>().data(),
          context->input(5).flat<T>().data(), context->input(6).flat<T>().data(),
          context->input(7).flat<T>().data(), context->input(8).flat<S>().data(),
          context->input(9).flat<S>().data(), hard_reset_,
          Events ? context->input(10).flat<T>().data() : nullptr,
          Events ? context->input(11).flat<T>().data() : nullptr,
          Events ? context->input(12).flat<T>().data() : nullptr,
          Events ? context->input(13).flat<T>().data() : nullptr,
          detach_reset_, detach_asc_reset_, pseudo_gauss_,
          outputs[0]->flat<T>().data(), outputs[1]->flat<T>().data(),
          outputs[2]->flat<S>().data(), outputs[3]->flat<S>().data(), outputs[4]->flat<S>().data()));
    } else {
      auto config = GetGpuLaunchConfig(
          voltage.NumElements(), device,
          NestForwardKernel<T, R, S, SoA, EmitPreResetVoltage, History>, 0, 0);
      int64 history_width = 0;
      const S* history = nullptr;
      S* spikes = nullptr;
      S* new_history = nullptr;
      if constexpr (History) {
        history_width = context->input(10).dim_size(1);
        history = context->input(10).flat<S>().data();
        spikes = outputs[6]->flat<S>().data();
        new_history = outputs[7]->flat<S>().data();
      }
      OP_REQUIRES_OK(context, GpuLaunchKernel(
          NestForwardKernel<T, R, S, SoA, EmitPreResetVoltage, History>,
          config.block_count, config.thread_per_block, 0,
          device.stream(), voltage.NumElements(), neurons, voltage.flat<T>().data(),
          context->input(1).flat<R>().data(), context->input(2).flat<T>().data(),
          context->input(3).flat<S>().data(), context->input(4).flat<S>().data(),
          context->input(5).flat<S>().data(), context->input(6).flat<T>().data(),
          context->input(7).flat<R>().data(), context->input(8).flat<T>().data(),
          context->input(9).flat<T>().data(), hard_reset_, outputs[0]->flat<T>().data(),
          outputs[1]->flat<T>().data(), outputs[2]->flat<R>().data(),
          outputs[3]->flat<T>().data(), outputs[4]->flat<S>().data(), outputs[5]->flat<S>().data(),
          history_width, history, spikes, new_history));
    }
  }

  bool hard_reset_;
  bool soa_;
  bool emit_pre_reset_voltage_ = false;
  bool detach_reset_ = true;
  bool detach_asc_reset_ = true;
  bool pseudo_gauss_ = false;
};

template <typename T, typename R, typename S, bool Backward, bool Events = false,
          bool History = false>
class NestStateTypeIndexedOp : public OpKernel {
 public:
  explicit NestStateTypeIndexedOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("hard_reset", &hard_reset_));
    if (!Backward) {
      OP_REQUIRES_OK(context, context->GetAttr("emit_pre_reset_voltage", &emit_pre_reset_voltage_));
    }
    if constexpr (Events) {
      OP_REQUIRES_OK(context, context->GetAttr("detach_reset", &detach_reset_));
      OP_REQUIRES_OK(context, context->GetAttr("detach_asc_reset", &detach_asc_reset_));
      OP_REQUIRES_OK(context, context->GetAttr("pseudo_gauss", &pseudo_gauss_));
    }
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& voltage = context->input(0);
    OP_REQUIRES(context, voltage.dims() == 2 && voltage.dim_size(1) > 0,
                errors::InvalidArgument("NEST voltage must have shape [batch, neurons]"));
    const int64 batch = voltage.dim_size(0);
    const int neurons = voltage.dim_size(1);
    const TensorShape asc_shape({batch, neurons * 2LL});
    const TensorShape psc_shape({batch, neurons * 4LL});
    const int coefficient_index = Backward ? 2 : 6;
    const int type_index_index = Backward ? 3 : 7;
    const Tensor& type_coefficients = context->input(coefficient_index);
    OP_REQUIRES(context, type_coefficients.dims() == 2 &&
                         type_coefficients.dim_size(0) > 0 &&
                         type_coefficients.dim_size(1) == 28,
                errors::InvalidArgument("NEST type coefficients must have shape [types, 28]"));
    const int type_count = static_cast<int>(type_coefficients.dim_size(0));
    const size_t shared_bytes = static_cast<size_t>(type_count) * 28 * sizeof(T);
    OP_REQUIRES(context, shared_bytes <= 48 * 1024,
                errors::InvalidArgument("NEST type coefficient table needs ",
                                        shared_bytes, " shared-memory bytes; max is 49152"));
    OP_REQUIRES(context, context->input(1).shape() == voltage.shape(),
                errors::InvalidArgument("NEST refractory shape differs from voltage"));
    OP_REQUIRES(context, context->input(type_index_index).shape() == TensorShape({neurons}),
                errors::InvalidArgument("NEST type_indices must have shape [neurons]"));
    if (Backward) {
      OP_REQUIRES(context,
          context->input(4).NumElements() == 1 && context->input(5).NumElements() == 1 &&
          context->input(6).shape() == voltage.shape() && context->input(7).shape() == voltage.shape() &&
          context->input(8).shape() == asc_shape && context->input(9).shape() == psc_shape &&
          context->input(10).shape() == psc_shape,
          errors::InvalidArgument("Invalid NEST type-indexed backward tensor shapes"));
      if constexpr (Events) {
        OP_REQUIRES(context,
            context->input(11).shape() == asc_shape &&
            TensorShapeUtils::IsScalar(context->input(12).shape()) &&
            TensorShapeUtils::IsScalar(context->input(13).shape()) &&
            TensorShapeUtils::IsScalar(context->input(14).shape()),
            errors::InvalidArgument("Invalid NEST type-indexed event VJP tensor shapes"));
      }
    } else {
      OP_REQUIRES(context,
          context->input(2).shape() == asc_shape && context->input(3).shape() == psc_shape &&
          context->input(4).shape() == psc_shape && context->input(5).shape() == psc_shape &&
          context->input(8).shape() == TensorShape({neurons}) &&
          context->input(9).NumElements() == 1 && context->input(10).NumElements() == 1,
          errors::InvalidArgument("Invalid NEST type-indexed forward tensor shapes; four bases are required"));
    }
    Tensor* outputs[8];
    const TensorShape shapes[6] = {
        voltage.shape(), Backward ? asc_shape : voltage.shape(),
        Backward ? psc_shape : voltage.shape(),
        Backward ? psc_shape : asc_shape, psc_shape, psc_shape};
    for (int output = 0; output < (Backward ? 5 : 6); ++output) {
      OP_REQUIRES_OK(context, context->allocate_output(output, shapes[output], &outputs[output]));
    }
    if constexpr (History) {
      const Tensor& history = context->input(11);
      OP_REQUIRES(context,
          history.dims() == 2 && history.dim_size(0) == batch &&
          history.dim_size(1) > 0 && history.dim_size(1) % neurons == 0,
          errors::InvalidArgument("NEST type-indexed history must have shape [batch, positive delays * neurons]"));
      OP_REQUIRES_OK(context, context->allocate_output(6, voltage.shape(), &outputs[6]));
      OP_REQUIRES_OK(context, context->allocate_output(7, history.shape(), &outputs[7]));
    }
    if (voltage.NumElements() == 0) return;
    if (!Backward && emit_pre_reset_voltage_) {
      Launch<true>(context, voltage, neurons, type_count, shared_bytes, outputs);
    } else {
      Launch<false>(context, voltage, neurons, type_count, shared_bytes, outputs);
    }
  }

 private:
  template <bool EmitPreResetVoltage>
  void Launch(OpKernelContext* context, const Tensor& voltage, int neurons,
              int type_count, size_t shared_bytes, Tensor** outputs) {
    auto& device = context->eigen_device<GPUDevice>();
    auto config = GetGpuLaunchConfig(voltage.NumElements(), device);
    if constexpr (Backward) {
      OP_REQUIRES_OK(context, GpuLaunchKernel(
          NestBackwardTypeIndexedKernel<T, R, S, Events>,
          config.block_count, config.thread_per_block, shared_bytes,
          device.stream(), voltage.NumElements(), neurons, type_count,
          voltage.flat<T>().data(), context->input(1).flat<R>().data(),
          context->input(2).flat<T>().data(), context->input(3).flat<int64>().data(),
          context->input(4).flat<T>().data(), context->input(5).flat<T>().data(),
          context->input(6).flat<T>().data(), context->input(7).flat<T>().data(),
          context->input(8).flat<T>().data(), context->input(9).flat<S>().data(),
          context->input(10).flat<S>().data(), hard_reset_,
          Events ? context->input(11).flat<T>().data() : nullptr,
          Events ? context->input(12).flat<T>().data() : nullptr,
          Events ? context->input(13).flat<T>().data() : nullptr,
          Events ? context->input(14).flat<T>().data() : nullptr,
          detach_reset_, detach_asc_reset_, pseudo_gauss_,
          outputs[0]->flat<T>().data(), outputs[1]->flat<T>().data(),
          outputs[2]->flat<S>().data(), outputs[3]->flat<S>().data(), outputs[4]->flat<S>().data()));
    } else {
      int64 history_width = 0;
      const S* history = nullptr;
      S* spikes = nullptr;
      S* new_history = nullptr;
      if constexpr (History) {
        history_width = context->input(11).dim_size(1);
        history = context->input(11).flat<S>().data();
        spikes = outputs[6]->flat<S>().data();
        new_history = outputs[7]->flat<S>().data();
      }
      OP_REQUIRES_OK(context, GpuLaunchKernel(
          NestForwardTypeIndexedKernel<T, R, S, EmitPreResetVoltage, History>,
          config.block_count, config.thread_per_block, shared_bytes,
          device.stream(), voltage.NumElements(), neurons, type_count,
          voltage.flat<T>().data(), context->input(1).flat<R>().data(),
          context->input(2).flat<T>().data(), context->input(3).flat<S>().data(),
          context->input(4).flat<S>().data(), context->input(5).flat<S>().data(),
          context->input(6).flat<T>().data(), context->input(7).flat<int64>().data(),
          context->input(8).flat<R>().data(), context->input(9).flat<T>().data(),
          context->input(10).flat<T>().data(), hard_reset_,
          outputs[0]->flat<T>().data(), outputs[1]->flat<T>().data(),
          outputs[2]->flat<R>().data(), outputs[3]->flat<T>().data(),
          outputs[4]->flat<S>().data(), outputs[5]->flat<S>().data(),
          history_width, history, spikes, new_history));
    }
  }

  bool hard_reset_;
  bool emit_pre_reset_voltage_ = false;
  bool detach_reset_ = true;
  bool detach_asc_reset_ = true;
  bool pseudo_gauss_ = false;
};

template <typename T, typename R, typename S, bool Events = false>
class NestStateHistoryBackwardOp : public OpKernel {
 public:
  explicit NestStateHistoryBackwardOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("hard_reset", &hard_reset_));
    OP_REQUIRES_OK(context, context->GetAttr("pseudo_gauss", &pseudo_gauss_));
    if constexpr (Events) {
      OP_REQUIRES_OK(context, context->GetAttr("detach_reset", &detach_reset_));
      OP_REQUIRES_OK(context, context->GetAttr("detach_asc_reset", &detach_asc_reset_));
    }
    string layout;
    OP_REQUIRES_OK(context, context->GetAttr("coefficients_layout", &layout));
    soa_ = layout == "soa";
    OP_REQUIRES(context, !soa_ || (std::is_same<T, float>::value),
                errors::InvalidArgument("NEST SoA coefficients require FP32 voltage/ASC"));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& voltage = context->input(0);
    OP_REQUIRES(context, voltage.dims() == 2 && voltage.dim_size(1) > 0,
                errors::InvalidArgument("NEST threshold must have shape [batch, neurons]"));
    const int64 batch = voltage.dim_size(0);
    const int neurons = voltage.dim_size(1);
    const TensorShape asc_shape({batch, neurons * 2LL});
    const TensorShape psc_shape({batch, neurons * 4LL});
    const TensorShape coefficient_shape =
        soa_ ? TensorShape({28, neurons}) : TensorShape({neurons, 28});
    OP_REQUIRES(context, context->input(1).shape() == voltage.shape(),
                errors::InvalidArgument("NEST refractory shape differs from threshold"));
    OP_REQUIRES(context, context->input(2).shape() == coefficient_shape,
          errors::InvalidArgument("NEST coefficients must have shape ",
                                  soa_ ? "[28, neurons]" : "[neurons, 28]"));
    OP_REQUIRES(context,
        context->input(3).NumElements() == 1 && context->input(4).NumElements() == 1 &&
        context->input(5).shape() == voltage.shape() && context->input(6).shape() == voltage.shape() &&
        context->input(7).shape() == asc_shape && context->input(8).shape() == psc_shape &&
        context->input(9).shape() == psc_shape,
        errors::InvalidArgument("Invalid NEST state/history backward tensor shapes"));
    if constexpr (Events) {
      OP_REQUIRES(context,
          context->input(10).shape() == asc_shape &&
          TensorShapeUtils::IsScalar(context->input(11).shape()) &&
          TensorShapeUtils::IsScalar(context->input(12).shape()) &&
          TensorShapeUtils::IsScalar(context->input(13).shape()),
          errors::InvalidArgument("Invalid NEST state/history event VJP tensor shapes"));
    } else {
      OP_REQUIRES(context,
          TensorShapeUtils::IsScalar(context->input(10).shape()) &&
          TensorShapeUtils::IsScalar(context->input(11).shape()),
          errors::InvalidArgument("Invalid NEST state/history surrogate scalar shapes"));
    }
    const int spike_index = Events ? 14 : 12;
    const int history_index = spike_index + 1;
    const Tensor& grad_history = context->input(history_index);
    OP_REQUIRES(context,
        context->input(spike_index).shape() == voltage.shape() &&
        grad_history.dims() == 2 && grad_history.dim_size(0) == batch &&
        grad_history.dim_size(1) > 0 && grad_history.dim_size(1) % neurons == 0,
        errors::InvalidArgument("Invalid NEST state/history spike or history gradient shape"));
    Tensor* outputs[6];
    const TensorShape shapes[6] = {
        voltage.shape(), asc_shape, psc_shape, psc_shape, psc_shape, grad_history.shape()};
    for (int output = 0; output < 6; ++output) {
      OP_REQUIRES_OK(context, context->allocate_output(output, shapes[output], &outputs[output]));
    }
    if (voltage.NumElements() == 0) return;
    if (soa_) {
      Launch<true>(context, voltage, neurons, grad_history.dim_size(1), spike_index, outputs);
    } else {
      Launch<false>(context, voltage, neurons, grad_history.dim_size(1), spike_index, outputs);
    }
  }

 private:
  template <bool SoA>
  void Launch(OpKernelContext* context, const Tensor& voltage, int neurons,
              int64 history_width, int spike_index, Tensor** outputs) {
    auto& device = context->eigen_device<GPUDevice>();
    auto config = GetGpuLaunchConfig(
        voltage.NumElements(), device, NestHistoryBackwardKernel<T, R, S, SoA, Events>, 0, 0);
    const int history_index = spike_index + 1;
    OP_REQUIRES_OK(context, GpuLaunchKernel(
        NestHistoryBackwardKernel<T, R, S, SoA, Events>,
        config.block_count, config.thread_per_block, 0,
        device.stream(), voltage.NumElements(), neurons, history_width,
        voltage.flat<T>().data(), context->input(1).flat<R>().data(),
        context->input(2).flat<T>().data(), context->input(3).flat<T>().data(),
        context->input(4).flat<T>().data(), context->input(5).flat<T>().data(),
        context->input(6).flat<T>().data(), context->input(7).flat<T>().data(),
        context->input(8).flat<S>().data(), context->input(9).flat<S>().data(),
        hard_reset_,
        Events ? context->input(10).flat<T>().data() : nullptr,
        Events ? context->input(11).flat<T>().data() : nullptr,
        (Events ? context->input(12) : context->input(10)).flat<T>().data(),
        (Events ? context->input(13) : context->input(11)).flat<T>().data(),
        context->input(spike_index).flat<S>().data(),
        context->input(history_index).flat<S>().data(),
        detach_reset_, detach_asc_reset_, pseudo_gauss_,
        outputs[0]->flat<T>().data(), outputs[1]->flat<T>().data(),
        outputs[2]->flat<S>().data(), outputs[3]->flat<S>().data(),
        outputs[4]->flat<S>().data(), outputs[5]->flat<S>().data()));
  }

  bool hard_reset_;
  bool soa_;
  bool detach_reset_ = true;
  bool detach_asc_reset_ = true;
  bool pseudo_gauss_ = false;
};

template <typename T, typename R, typename S, bool Events = false>
class NestStateHistoryTypeIndexedBackwardOp : public OpKernel {
 public:
  explicit NestStateHistoryTypeIndexedBackwardOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("hard_reset", &hard_reset_));
    OP_REQUIRES_OK(context, context->GetAttr("pseudo_gauss", &pseudo_gauss_));
    if constexpr (Events) {
      OP_REQUIRES_OK(context, context->GetAttr("detach_reset", &detach_reset_));
      OP_REQUIRES_OK(context, context->GetAttr("detach_asc_reset", &detach_asc_reset_));
    }
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& voltage = context->input(0);
    OP_REQUIRES(context, voltage.dims() == 2 && voltage.dim_size(1) > 0,
                errors::InvalidArgument("NEST type-indexed threshold must have shape [batch, neurons]"));
    const int64 batch = voltage.dim_size(0);
    const int neurons = voltage.dim_size(1);
    const TensorShape asc_shape({batch, neurons * 2LL});
    const TensorShape psc_shape({batch, neurons * 4LL});
    const Tensor& type_coefficients = context->input(2);
    OP_REQUIRES(context, type_coefficients.dims() == 2 &&
                         type_coefficients.dim_size(0) > 0 &&
                         type_coefficients.dim_size(1) == 28,
                errors::InvalidArgument("NEST type coefficients must have shape [types, 28]"));
    const int type_count = static_cast<int>(type_coefficients.dim_size(0));
    const size_t shared_bytes = static_cast<size_t>(type_count) * 28 * sizeof(T);
    OP_REQUIRES(context, shared_bytes <= 48 * 1024,
                errors::InvalidArgument("NEST type coefficient table needs ",
                                        shared_bytes, " shared-memory bytes; max is 49152"));
    OP_REQUIRES(context, context->input(1).shape() == voltage.shape(),
                errors::InvalidArgument("NEST refractory shape differs from threshold"));
    OP_REQUIRES(context, context->input(3).shape() == TensorShape({neurons}),
                errors::InvalidArgument("NEST type_indices must have shape [neurons]"));
    OP_REQUIRES(context,
        context->input(4).NumElements() == 1 && context->input(5).NumElements() == 1 &&
        context->input(6).shape() == voltage.shape() && context->input(7).shape() == voltage.shape() &&
        context->input(8).shape() == asc_shape && context->input(9).shape() == psc_shape &&
        context->input(10).shape() == psc_shape,
        errors::InvalidArgument("Invalid NEST type-indexed state/history backward tensor shapes"));
    if constexpr (Events) {
      OP_REQUIRES(context,
          context->input(11).shape() == asc_shape &&
          TensorShapeUtils::IsScalar(context->input(12).shape()) &&
          TensorShapeUtils::IsScalar(context->input(13).shape()) &&
          TensorShapeUtils::IsScalar(context->input(14).shape()),
          errors::InvalidArgument("Invalid NEST type-indexed state/history event VJP tensor shapes"));
    } else {
      OP_REQUIRES(context,
          TensorShapeUtils::IsScalar(context->input(11).shape()) &&
          TensorShapeUtils::IsScalar(context->input(12).shape()),
          errors::InvalidArgument("Invalid NEST type-indexed state/history surrogate scalar shapes"));
    }
    const int spike_index = Events ? 15 : 13;
    const int history_index = spike_index + 1;
    const Tensor& grad_history = context->input(history_index);
    OP_REQUIRES(context,
        context->input(spike_index).shape() == voltage.shape() &&
        grad_history.dims() == 2 && grad_history.dim_size(0) == batch &&
        grad_history.dim_size(1) > 0 && grad_history.dim_size(1) % neurons == 0,
        errors::InvalidArgument("Invalid NEST type-indexed state/history spike or history gradient shape"));
    Tensor* outputs[6];
    const TensorShape shapes[6] = {
        voltage.shape(), asc_shape, psc_shape, psc_shape, psc_shape, grad_history.shape()};
    for (int output = 0; output < 6; ++output) {
      OP_REQUIRES_OK(context, context->allocate_output(output, shapes[output], &outputs[output]));
    }
    if (voltage.NumElements() == 0) return;
    Launch(context, voltage, neurons, type_count, shared_bytes,
           grad_history.dim_size(1), spike_index, outputs);
  }

 private:
  void Launch(OpKernelContext* context, const Tensor& voltage, int neurons,
              int type_count, size_t shared_bytes, int64 history_width,
              int spike_index, Tensor** outputs) {
    auto& device = context->eigen_device<GPUDevice>();
    auto config = GetGpuLaunchConfig(voltage.NumElements(), device);
    const int history_index = spike_index + 1;
    OP_REQUIRES_OK(context, GpuLaunchKernel(
        NestHistoryBackwardTypeIndexedKernel<T, R, S, Events>,
        config.block_count, config.thread_per_block, shared_bytes,
        device.stream(), voltage.NumElements(), neurons, type_count, history_width,
        voltage.flat<T>().data(), context->input(1).flat<R>().data(),
        context->input(2).flat<T>().data(), context->input(3).flat<int64>().data(),
        context->input(4).flat<T>().data(), context->input(5).flat<T>().data(),
        context->input(6).flat<T>().data(), context->input(7).flat<T>().data(),
        context->input(8).flat<T>().data(), context->input(9).flat<S>().data(),
        context->input(10).flat<S>().data(), hard_reset_,
        Events ? context->input(11).flat<T>().data() : nullptr,
        Events ? context->input(12).flat<T>().data() : nullptr,
        (Events ? context->input(13) : context->input(11)).flat<T>().data(),
        (Events ? context->input(14) : context->input(12)).flat<T>().data(),
        context->input(spike_index).flat<S>().data(),
        context->input(history_index).flat<S>().data(),
        detach_reset_, detach_asc_reset_, pseudo_gauss_,
        outputs[0]->flat<T>().data(), outputs[1]->flat<T>().data(),
        outputs[2]->flat<S>().data(), outputs[3]->flat<S>().data(),
        outputs[4]->flat<S>().data(), outputs[5]->flat<S>().data()));
  }

  bool hard_reset_;
  bool detach_reset_ = true;
  bool detach_asc_reset_ = true;
  bool pseudo_gauss_ = false;
};


#define REGISTER_NEST(T, R, S) \
  REGISTER_KERNEL_BUILDER(Name("DpointnetNestStateHistoryForward").Device(DEVICE_GPU) \
      .TypeConstraint<T>("T").TypeConstraint<R>("R").TypeConstraint<S>("S"), NestStateOp<T, R, S, false, false, true>); \
  REGISTER_KERNEL_BUILDER(Name("DpointnetNestStateForward").Device(DEVICE_GPU) \
      .TypeConstraint<T>("T").TypeConstraint<R>("R").TypeConstraint<S>("S"), NestStateOp<T, R, S, false>); \
  REGISTER_KERNEL_BUILDER(Name("DpointnetNestStateBackward").Device(DEVICE_GPU) \
      .TypeConstraint<T>("T").TypeConstraint<R>("R").TypeConstraint<S>("S"), NestStateOp<T, R, S, true>); \
  REGISTER_KERNEL_BUILDER(Name("DpointnetNestStateBackwardEvents").Device(DEVICE_GPU) \
      .TypeConstraint<T>("T").TypeConstraint<R>("R").TypeConstraint<S>("S"), NestStateOp<T, R, S, true, true>); \
  REGISTER_KERNEL_BUILDER(Name("DpointnetNestStateForwardTypeIndexed").Device(DEVICE_GPU) \
      .TypeConstraint<T>("T").TypeConstraint<R>("R").TypeConstraint<S>("S"), NestStateTypeIndexedOp<T, R, S, false>); \
  REGISTER_KERNEL_BUILDER(Name("DpointnetNestStateHistoryForwardTypeIndexed").Device(DEVICE_GPU) \
      .TypeConstraint<T>("T").TypeConstraint<R>("R").TypeConstraint<S>("S"), NestStateTypeIndexedOp<T, R, S, false, false, true>); \
  REGISTER_KERNEL_BUILDER(Name("DpointnetNestStateBackwardTypeIndexed").Device(DEVICE_GPU) \
      .TypeConstraint<T>("T").TypeConstraint<R>("R").TypeConstraint<S>("S"), NestStateTypeIndexedOp<T, R, S, true>); \
  REGISTER_KERNEL_BUILDER(Name("DpointnetNestStateBackwardEventsTypeIndexed").Device(DEVICE_GPU) \
      .TypeConstraint<T>("T").TypeConstraint<R>("R").TypeConstraint<S>("S"), NestStateTypeIndexedOp<T, R, S, true, true>); \
  REGISTER_KERNEL_BUILDER(Name("DpointnetNestStateHistoryBackward").Device(DEVICE_GPU) \
      .TypeConstraint<T>("T").TypeConstraint<R>("R").TypeConstraint<S>("S"), NestStateHistoryBackwardOp<T, R, S>); \
  REGISTER_KERNEL_BUILDER(Name("DpointnetNestStateHistoryBackwardEvents").Device(DEVICE_GPU) \
      .TypeConstraint<T>("T").TypeConstraint<R>("R").TypeConstraint<S>("S"), NestStateHistoryBackwardOp<T, R, S, true>); \
  REGISTER_KERNEL_BUILDER(Name("DpointnetNestStateHistoryBackwardTypeIndexed").Device(DEVICE_GPU) \
      .TypeConstraint<T>("T").TypeConstraint<R>("R").TypeConstraint<S>("S"), NestStateHistoryTypeIndexedBackwardOp<T, R, S>); \
  REGISTER_KERNEL_BUILDER(Name("DpointnetNestStateHistoryBackwardEventsTypeIndexed").Device(DEVICE_GPU) \
      .TypeConstraint<T>("T").TypeConstraint<R>("R").TypeConstraint<S>("S"), NestStateHistoryTypeIndexedBackwardOp<T, R, S, true>);
REGISTER_NEST(float, int8, float)
REGISTER_NEST(float, int16, float)
REGISTER_NEST(Eigen::half, int8, Eigen::half)
REGISTER_NEST(Eigen::half, int16, Eigen::half)
REGISTER_NEST(float, int8, Eigen::half)
REGISTER_NEST(float, int16, Eigen::half)
#undef REGISTER_NEST

enum class VoltagePenaltyMode : int { kRange = 0, kThreshold = 1 };

__device__ __forceinline__ float VoltagePenaltyTerm(float value, VoltagePenaltyMode mode) {
  if (mode == VoltagePenaltyMode::kRange) {
    const float centered = value - 0.5f;
    const float outside = fmaxf(fabsf(centered) - 0.5f, 0.0f);
    return outside * outside;
  }
  const float offset = value - 1.0f;
  return offset * offset;
}

// Two-stage deterministic reduction. One block per sample left most SMs idle
// (32 blocks on the full network, ~89 us/step on an RTX 3090); partial sums
// over neuron chunks spread the read over the whole device and a fixed-order
// second pass keeps the result deterministic.
constexpr int kVoltagePenaltyThreads = 256;
constexpr int kVoltagePenaltyMaxChunks = 64;

template <typename T>
__global__ void VoltagePenaltyPartialKernel(
    int neurons, int chunk_neurons, const T* voltage, VoltagePenaltyMode mode,
    float* partial) {
  __shared__ float shared[kVoltagePenaltyThreads / 32];
  const int chunk = blockIdx.x;
  const int sample = blockIdx.y;
  const int begin = chunk * chunk_neurons;
  const int end = min(begin + chunk_neurons, neurons);
  const T* row = voltage + static_cast<int64>(sample) * neurons;
  float sum = 0.0f;
  for (int neuron = begin + threadIdx.x; neuron < end; neuron += blockDim.x) {
    sum += VoltagePenaltyTerm(ToFloat(row[neuron]), mode);
  }
  #pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    sum += __shfl_down_sync(0xffffffffu, sum, offset);
  }
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  if (lane == 0) shared[warp] = sum;
  __syncthreads();
  if (warp == 0) {
    sum = lane < (blockDim.x >> 5) ? shared[lane] : 0.0f;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      sum += __shfl_down_sync(0xffffffffu, sum, offset);
    }
    if (lane == 0) partial[sample * gridDim.x + chunk] = sum;
  }
}

__global__ void VoltagePenaltyFinalizeKernel(
    int chunks, int neurons, const float* partial, float* penalty) {
  const int sample = blockIdx.x;
  const int lane = threadIdx.x;
  float sum = 0.0f;
  for (int chunk = lane; chunk < chunks; chunk += 32) {
    sum += partial[sample * chunks + chunk];
  }
  #pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    sum += __shfl_down_sync(0xffffffffu, sum, offset);
  }
  if (lane == 0) penalty[sample] = sum / static_cast<float>(neurons);
}

template <typename T>
__global__ void VoltagePenaltyBackwardKernel(
    int64 count, int neurons, const T* voltage, const float* grad_penalty,
    VoltagePenaltyMode mode, T* voltage_grad) {
  const float inverse_neurons = 1.0f / static_cast<float>(neurons);
  GPU_1D_KERNEL_LOOP(index, count) {
    const int sample = index / neurons;
    const float upstream = grad_penalty[sample] * inverse_neurons;
    const float value = ToFloat(voltage[index]);
    float grad;
    if (mode == VoltagePenaltyMode::kRange) {
      const float centered = value - 0.5f;
      const float outside = fmaxf(fabsf(centered) - 0.5f, 0.0f);
      const float sign = (centered > 0.0f) - (centered < 0.0f);
      grad = upstream * 2.0f * outside * sign;
    } else {
      grad = upstream * 2.0f * (value - 1.0f);
    }
    voltage_grad[index] = FromFloat<T>(grad);
  }
}

template <typename T, bool Backward>
class VoltagePenaltyOp : public OpKernel {
 public:
  explicit VoltagePenaltyOp(OpKernelConstruction* context) : OpKernel(context) {
    string mode;
    OP_REQUIRES_OK(context, context->GetAttr("mode", &mode));
    mode_ = mode == "threshold" ? VoltagePenaltyMode::kThreshold
                                : VoltagePenaltyMode::kRange;
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& voltage = context->input(0);
    OP_REQUIRES(context, voltage.dims() == 2 && voltage.dim_size(1) > 0,
                errors::InvalidArgument("voltage must have shape [batch, neurons]"));
    const int batch = voltage.dim_size(0);
    const int neurons = voltage.dim_size(1);
    auto& device = context->eigen_device<GPUDevice>();
    if constexpr (Backward) {
      OP_REQUIRES(context, context->input(1).shape() == TensorShape({batch}),
                  errors::InvalidArgument("grad_penalty must have shape [batch]"));
      Tensor* output;
      OP_REQUIRES_OK(context, context->allocate_output(0, voltage.shape(), &output));
      auto config = GetGpuLaunchConfig(voltage.NumElements(), device);
      OP_REQUIRES_OK(context, GpuLaunchKernel(
          VoltagePenaltyBackwardKernel<T>, config.block_count, config.thread_per_block, 0,
          device.stream(), voltage.NumElements(), neurons, voltage.flat<T>().data(),
          context->input(1).flat<float>().data(), mode_, output->flat<T>().data()));
    } else {
      Tensor* output;
      OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({batch}), &output));
      const int chunks = std::max(1, std::min(
          kVoltagePenaltyMaxChunks,
          (neurons + 4 * kVoltagePenaltyThreads - 1) / (4 * kVoltagePenaltyThreads)));
      const int chunk_neurons = (neurons + chunks - 1) / chunks;
      Tensor partial;
      OP_REQUIRES_OK(context, context->allocate_temp(
          DT_FLOAT, TensorShape({static_cast<int64>(batch) * chunks}), &partial));
      OP_REQUIRES_OK(context, GpuLaunchKernel(
          VoltagePenaltyPartialKernel<T>, dim3(chunks, batch), kVoltagePenaltyThreads, 0,
          device.stream(), neurons, chunk_neurons, voltage.flat<T>().data(), mode_,
          partial.flat<float>().data()));
      OP_REQUIRES_OK(context, GpuLaunchKernel(
          VoltagePenaltyFinalizeKernel, batch, 32, 0, device.stream(), chunks, neurons,
          partial.flat<float>().data(), output->flat<float>().data()));
    }
  }

 private:
  VoltagePenaltyMode mode_;
};

#define REGISTER_VOLTAGE_PENALTY(T) \
  REGISTER_KERNEL_BUILDER(Name("DpointnetVoltagePenaltyForward").Device(DEVICE_GPU) \
      .TypeConstraint<T>("T"), VoltagePenaltyOp<T, false>); \
  REGISTER_KERNEL_BUILDER(Name("DpointnetVoltagePenaltyBackward").Device(DEVICE_GPU) \
      .TypeConstraint<T>("T"), VoltagePenaltyOp<T, true>);
REGISTER_VOLTAGE_PENALTY(float)
REGISTER_VOLTAGE_PENALTY(Eigen::half)
#undef REGISTER_VOLTAGE_PENALTY

template <typename T, typename R, typename S, int Basis>
__global__ void StateForwardKernel(
    int64 count, int neurons, const S* z, const T* v, const R* r,
    const T* asc, const S* rise, const S* psc, const S* inputs,
    const T* syn_decay, const T* initial, const T* asc_decay,
    const T* asc_amps, const T* decay, const T* current_factor,
    const R* t_ref, const T* dt, const T* v_reset, bool hard_reset,
    T* new_v, R* new_r, T* new_asc, S* new_rise, S* new_psc) {
  GPU_1D_KERNEL_LOOP(index, count) {
    const int neuron = index % neurons;
    const int parameter_base = neuron * Basis;
    const int64 state_base = index * Basis;
    const float reset = ToFloat(z[index]);
    float current = ToFloat(asc[2 * index]) + ToFloat(asc[2 * index + 1]);
    #pragma unroll
    for (int basis = 0; basis < Basis; ++basis) {
      const int parameter = parameter_base + basis;
      const int64 state = state_base + basis;
      const float synaptic_decay = ToFloat(syn_decay[parameter]);
      const float old_rise = ToFloat(rise[state]);
      current += ToFloat(psc[state]);
      new_rise[state] = FromFloat<S>(
          old_rise * synaptic_decay +
          ToFloat(inputs[state]) * ToFloat(initial[parameter]));
      new_psc[state] = FromFloat<S>(
          ToFloat(psc[state]) * synaptic_decay +
          ToFloat(*dt) * synaptic_decay * old_rise);
    }
    const int refractory = max(
        static_cast<int>(r[index]) +
            static_cast<int>(reset) * static_cast<int>(t_ref[neuron]) - 1,
        0);
    float voltage = ToFloat(decay[neuron]) * ToFloat(v[index]) +
                    ToFloat(current_factor[neuron]) * current - reset;
    if (hard_reset && refractory > 0) {
      voltage = ToFloat(*v_reset);
    }
    new_v[index] = FromFloat<T>(voltage);
    new_r[index] = static_cast<R>(refractory);
    new_asc[2 * index] = FromFloat<T>(
        ToFloat(asc_decay[2 * neuron]) * ToFloat(asc[2 * index]) +
        reset * ToFloat(asc_amps[2 * neuron]));
    new_asc[2 * index + 1] = FromFloat<T>(
        ToFloat(asc_decay[2 * neuron + 1]) * ToFloat(asc[2 * index + 1]) +
        reset * ToFloat(asc_amps[2 * neuron + 1]));
  }
}

template <typename T, typename R, typename S, int Basis>
__global__ void StateBackwardKernel(
    int64 count, int neurons, const S* z, const R* r,
    const T* syn_decay, const T* initial, const T* asc_decay, const T* asc_amps,
    const T* decay, const T* current_factor, const R* t_ref, const T* dt,
    const T* voltage_gradient_retention, const T* grad_v, const T* grad_asc,
    const S* grad_rise, const S* grad_psc, bool hard_reset,
    bool detach_reset, bool detach_asc_reset, S* z_gradient,
    T* v_gradient, T* asc_gradient, S* rise_gradient, S* psc_gradient,
    S* input_gradient) {
  GPU_1D_KERNEL_LOOP(index, count) {
    const int neuron = index % neurons;
    const int refractory = max(
        static_cast<int>(r[index]) +
            static_cast<int>(ToFloat(z[index])) *
                static_cast<int>(t_ref[neuron]) - 1,
        0);
    const float active_voltage_gradient =
        hard_reset && refractory > 0 ? 0.0f : ToFloat(grad_v[index]);
    const float current_gradient =
        active_voltage_gradient * ToFloat(current_factor[neuron]);
    float event_gradient = detach_reset ? 0.0f : -active_voltage_gradient;
    if (!detach_asc_reset) {
      event_gradient += ToFloat(grad_asc[2 * index]) * ToFloat(asc_amps[2 * neuron])
          + ToFloat(grad_asc[2 * index + 1]) * ToFloat(asc_amps[2 * neuron + 1]);
    }
    z_gradient[index] = FromFloat<S>(event_gradient);
    v_gradient[index] = FromFloat<T>(
        active_voltage_gradient * ToFloat(decay[neuron]) *
        ToFloat(*voltage_gradient_retention));
    asc_gradient[2 * index] = FromFloat<T>(
        current_gradient +
        ToFloat(grad_asc[2 * index]) * ToFloat(asc_decay[2 * neuron]));
    asc_gradient[2 * index + 1] = FromFloat<T>(
        current_gradient + ToFloat(grad_asc[2 * index + 1]) *
                               ToFloat(asc_decay[2 * neuron + 1]));
    const int parameter_base = neuron * Basis;
    const int64 state_base = index * Basis;
    #pragma unroll
    for (int basis = 0; basis < Basis; ++basis) {
      const int parameter = parameter_base + basis;
      const int64 state = state_base + basis;
      const float synaptic_decay = ToFloat(syn_decay[parameter]);
      rise_gradient[state] = FromFloat<S>(
          ToFloat(grad_rise[state]) * synaptic_decay +
          ToFloat(grad_psc[state]) * ToFloat(*dt) * synaptic_decay);
      psc_gradient[state] = FromFloat<S>(
          ToFloat(grad_psc[state]) * synaptic_decay + current_gradient);
      input_gradient[state] = FromFloat<S>(
          ToFloat(grad_rise[state]) * ToFloat(initial[parameter]));
    }
  }
}

template <typename T, typename S>
__global__ void SpikeForwardKernel(
    int64 count, int64 neurons, int64 width, const T* voltage,
    const bool* refractory, const S* history, S* spikes, S* new_history) {
  GPU_1D_KERNEL_LOOP(index, count) {
    const int64 batch = index / width;
    const int64 column = index - batch * width;
    if (column < neurons) {
      const int64 neuron_index = batch * neurons + column;
      const S spike = static_cast<S>(
          !refractory[neuron_index] && ToFloat(voltage[neuron_index]) > 0.0f);
      spikes[neuron_index] = spike;
      new_history[index] = spike;
    } else {
      new_history[index] = history[batch * width + column - neurons];
    }
  }
}

template <typename T, typename S>
__global__ void SpikeBackwardKernel(
    int64 history_count, int64 neuron_count, int64 neurons, int64 width,
    const T* voltage, const bool* refractory, const S* spike_grad,
    const S* history_grad, const T* dampening, const T* gauss_std,
    bool pseudo_gauss, T* voltage_grad, S* old_history_grad) {
  GPU_1D_KERNEL_LOOP(index, history_count) {
    const int64 batch = index / width;
    const int64 column = index - batch * width;
    old_history_grad[index] = column + neurons < width
                                  ? history_grad[index + neurons]
                                  : FromFloat<S>(0.0f);
    if (index < neuron_count) {
      const int64 neuron_batch = index / neurons;
      const int64 neuron = index - neuron_batch * neurons;
      const T square = FromFloat<T>(
          ToFloat(NestMul(voltage[index], voltage[index])) /
          ToFloat(NestMul(*gauss_std, *gauss_std)));
      const T triangle = pseudo_gauss
          ? FromFloat<T>(expf(-ToFloat(square)))
          : FromFloat<T>(fmaxf(1.0f - fabsf(ToFloat(voltage[index])), 0.0f));
      const T derivative = FromFloat<T>(
          ToFloat(dampening[0]) * ToFloat(triangle));
      const T upstream = FromFloat<T>(
          ToFloat(spike_grad[index]) +
          ToFloat(history_grad[neuron_batch * width + neuron]));
      voltage_grad[index] = FromFloat<T>(
          refractory[index] ? 0.0f
                            : ToFloat(upstream) * ToFloat(derivative));
    }
  }
}

template <typename T, typename R, typename S>
class StateForwardOp : public OpKernel {
 public:
  explicit StateForwardOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("hard_reset", &hard_reset_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& voltage = context->input(1);
    const Tensor& psc = context->input(5);
    const int neurons = voltage.dim_size(1);
    const int basis = psc.dim_size(1) / neurons;
    OP_REQUIRES(context, basis == 4,
                errors::InvalidArgument("fused GLIF state requires four bases"));
    Tensor *new_v, *new_r, *new_asc, *new_rise, *new_psc;
    OP_REQUIRES_OK(context, context->allocate_output(0, voltage.shape(), &new_v));
    OP_REQUIRES_OK(context, context->allocate_output(1, context->input(2).shape(), &new_r));
    OP_REQUIRES_OK(context, context->allocate_output(2, context->input(3).shape(), &new_asc));
    OP_REQUIRES_OK(context, context->allocate_output(3, context->input(4).shape(), &new_rise));
    OP_REQUIRES_OK(context, context->allocate_output(4, psc.shape(), &new_psc));
    auto& device = context->eigen_device<GPUDevice>();
    auto config = GetGpuLaunchConfig(voltage.NumElements(), device);
    OP_REQUIRES_OK(context, GpuLaunchKernel(
        StateForwardKernel<T, R, S, 4>, config.block_count,
        config.thread_per_block, 0, device.stream(), voltage.NumElements(),
        neurons, context->input(0).flat<S>().data(), voltage.flat<T>().data(),
        context->input(2).flat<R>().data(), context->input(3).flat<T>().data(),
        context->input(4).flat<S>().data(), psc.flat<S>().data(),
        context->input(6).flat<S>().data(), context->input(7).flat<T>().data(),
        context->input(8).flat<T>().data(), context->input(9).flat<T>().data(),
        context->input(10).flat<T>().data(), context->input(11).flat<T>().data(),
        context->input(12).flat<T>().data(), context->input(13).flat<R>().data(),
        context->input(14).flat<T>().data(), context->input(15).flat<T>().data(),
        hard_reset_, new_v->flat<T>().data(), new_r->flat<R>().data(),
        new_asc->flat<T>().data(), new_rise->flat<S>().data(),
        new_psc->flat<S>().data()));
  }

 private:
  bool hard_reset_;
};

template <typename T, typename R, typename S>
class StateBackwardOp : public OpKernel {
 public:
  explicit StateBackwardOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("hard_reset", &hard_reset_));
    OP_REQUIRES_OK(context, context->GetAttr("detach_reset", &detach_reset_));
    OP_REQUIRES_OK(context, context->GetAttr("detach_asc_reset", &detach_asc_reset_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& grad_v = context->input(11);
    const Tensor& grad_rise = context->input(13);
    const int neurons = grad_v.dim_size(1);
    const int basis = grad_rise.dim_size(1) / neurons;
    OP_REQUIRES(context, basis == 4,
                errors::InvalidArgument("fused GLIF state requires four bases"));
    Tensor *z_grad, *v_grad, *asc_grad, *rise_grad, *psc_grad, *input_grad;
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_v.shape(), &z_grad));
    OP_REQUIRES_OK(context, context->allocate_output(1, grad_v.shape(), &v_grad));
    OP_REQUIRES_OK(context, context->allocate_output(2, context->input(12).shape(), &asc_grad));
    OP_REQUIRES_OK(context, context->allocate_output(3, grad_rise.shape(), &rise_grad));
    OP_REQUIRES_OK(context, context->allocate_output(4, grad_rise.shape(), &psc_grad));
    OP_REQUIRES_OK(context, context->allocate_output(5, grad_rise.shape(), &input_grad));
    auto& device = context->eigen_device<GPUDevice>();
    auto config = GetGpuLaunchConfig(grad_v.NumElements(), device);
    OP_REQUIRES_OK(context, GpuLaunchKernel(
        StateBackwardKernel<T, R, S, 4>, config.block_count,
        config.thread_per_block, 0, device.stream(), grad_v.NumElements(),
        neurons, context->input(0).flat<S>().data(),
        context->input(1).flat<R>().data(), context->input(2).flat<T>().data(),
        context->input(3).flat<T>().data(), context->input(4).flat<T>().data(),
        context->input(5).flat<T>().data(),
        context->input(6).flat<T>().data(), context->input(7).flat<T>().data(),
        context->input(8).flat<R>().data(), context->input(9).flat<T>().data(),
        context->input(10).flat<T>().data(), grad_v.flat<T>().data(),
        context->input(12).flat<T>().data(), grad_rise.flat<S>().data(),
        context->input(14).flat<S>().data(), hard_reset_, detach_reset_, detach_asc_reset_,
        z_grad->flat<S>().data(), v_grad->flat<T>().data(),
        asc_grad->flat<T>().data(), rise_grad->flat<S>().data(),
        psc_grad->flat<S>().data(), input_grad->flat<S>().data()));
  }

 private:
  bool hard_reset_;
  bool detach_reset_;
  bool detach_asc_reset_;
};

template <typename T, typename S>
class SpikeForwardOp : public OpKernel {
 public:
  explicit SpikeForwardOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& voltage = context->input(0);
    const Tensor& refractory = context->input(1);
    const Tensor& history = context->input(2);
    OP_REQUIRES(context, voltage.dims() == 2 && history.dims() == 2,
          errors::InvalidArgument("voltage and history must have rank two"));
    OP_REQUIRES(context, voltage.shape() == refractory.shape(),
                errors::InvalidArgument("voltage and refractory shapes differ"));
    const int64 neurons = voltage.dim_size(1);
    const int64 width = history.dim_size(1);
    OP_REQUIRES(context, voltage.dim_size(0) == history.dim_size(0),
          errors::InvalidArgument("voltage and history batch sizes differ"));
    OP_REQUIRES(context, neurons > 0 && width > 0 && width % neurons == 0,
                errors::InvalidArgument("history width must be a positive multiple of neurons"));
    Tensor* spikes;
    Tensor* new_history;
    OP_REQUIRES_OK(context, context->allocate_output(0, voltage.shape(), &spikes));
    OP_REQUIRES_OK(context, context->allocate_output(1, history.shape(), &new_history));
    if (voltage.NumElements() == 0) return;
    auto& device = context->eigen_device<GPUDevice>();
    auto config = GetGpuLaunchConfig(history.NumElements(), device);
    OP_REQUIRES_OK(context, GpuLaunchKernel(
        SpikeForwardKernel<T, S>, config.block_count, config.thread_per_block, 0,
        device.stream(), history.NumElements(), neurons, width,
        voltage.flat<T>().data(), refractory.flat<bool>().data(),
        history.flat<S>().data(), spikes->flat<S>().data(),
        new_history->flat<S>().data()));
  }
};

template <typename T, typename S>
class SpikeBackwardOp : public OpKernel {
 public:
  explicit SpikeBackwardOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("pseudo_gauss", &pseudo_gauss_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& voltage = context->input(0);
    const Tensor& refractory = context->input(1);
    const Tensor& spike_grad = context->input(2);
    const Tensor& history_grad = context->input(3);
    const Tensor& dampening = context->input(4);
    const Tensor& gauss_std = context->input(5);
    OP_REQUIRES(context, gauss_std.NumElements() == 1,
          errors::InvalidArgument("gauss_std must contain one value"));
    OP_REQUIRES(context, voltage.dims() == 2 && history_grad.dims() == 2,
          errors::InvalidArgument("voltage and history gradient must have rank two"));
    OP_REQUIRES(context,
          voltage.shape() == refractory.shape() && voltage.shape() == spike_grad.shape(),
          errors::InvalidArgument("voltage, refractory and spike gradient shapes differ"));
    const int64 neurons = voltage.dim_size(1);
    const int64 width = history_grad.dim_size(1);
    OP_REQUIRES(context, voltage.dim_size(0) == history_grad.dim_size(0),
          errors::InvalidArgument("voltage and history gradient batch sizes differ"));
    OP_REQUIRES(context, neurons > 0 && width > 0 && width % neurons == 0,
          errors::InvalidArgument("history width must be a positive multiple of neurons"));
    OP_REQUIRES(context, dampening.NumElements() == 1,
          errors::InvalidArgument("dampening must contain one value"));
    Tensor* voltage_grad;
    Tensor* old_history_grad;
    OP_REQUIRES_OK(context, context->allocate_output(0, voltage.shape(), &voltage_grad));
    OP_REQUIRES_OK(context, context->allocate_output(1, history_grad.shape(), &old_history_grad));
    if (voltage.NumElements() == 0) return;
    auto& device = context->eigen_device<GPUDevice>();
    auto config = GetGpuLaunchConfig(history_grad.NumElements(), device);
    OP_REQUIRES_OK(context, GpuLaunchKernel(
        SpikeBackwardKernel<T, S>, config.block_count, config.thread_per_block, 0,
        device.stream(), history_grad.NumElements(), voltage.NumElements(),
        neurons, width, voltage.flat<T>().data(),
        refractory.flat<bool>().data(), spike_grad.flat<S>().data(),
        history_grad.flat<S>().data(), dampening.flat<T>().data(),
        gauss_std.flat<T>().data(), pseudo_gauss_,
        voltage_grad->flat<T>().data(), old_history_grad->flat<S>().data()));
  }
 private:
  bool pseudo_gauss_;
};

#define REGISTER_STATE(T, R, S)                                             \
  REGISTER_KERNEL_BUILDER(                                                  \
      Name("DpointnetGlifStateForward")                                    \
          .Device(DEVICE_GPU)                                               \
          .TypeConstraint<T>("T")                                          \
          .TypeConstraint<S>("S")                                          \
          .TypeConstraint<R>("R"),                                         \
      StateForwardOp<T, R, S>);                                             \
  REGISTER_KERNEL_BUILDER(                                                  \
      Name("DpointnetGlifStateBackward")                                   \
          .Device(DEVICE_GPU)                                               \
          .TypeConstraint<T>("T")                                          \
          .TypeConstraint<S>("S")                                          \
          .TypeConstraint<R>("R"),                                         \
      StateBackwardOp<T, R, S>);

REGISTER_STATE(float, int8, float)
REGISTER_STATE(float, int16, float)
REGISTER_STATE(Eigen::half, int8, Eigen::half)
REGISTER_STATE(Eigen::half, int16, Eigen::half)
REGISTER_STATE(float, int8, Eigen::half)
REGISTER_STATE(float, int16, Eigen::half)
#undef REGISTER_STATE

#define REGISTER_SPIKE(T, S)                                                \
  REGISTER_KERNEL_BUILDER(                                                  \
      Name("DpointnetSpikeShift").Device(DEVICE_GPU).TypeConstraint<T>("T").TypeConstraint<S>("S"), \
      SpikeForwardOp<T, S>);                                                \
  REGISTER_KERNEL_BUILDER(                                                  \
      Name("DpointnetSpikeShiftBackwardV2")                                 \
          .Device(DEVICE_GPU)                                               \
          .TypeConstraint<T>("T").TypeConstraint<S>("S"),                    \
      SpikeBackwardOp<T, S>);

REGISTER_SPIKE(float, float)
REGISTER_SPIKE(Eigen::half, Eigen::half)
REGISTER_SPIKE(float, Eigen::half)
#undef REGISTER_SPIKE

#endif
