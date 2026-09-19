#if GOOGLE_CUDA
#define EIGEN_USE_GPU

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

template <typename T, typename R>
__global__ void NestForwardKernel(
    int64 count, int neurons, const T* voltage, const R* refractory,
    const T* asc, const T* rise, const T* psc, const T* currents,
    const T* coefficients, const R* t_ref, const T* dt, const T* v_th,
    bool hard_reset, T* threshold, T* new_v, R* new_r, T* new_asc,
    T* new_rise, T* new_psc) {
  GPU_1D_KERNEL_LOOP(index, count) {
    const int neuron = index % neurons;
    const T* params = coefficients + neuron * 28;
    const bool active = refractory[index] <= 0;
    const T mean_asc = NestAdd(NestMul(asc[index * 2], params[14]),
                               NestMul(asc[index * 2 + 1], params[15]));
    T contributions[4];
    #pragma unroll
    for (int basis = 0; basis < 4; ++basis) {
      const int64 offset = index * 4 + basis;
      contributions[basis] = NestAdd(NestMul(psc[offset], params[18 + basis]),
                                      NestMul(rise[offset], params[22 + basis]));
      new_rise[offset] = NestAdd(NestMul(rise[offset], params[basis]),
                                  NestMul(currents[offset], params[4 + basis]));
      new_psc[offset] = NestAdd(NestMul(psc[offset], params[basis]),
                                 NestMul(NestMul(*dt, params[basis]), rise[offset]));
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
    threshold[index] = threshold_value;
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

template <typename T, typename R>
__global__ void NestBackwardKernel(
    int64 count, int neurons, const T* threshold, const R* refractory,
    const T* coefficients, const T* dt, const T* retention,
    const T* grad_threshold, const T* grad_v, const T* grad_asc,
    const T* grad_rise, const T* grad_psc, bool hard_reset,
    T* voltage_grad, T* asc_grad, T* rise_grad, T* psc_grad, T* currents_grad) {
  GPU_1D_KERNEL_LOOP(index, count) {
    const T* params = coefficients + (index % neurons) * 28;
    const bool active = refractory[index] <= 0;
    const bool fired = active && ToFloat(threshold[index]) > 0.0f;
    T candidate_grad = NestAdd(grad_threshold[index],
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
      rise_grad[offset] = NestAdd(
          NestAdd(NestMul(grad_rise[offset], params[basis]),
                  NestMul(grad_psc[offset], NestMul(*dt, params[basis]))),
          NestMul(candidate_grad, params[22 + basis]));
      psc_grad[offset] = NestAdd(NestMul(grad_psc[offset], params[basis]),
                                  NestMul(candidate_grad, params[18 + basis]));
      currents_grad[offset] = NestMul(grad_rise[offset], params[4 + basis]);
    }
  }
}

template <typename T, typename R, bool Backward>
class NestStateOp : public OpKernel {
 public:
  explicit NestStateOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("hard_reset", &hard_reset_));
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
    OP_REQUIRES(context, context->input(coefficient_index).shape() == TensorShape({neurons, 28}),
          errors::InvalidArgument("NEST coefficients must have shape [neurons, 28]"));
    if (Backward) {
      OP_REQUIRES(context,
          context->input(3).NumElements() == 1 && context->input(4).NumElements() == 1 &&
          context->input(5).shape() == voltage.shape() && context->input(6).shape() == voltage.shape() &&
          context->input(7).shape() == asc_shape && context->input(8).shape() == psc_shape &&
          context->input(9).shape() == psc_shape,
          errors::InvalidArgument("Invalid NEST backward tensor shapes"));
    } else {
      OP_REQUIRES(context,
          context->input(2).shape() == asc_shape && context->input(3).shape() == psc_shape &&
          context->input(4).shape() == psc_shape && context->input(5).shape() == psc_shape &&
          context->input(7).shape() == TensorShape({neurons}) &&
          context->input(8).NumElements() == 1 && context->input(9).NumElements() == 1,
          errors::InvalidArgument("Invalid NEST forward tensor shapes; four bases are required"));
    }
    Tensor* outputs[6];
    const TensorShape shapes[6] = {
        voltage.shape(), Backward ? asc_shape : voltage.shape(),
        Backward ? psc_shape : voltage.shape(),
        Backward ? psc_shape : asc_shape, psc_shape, psc_shape};
    for (int output = 0; output < (Backward ? 5 : 6); ++output) {
      OP_REQUIRES_OK(context, context->allocate_output(output, shapes[output], &outputs[output]));
    }
    if (voltage.NumElements() == 0) return;
    auto& device = context->eigen_device<GPUDevice>();
    auto config = GetGpuLaunchConfig(voltage.NumElements(), device);
    if (Backward) {
      OP_REQUIRES_OK(context, GpuLaunchKernel(
          NestBackwardKernel<T, R>, config.block_count, config.thread_per_block, 0,
          device.stream(), voltage.NumElements(), neurons, voltage.flat<T>().data(),
          context->input(1).flat<R>().data(), context->input(2).flat<T>().data(),
          context->input(3).flat<T>().data(), context->input(4).flat<T>().data(),
          context->input(5).flat<T>().data(), context->input(6).flat<T>().data(),
          context->input(7).flat<T>().data(), context->input(8).flat<T>().data(),
          context->input(9).flat<T>().data(), hard_reset_,
          outputs[0]->flat<T>().data(), outputs[1]->flat<T>().data(),
          outputs[2]->flat<T>().data(), outputs[3]->flat<T>().data(), outputs[4]->flat<T>().data()));
    } else {
      OP_REQUIRES_OK(context, GpuLaunchKernel(
          NestForwardKernel<T, R>, config.block_count, config.thread_per_block, 0,
          device.stream(), voltage.NumElements(), neurons, voltage.flat<T>().data(),
          context->input(1).flat<R>().data(), context->input(2).flat<T>().data(),
          context->input(3).flat<T>().data(), context->input(4).flat<T>().data(),
          context->input(5).flat<T>().data(), context->input(6).flat<T>().data(),
          context->input(7).flat<R>().data(), context->input(8).flat<T>().data(),
          context->input(9).flat<T>().data(), hard_reset_, outputs[0]->flat<T>().data(),
          outputs[1]->flat<T>().data(), outputs[2]->flat<R>().data(),
          outputs[3]->flat<T>().data(), outputs[4]->flat<T>().data(), outputs[5]->flat<T>().data()));
    }
  }

 private:
  bool hard_reset_;
};

#define REGISTER_NEST(T, R) \
  REGISTER_KERNEL_BUILDER(Name("DpointnetNestStateForward").Device(DEVICE_GPU) \
      .TypeConstraint<T>("T").TypeConstraint<R>("R"), NestStateOp<T, R, false>); \
  REGISTER_KERNEL_BUILDER(Name("DpointnetNestStateBackward").Device(DEVICE_GPU) \
      .TypeConstraint<T>("T").TypeConstraint<R>("R"), NestStateOp<T, R, true>);
REGISTER_NEST(float, int8)
REGISTER_NEST(float, int16)
REGISTER_NEST(Eigen::half, int8)
REGISTER_NEST(Eigen::half, int16)
#undef REGISTER_NEST

template <typename T, typename R, int Basis>
__global__ void StateForwardKernel(
    int64 count, int neurons, const T* z, const T* v, const R* r,
    const T* asc, const T* rise, const T* psc, const T* inputs,
    const T* syn_decay, const T* initial, const T* asc_decay,
    const T* asc_amps, const T* decay, const T* current_factor,
    const R* t_ref, const T* dt, const T* v_reset, bool hard_reset,
    T* new_v, R* new_r, T* new_asc, T* new_rise, T* new_psc) {
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
      new_rise[state] = FromFloat<T>(
          old_rise * synaptic_decay +
          ToFloat(inputs[state]) * ToFloat(initial[parameter]));
      new_psc[state] = FromFloat<T>(
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

template <typename T, typename R, int Basis>
__global__ void StateBackwardKernel(
    int64 count, int neurons, const T* z, const R* r,
    const T* syn_decay, const T* initial, const T* asc_decay,
    const T* decay, const T* current_factor, const R* t_ref, const T* dt,
    const T* voltage_gradient_retention, const T* grad_v, const T* grad_asc,
    const T* grad_rise, const T* grad_psc, bool hard_reset, T* z_gradient,
    T* v_gradient, T* asc_gradient, T* rise_gradient, T* psc_gradient,
    T* input_gradient) {
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
    z_gradient[index] = FromFloat<T>(0.0f);
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
      rise_gradient[state] = FromFloat<T>(
          ToFloat(grad_rise[state]) * synaptic_decay +
          ToFloat(grad_psc[state]) * ToFloat(*dt) * synaptic_decay);
      psc_gradient[state] = FromFloat<T>(
          ToFloat(grad_psc[state]) * synaptic_decay + current_gradient);
      input_gradient[state] = FromFloat<T>(
          ToFloat(grad_rise[state]) * ToFloat(initial[parameter]));
    }
  }
}

template <typename T>
__global__ void SpikeForwardKernel(
    int64 count, int64 neurons, int64 width, const T* voltage,
    const bool* refractory, const T* history, T* spikes, T* new_history) {
  GPU_1D_KERNEL_LOOP(index, count) {
    const int64 batch = index / width;
    const int64 column = index - batch * width;
    if (column < neurons) {
      const int64 neuron_index = batch * neurons + column;
      const T spike = static_cast<T>(
          !refractory[neuron_index] && ToFloat(voltage[neuron_index]) > 0.0f);
      spikes[neuron_index] = spike;
      new_history[index] = spike;
    } else {
      new_history[index] = history[batch * width + column - neurons];
    }
  }
}

template <typename T>
__global__ void SpikeBackwardKernel(
    int64 history_count, int64 neuron_count, int64 neurons, int64 width,
    const T* voltage, const bool* refractory, const T* spike_grad,
    const T* history_grad, const T* dampening, T* voltage_grad,
    T* old_history_grad) {
  GPU_1D_KERNEL_LOOP(index, history_count) {
    const int64 batch = index / width;
    const int64 column = index - batch * width;
    old_history_grad[index] = column + neurons < width
                                  ? history_grad[index + neurons]
                                  : FromFloat<T>(0.0f);
    if (index < neuron_count) {
      const int64 neuron_batch = index / neurons;
      const int64 neuron = index - neuron_batch * neurons;
      const T triangle = FromFloat<T>(
          fmaxf(1.0f - fabsf(ToFloat(voltage[index])), 0.0f));
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

template <typename T, typename R>
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
        StateForwardKernel<T, R, 4>, config.block_count,
        config.thread_per_block, 0, device.stream(), voltage.NumElements(),
        neurons, context->input(0).flat<T>().data(), voltage.flat<T>().data(),
        context->input(2).flat<R>().data(), context->input(3).flat<T>().data(),
        context->input(4).flat<T>().data(), psc.flat<T>().data(),
        context->input(6).flat<T>().data(), context->input(7).flat<T>().data(),
        context->input(8).flat<T>().data(), context->input(9).flat<T>().data(),
        context->input(10).flat<T>().data(), context->input(11).flat<T>().data(),
        context->input(12).flat<T>().data(), context->input(13).flat<R>().data(),
        context->input(14).flat<T>().data(), context->input(15).flat<T>().data(),
        hard_reset_, new_v->flat<T>().data(), new_r->flat<R>().data(),
        new_asc->flat<T>().data(), new_rise->flat<T>().data(),
        new_psc->flat<T>().data()));
  }

 private:
  bool hard_reset_;
};

template <typename T, typename R>
class StateBackwardOp : public OpKernel {
 public:
  explicit StateBackwardOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("hard_reset", &hard_reset_));
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
        StateBackwardKernel<T, R, 4>, config.block_count,
        config.thread_per_block, 0, device.stream(), grad_v.NumElements(),
        neurons, context->input(0).flat<T>().data(),
        context->input(1).flat<R>().data(), context->input(2).flat<T>().data(),
        context->input(3).flat<T>().data(), context->input(4).flat<T>().data(),
        context->input(6).flat<T>().data(), context->input(7).flat<T>().data(),
        context->input(8).flat<R>().data(), context->input(9).flat<T>().data(),
        context->input(10).flat<T>().data(), grad_v.flat<T>().data(),
        context->input(12).flat<T>().data(), grad_rise.flat<T>().data(),
        context->input(14).flat<T>().data(), hard_reset_,
        z_grad->flat<T>().data(), v_grad->flat<T>().data(),
        asc_grad->flat<T>().data(), rise_grad->flat<T>().data(),
        psc_grad->flat<T>().data(), input_grad->flat<T>().data()));
  }

 private:
  bool hard_reset_;
};

template <typename T>
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
        SpikeForwardKernel<T>, config.block_count, config.thread_per_block, 0,
        device.stream(), history.NumElements(), neurons, width,
        voltage.flat<T>().data(), refractory.flat<bool>().data(),
        history.flat<T>().data(), spikes->flat<T>().data(),
        new_history->flat<T>().data()));
  }
};

template <typename T>
class SpikeBackwardOp : public OpKernel {
 public:
  explicit SpikeBackwardOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& voltage = context->input(0);
    const Tensor& refractory = context->input(1);
    const Tensor& spike_grad = context->input(2);
    const Tensor& history_grad = context->input(3);
    const Tensor& dampening = context->input(4);
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
        SpikeBackwardKernel<T>, config.block_count, config.thread_per_block, 0,
        device.stream(), history_grad.NumElements(), voltage.NumElements(),
        neurons, width, voltage.flat<T>().data(),
        refractory.flat<bool>().data(), spike_grad.flat<T>().data(),
        history_grad.flat<T>().data(), dampening.flat<T>().data(),
        voltage_grad->flat<T>().data(), old_history_grad->flat<T>().data()));
  }
};

#define REGISTER_STATE(T, R)                                                \
  REGISTER_KERNEL_BUILDER(                                                  \
      Name("DpointnetGlifStateForward")                                    \
          .Device(DEVICE_GPU)                                               \
          .TypeConstraint<T>("T")                                          \
          .TypeConstraint<R>("R"),                                         \
      StateForwardOp<T, R>);                                                \
  REGISTER_KERNEL_BUILDER(                                                  \
      Name("DpointnetGlifStateBackward")                                   \
          .Device(DEVICE_GPU)                                               \
          .TypeConstraint<T>("T")                                          \
          .TypeConstraint<R>("R"),                                         \
      StateBackwardOp<T, R>);

REGISTER_STATE(float, int8)
REGISTER_STATE(float, int16)
REGISTER_STATE(Eigen::half, int8)
REGISTER_STATE(Eigen::half, int16)
#undef REGISTER_STATE

#define REGISTER_SPIKE(T)                                                   \
  REGISTER_KERNEL_BUILDER(                                                  \
      Name("DpointnetSpikeShift").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
      SpikeForwardOp<T>);                                                   \
  REGISTER_KERNEL_BUILDER(                                                  \
      Name("DpointnetSpikeShiftBackward")                                  \
          .Device(DEVICE_GPU)                                               \
          .TypeConstraint<T>("T"),                                         \
      SpikeBackwardOp<T>);

REGISTER_SPIKE(float)
REGISTER_SPIKE(Eigen::half)
#undef REGISTER_SPIKE

#endif
