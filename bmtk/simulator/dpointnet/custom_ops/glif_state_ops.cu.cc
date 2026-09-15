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
    OP_REQUIRES(context, voltage.shape() == refractory.shape(),
                errors::InvalidArgument("voltage and refractory shapes differ"));
    const int64 neurons = voltage.dim_size(1);
    const int64 width = history.dim_size(1);
    OP_REQUIRES(context, neurons > 0 && width % neurons == 0,
                errors::InvalidArgument("history width must be a positive multiple of neurons"));
    Tensor* spikes;
    Tensor* new_history;
    OP_REQUIRES_OK(context, context->allocate_output(0, voltage.shape(), &spikes));
    OP_REQUIRES_OK(context, context->allocate_output(1, history.shape(), &new_history));
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
    Tensor* voltage_grad;
    Tensor* old_history_grad;
    OP_REQUIRES_OK(context, context->allocate_output(0, voltage.shape(), &voltage_grad));
    OP_REQUIRES_OK(context, context->allocate_output(1, history_grad.shape(), &old_history_grad));
    const int64 neurons = voltage.dim_size(1);
    const int64 width = history_grad.dim_size(1);
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
