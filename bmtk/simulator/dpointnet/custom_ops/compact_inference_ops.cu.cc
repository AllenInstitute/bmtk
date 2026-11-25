#define EIGEN_USE_GPU
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"

using namespace tensorflow;
using GPUDevice = Eigen::GpuDevice;

template <typename T>
__global__ void CompactProject(const T* spikes, const int64_t* ptr,
    const uint32_t* targets, const T* weights, const T* basis,
    int64_t rows, int64_t batch, int64_t neurons, int64_t types, int64_t bases, float* out) {
  for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < rows * batch; index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const float spike = static_cast<float>(spikes[index]);
    if (spike == 0.0f) continue;
    const int64_t row = index % rows, slot = index / rows;
    for (int64_t edge = ptr[row]; edge < ptr[row + 1]; ++edge) {
      const int64_t target = targets[edge] / types, type = targets[edge] % types;
      for (int64_t receptor = 0; receptor < bases; ++receptor) {
        const float factor = static_cast<float>(basis[type * bases + receptor]);
        if (factor != 0.0f)
          atomicAdd(out + (slot * neurons + target) * bases + receptor,
              spike * static_cast<float>(weights[edge]) * factor);
      }
    }
  }
}

template <typename T>
class CompactInferenceGPU : public OpKernel {
 public:
  explicit CompactInferenceGPU(OpKernelConstruction* c) : OpKernel(c) {
    OP_REQUIRES_OK(c, c->GetAttr("n_neurons", &neurons_));
  }
  void Compute(OpKernelContext* c) override {
    const Tensor &spikes = c->input(0), &ptr = c->input(1),
                 &targets = c->input(2), &weights = c->input(3), &basis = c->input(4);
    OP_REQUIRES(c, spikes.dims() == 2 && ptr.dims() == 1 &&
        targets.dims() == 1 && weights.dims() == 1 && basis.dims() == 2 &&
        basis.dim_size(0) > 0 && basis.dim_size(1) > 0 &&
        spikes.dim_size(1) % neurons_ == 0 &&
        ptr.NumElements() == spikes.dim_size(1) + 1 &&
        targets.NumElements() == weights.NumElements(),
        errors::InvalidArgument("Invalid compact inference dimensions"));
    Tensor* out;
    OP_REQUIRES_OK(c, c->allocate_output(0,
        {spikes.dim_size(0) * neurons_, basis.dim_size(1)}, &out));
    Tensor accumulator;
    OP_REQUIRES_OK(c, c->allocate_temp(DT_FLOAT, out->shape(), &accumulator));
    auto device = c->eigen_device<GPUDevice>();
    accumulator.flat<float>().device(device) = accumulator.flat<float>().constant(0);
    const int64_t work = spikes.NumElements();
    if (work > 0) {
      const int blocks = static_cast<int>(std::min<int64_t>((work + 255) / 256, 4096));
      OP_REQUIRES_OK(c, GpuLaunchKernel(CompactProject<T>, blocks, 256, 0, device.stream(),
          spikes.flat<T>().data(), ptr.flat<int64_t>().data(), targets.flat<uint32>().data(),
          weights.flat<T>().data(), basis.flat<T>().data(), spikes.dim_size(1),
          spikes.dim_size(0), neurons_, basis.dim_size(0), basis.dim_size(1),
          accumulator.flat<float>().data()));
    }
    out->flat<T>().device(device) = accumulator.flat<float>().cast<T>();
  }
 private:
  int64_t neurons_;
};

REGISTER_KERNEL_BUILDER(Name("CompactInferenceCurrents").Device(DEVICE_GPU)
    .TypeConstraint<float>("T"), CompactInferenceGPU<float>);
REGISTER_KERNEL_BUILDER(Name("CompactInferenceCurrents").Device(DEVICE_GPU)
    .TypeConstraint<Eigen::half>("T"), CompactInferenceGPU<Eigen::half>);
