#pragma once

namespace tensorflow {
template <typename T, typename I, bool Gradient>
__global__ void MasterProjectionKernel(
    int64_t count, int64_t width, int posts, int64_t edges, const T* spikes,
    const float* weights, const I* meta, const T* basis, const float* dy,
    float* currents, float* dz, float* dw) {
  const I* splits = meta + 2 * edges;
  const I* canonical = splits + width + 1;
  for (int64_t index : GpuGridRangeX(count)) {
    const int64_t sample = index / width, row = index % width;
    const float spike = ToFloat(spikes[index]);
    float credit = 0;
    for (int64_t e = splits[row]; e < splits[row + 1]; ++e) {
      const int64_t post = meta[e], type = meta[edges + e], edge = canonical[e];
      float projected = 0;
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        const float coefficient = ToFloat(basis[type * 4 + k]);
        const int64_t target = (sample * posts + post) * 4 + k;
        if constexpr (Gradient) projected += dy[target] * coefficient;
        else if (spike != 0) atomicAdd(currents + target, spike * weights[edge] * coefficient);
      }
      if constexpr (Gradient) {
        credit += weights[edge] * projected;
        atomicAdd(dw + edge, spike * projected);
      }
    }
    if constexpr (Gradient) dz[index] = credit;
  }
}

template <typename T, typename I, bool Gradient>
class MasterProjectionGpu : public OpKernel {
 public:
  explicit MasterProjectionGpu(OpKernelConstruction* c) : OpKernel(c) {
    OP_REQUIRES_OK(c, c->GetAttr("n_post", &posts_));
    OP_REQUIRES_OK(c, c->GetAttr("n_edges", &edges_));
  }
  void Compute(OpKernelContext* c) override {
    const auto& z = c->input(0);
    OP_REQUIRES(c, z.dims() == 2 && c->input(1).dims() == 1 &&
        c->input(2).dims() == 1 && c->input(3).dims() == 2,
        errors::InvalidArgument("Master projection expects ranks 2,1,1,2"));
    OP_REQUIRES(c, c->input(1).NumElements() == edges_ &&
        c->input(2).NumElements() >= 3 * edges_ + z.dim_size(1) + 1 &&
        c->input(3).dim_size(1) == 4,
        errors::InvalidArgument("Master projection operand dimensions disagree"));
    Tensor *out, *dw = nullptr;
    const int64_t currents = z.dim_size(0) * posts_ * 4;
    auto device = c->eigen_device<GPUDevice>();
    if constexpr (Gradient) {
      OP_REQUIRES(c, c->input(4).shape() == TensorShape({z.dim_size(0) * posts_, 4}),
                  errors::InvalidArgument("Master projection cotangent shape disagrees"));
      OP_REQUIRES_OK(c, c->allocate_output(0, z.shape(), &out));
      OP_REQUIRES_OK(c, c->allocate_output(1, c->input(1).shape(), &dw));
      OP_REQUIRES_OK(c, GpuLaunchKernel(SetZeroKernel<float>,
          BlockCountFor(edges_, 256, device), 256, 0, device.stream(),
          edges_, dw->flat<float>().data()));
    } else {
      OP_REQUIRES_OK(c, c->allocate_output(0, TensorShape({z.dim_size(0) * posts_, 4}), &out));
      OP_REQUIRES_OK(c, GpuLaunchKernel(SetZeroKernel<float>,
          BlockCountFor(currents, 256, device), 256, 0, device.stream(),
          currents, out->flat<float>().data()));
    }
    if (z.NumElements()) {
      OP_REQUIRES_OK(c, GpuLaunchKernel((MasterProjectionKernel<T, I, Gradient>),
          BlockCountFor(z.NumElements(), 256, device), 256, 0, device.stream(),
          z.NumElements(), z.dim_size(1), posts_, edges_, z.flat<T>().data(),
          c->input(1).flat<float>().data(), c->input(2).flat<I>().data(),
          c->input(3).flat<T>().data(),
          Gradient ? c->input(4).flat<float>().data() : nullptr,
          Gradient ? nullptr : out->flat<float>().data(),
          Gradient ? out->flat<float>().data() : nullptr,
          Gradient ? dw->flat<float>().data() : nullptr));
    }
  }
 private:
  int posts_;
  int64_t edges_;
};

#define REGISTER_MASTER_GPU(T, I)                                           \
  REGISTER_KERNEL_BUILDER(Name("DpointnetMasterProjection").Device(DEVICE_GPU) \
      .TypeConstraint<T>("T").TypeConstraint<I>("I"), MasterProjectionGpu<T, I, false>); \
  REGISTER_KERNEL_BUILDER(Name("DpointnetMasterProjectionGrad").Device(DEVICE_GPU) \
      .TypeConstraint<T>("T").TypeConstraint<I>("I"), MasterProjectionGpu<T, I, true>);
REGISTER_MASTER_GPU(Eigen::half, uint32)
REGISTER_MASTER_GPU(Eigen::half, int64_t)
REGISTER_MASTER_GPU(float, uint32)
REGISTER_MASTER_GPU(float, int64_t)
#undef REGISTER_MASTER_GPU
}  // namespace tensorflow
