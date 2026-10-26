// Adapted from Javier Galvan's v1_model_utils/cuda_lgn/lgn_common.cuh (commit d8fb466b; credit: Javier).
// DPointNet renames the operators with the Dpointnet prefix and keeps the math unchanged.
// Shared by the LGN ops: output modes, the spike probability, and the chunked
// Bernoulli uniforms that the spike ops write into a batch in place.
#pragma once

#include <limits>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"

namespace lgn {

constexpr int kMaxEntries = 16;

enum class Output { kSpikes, kProbabilities, kRates };

// The chunk's uniforms: tensors of `per_entry` samples each.
template <typename U> struct Uniforms {
  const U* entry[kMaxEntries];
  int per_entry;
  // Constant indices only: a dynamically indexed parameter array would be
  // copied to local memory.
  __device__ const U* Sample(int sample, int64_t stride) const {
    const int index = sample / per_entry;
    const U* pointer = entry[0];
#pragma unroll
    for (int e = 1; e < kMaxEntries; ++e)
      if (e == index) pointer = entry[e];
    return pointer + static_cast<int64_t>(sample % per_entry) * stride;
  }
};

// p = 1 - exp(-rate dt), dt = 1 ms, to about half an ulp, without a division
// or the general expm1f. x = -rate / 1000 is kept as x_hi + x_lo (x_lo the
// rounding error of x_hi plus the split constant's low part), and for
// |x| <= 0.5 (rates <= 500 Hz) p = -x - x^2 (1/2 + x/6 + ...) =
// -(x_hi + (x_hi^2 s + x_lo)), s the degree-8 Taylor series (truncation
// < 2e-9 relative), so the only rounding of size ulp(p) is the final one;
// otherwise 1 - expf(x).
__device__ inline float SpikeProbability(float rate) {
  constexpr float kHigh = -1e-3f;
  constexpr float kLow = static_cast<float>(-1e-3 - static_cast<double>(kHigh));
  const float high = rate * kHigh;
  const float low = fmaf(rate, kHigh, -high) + rate * kLow;
  if (high < -0.5f) return 1.f - expf(high + low);
  float series = 1.f / 362880.f;
  series = fmaf(series, high, 1.f / 40320.f);
  series = fmaf(series, high, 1.f / 5040.f);
  series = fmaf(series, high, 1.f / 720.f);
  series = fmaf(series, high, 1.f / 120.f);
  series = fmaf(series, high, 1.f / 24.f);
  series = fmaf(series, high, 1.f / 6.f);
  series = fmaf(series, high, 0.5f);
  return -(high + fmaf(high * high, series, low));
}

// Reads the spike op's `uniforms` list (at most 16 equal [time, units] or
// [samples, time, units] tensors, the samples [offset, offset + *count) of a
// [batch, time, units] batch) and returns in *spikes input `spikes_input`
// forwarded in place, or a new output when that input is empty, so a batch is
// sampled chunk by chunk without ever holding every sample's uniforms.
template <typename U>
tensorflow::Status PrepareSpikes(tensorflow::OpKernelContext* c, int spikes_input,
                                 const tensorflow::TensorShape& shape, int offset,
                                 Uniforms<U>* uniforms, int* count,
                                 tensorflow::Tensor** spikes) {
  using tensorflow::errors::InvalidArgument;
  const tensorflow::Tensor& previous = c->input(spikes_input);
  tensorflow::OpInputList entries;
  TF_RETURN_IF_ERROR(c->input_list("uniforms", &entries));
  const tensorflow::TensorShape& entry_shape = entries[0].shape();
  const int dims = entry_shape.dims();
  const int64_t per_entry = dims == 3 ? entry_shape.dim_size(0) : 1;
  if (entries.size() > kMaxEntries || dims < 2 || dims > 3 || per_entry <= 0 ||
      entry_shape.dim_size(dims - 2) != shape.dim_size(1) ||
      entry_shape.dim_size(dims - 1) != shape.dim_size(2) ||
      offset + entries.size() * per_entry > shape.dim_size(0))
    return InvalidArgument("uniforms must be at most 16 [time, units] or "
                           "[samples, time, units] tensors within the batch");
  *uniforms = Uniforms<U>{{}, static_cast<int>(per_entry)};
  for (int e = 0; e < entries.size(); ++e) {
    if (entries[e].shape() != entry_shape)
      return InvalidArgument("uniform tensors must have one shape");
    uniforms->entry[e] = entries[e].flat<U>().data();
  }
  *count = static_cast<int>(entries.size() * per_entry);
  if (previous.NumElements() == 0) return c->allocate_output(0, shape, spikes);
  if (previous.shape() != shape)
    return InvalidArgument("spikes_in must be empty or [batch, time, units]");
  TF_RETURN_IF_ERROR(c->forward_input_or_allocate_output({spikes_input}, 0, shape, spikes));
  if ((*spikes)->flat<bool>().data() != previous.flat<bool>().data() &&
      cudaMemcpyAsync((*spikes)->flat<bool>().data(), previous.flat<bool>().data(),
                      previous.TotalBytes(), cudaMemcpyDeviceToDevice,
                      c->eigen_device<Eigen::GpuDevice>().stream()) != cudaSuccess)
    // Not forwardable: keep the samples earlier chunks wrote.
    return tensorflow::errors::Internal("spikes_in copy failed");
  return tensorflow::OkStatus();
}

}  // namespace lgn
