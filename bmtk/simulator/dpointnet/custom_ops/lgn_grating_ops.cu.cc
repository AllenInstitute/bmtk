// Adapted from Javier Galvan's v1_model_utils/cuda_lgn/lgn_grating_ops.cu.cc (commit d8fb466b; credit: Javier).
// DPointNet renames the operators with the Dpointnet prefix and keeps the math unchanged.
#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#include <cuda_fp16.h>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"
#include "lgn_common.cuh"

using namespace tensorflow;
using GPUDevice = Eigen::GpuDevice;

namespace {

using lgn::Output;
using lgn::SpikeProbability;
using lgn::Uniforms;

constexpr double kPi = 3.141592653589793238462643383279502884;
constexpr int kUnits = 4;    // consecutive units per response thread
constexpr int kSteps = 4;    // timesteps per response thread
constexpr int kResponseThreads = 128;
constexpr int kResponseBlocks = 4;  // per SM: >= 16 warps to hide the streaming latency
constexpr int kSampleGroup = 16;     // samples per thread (blockIdx.z splits the chunk)
constexpr int kTableThreads = 256;

__device__ double2 Mul(double2 a, double2 b) {
  return make_double2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

__device__ double2 Exp(double angle) {
  double s, c;
  sincos(angle, &s, &c);
  return make_double2(c, s);
}

template <typename T> __device__ double Degrees(T value) {
  return static_cast<double>(static_cast<float>(value));
}

// One (bin, sample) block: the complex grating e^{i k.r} filtered along each
// axis by the bin's separable Gaussian, zero-padded to the frame. Entry x < cols
// is the horizontal factor at column x (times the phase e^{i phi}), entry
// cols + y the vertical factor at row y; their product is the filtered complex
// grating at (y, x).
template <typename T>
__global__ void TablesKernel(const T* __restrict__ theta, const T* __restrict__ phase,
                             const double* __restrict__ wavenumber, float sign,
                             float offset, int bins, int taps,
                             const double* __restrict__ vertical,
                             const double* __restrict__ horizontal, int rows, int cols,
                             double2* __restrict__ tables) {
  extern __shared__ double2 shifts[];  // [2 * taps]: e^{i f (j - c)}, x then y
  const int bin = blockIdx.x, sample = blockIdx.y, center = taps / 2;
  const double orientation = kPi * (180.0 - (sign * Degrees(theta[sample]) + offset)) / 180.0;
  const double phi = kPi * (180.0 - Degrees(phase[sample])) / 180.0;
  const double fx = *wavenumber * cos(orientation);
  const double fy = *wavenumber * sin(orientation);
  for (int j = threadIdx.x; j < taps; j += blockDim.x) {
    shifts[j] = Exp(fx * (j - center));
    shifts[taps + j] = Exp(fy * (j - center));
  }
  __syncthreads();
  for (int e = threadIdx.x; e < cols + rows; e += blockDim.x) {
    const bool along_x = e < cols;
    const int position = along_x ? e : e - cols, size = along_x ? cols : rows;
    const double* tap = (along_x ? horizontal : vertical) + bin * taps;
    const double2* shift = shifts + (along_x ? 0 : taps);
    double2 sum = make_double2(0.0, 0.0);
    const int first = max(0, center - position);
    const int last = min(taps, size + center - position);
    for (int j = first; j < last; ++j) {
      sum.x += tap[j] * shift[j].x;
      sum.y += tap[j] * shift[j].y;
    }
    tables[(sample * bins + bin) * (cols + rows) + e] =
        Mul(sum, Exp(along_x ? fx * position + phi : fy * position));
  }
}

// corner = [x0, x1, y0, y1]; weights of (y0,x0), (y1,x0), (y0,x1), (y1,x1).
__device__ float2 Sample(const double2* table, int cols, const int64_t* corner,
                         const double* weight) {
  const double2 y0 = table[cols + corner[2]], y1 = table[cols + corner[3]];
  const double2 left = make_double2(weight[0] * y0.x + weight[1] * y1.x,
                                    weight[0] * y0.y + weight[1] * y1.y);
  const double2 right = make_double2(weight[2] * y0.x + weight[3] * y1.x,
                                     weight[2] * y0.y + weight[3] * y1.y);
  const double2 a = Mul(table[corner[0]], left), b = Mul(table[corner[1]], right);
  return make_float2(a.x + b.x, a.y + b.y);
}

// coefficients [batch, 2, units] complex: plane 0 the dominant subunit, plane 1
// the non-dominant one (zero weights unless composite).
__global__ void CoefficientsKernel(int64_t count, int units, int bins, int rows, int cols,
                                   const int64_t* __restrict__ bin,
                                   const int64_t* __restrict__ corners,
                                   const double* __restrict__ weights,
                                   const double2* __restrict__ tables,
                                   float2* __restrict__ coefficients) {
  GPU_1D_KERNEL_LOOP(i, count) {
    const int64_t sample = i / units;
    const int unit = i - sample * units;
    const double2* table = tables + (sample * bins + bin[unit]) * (cols + rows);
    float2* out = coefficients + 2 * sample * units + unit;
    out[0] = Sample(table, cols, corners + 8 * unit, weights + 8 * unit);
    out[units] = Sample(table, cols, corners + 8 * unit + 4, weights + 8 * unit + 4);
  }
}

__device__ float4 Load4(const Eigen::half* u) {
  const ::uint2 raw = __ldcs(reinterpret_cast<const ::uint2*>(u));
  const float2 a = __half22float2(*reinterpret_cast<const __half2*>(&raw.x));
  const float2 b = __half22float2(*reinterpret_cast<const __half2*>(&raw.y));
  return make_float4(a.x, a.y, b.x, b.y);
}
__device__ float4 Load4(const float* u) { return __ldcs(reinterpret_cast<const float4*>(u)); }

__device__ float Im(float2 a, float2 r) { return fmaf(a.x, r.y, a.y * r.x); }

// Each thread holds the temporal responses of kUnits consecutive units over
// kSteps timesteps in registers and sweeps the samples [begin, end), so R is
// read once per launch while the uniforms and outputs stream through. kVector:
// units % kUnits == 0, so every thread's units are whole and 4-aligned.
template <typename U, Output kOutput, bool kVector>
__global__ void __launch_bounds__(kResponseThreads, kResponseBlocks)
ResponseKernel(int units, int steps, int begin, int end,
               const float2* __restrict__ coefficients,
               const float2* __restrict__ response,
               const float2* __restrict__ composite_response, int composites,
               const int64_t* __restrict__ composite_slot,
               const float* __restrict__ spontaneous, Uniforms<U> uniforms,
               void* __restrict__ output) {
  const int first = (blockIdx.x * blockDim.x + threadIdx.x) * kUnits;
  if (first >= units) return;
  const int t0 = blockIdx.y * kSteps;
  const int width = kVector ? kUnits : min(kUnits, units - first);
  float2 r[kSteps][kUnits];
  int slot[kUnits];
  float spont[kUnits];
  bool composite = false;
#pragma unroll
  for (int j = 0; j < kUnits; ++j) {
    const bool live = j < width;
    slot[j] = live ? static_cast<int>(composite_slot[first + j]) : -1;
    spont[j] = live ? spontaneous[first + j] : 0.f;
    composite |= slot[j] >= 0;
#pragma unroll
    for (int s = 0; s < kSteps; ++s)
      r[s][j] = live && t0 + s < steps
          ? response[static_cast<int64_t>(t0 + s) * units + first + j] : make_float2(0.f, 0.f);
  }
  const int group_begin = begin + blockIdx.z * kSampleGroup;
  const int group_end = min(end, group_begin + kSampleGroup);
  for (int b = group_begin; b < group_end; ++b) {
    const float2* plane = coefficients + 2 * static_cast<int64_t>(b) * units + first;
    float2 a[kUnits], ac[kUnits];
#pragma unroll
    for (int j = 0; j < kUnits; ++j) {
      a[j] = j < width ? plane[j] : make_float2(0.f, 0.f);
      ac[j] = composite && slot[j] >= 0 ? plane[units + j] : make_float2(0.f, 0.f);
    }
    // Issue every timestep's uniform load before any arithmetic.
    float4 u[kSteps];
    const int sample = b - begin;
    const U* base = kOutput == Output::kSpikes
        ? uniforms.Sample(sample, static_cast<int64_t>(steps) * units) + first : nullptr;
    if constexpr (kOutput == Output::kSpikes && kVector) {
#pragma unroll
      for (int s = 0; s < kSteps; ++s)
        if (t0 + s < steps) u[s] = Load4(base + static_cast<int64_t>(t0 + s) * units);
    }
#pragma unroll
    for (int s = 0; s < kSteps; ++s) {
      const int t = t0 + s;
      if (t >= steps) continue;
      float value[kUnits];
#pragma unroll
      for (int j = 0; j < kUnits; ++j) {
        // Im(A R) + spont, rectified, per subunit.
        float rate = fmaxf(Im(a[j], r[s][j]) + spont[j], 0.f);
        if (composite && slot[j] >= 0)
          rate += fmaxf(Im(ac[j], composite_response[static_cast<int64_t>(t) * composites + slot[j]])
                        + spont[j], 0.f);
        value[j] = kOutput == Output::kRates ? rate : SpikeProbability(rate);
      }
      const int64_t out = (static_cast<int64_t>(b) * steps + t) * units + first;
      if constexpr (kOutput == Output::kSpikes) {
        bool* spikes = static_cast<bool*>(output) + out;
        if constexpr (kVector) {
          __stcs(reinterpret_cast<unsigned int*>(spikes),
                 (u[s].x < value[0]) | (u[s].y < value[1]) << 8 |
                 (u[s].z < value[2]) << 16 | (u[s].w < value[3]) << 24);
        } else {
          const U* uniform = base + static_cast<int64_t>(t) * units;
          for (int j = 0; j < width; ++j) spikes[j] = static_cast<float>(uniform[j]) < value[j];
        }
      } else {
        float* probabilities = static_cast<float*>(output) + out;
        if constexpr (kVector) {
          __stcs(reinterpret_cast<float4*>(probabilities),
                 make_float4(value[0], value[1], value[2], value[3]));
        } else {
          for (int j = 0; j < width; ++j) probabilities[j] = value[j];
        }
      }
    }
  }
}

Status ValidateResponse(OpKernelContext* c, int first) {
  const Tensor& coefficients = c->input(first);
  const Tensor& response = c->input(first + 1);
  const Tensor& composite = c->input(first + 2);
  if (coefficients.dims() != 4 || coefficients.dim_size(1) != 2 || coefficients.dim_size(3) != 2)
    return errors::InvalidArgument("coefficients must be [batch, 2, units, 2]");
  const int64_t units = coefficients.dim_size(2);
  if (response.dims() != 3 || response.dim_size(1) != units || response.dim_size(2) != 2)
    return errors::InvalidArgument("response must be [time, units, 2]");
  if (composite.dims() != 3 || composite.dim_size(0) != response.dim_size(0) ||
      composite.dim_size(2) != 2)
    return errors::InvalidArgument("composite_response must be [time, composites, 2]");
  for (int index : {first + 3, first + 4})
    if (c->input(index).dims() != 1 || c->input(index).dim_size(0) != units)
      return errors::InvalidArgument("per-unit constants must be [units]");
  if (response.NumElements() / 2 > std::numeric_limits<int>::max() ||
      coefficients.dim_size(0) > std::numeric_limits<int>::max())
    return errors::InvalidArgument("LGN response too large");
  return OkStatus();
}

template <typename U, Output kOutput>
Status LaunchResponse(OpKernelContext* c, int first, int begin, int end,
                      Uniforms<U> uniforms, void* output) {
  const Tensor& response = c->input(first + 1);
  const int steps = response.dim_size(0), units = response.dim_size(1);
  if (steps == 0 || units == 0 || end <= begin) return OkStatus();
  const dim3 blocks((units + kUnits * kResponseThreads - 1) / (kUnits * kResponseThreads),
                    (steps + kSteps - 1) / kSteps, (end - begin + kSampleGroup - 1) / kSampleGroup);
  auto kernel = units % kUnits == 0 ? ResponseKernel<U, kOutput, true>
                                    : ResponseKernel<U, kOutput, false>;
  return GpuLaunchKernel(
      kernel, blocks, kResponseThreads, 0,
      c->eigen_device<GPUDevice>().stream(), units, steps, begin, end,
      reinterpret_cast<const float2*>(c->input(first).flat<float>().data()),
      reinterpret_cast<const float2*>(response.flat<float>().data()),
      reinterpret_cast<const float2*>(c->input(first + 2).flat<float>().data()),
      static_cast<int>(c->input(first + 2).dim_size(1)),
      c->input(first + 3).flat<int64_t>().data(),
      c->input(first + 4).flat<float>().data(), uniforms, output);
}


template <typename T>
class CoefficientsOp : public OpKernel {
 public:
  explicit CoefficientsOp(OpKernelConstruction* c) : OpKernel(c) {
    OP_REQUIRES_OK(c, c->GetAttr("theta_sign", &sign_));
    OP_REQUIRES_OK(c, c->GetAttr("theta_offset", &offset_));
    OP_REQUIRES_OK(c, c->GetAttr("rows", &rows_));
    OP_REQUIRES_OK(c, c->GetAttr("cols", &cols_));
  }
  void Compute(OpKernelContext* c) override {
    const Tensor& theta = c->input(0);
    const Tensor& vertical = c->input(3);
    const Tensor& bin = c->input(5);
    const int64_t batch = theta.NumElements(), units = bin.NumElements();
    OP_REQUIRES(c, c->input(1).NumElements() == batch && c->input(2).NumElements() == 1,
                errors::InvalidArgument("theta, phase must be [batch]; wavenumber a scalar"));
    OP_REQUIRES(c, vertical.dims() == 2 && vertical.shape() == c->input(4).shape(),
                errors::InvalidArgument("taps must be [bins, taps], alike"));
    OP_REQUIRES(c, c->input(6).NumElements() == 8 * units &&
                c->input(7).NumElements() == 8 * units,
                errors::InvalidArgument("corners and weights must be [units, 8]"));
    const int bins = vertical.dim_size(0), taps = vertical.dim_size(1);
    Tensor* coefficients;
    OP_REQUIRES_OK(c, c->allocate_output(0, TensorShape({batch, 2, units, 2}), &coefficients));
    if (batch == 0 || units == 0) return;
    Tensor tables;
    OP_REQUIRES_OK(c, c->allocate_temp(
        DT_DOUBLE, TensorShape({batch, bins, rows_ + cols_, 2}), &tables));
    auto& d = c->eigen_device<GPUDevice>();
    auto* table = reinterpret_cast<double2*>(tables.flat<double>().data());
    OP_REQUIRES_OK(c, GpuLaunchKernel(
        TablesKernel<T>, dim3(bins, batch), kTableThreads,
        2 * taps * sizeof(double2), d.stream(), theta.flat<T>().data(),
        c->input(1).flat<T>().data(),
        c->input(2).flat<double>().data(), sign_, offset_, bins, taps,
        vertical.flat<double>().data(), c->input(4).flat<double>().data(),
        rows_, cols_, table));
    auto config = GetGpuLaunchConfig(batch * units, d);
    OP_REQUIRES_OK(c, GpuLaunchKernel(
        CoefficientsKernel, config.block_count, config.thread_per_block, 0,
        d.stream(), batch * units, static_cast<int>(units), bins, rows_, cols_,
        bin.flat<int64_t>().data(),
        c->input(6).flat<int64_t>().data(), c->input(7).flat<double>().data(),
        table, reinterpret_cast<float2*>(coefficients->flat<float>().data())));
  }
 private:
  float sign_, offset_;
  int rows_, cols_;
};

class ProbabilitiesOp : public OpKernel {
 public:
  explicit ProbabilitiesOp(OpKernelConstruction* c) : OpKernel(c) {
    OP_REQUIRES_OK(c, c->GetAttr("rates", &rates_));
  }
  void Compute(OpKernelContext* c) override {
    OP_REQUIRES_OK(c, ValidateResponse(c, 0));
    const int batch = c->input(0).dim_size(0);
    Tensor* output;
    OP_REQUIRES_OK(c, c->allocate_output(0, TensorShape(
        {batch, c->input(1).dim_size(0), c->input(1).dim_size(1)}), &output));
    void* data = output->flat<float>().data();
    OP_REQUIRES_OK(c, rates_
        ? LaunchResponse<float, Output::kRates>(c, 0, 0, batch, Uniforms<float>{}, data)
        : LaunchResponse<float, Output::kProbabilities>(c, 0, 0, batch, Uniforms<float>{}, data));
  }
 private:
  bool rates_;
};

template <typename U>
class SpikesOp : public OpKernel {
 public:
  explicit SpikesOp(OpKernelConstruction* c) : OpKernel(c) {
    OP_REQUIRES_OK(c, c->GetAttr("offset", &offset_));
  }
  void Compute(OpKernelContext* c) override {
    OP_REQUIRES_OK(c, ValidateResponse(c, 1));
    const int64_t batch = c->input(1).dim_size(0);
    const TensorShape shape({batch, c->input(2).dim_size(0), c->input(2).dim_size(1)});
    Uniforms<U> uniforms;
    int count;
    Tensor* spikes;
    OP_REQUIRES_OK(c, lgn::PrepareSpikes(c, 0, shape, offset_, &uniforms, &count, &spikes));
    OP_REQUIRES_OK(c, LaunchResponse<U, Output::kSpikes>(
        c, 1, offset_, offset_ + count, uniforms,
        spikes->flat<bool>().data()));
  }
 private:
  int offset_;
};

}  // namespace

#define REG(T) REGISTER_KERNEL_BUILDER(Name("DpointnetLgnGratingCoefficients").Device(DEVICE_GPU).TypeConstraint<T>("T"), CoefficientsOp<T>);
REG(float); REG(Eigen::half);
#undef REG
REGISTER_KERNEL_BUILDER(Name("DpointnetLgnGratingProbabilities").Device(DEVICE_GPU), ProbabilitiesOp);
#define REG(U) REGISTER_KERNEL_BUILDER(Name("DpointnetLgnGratingSpikes").Device(DEVICE_GPU).TypeConstraint<U>("U"), SpikesOp<U>);
REG(float); REG(Eigen::half);
#undef REG
#endif
