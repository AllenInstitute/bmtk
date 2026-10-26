// Adapted from Javier Galvan's v1_model_utils/cuda_lgn/lgn_movie_ops.cu.cc (commit d8fb466b; credit: Javier).
// DPointNet renames the operators with the Dpointnet prefix and keeps the math unchanged.
#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#include <cstdlib>
#include <cuda_fp16.h>

#include <algorithm>
#include <limits>
#include <string>
#include <utility>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"
#include "lgn_common.cuh"

using namespace tensorflow;
using GPUDevice = Eigen::GpuDevice;

namespace {

using lgn::Output;
using lgn::SpikeProbability;
using lgn::Uniforms;

// Spatial pass: one block per frame.
constexpr int kSpatialThreads = 1024;
constexpr int kRows = 10;     // vertical outputs per item (one column)
constexpr int kCols = 11;     // horizontal outputs per item; odd, so lanes hit distinct banks
constexpr int kItems = 1;     // horizontal items per thread
constexpr int kMaxHalf = 13;  // filters up to 27 taps
constexpr int kPad = kMaxHalf;  // zero halo of the shared frame and filtered buffers

// Temporal pass: a lane per subunit column, kSteps timesteps x kSamples samples per thread.
constexpr int kTile = 32;
constexpr int kTimeWarps = 4;
constexpr int kSteps = 16;
constexpr int kSamples = 1;
constexpr int kLagPadding = 32;  // zero lags the kernels carry after their last nonzero one
static_assert(2 * kSteps - 1 <= kLagPadding && kSteps % 4 == 0,
              "lag blocks prefetch the next block's weights in whole float4s");

// The per-chunk spatial responses stay below this many bytes.
constexpr int64_t kChunkBytes = int64_t{256} << 20;

// A packed sample: bits 0-29 the flat index of its top-left corner (y0, x0),
// bit 30 x1 = x0 + 1, bit 31 y1 = y0 + 1, bits 32-62 its subunit column.
constexpr int64_t kIndexMask = (int64_t{1} << 30) - 1;

__device__ float LoadUniform(const float* u) { return __ldcs(u); }
__device__ float LoadUniform(const Eigen::half* u) {
  return __half2float(__ldcs(reinterpret_cast<const __half*>(u)));
}

// The separable passes of one bin, its half-width H a template parameter so
// the 2H + 1 taps sit in registers and every loaded pixel feeds up to 2H + 1
// independent accumulators. Vertical: each item is kRows rows of one column,
// frame -> filtered. Horizontal: each item is kCols columns of one row,
// filtered -> filtered in place (staged in registers across a barrier). Both
// buffers carry zero halos, so neither pass checks bounds: frame
// [rows + 2 kPad][cols] and filtered [rows][cols + 2 kPad], both pointing at
// their first interior element.
template <int H>
__device__ void Vertical(const float* frame, float* filtered, int rows, int cols,
                         const float* taps) {
  float tap[2 * H + 1];
#pragma unroll
  for (int j = 0; j <= 2 * H; ++j) tap[j] = taps[j - H];
  const int groups = (rows + kRows - 1) / kRows, pitch = cols + 2 * kPad;
  for (int item = threadIdx.x; item < groups * cols; item += blockDim.x) {
    const int group = item / cols, x = item - group * cols, y0 = group * kRows;
    float acc[kRows] = {};
#pragma unroll
    for (int k = 0; k < kRows + 2 * H; ++k) {
      const float value = frame[(y0 - H + k) * cols + x];
#pragma unroll
      for (int r = 0; r < kRows; ++r)
        if (k - r >= 0 && k - r <= 2 * H) acc[r] = fmaf(tap[k - r], value, acc[r]);
    }
#pragma unroll
    for (int r = 0; r < kRows; ++r)
      if (y0 + r < rows) filtered[(y0 + r) * pitch + x] = acc[r];
  }
}

template <int H>
__device__ void Horizontal(float* filtered, int rows, int cols, const float* taps) {
  float tap[2 * H + 1];
#pragma unroll
  for (int j = 0; j <= 2 * H; ++j) tap[j] = taps[j - H];
  const int groups = (cols + kCols - 1) / kCols, pitch = cols + 2 * kPad;
  float acc[kItems][kCols] = {};
#pragma unroll
  for (int n = 0; n < kItems; ++n) {
    const int item = threadIdx.x + n * blockDim.x;
    if (item >= rows * groups) break;
    const int y = item / groups, x0 = (item - y * groups) * kCols;
    const float* row = filtered + y * pitch;
#pragma unroll
    for (int k = 0; k < kCols + 2 * H; ++k) {
      const float value = row[x0 - H + k];
#pragma unroll
      for (int r = 0; r < kCols; ++r)
        if (k - r >= 0 && k - r <= 2 * H) acc[n][r] = fmaf(tap[k - r], value, acc[n][r]);
    }
  }
  __syncthreads();
#pragma unroll
  for (int n = 0; n < kItems; ++n) {
    const int item = threadIdx.x + n * blockDim.x;
    if (item >= rows * groups) break;
    const int y = item / groups, x0 = (item - y * groups) * kCols;
#pragma unroll
    for (int r = 0; r < kCols; ++r)
      if (x0 + r < cols) filtered[y * pitch + x0 + r] = acc[n][r];
  }
}

template <int H>
__device__ void Separable(const float* frame, float* filtered, int rows, int cols,
                          const float* vertical, const float* horizontal, int half) {
  if constexpr (H < kMaxHalf) {
    if (half != H)
      return Separable<H + 1>(frame, filtered, rows, cols, vertical, horizontal, half);
  }
  Vertical<H>(frame, filtered, rows, cols, vertical);
  __syncthreads();
  Horizontal<H>(filtered, rows, cols, horizontal);
}

// Shared-memory floats of SpatialKernel for frames of `rows` x `cols` processed
// in bands of `band` rows (`band == rows`: the whole frame at once).
// The frame buffer holds a whole number of Vertical's kRows-row groups (plus the
// halo): the pass reads every row of a group's kRows + 2H window unguarded, so
// the rows past a partial last group must exist (zero) in shared memory.
__host__ __device__ inline int FrameRows(int brows) {
  return (brows + kRows - 1) / kRows * kRows + 2 * kPad;
}

__host__ __device__ inline int SpatialSharedFloats(int rows, int cols, int band, int bins,
                                                   int width) {
  const int brows = min(band + 1, rows), pitch = cols + 2 * kPad;
  return kPad + brows * pitch + FrameRows(brows) * cols + 2 * bins * width;
}

// One block per frame (sample g, time t), staged in shared memory as float32,
// in bands of `band` rows (the whole frame when it fits the device's shared
// memory): for each band and spatial-size bin, the separable Gaussian filter of
// the band plus its kPad-row halo (zero padding, 'SAME' alignment) into shared
// memory, then each of the bin's samples whose first corner row lies in the
// band as the weighted sum of its four filtered corners, written to
// spatial[g, t / 4, column, t % 4] (see TemporalKernel). A band computes
// band + 1 filtered rows so a sample's second corner row (y0 + 1) is present.
template <typename T>
__global__ void __launch_bounds__(kSpatialThreads)
SpatialKernel(const T* __restrict__ movie, int64_t first_frame, int steps, int columns,
              int rows, int cols, int band, int bins, int width,
              const float* __restrict__ taps, const int64_t* __restrict__ half_widths,
              const int64_t* __restrict__ bin_offsets, const int64_t* __restrict__ samples,
              const float4* __restrict__ weights, float* __restrict__ spatial,
              int64_t sample_stride) {
  extern __shared__ float shared[];
  const int pixels = rows * cols, pitch = cols + 2 * kPad;
  const int brows_max = min(band + 1, rows);
  float* filtered = shared + kPad;                                     // [brows][pitch]
  float* frame = shared + brows_max * pitch + kPad * cols;             // [FrameRows(brows)][cols]
  float* tap = shared + brows_max * pitch + FrameRows(brows_max) * cols;  // [2, bins, width]
  const int region = brows_max * pitch + FrameRows(brows_max) * cols;
  const T* source = movie + (first_frame + blockIdx.x) * pixels;
  for (int i = threadIdx.x; i < 2 * bins * width; i += blockDim.x) tap[i] = taps[i];
  const int g = blockIdx.x / steps, t = blockIdx.x - g * steps;
  float* out = spatial + g * sample_stride + (t >> 2) * 4 * int64_t{columns} + (t & 3);
  const int center = width / 2;
  for (int y0 = 0; y0 < rows; y0 += band) {
    const int brows = min(band + 1, rows - y0);  // filtered rows of this band
    const int ylast = min(y0 + band, rows);      // samples with first corner row in [y0, ylast)
    __syncthreads();
    for (int i = threadIdx.x; i < region; i += blockDim.x) shared[i] = 0.f;
    __syncthreads();
    for (int i = threadIdx.x; i < (brows + 2 * kPad) * cols; i += blockDim.x) {
      const int r = i / cols - kPad, x = i - (r + kPad) * cols, y = y0 + r;
      if (y >= 0 && y < rows) frame[r * cols + x] = static_cast<float>(source[y * cols + x]);
    }
    __syncthreads();
    for (int bin = 0; bin < bins; ++bin) {
      Separable<0>(frame, filtered, brows, cols, tap + bin * width + center,
                      tap + (bins + bin) * width + center, static_cast<int>(half_widths[bin]));
      __syncthreads();
      for (int64_t e = bin_offsets[bin] + threadIdx.x; e < bin_offsets[bin + 1]; e += blockDim.x) {
        const int64_t sample = samples[e];
        const int flat = static_cast<int>(sample & kIndexMask);
        const int y = flat / cols;
        if (y < y0 || y >= ylast) continue;
        const int base = (y - y0) * pitch + (flat - y * cols);
        const int dx = static_cast<int>(sample >> 30) & 1, dy = static_cast<int>(sample >> 31) & 1;
        const float4 w = weights[e];
        // (y0, x0), (y1, x0), (y0, x1), (y1, x1)
        float value = w.x * filtered[base];
        value = fmaf(w.y, filtered[base + dy * pitch], value);
        value = fmaf(w.z, filtered[base + dx], value);
        value = fmaf(w.w, filtered[base + dy * pitch + dx], value);
        out[(sample >> 32) * 4] = value;
      }
      __syncthreads();
    }
  }
}

// v[r] = element r of a time-blocked array whose 4-element blocks are `stride` apart.
__device__ __forceinline__ void Load(const float* p, int64_t stride, float (&v)[kSteps]) {
#pragma unroll
  for (int r = 0; r < kSteps; r += 4) {
    const float4 q = __ldg(reinterpret_cast<const float4*>(p + r / 4 * stride));
    v[r] = q.x, v[r + 1] = q.y, v[r + 2] = q.z, v[r + 3] = q.w;
  }
}

// Loads s[t, t + kSteps) of every sample, zero before the first frame (t and
// the lags are multiples of kSteps, so such a block lies wholly before or after it).
__device__ __forceinline__ void LoadFrames(const float* const (&base)[kSamples], int t,
                                           int64_t stride, float (&v)[kSamples][kSteps]) {
#pragma unroll
  for (int i = 0; i < kSamples; ++i) {
    if (t >= 0) {
      Load(base[i] + t / 4 * stride, stride, v[i]);
    } else {
#pragma unroll
      for (int r = 0; r < kSteps; ++r) v[i][r] = 0.f;
    }
  }
}

// kSteps lags [m0, m0 + kSteps) of the causal temporal convolution for one
// thread: kSteps consecutive outputs t0 + r of kSamples samples, with
// high = s[t0 - m0 + r], low = s[t0 - m0 - kSteps + r] and k the lags' weights,
// so lag m0 + q of output r reads s[t0 + r - m0 - q] = high[r - q] or
// low[kSteps + r - q] at static indices. It first issues the next block's loads
// (`next` = s[t0 - m0 - 2 kSteps + r], `next_k`), so their latency hides behind
// this block's FMAs; the caller rotates the roles of the windows. The block is
// summed on its own before it joins the total, which keeps float32 rounding to
// that of about kSteps + lags / kSteps terms.
__device__ __forceinline__ void LagBlock(
    int m0, int t0, const float* kernel, int64_t weights, const float* const (&base)[kSamples],
    int64_t frames,
    const float (&high)[kSamples][kSteps], const float (&low)[kSamples][kSteps],
    float (&next)[kSamples][kSteps], const float (&k)[kSteps], float (&next_k)[kSteps],
    float (&total)[kSamples][kSteps]) {
  Load(kernel + (m0 + kSteps) / 4 * weights, weights, next_k);
  LoadFrames(base, t0 - m0 - 2 * kSteps, frames, next);
#pragma unroll
  for (int i = 0; i < kSamples; ++i) {
    float block[kSteps] = {};
#pragma unroll
    for (int q = 0; q < kSteps; ++q)
#pragma unroll
      for (int r = 0; r < kSteps; ++r)
        block[r] = fmaf(k[q], r >= q ? high[i][r - q] : low[i][kSteps + r - q], block[r]);
#pragma unroll
    for (int r = 0; r < kSteps; ++r) total[i][r] += block[r];
  }
}

// The causal temporal convolution of kTile subunit columns (one per lane):
//   out[t] = sum over lags m in the tile's range of K[m] s[t - m], s = 0 before 0.
// Both are time-blocked, [time / 4, columns, 4], so a lane reads 4 consecutive
// steps with one float4 and a warp's 32 columns are 512 contiguous bytes:
// s = spatial[sample][t / 4][column][t % 4] (span >= steps frames, a multiple
// of kSteps; frames past `steps` only reach outputs past them), and
// K = kernels[m / 4][column][m % 4] (lags a multiple of kSteps, ending with
// kLagPadding zeros, so a lag block's prefetch never reads past them).
// `spatial` and `kernels` point at the pass's first column; `width` is the
// columns of a time block of `spatial` (units + composites). `kComposite` is the non-dominant pass,
// which writes its rectified rates [samples, time, composites]; the dominant
// pass adds them and writes the output for the samples
// [first_sample, first_sample + samples).
template <Output kOutput, bool kComposite, typename U>
__global__ void __launch_bounds__(kTile * kTimeWarps)
TemporalKernel(int steps, int width, int columns, int64_t sample_stride, int samples,
               const float* __restrict__ spatial, const float* __restrict__ kernels,
               const int64_t* __restrict__ lag_ranges,
               const float* __restrict__ spontaneous, const int64_t* __restrict__ slot,
               float* __restrict__ composite_rates, int composites,
               Uniforms<U> uniforms, int uniform_offset, void* __restrict__ output,
               int64_t first_sample) {
  const int column = blockIdx.y * kTile + threadIdx.x;
  const int t0 = (blockIdx.x * kTimeWarps + threadIdx.y) * kSteps;
  const int b0 = blockIdx.z * kSamples;
  if (column >= columns || t0 >= steps) return;
  const int first = static_cast<int>(lag_ranges[2 * blockIdx.y]) / kSteps * kSteps;
  const int last = min(static_cast<int>(lag_ranges[2 * blockIdx.y + 1]), t0 + kSteps - 1);
  const float* base[kSamples];  // samples past the chunk reread its last one
#pragma unroll
  for (int i = 0; i < kSamples; ++i)
    base[i] = spatial + min(b0 + i, samples - 1) * sample_stride + 4 * column;
  const float* kernel = kernels + 4 * column;
  const int64_t frames = 4 * int64_t{width}, weights = 4 * int64_t{columns};
  // Windows w0, w1, w2 and weights k0, k1 rotate through the roles of LagBlock.
  float total[kSamples][kSteps] = {}, w0[kSamples][kSteps], w1[kSamples][kSteps],
      w2[kSamples][kSteps], k0[kSteps], k1[kSteps];
  LoadFrames(base, t0 - first, frames, w0);
  LoadFrames(base, t0 - first - kSteps, frames, w1);
  Load(kernel + first / 4 * weights, weights, k0);
  for (int m0 = first; m0 <= last;) {
    LagBlock(m0, t0, kernel, weights, base, frames, w0, w1, w2, k0, k1, total);
    if ((m0 += kSteps) > last) break;
    LagBlock(m0, t0, kernel, weights, base, frames, w1, w2, w0, k1, k0, total);
    if ((m0 += kSteps) > last) break;
    LagBlock(m0, t0, kernel, weights, base, frames, w2, w0, w1, k0, k1, total);
    if ((m0 += kSteps) > last) break;
    LagBlock(m0, t0, kernel, weights, base, frames, w0, w1, w2, k1, k0, total);
    if ((m0 += kSteps) > last) break;
    LagBlock(m0, t0, kernel, weights, base, frames, w1, w2, w0, k0, k1, total);
    if ((m0 += kSteps) > last) break;
    LagBlock(m0, t0, kernel, weights, base, frames, w2, w0, w1, k1, k0, total);
    if ((m0 += kSteps) > last) break;
  }
  const float spont = spontaneous[column];
  const int composite = kComposite ? -1 : static_cast<int>(slot[column]);
#pragma unroll
  for (int i = 0; i < kSamples; ++i) {
    if (b0 + i >= samples) break;
    const int64_t sample = b0 + i;
#pragma unroll
    for (int r = 0; r < kSteps; ++r) {
      const int t = t0 + r;
      if (t >= steps) break;
      float rate = fmaxf(total[i][r] + spont, 0.f);
      if constexpr (kComposite) {
        composite_rates[(sample * steps + t) * composites + column] = rate;
        continue;
      }
      if (composite >= 0) rate += composite_rates[(sample * steps + t) * composites + composite];
      const int64_t out = ((first_sample + sample) * steps + t) * columns + column;
      if constexpr (kOutput == Output::kRates) {
        __stcs(static_cast<float*>(output) + out, rate);
      } else if constexpr (kOutput == Output::kProbabilities) {
        __stcs(static_cast<float*>(output) + out, SpikeProbability(rate));
      } else {
        const U* uniform = uniforms.Sample(static_cast<int>(first_sample + sample) - uniform_offset,
                                           static_cast<int64_t>(steps) * columns);
        static_cast<bool*>(output)[out] =
            LoadUniform(uniform + static_cast<int64_t>(t) * columns + column) <
            SpikeProbability(rate);
      }
    }
  }
}

// The op inputs from `first` on (after spikes_in for the spike op).
enum Input {
  kMovie, kTaps, kHalfWidths, kBinOffsets, kSampleIndex, kSampleWeights, kKernels,
  kLagRanges, kSpontaneous, kSlot, kCompositeKernels, kCompositeLagRanges,
  kCompositeSpontaneous,
};

const Tensor& In(OpKernelContext* c, int first, int index) { return c->input(first + index); }

Status Validate(OpKernelContext* c, int first, int rows, int cols) {
  const Tensor& movie = In(c, first, kMovie);
  if (movie.dims() != 4 || movie.dim_size(2) != rows || movie.dim_size(3) != cols)
    return errors::InvalidArgument("movie must be [batch, time, ", rows, ", ", cols, "]");
  if (movie.dim_size(2) * ((movie.dim_size(3) + kCols - 1) / kCols) > kSpatialThreads * kItems)
    return errors::InvalidArgument("frames above ", kSpatialThreads * kItems, " rows x ",
                                   kCols, "-column groups");
  const Tensor& taps = In(c, first, kTaps);
  if (taps.dims() != 3 || taps.dim_size(0) != 2 || taps.dim_size(2) % 2 == 0 ||
      taps.dim_size(2) > 2 * kMaxHalf + 1)
    return errors::InvalidArgument("taps must be [2, bins, odd width <= ", 2 * kMaxHalf + 1, "]");
  const int64_t bins = taps.dim_size(1);
  if (In(c, first, kHalfWidths).NumElements() != bins ||
      In(c, first, kBinOffsets).NumElements() != bins + 1)
    return errors::InvalidArgument("half_widths must be [bins], bin_offsets [bins + 1]");
  if (In(c, first, kSampleWeights).NumElements() != 4 * In(c, first, kSampleIndex).NumElements())
    return errors::InvalidArgument("sample_weights must be [samples, 4]");
  const int64_t units = In(c, first, kSpontaneous).NumElements();
  const int64_t composites = In(c, first, kCompositeSpontaneous).NumElements();
  for (auto [index, columns] : {std::pair{kKernels, units}, {kCompositeKernels, composites}}) {
    const Tensor& kernels = In(c, first, index);
    if (kernels.dims() != 3 || kernels.dim_size(1) != columns || kernels.dim_size(2) != 4 ||
        kernels.dim_size(0) % (kSteps / 4) != 0 || kernels.dim_size(0) * 4 < kLagPadding)
      return errors::InvalidArgument("kernels must be [lags / 4, columns, 4], lags a multiple of ",
                                     kSteps, " ending with ", kLagPadding, " zeros");
  }
  if (In(c, first, kSlot).NumElements() != units)
    return errors::InvalidArgument("composite_slot must be [units]");
  if (In(c, first, kLagRanges).NumElements() != 2 * ((units + kTile - 1) / kTile) ||
      In(c, first, kCompositeLagRanges).NumElements() != 2 * ((composites + kTile - 1) / kTile))
    return errors::InvalidArgument("lag ranges must be [ceil(columns / 32), 2]");
  const int64_t span = (movie.dim_size(1) + kSteps - 1) / kSteps * kSteps;
  if (units + composites >= (int64_t{1} << 31) || movie.dim_size(0) >= (int64_t{1} << 31) ||
      span * (units + composites) >= (int64_t{1} << 31) ||
      In(c, first, kKernels).NumElements() >= (int64_t{1} << 31))
    return errors::InvalidArgument("LGN movie response too large");
  return OkStatus();
}

// The samples [begin, end) of the batch, chunk by chunk: spatial responses
// [chunk, span / 4, units + composites, 4], then the non-dominant and the dominant
// temporal passes; `output` is the full [batch, time, units] tensor. The movie's
// first sample is sample `movie_first` of the batch.
template <typename T, typename U, Output kOutput>
Status Launch(OpKernelContext* c, int first, int begin, int end, Uniforms<U> uniforms,
              int uniform_offset, void* output, int movie_first = 0) {
  const Tensor& movie = In(c, first, kMovie);
  const int steps = movie.dim_size(1), rows = movie.dim_size(2), cols = movie.dim_size(3);
  const int units = In(c, first, kSpontaneous).NumElements();
  const int composites = In(c, first, kCompositeSpontaneous).NumElements();
  if (end <= begin || steps == 0 || units == 0) return OkStatus();
  const int span = (steps + kSteps - 1) / kSteps * kSteps;
  const int width = units + composites;
  const int64_t sample_stride = int64_t{span} * width;
  const int64_t sample_bytes = 4 * (sample_stride + int64_t{steps} * composites);
  const int chunk =
      static_cast<int>(std::clamp<int64_t>(kChunkBytes / sample_bytes, 1, end - begin));
  Tensor spatial, composite_rates;
  TF_RETURN_IF_ERROR(c->allocate_temp(DT_FLOAT, TensorShape({chunk * sample_stride}), &spatial));
  TF_RETURN_IF_ERROR(c->allocate_temp(
      DT_FLOAT, TensorShape({int64_t{chunk} * steps * composites}), &composite_rates));
  const Tensor& taps = In(c, first, kTaps);
  const int bins = static_cast<int>(taps.dim_size(1)), tap_width = static_cast<int>(taps.dim_size(2));
  // The whole frame when the device's opt-in shared memory holds it (sm_86+ for
  // 80 x 120), else the largest band of rows that fits (V100: 2 bands, Pascal: 3).
  int device = 0, optin = 48 * 1024;
  if (cudaGetDevice(&device) == cudaSuccess)
    cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
  int band = rows;
  // Test hook: DPOINTNET_LGN_MOVIE_MAX_BAND caps the band to exercise the banded
  // path on devices whose shared memory holds the whole frame.
  if (const char* cap = std::getenv("DPOINTNET_LGN_MOVIE_MAX_BAND"))
    band = std::max(1, std::min(band, std::atoi(cap)));
  while (band > 0 &&
         SpatialSharedFloats(rows, cols, band, bins, tap_width) * sizeof(float) > optin)
    --band;
  if (band < 1)
    return errors::InvalidArgument("frames of ", rows, " x ", cols,
                                   " exceed shared memory even one row at a time");
  const int shared = SpatialSharedFloats(rows, cols, band, bins, tap_width) * sizeof(float);
  if (cudaFuncSetAttribute(SpatialKernel<T>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                           shared) != cudaSuccess)
    return errors::InvalidArgument("frames of ", rows, " x ", cols, " exceed shared memory");
  auto stream = c->eigen_device<GPUDevice>().stream();
  float* spatial_data = spatial.flat<float>().data();
  float* composite_data = composite_rates.flat<float>().data();
  const dim3 threads(kTile, kTimeWarps);
  const int time_blocks = (steps + kSteps * kTimeWarps - 1) / (kSteps * kTimeWarps);
  for (int b0 = begin; b0 < end; b0 += chunk) {
    const int samples = std::min(chunk, end - b0);
    TF_RETURN_IF_ERROR(GpuLaunchKernel(
        SpatialKernel<T>, samples * steps, kSpatialThreads, shared, stream,
        reinterpret_cast<const T*>(movie.flat<T>().data()), int64_t{b0 - movie_first} * steps,
        steps, width,
        rows, cols, band, bins, tap_width,
        taps.flat<float>().data(), In(c, first, kHalfWidths).flat<int64_t>().data(),
        In(c, first, kBinOffsets).flat<int64_t>().data(),
        In(c, first, kSampleIndex).flat<int64_t>().data(),
        reinterpret_cast<const float4*>(In(c, first, kSampleWeights).flat<float>().data()),
        spatial_data, sample_stride));
    const int sample_blocks = (samples + kSamples - 1) / kSamples;
    if (composites > 0) {
      const Tensor& kernels = In(c, first, kCompositeKernels);
      auto composite_pass = TemporalKernel<kOutput, true, U>;
      TF_RETURN_IF_ERROR(GpuLaunchKernel(
          composite_pass, dim3(time_blocks, (composites + kTile - 1) / kTile, sample_blocks),
          threads, 0, stream, steps, width, composites, sample_stride, samples,
          spatial_data + 4 * int64_t{units}, kernels.flat<float>().data(),
          In(c, first, kCompositeLagRanges).flat<int64_t>().data(),
          In(c, first, kCompositeSpontaneous).flat<float>().data(), nullptr, composite_data,
          composites, uniforms, uniform_offset, nullptr, int64_t{0}));
    }
    const Tensor& kernels = In(c, first, kKernels);
    auto dominant_pass = TemporalKernel<kOutput, false, U>;
    TF_RETURN_IF_ERROR(GpuLaunchKernel(
        dominant_pass, dim3(time_blocks, (units + kTile - 1) / kTile, sample_blocks), threads,
        0, stream, steps, width, units, sample_stride, samples, spatial_data,
        kernels.flat<float>().data(), In(c, first, kLagRanges).flat<int64_t>().data(),
        In(c, first, kSpontaneous).flat<float>().data(),
        In(c, first, kSlot).flat<int64_t>().data(), composite_data, composites, uniforms,
        uniform_offset, output, int64_t{b0}));
  }
  return OkStatus();
}

// [batch, time, units]; batch > 0 overrides the movie's own batch.
TensorShape OutputShape(OpKernelContext* c, int first, int64_t batch = 0) {
  const Tensor& movie = In(c, first, kMovie);
  return TensorShape({batch > 0 ? batch : movie.dim_size(0), movie.dim_size(1),
                      In(c, first, kSpontaneous).NumElements()});
}

// The frame size the constants were built for.
class MovieOp : public OpKernel {
 public:
  explicit MovieOp(OpKernelConstruction* c) : OpKernel(c) {
    OP_REQUIRES_OK(c, c->GetAttr("rows", &rows_));
    OP_REQUIRES_OK(c, c->GetAttr("cols", &cols_));
  }
 protected:
  int rows_, cols_;
};

template <typename T>
class ResponseOp : public MovieOp {
 public:
  explicit ResponseOp(OpKernelConstruction* c) : MovieOp(c) {
    std::string output;
    OP_REQUIRES_OK(c, c->GetAttr("output", &output));
    rates_ = output == "rates";
  }
  void Compute(OpKernelContext* c) override {
    OP_REQUIRES_OK(c, Validate(c, 0, rows_, cols_));
    Tensor* output;
    OP_REQUIRES_OK(c, c->allocate_output(0, OutputShape(c, 0), &output));
    const int batch = output->dim_size(0);
    float* data = output->flat<float>().data();
    OP_REQUIRES_OK(c, rates_
        ? Launch<T, float, Output::kRates>(c, 0, 0, batch, Uniforms<float>{}, 0, data)
        : Launch<T, float, Output::kProbabilities>(c, 0, 0, batch, Uniforms<float>{}, 0, data));
  }
 private:
  bool rates_;
};

template <typename T, typename U>
class SpikesOp : public MovieOp {
 public:
  explicit SpikesOp(OpKernelConstruction* c) : MovieOp(c) {
    OP_REQUIRES_OK(c, c->GetAttr("offset", &offset_));
    OP_REQUIRES_OK(c, c->GetAttr("batch", &batch_));
  }
  void Compute(OpKernelContext* c) override {
    OP_REQUIRES_OK(c, Validate(c, 1, rows_, cols_));
    Uniforms<U> uniforms;
    int count;
    Tensor* spikes;
    OP_REQUIRES_OK(c, lgn::PrepareSpikes(c, 0, OutputShape(c, 1, batch_), offset_, &uniforms,
                                         &count, &spikes));
    // A chunk movie holds exactly the chunk's samples.
    OP_REQUIRES(c, batch_ == 0 || In(c, 1, kMovie).dim_size(0) == count,
                errors::InvalidArgument("with batch > 0 the movie must hold the chunk's ", count,
                                        " samples"));
    OP_REQUIRES_OK(c, (Launch<T, U, Output::kSpikes>(c, 1, offset_, offset_ + count, uniforms,
                                                     offset_, spikes->flat<bool>().data(),
                                                     batch_ > 0 ? offset_ : 0)));
  }
 private:
  int offset_, batch_;
};

}  // namespace

#define REG(T)                                                                     \
  REGISTER_KERNEL_BUILDER(                                                         \
      Name("DpointnetLgnMovieResponse").Device(DEVICE_GPU).TypeConstraint<T>("T"), ResponseOp<T>);
REG(float); REG(Eigen::half);
#undef REG
#define REG(T, U)                                                                  \
  REGISTER_KERNEL_BUILDER(Name("DpointnetLgnMovieSpikes")                                   \
                              .Device(DEVICE_GPU)                                  \
                              .TypeConstraint<T>("T")                              \
                              .TypeConstraint<U>("U"),                             \
                          SpikesOp<T, U>);
REG(float, float); REG(float, Eigen::half); REG(Eigen::half, float); REG(Eigen::half, Eigen::half);
#undef REG
#endif
