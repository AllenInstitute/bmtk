// Adapted from Javier Galvan's v1_model_utils/cuda_lgn/lgn_movie_ops.cc (commit d8fb466b; credit: Javier).
// DPointNet renames the operators with the Dpointnet prefix and keeps the math unchanged.
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;

// The LGN response to an arbitrary movie, [batch, time, rows, cols], the same
// pipeline as lgn.LGN.spatial_response + firing_rates_from_spatial in float32:
//
// 1. Spatial: per frame and spatial-size bin, the separable Gaussian filter
//    (vertical then horizontal taps, zero padding, TF 'SAME' alignment), then
//    every subunit's bilinear sample. A sample is a weighted sum of four
//    filtered pixels (`sample_corners`, flat row * cols + col); its weights fold
//    in the corner weights, the edge normalization (bmtk_compat) and the
//    unit's amplitude, as everything up to the rectification is linear.
//    `bin_offsets` [bins + 1] delimits each bin's samples, and
//    `sample_columns` maps a sample to its subunit column: the unit index for
//    dominant subunits, units + slot for composite non-dominant ones.
// 2. Temporal: per subunit column, the causal convolution with its kernel
//    ([lags / 4, columns, 4]: lag m, in block m / 4 at m % 4, multiplies the
//    frame m steps back; lags a multiple of 16 ending with at least 32 zeros),
//    zero history before the first frame. Lags outside a 32-column tile's `lag_ranges` [first, last]
//    are exactly zero and skipped.
// 3. rate = max(dominant + spont, 0) + [composite] max(non-dominant + spont', 0),
//    in Hz; `output` = "probabilities" gives p = 1 - exp(-rate dt), dt = 1 ms,
//    to about half an ulp.
//
// Everything accumulates in float32; the movie may be float16 or float32. The
// constants are those of `rows` x `cols` frames, which the movie must match.
#define LGN_MOVIE_INPUTS                                                     \
  Input("movie: T")                                                          \
      .Input("taps: float")                                                  \
      .Input("half_widths: int64")                                           \
      .Input("bin_offsets: int64")                                           \
      .Input("samples: int64")                                               \
      .Input("sample_weights: float")                                        \
      .Input("kernels: float")                                               \
      .Input("lag_ranges: int64")                                            \
      .Input("spontaneous: float")                                           \
      .Input("composite_slot: int64")                                        \
      .Input("composite_kernels: float")                                     \
      .Input("composite_lag_ranges: int64")                                  \
      .Input("composite_spontaneous: float")

namespace {
Status MovieShape(shape_inference::InferenceContext* c, int movie, int units) {
  shape_inference::ShapeHandle shape;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(movie), 4, &shape));
  c->set_output(0, c->MakeShape({c->Dim(shape, 0), c->Dim(shape, 1),
                                 c->Dim(c->input(units), 0)}));
  return absl::OkStatus();
}
}  // namespace

// Rates (Hz) or spike probabilities, float32 [batch, time, units].
REGISTER_OP("DpointnetLgnMovieResponse")
    .Attr("T: {half, float}")
    .Attr("rows: int")
    .Attr("cols: int")
    .Attr("output: {'rates', 'probabilities'} = 'rates'")
    .LGN_MOVIE_INPUTS
    .Output("response: float")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      return MovieShape(c, 0, 8);
    });

// Spikes uniforms < p for the samples [offset, offset + chunk) of the movie
// batch, the uniforms given as at most 16 equal [time, units] (one sample each)
// or [samples, time, units] tensors, written into `spikes_in` in place (or into
// a new [batch, time, units] tensor when `spikes_in` is empty): the same
// chunked sampling as LgnGratingSpikes. The probability never leaves registers.
// With `batch` > 0 the movie holds only the chunk's samples, [chunk, time,
// rows, cols], and the spikes are [batch, time, units], so a caller can build
// one chunk of movie at a time; with 0 the movie is the whole batch.
REGISTER_OP("DpointnetLgnMovieSpikes")
    .Attr("T: {half, float}")
    .Attr("rows: int")
    .Attr("cols: int")
    .Attr("U: {half, float}")
    .Attr("offset: int >= 0")
    .Attr("entries: int >= 1")
    .Attr("batch: int >= 0 = 0")
    .Input("spikes_in: bool")
    .LGN_MOVIE_INPUTS
    .Input("uniforms: entries * U")
    .Output("spikes: bool")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      TF_RETURN_IF_ERROR(MovieShape(c, 1, 9));
      int batch;
      TF_RETURN_IF_ERROR(c->GetAttr("batch", &batch));
      if (batch > 0) {
        shape_inference::ShapeHandle shape;
        TF_RETURN_IF_ERROR(c->ReplaceDim(c->output(0), 0, c->MakeDim(batch), &shape));
        c->set_output(0, shape);
      }
      return absl::OkStatus();
    });
