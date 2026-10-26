// Adapted from Javier Galvan's v1_model_utils/cuda_lgn/lgn_grating_ops.cc (commit d8fb466b; credit: Javier).
// DPointNet renames the operators with the Dpointnet prefix and keeps the math unchanged.
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;

// The LGN response to a drifting sinusoidal grating, factored exactly.
//
// A grating frame is Im(e^{i(k.r + w f + phi)}), and every stage up to the
// rectification is linear, so each unit's filtered response is
//   out_n(t) = Im(A_n R_n(t)),
// where the complex spatial coefficient A_n depends only on the sample's
// orientation and phase, and the complex temporal response R_n(t) - the unit's
// temporal kernel applied to e^{iwf} inside the stimulus window, zero-padded
// gray screen outside it - depends only on the unit and the stimulus timing.
// R is precomputed once in float64 (see wrapper.py).
//
// LgnGratingCoefficients computes A (and A' of the non-dominant subunit of
// composite units) in float64: the separable Gaussian filter of the complex
// grating, zero-padded to the frame (TF 'SAME'), sampled at each unit's four
// bilinear corners with the corner weights, edge normalization, amplitude and
// contrast folded into `weights`. It returns float32 [batch, 2, units, 2]:
// A in plane 0 and A' in plane 1 (zero for simple units), as [Re, Im].
REGISTER_OP("DpointnetLgnGratingCoefficients")
    .Attr("T: {half, float}")
    .Attr("theta_sign: float = 1.0").Attr("theta_offset: float = 0.0")
    .Attr("rows: int").Attr("cols: int")
    .Input("theta: T").Input("phase: T").Input("wavenumber: double")
    .Input("vertical_taps: double").Input("horizontal_taps: double")
    .Input("bins: int64").Input("corners: int64").Input("weights: double")
    .Output("coefficients: float")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->MakeShape({c->Dim(c->input(0), 0), 2,
                                     c->Dim(c->input(5), 0), 2}));
      return absl::OkStatus();
    });

// The spike probabilities p = 1 - exp(-rate dt), dt = 1 ms, or with
// `rates` the rates in Hz, [batch, time, units], of
//   rate = max(Im(A R) + spont, 0) + [composite] max(Im(A' R') + spont, 0).
// R' is compact over the composite units; composite_slot maps a unit to its
// column of R', or -1.
REGISTER_OP("DpointnetLgnGratingProbabilities")
    .Attr("rates: bool = false")
    .Input("coefficients: float").Input("response: float")
    .Input("composite_response: float").Input("composite_slot: int64")
    .Input("spontaneous: float")
    .Output("probabilities: float")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->MakeShape({c->Dim(c->input(0), 0),
                                     c->Dim(c->input(1), 0),
                                     c->Dim(c->input(1), 1)}));
      return absl::OkStatus();
    });

// Spikes uniforms < p for the samples [offset, offset + chunk) of the batch,
// the uniforms given as at most 16 equal tensors, [time, units] (one sample
// each, so none is ever copied into a stack) or [samples, time, units]. They are
// written into `spikes_in` in place (or into a new [batch, time, units]
// tensor when `spikes_in` is empty), so a batch is sampled in chunks without
// ever holding every sample's uniforms. The probability never leaves
// registers; the uniforms are TensorFlow's, in their own dtype.
REGISTER_OP("DpointnetLgnGratingSpikes")
    .Attr("U: {half, float}").Attr("offset: int >= 0")
    .Attr("entries: int >= 1")
    .Input("spikes_in: bool")
    .Input("coefficients: float").Input("response: float")
    .Input("composite_response: float").Input("composite_slot: int64")
    .Input("spontaneous: float").Input("uniforms: entries * U")
    .Output("spikes: bool")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->MakeShape({c->Dim(c->input(1), 0),
                                     c->Dim(c->input(2), 0),
                                     c->Dim(c->input(2), 1)}));
      return absl::OkStatus();
    });
