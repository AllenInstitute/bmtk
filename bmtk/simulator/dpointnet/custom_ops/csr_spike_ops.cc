#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

using tensorflow::shape_inference::DimensionHandle;
using tensorflow::shape_inference::InferenceContext;
using tensorflow::shape_inference::ShapeHandle;

REGISTER_OP("DpointnetCsrReorder")
    .Input("values: T")
    .Input("metadata: resource")
    .Attr("T: {half, float}")
    .Attr("Tindex: {uint32, int64}")
    .Attr("n_edges: int >= 0")
    .Output("reordered: T")
    .SetShapeFn([](InferenceContext* context) -> absl::Status {
      ShapeHandle values;
      TF_RETURN_IF_ERROR(context->WithRank(context->input(0), 1, &values));
      context->set_output(0, values);
      return absl::OkStatus();
    });

REGISTER_OP("DpointnetCsrSpikeForward")
    .Input("spikes: T")
    .Input("master_weights: Tmaster")
    .Input("metadata: resource")
    .Input("weights: T")
    .Input("basis: T")
    .Attr("T: {half, float}")
    .Attr("Tmaster: {half, float}")
    .Attr("Tindex: {uint32, int64}")
    .Attr("n_post: int >= 1")
    .Attr("n_edges: int >= 0")
    .Attr("compute_spike_gradient: bool")
    .Output("currents: T")
    .SetShapeFn([](InferenceContext* context) -> absl::Status {
      ShapeHandle spikes;
      ShapeHandle basis;
      TF_RETURN_IF_ERROR(context->WithRank(context->input(0), 2, &spikes));
      TF_RETURN_IF_ERROR(context->WithRank(context->input(4), 2, &basis));
      int n_post;
      TF_RETURN_IF_ERROR(context->GetAttr("n_post", &n_post));
      DimensionHandle flattened_batch;
      TF_RETURN_IF_ERROR(context->Multiply(
          context->Dim(spikes, 0), n_post, &flattened_batch));
      context->set_output(
          0, context->Matrix(flattened_batch, context->Dim(basis, 1)));
      return absl::OkStatus();
    });

REGISTER_OP("DpointnetCsrSpikeGrad")
    .Input("spikes: T")
    .Input("current_grad: T")
    .Input("metadata: resource")
    .Input("weights: T")
    .Input("basis: T")
    .Attr("T: {half, float}")
    .Attr("Tindex: {uint32, int64}")
    .Attr("n_post: int >= 1")
    .Attr("n_edges: int >= 0")
    .Output("spike_grad: T")
    .Output("weight_grad: float")
    .SetShapeFn([](InferenceContext* context) -> absl::Status {
      ShapeHandle spikes;
      ShapeHandle weights;
      TF_RETURN_IF_ERROR(context->WithRank(context->input(0), 2, &spikes));
      TF_RETURN_IF_ERROR(context->WithRank(context->input(3), 1, &weights));
      context->set_output(0, spikes);
      context->set_output(1, weights);
      return absl::OkStatus();
    });

REGISTER_OP("DpointnetCsrWeightGrad")
    .Input("spikes: T")
    .Input("current_grad: T")
    .Input("metadata: resource")
    .Input("basis: T")
    .Attr("T: {half, float}")
    .Attr("Tindex: {uint32, int64}")
    .Attr("n_post: int >= 1")
    .Attr("n_edges: int >= 0")
    .Output("weight_grad: float")
    .SetShapeFn([](InferenceContext* context) -> absl::Status {
      ShapeHandle edge_ids;
      int64_t n_edges;
      TF_RETURN_IF_ERROR(context->GetAttr("n_edges", &n_edges));
      context->set_output(0, context->Vector(n_edges));
      return absl::OkStatus();
    });
