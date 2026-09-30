#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_handle.h"
#include "tensorflow/core/framework/resource_base.h"
#include <mutex>
#include <vector>

using tensorflow::shape_inference::DimensionHandle;
using tensorflow::shape_inference::InferenceContext;
using tensorflow::shape_inference::ShapeHandle;

REGISTER_OP("DpointnetCurrentTapeCreate")
    .Input("size: int32")
    .Attr("T: {half, float}")
    .Output("handle: resource")
    .SetIsStateful()
    .SetShapeFn([](InferenceContext* c) -> absl::Status {
      ShapeHandle size;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &size));
      c->set_output(0, c->Scalar());
      c->set_output_handle_shapes_and_types(
          0, {{c->Scalar(), tensorflow::DT_INT32}});
      return absl::OkStatus();
    });

REGISTER_OP("DpointnetCurrentTapeWrite")
    .Input("handle: resource")
    .Input("index: int32")
    .Input("currents: T")
    .Input("flow: int32")
    .Attr("T: {half, float}")
    .Output("next_flow: int32")
    .SetIsStateful()
    .SetShapeFn([](InferenceContext* c) -> absl::Status {
      ShapeHandle ignored;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &ignored));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &ignored));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 3, &ignored));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 0, &ignored));
      c->set_output(0, c->Scalar());
      return absl::OkStatus();
    });

REGISTER_OP("DpointnetCurrentTapeRead")
    .Input("handle: resource")
    .Input("index: int32")
    .Input("flow: int32")
    .Attr("T: {half, float}")
    .Output("currents: T")
    .SetIsStateful()
    .SetShapeFn([](InferenceContext* c) -> absl::Status {
      ShapeHandle ignored;
      for (int i = 0; i < 3; ++i) {
        TF_RETURN_IF_ERROR(c->WithRank(c->input(i), 0, &ignored));
      }
      c->set_output(0, c->UnknownShapeOfRank(3));
      return absl::OkStatus();
    });

namespace tensorflow {
namespace {

class CurrentTape : public ResourceBase {
 public:
  CurrentTape(int size, DataType dtype) : chunks_(size), dtype_(dtype) {}
  std::string DebugString() const override { return "DPointNet CPU current tape"; }
  std::mutex mutex;
  std::vector<Tensor> chunks_;
  DataType dtype_;
  int written = 0;
};

class CurrentTapeCreateOp : public OpKernel {
 public:
  explicit CurrentTapeCreateOp(OpKernelConstruction* c) : OpKernel(c) {
    OP_REQUIRES_OK(c, c->GetAttr("T", &dtype_));
  }
  void Compute(OpKernelContext* c) override {
    OP_REQUIRES(c, TensorShapeUtils::IsScalar(c->input(0).shape()),
                errors::InvalidArgument("Tape size must be scalar"));
    const int size = c->input(0).scalar<int32>()();
    OP_REQUIRES(c, size > 0, errors::InvalidArgument("Tape size must be positive"));
    Tensor* handle;
    OP_REQUIRES_OK(c, c->allocate_output(0, TensorShape({}), &handle));
    handle->scalar<ResourceHandle>()() = ResourceHandle::MakeRefCountingHandle(
        new CurrentTape(size, dtype_), c->device()->name(),
        {{DT_INT32, PartialTensorShape({})}});
  }
 private:
  DataType dtype_;
};

class CurrentTapeWriteOp : public OpKernel {
 public:
  explicit CurrentTapeWriteOp(OpKernelConstruction* c) : OpKernel(c) {}
  void Compute(OpKernelContext* c) override {
    for (int i : {0, 1, 3}) {
      OP_REQUIRES(c, TensorShapeUtils::IsScalar(c->input(i).shape()),
                  errors::InvalidArgument("Tape handle, index and flow must be scalar"));
    }
    auto result = c->input(0).scalar<ResourceHandle>()().GetResource<CurrentTape>();
    OP_REQUIRES_OK(c, result.status());
    auto* tape = result.value();
    const int index = c->input(1).scalar<int32>()();
    const int flow = c->input(3).scalar<int32>()();
    std::lock_guard<std::mutex> lock(tape->mutex);
    OP_REQUIRES(c, index == flow && index == tape->written &&
                   index >= 0 && index < tape->chunks_.size(),
                errors::InvalidArgument("Current tape writes must be sequential and in range"));
    OP_REQUIRES(c, c->input(2).dims() == 3 && c->input(2).dtype() == tape->dtype_,
                errors::InvalidArgument("Current tape dtype or rank mismatch"));
    // Tensor ownership is shared with the CPU input buffer, not copied elementwise.
    tape->chunks_[index] = c->input(2);
    ++tape->written;
    Tensor* next;
    OP_REQUIRES_OK(c, c->allocate_output(0, TensorShape({}), &next));
    next->scalar<int32>()() = index + 1;
  }
};

class CurrentTapeReadOp : public OpKernel {
 public:
  explicit CurrentTapeReadOp(OpKernelConstruction* c) : OpKernel(c) {}
  void Compute(OpKernelContext* c) override {
    for (int i = 0; i < 3; ++i) {
      OP_REQUIRES(c, TensorShapeUtils::IsScalar(c->input(i).shape()),
                  errors::InvalidArgument("Tape handle, index and flow must be scalar"));
    }
    auto result = c->input(0).scalar<ResourceHandle>()().GetResource<CurrentTape>();
    OP_REQUIRES_OK(c, result.status());
    auto* tape = result.value();
    const int index = c->input(1).scalar<int32>()();
    const int flow = c->input(2).scalar<int32>()();
    std::lock_guard<std::mutex> lock(tape->mutex);
    OP_REQUIRES(c, index >= 0 && index < flow && flow <= tape->written,
                errors::InvalidArgument("Current chunk not recorded"));
    OP_REQUIRES(c, tape->dtype_ == output_type(0),
                errors::InvalidArgument("Current tape output dtype mismatch"));
    c->set_output(0, tape->chunks_[index]);
  }
};

REGISTER_KERNEL_BUILDER(Name("DpointnetCurrentTapeCreate").Device(DEVICE_CPU), CurrentTapeCreateOp);
REGISTER_KERNEL_BUILDER(Name("DpointnetCurrentTapeWrite").Device(DEVICE_CPU), CurrentTapeWriteOp);
REGISTER_KERNEL_BUILDER(Name("DpointnetCurrentTapeRead").Device(DEVICE_CPU), CurrentTapeReadOp);
}  // namespace
}  // namespace tensorflow

REGISTER_OP("DpointnetCsrReorder")
    .Input("values: T")
    .Input("metadata: resource")
    .Attr("T: {half, float}")
    .Attr("Tindex: {uint32, int64}")
    .Attr("n_edges: int >= 0")
  .Attr("n_sources: int >= 1")
  .Attr("n_pairs: int >= 0")
    .Output("reordered: T")
    .SetShapeFn([](InferenceContext* context) -> absl::Status {
      ShapeHandle values;
      TF_RETURN_IF_ERROR(context->WithRank(context->input(0), 1, &values));
      context->set_output(0, values);
      return absl::OkStatus();
    });

REGISTER_OP("DpointnetCsrRestore")
    .Input("values: T")
    .Input("metadata: resource")
    .Attr("T: {half, float}")
    .Attr("Tindex: {uint32, int64}")
    .Attr("n_edges: int >= 0")
    .Attr("n_sources: int >= 1")
    .Attr("n_pairs: int >= 0")
    .Output("restored: T")
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
    .Input("spike_gradient_scale: T")
    .Input("active_rows: int64")
    .Input("incoming_pre_ids: Tindex")
    .Input("incoming_edge_ids: Tindex")
    .Input("incoming_types: Tindex")
    .Input("initial: T")
    .Attr("T: {half, float}")
    .Attr("Tmaster: {half, float}")
    .Attr("Tindex: {uint32, int64}")
    .Attr("n_post: int >= 1")
    .Attr("n_edges: int >= 0")
    .Attr("n_pairs: int >= 0")
    .Attr("compute_spike_gradient: bool")
    .Attr("compute_weight_gradient: bool = true")
    .Attr("use_grouped_batch32_forward: bool = false")
    .Attr("use_fixed4_forward: bool = false")
    .Attr("use_forward_run_aggregation: bool = false")
    .Attr("use_device_active_queue_forward: bool = false")
    .Attr("use_packed_sm120_backward: bool = false")
    .Attr("write_csr_weight_gradient: bool = false")
    .Attr("use_small_batch_backward: bool = false")
    .Attr("use_javier_batch32_backward: bool = false")
    .Attr("vjp_only: bool = false")
    .Output("currents: T")
    .SetShapeFn([](InferenceContext* context) -> absl::Status {
      ShapeHandle spikes;
      ShapeHandle basis;
      TF_RETURN_IF_ERROR(context->WithRank(context->input(0), 2, &spikes));
      TF_RETURN_IF_ERROR(context->WithRank(context->input(4), 2, &basis));
        ShapeHandle spike_gradient_scale;
        TF_RETURN_IF_ERROR(context->WithRank(
          context->input(5), 0, &spike_gradient_scale));
      ShapeHandle active_rows;
      TF_RETURN_IF_ERROR(context->WithRank(context->input(6), 1, &active_rows));
      ShapeHandle incoming_pre_ids;
      ShapeHandle incoming_edge_ids;
      ShapeHandle incoming_types;
      TF_RETURN_IF_ERROR(context->WithRank(context->input(7), 1, &incoming_pre_ids));
      TF_RETURN_IF_ERROR(context->WithRank(context->input(8), 1, &incoming_edge_ids));
      TF_RETURN_IF_ERROR(context->WithRank(context->input(9), 1, &incoming_types));
      ShapeHandle initial = context->input(10);
      if (context->RankKnown(initial) && context->Rank(initial) != 1 &&
          context->Rank(initial) != 2) {
        return absl::InvalidArgumentError(
            "initial currents must be an empty vector or rank two");
      }
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
    .Input("spike_gradient_scale: T")
    .Attr("T: {half, float}")
    .Attr("Tindex: {uint32, int64}")
    .Attr("n_post: int >= 1")
    .Attr("n_edges: int >= 0")
    .Attr("n_pairs: int >= 0")
    .Attr("use_packed_sm120_backward: bool = false")
    .Attr("write_csr_weight_gradient: bool = false")
    .Attr("use_small_batch_backward: bool = false")
    .Attr("use_javier_batch32_backward: bool = false")
    .Output("spike_grad: T")
    .Output("weight_grad: float")
    .SetShapeFn([](InferenceContext* context) -> absl::Status {
      ShapeHandle spikes;
      ShapeHandle weights;
      TF_RETURN_IF_ERROR(context->WithRank(context->input(0), 2, &spikes));
      TF_RETURN_IF_ERROR(context->WithRank(context->input(3), 1, &weights));
        ShapeHandle spike_gradient_scale;
        TF_RETURN_IF_ERROR(context->WithRank(
          context->input(5), 0, &spike_gradient_scale));
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
    .Attr("n_pairs: int >= 0")
    .Attr("use_packed_sm120_backward: bool = false")
    .Output("weight_grad: float")
    .SetShapeFn([](InferenceContext* context) -> absl::Status {
      ShapeHandle edge_ids;
      int64_t n_edges;
      TF_RETURN_IF_ERROR(context->GetAttr("n_edges", &n_edges));
      context->set_output(0, context->Vector(n_edges));
      return absl::OkStatus();
    });

REGISTER_OP("DpointnetCsrSpikeGradAccumulate")
    .Input("spikes: T")
    .Input("current_grad: T")
    .Input("metadata: resource")
    .Input("weights: T")
    .Input("basis: T")
    .Input("spike_gradient_scale: T")
    .Input("accumulator: float")
    .Attr("T: {half, float}")
    .Attr("Tindex: {uint32}")
    .Attr("n_post: int >= 1")
    .Attr("n_edges: int >= 0")
    .Attr("n_pairs: int >= 1")
    .Attr("use_packed_sm120_backward: bool = false")
    .Attr("write_csr_weight_gradient: bool = true")
    .Attr("use_small_batch_backward: bool = false")
    .Attr("use_javier_batch32_backward: bool = false")
    .Output("spike_grad: T")
    .Output("weight_grad: float")
    .SetShapeFn([](InferenceContext* context) -> absl::Status {
      ShapeHandle spikes, weights, accumulator;
      TF_RETURN_IF_ERROR(context->WithRank(context->input(0), 2, &spikes));
      TF_RETURN_IF_ERROR(context->WithRank(context->input(3), 1, &weights));
      TF_RETURN_IF_ERROR(context->WithRank(context->input(6), 1, &accumulator));
      TF_RETURN_IF_ERROR(context->Merge(weights, accumulator, &weights));
      context->set_output(0, spikes);
      context->set_output(1, weights);
      return absl::OkStatus();
    });
