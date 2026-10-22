#pragma once

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

REGISTER_OP("DpointnetMasterProjection")
    .Input("spikes: T")
    .Input("weights: float")
    .Input("metadata: I")
    .Input("basis: T")
    .Attr("T: {half, float}")
    .Attr("I: {uint32, int64}")
    .Attr("n_post: int >= 1")
    .Attr("n_edges: int >= 0")
    .Output("currents: float")
    .SetShapeFn([](tensorflow::shape_inference::InferenceContext* c) -> absl::Status {
      tensorflow::shape_inference::ShapeHandle z, w, m, b;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &z));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &w));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &m));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 2, &b));
      int posts;
      TF_RETURN_IF_ERROR(c->GetAttr("n_post", &posts));
      tensorflow::shape_inference::DimensionHandle rows;
      TF_RETURN_IF_ERROR(c->Multiply(c->Dim(z, 0), posts, &rows));
      c->set_output(0, c->Matrix(rows, c->Dim(b, 1)));
      return absl::OkStatus();
    });

REGISTER_OP("DpointnetMasterProjectionGrad")
    .Input("spikes: T")
    .Input("weights: float")
    .Input("metadata: I")
    .Input("basis: T")
    .Input("current_grad: float")
    .Attr("T: {half, float}")
    .Attr("I: {uint32, int64}")
    .Attr("n_post: int >= 1")
    .Attr("n_edges: int >= 0")
    .Output("spike_grad: float")
    .Output("weight_grad: float")
    .SetShapeFn([](tensorflow::shape_inference::InferenceContext* c) -> absl::Status {
      c->set_output(0, c->input(0));
      c->set_output(1, c->input(1));
      return absl::OkStatus();
    });

namespace tensorflow {

inline absl::Status ValidateMasterProjection(
    OpKernelContext* c, int posts, int64_t edges, bool gradient) {
  const auto& z = c->input(0);
  const auto& w = c->input(1);
  const auto& m = c->input(2);
  const auto& b = c->input(3);
  if (z.dims() != 2 || w.dims() != 1 || m.dims() != 1 || b.dims() != 2)
    return errors::InvalidArgument("Master projection expects ranks 2,1,1,2");
  if (w.NumElements() != edges ||
      m.NumElements() < 3 * edges + z.dim_size(1) + 1 || b.dim_size(1) != 4)
    return errors::InvalidArgument("Master projection operand dimensions disagree");
  if (gradient && c->input(4).shape() != TensorShape({z.dim_size(0) * posts, 4}))
    return errors::InvalidArgument("Master projection cotangent shape disagrees");
  return absl::OkStatus();
}

// Every index-valued metadata field is checked against the operand dimensions
// before any is used as an offset; int64 metadata may also hold negatives.
template <typename I>
inline absl::Status ValidateMasterMetadata(
    const I* meta, int64_t width, int64_t posts, int64_t types, int64_t edges) {
  const I* splits = meta + 2 * edges;
  const I* canonical = splits + width + 1;
  for (int64_t row = 0; row <= width; ++row) {
    const int64_t split = static_cast<int64_t>(splits[row]);
    if (split < 0 || split > edges ||
        (row > 0 && split < static_cast<int64_t>(splits[row - 1])))
      return errors::InvalidArgument(
          "Invalid master projection metadata: row splits must be nondecreasing in [0, ",
          edges, "]");
  }
  for (int64_t e = 0; e < edges; ++e) {
    const int64_t post = static_cast<int64_t>(meta[e]);
    const int64_t type = static_cast<int64_t>(meta[edges + e]);
    const int64_t edge = static_cast<int64_t>(canonical[e]);
    if (post < 0 || post >= posts)
      return errors::InvalidArgument(
          "Invalid master projection metadata: post index ", post, " outside [0, ", posts, ")");
    if (type < 0 || type >= types)
      return errors::InvalidArgument(
          "Invalid master projection metadata: synapse type ", type, " outside [0, ", types,
          ") basis rows");
    if (edge < 0 || edge >= edges)
      return errors::InvalidArgument(
          "Invalid master projection metadata: canonical edge ID ", edge, " outside [0, ",
          edges, ")");
  }
  return absl::OkStatus();
}

template <typename T, typename I, bool Gradient>
class MasterProjectionCpu : public OpKernel {
 public:
  explicit MasterProjectionCpu(OpKernelConstruction* c) : OpKernel(c) {
    OP_REQUIRES_OK(c, c->GetAttr("n_post", &posts_));
    OP_REQUIRES_OK(c, c->GetAttr("n_edges", &edges_));
  }
  void Compute(OpKernelContext* c) override {
    OP_REQUIRES_OK(c, ValidateMasterProjection(c, posts_, edges_, Gradient));
    const auto& z = c->input(0);
    const auto* weights = c->input(1).flat<float>().data();
    const auto* meta = c->input(2).flat<I>().data();
    const auto* basis = c->input(3).flat<T>().data();
    const int64_t width = z.dim_size(1), batch = z.dim_size(0);
    OP_REQUIRES_OK(c, ValidateMasterMetadata(
        meta, width, posts_, c->input(3).dim_size(0), edges_));
    const I* splits = meta + 2 * edges_;
    const I* canonical = splits + width + 1;
    Tensor *out, *dw = nullptr;
    OP_REQUIRES_OK(c, c->allocate_output(
        0, Gradient ? z.shape() : TensorShape({batch * posts_, 4}), &out));
    if constexpr (Gradient) {
      OP_REQUIRES_OK(c, c->allocate_output(1, c->input(1).shape(), &dw));
      dw->flat<float>().setZero();
    } else {
      out->flat<float>().setZero();
    }
    for (int64_t row = 0; row < width; ++row) {
      for (int64_t sample = 0; sample < batch; ++sample) {
        float dz = 0;
        const float spike = static_cast<float>(z.flat<T>()(sample * width + row));
        for (int64_t e = splits[row]; e < splits[row + 1]; ++e) {
          const int64_t post = meta[e], type = meta[edges_ + e], edge = canonical[e];
          float projected = 0;
          for (int k = 0; k < 4; ++k) {
            const float coefficient = static_cast<float>(basis[type * 4 + k]);
            const int64_t target = (sample * posts_ + post) * 4 + k;
            if constexpr (Gradient) {
              projected += c->input(4).flat<float>()(target) * coefficient;
            } else {
              out->flat<float>()(target) += spike * weights[edge] * coefficient;
            }
          }
          if constexpr (Gradient) {
            dz += weights[edge] * projected;
            dw->flat<float>()(edge) += spike * projected;
          }
        }
        if constexpr (Gradient) out->flat<float>()(sample * width + row) = dz;
      }
    }
  }
 private:
  int posts_;
  int64_t edges_;
};

#define REGISTER_MASTER_CPU(T, I)                                           \
  REGISTER_KERNEL_BUILDER(Name("DpointnetMasterProjection").Device(DEVICE_CPU) \
      .TypeConstraint<T>("T").TypeConstraint<I>("I"), MasterProjectionCpu<T, I, false>); \
  REGISTER_KERNEL_BUILDER(Name("DpointnetMasterProjectionGrad").Device(DEVICE_CPU) \
      .TypeConstraint<T>("T").TypeConstraint<I>("I"), MasterProjectionCpu<T, I, true>);
REGISTER_MASTER_CPU(Eigen::half, uint32)
REGISTER_MASTER_CPU(Eigen::half, int64_t)
REGISTER_MASTER_CPU(float, uint32)
REGISTER_MASTER_CPU(float, int64_t)
#undef REGISTER_MASTER_CPU
}  // namespace tensorflow
