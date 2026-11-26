#include <vector>
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;

REGISTER_OP("CompactInferenceCurrents")
    .Input("spikes: T")
    .Input("row_ptr: int64")
    .Input("targets: uint32")
    .Input("weights: T")
    .Input("basis: T")
    .Output("currents: T")
    .Attr("T: {float, half}")
    .Attr("n_neurons: int >= 1")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle spikes, basis, vector;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &spikes));
      for (int i = 1; i < 4; ++i)
        TF_RETURN_IF_ERROR(c->WithRank(c->input(i), 1, &vector));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(4), 2, &basis));
      int64_t neurons;
      TF_RETURN_IF_ERROR(c->GetAttr("n_neurons", &neurons));
      shape_inference::DimensionHandle rows;
      TF_RETURN_IF_ERROR(c->Multiply(c->Dim(spikes, 0), neurons, &rows));
      c->set_output(0, c->Matrix(rows, c->Dim(basis, 1)));
      return absl::OkStatus();
    });

template <typename T>
class CompactInferenceCPU : public OpKernel {
 public:
  explicit CompactInferenceCPU(OpKernelConstruction* c) : OpKernel(c) {
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
    const int64_t rows = spikes.dim_size(1), batch = spikes.dim_size(0),
                  types = basis.dim_size(0), bases = basis.dim_size(1),
                  edges = weights.NumElements(), out_size = neurons_ * bases;
    auto offsets = ptr.flat<int64_t>();
    auto packed = targets.flat<uint32>();
    // Connectivity is validated once when the cell is built; per call, only the rows
    // and edges an active source visits are checked, so frozen inference never scans
    // every edge at every step.
    OP_REQUIRES(c, offsets(0) == 0 && offsets(rows) == edges,
                errors::InvalidArgument("Invalid compact row endpoints"));
    const uint64_t packed_limit = static_cast<uint64_t>(neurons_) * types;
    Tensor* out;
    OP_REQUIRES_OK(c, c->allocate_output(0, {batch * neurons_, bases}, &out));
    std::vector<float> accumulated(batch * out_size, 0.0f);
    auto input = spikes.flat<T>();
    auto values = weights.flat<T>();
    auto factors = basis.flat<T>();
    for (int64_t slot = 0; slot < batch; ++slot)
      for (int64_t row = 0; row < rows; ++row) {
        const float spike = static_cast<float>(input(slot * rows + row));
        if (spike == 0.0f) continue;
        const int64_t begin = offsets(row), end = offsets(row + 1);
        OP_REQUIRES(c, begin >= 0 && begin <= end && end <= edges,
                    errors::InvalidArgument("Invalid compact row offsets"));
        for (int64_t edge = begin; edge < end; ++edge) {
          OP_REQUIRES(c, packed(edge) < packed_limit,
                      errors::InvalidArgument("Invalid compact target/type"));
          const int64_t target = packed(edge) / types, type = packed(edge) % types;
          for (int64_t receptor = 0; receptor < bases; ++receptor)
            accumulated[slot * out_size + target * bases + receptor] +=
                spike * static_cast<float>(values(edge)) *
                static_cast<float>(factors(type * bases + receptor));
        }
      }
    auto output = out->flat<T>();
    for (int64_t i = 0; i < output.size(); ++i) output(i) = static_cast<T>(accumulated[i]);
  }
 private:
  int64_t neurons_;
};

REGISTER_KERNEL_BUILDER(Name("CompactInferenceCurrents").Device(DEVICE_CPU)
    .TypeConstraint<float>("T"), CompactInferenceCPU<float>);
REGISTER_KERNEL_BUILDER(Name("CompactInferenceCurrents").Device(DEVICE_CPU)
    .TypeConstraint<Eigen::half>("T"), CompactInferenceCPU<Eigen::half>);
