#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;

namespace {
// Internal experiment only: AoS [neurons, 28] remains the default. SoA [28,
// neurons] changes addresses, not coefficient values or neuron/state ordering.
// Callers must supply the matching layout to both forward and backward.
Status NestCoefficientShape(shape_inference::InferenceContext* context,
                            int coefficient_index) {
  string layout;
  DataType dtype;
  TF_RETURN_IF_ERROR(context->GetAttr("coefficients_layout", &layout));
  TF_RETURN_IF_ERROR(context->GetAttr("T", &dtype));
  if (layout == "soa" && dtype != DT_FLOAT) {
    return errors::InvalidArgument("NEST SoA coefficients require FP32 voltage/ASC");
  }
  shape_inference::ShapeHandle voltage, coefficients;
  TF_RETURN_IF_ERROR(context->WithRank(context->input(0), 2, &voltage));
  TF_RETURN_IF_ERROR(context->WithRank(context->input(coefficient_index), 2,
                                      &coefficients));
  shape_inference::DimensionHandle dimension;
  TF_RETURN_IF_ERROR(context->WithValue(
      context->Dim(coefficients, layout == "soa" ? 0 : 1), 28, &dimension));
  TF_RETURN_IF_ERROR(context->Merge(
      context->Dim(coefficients, layout == "soa" ? 1 : 0),
      context->Dim(voltage, 1), &dimension));
  return OkStatus();
}

Status NestTypeIndexedCoefficientShape(shape_inference::InferenceContext* context,
                                       int coefficient_index,
                                       int type_index_index) {
  shape_inference::ShapeHandle voltage, coefficients, type_indices;
  TF_RETURN_IF_ERROR(context->WithRank(context->input(0), 2, &voltage));
  TF_RETURN_IF_ERROR(context->WithRank(context->input(coefficient_index), 2,
                                      &coefficients));
  TF_RETURN_IF_ERROR(context->WithRank(context->input(type_index_index), 1,
                                      &type_indices));
  shape_inference::DimensionHandle dimension;
  TF_RETURN_IF_ERROR(context->WithValue(context->Dim(coefficients, 1), 28,
                                       &dimension));
  TF_RETURN_IF_ERROR(context->Merge(context->Dim(type_indices, 0),
                                    context->Dim(voltage, 1), &dimension));
  return OkStatus();
}
}  // namespace

namespace {
Status NestDeviceDispatchShape(shape_inference::InferenceContext* context,
                               bool backward, bool history, bool events) {
  TF_RETURN_IF_ERROR(NestTypeIndexedCoefficientShape(
      context, backward ? 2 : 6, backward ? 3 : 7));
  const int fallback = context->num_inputs() - 2;
  shape_inference::ShapeHandle voltage, coefficients, predicate;
  TF_RETURN_IF_ERROR(context->WithRank(context->input(0), 2, &voltage));
  TF_RETURN_IF_ERROR(context->WithRank(context->input(fallback), 2, &coefficients));
  TF_RETURN_IF_ERROR(context->WithRank(context->input(fallback + 1), 0, &predicate));
  shape_inference::DimensionHandle dimension;
  TF_RETURN_IF_ERROR(context->WithValue(context->Dim(coefficients, 1), 28, &dimension));
  TF_RETURN_IF_ERROR(context->Merge(context->Dim(coefficients, 0),
                                    context->Dim(voltage, 1), &dimension));
  if (backward) {
    context->set_output(0, voltage);
    context->set_output(1, context->input(8));
    context->set_output(2, context->input(9));
    context->set_output(3, context->input(10));
    context->set_output(4, context->input(9));
    if (history) context->set_output(5, context->input(events ? 16 : 14));
  } else {
    context->set_output(0, voltage);
    for (int output = 1; output < 6; ++output) {
      context->set_output(output, context->input(output - 1));
    }
    if (history) {
      context->set_output(6, voltage);
      context->set_output(7, context->input(11));
    }
  }
  return OkStatus();
}
}  // namespace

#define NEST_DEVICE_TYPES \
    .Attr("T: {half, float}") .Attr("S: {half, float}") \
    .Attr("R: {int8, int16}") .Attr("hard_reset: bool = false")
#define NEST_DEVICE_FORWARD_INPUTS \
    .Input("v: T") .Input("r: R") .Input("asc: T") \
    .Input("psc_rise: S") .Input("psc: S") .Input("currents: S") \
    .Input("type_coefficients: T") .Input("type_indices: int64") \
    .Input("t_ref: R") .Input("dt: T") .Input("v_th: T")
#define NEST_DEVICE_FORWARD_OUTPUTS \
    .Output("threshold_voltage: T") .Output("new_v: T") .Output("new_r: R") \
    .Output("new_asc: T") .Output("new_rise: S") .Output("new_psc: S")
#define NEST_DEVICE_BACKWARD_INPUTS \
    .Input("threshold_voltage: T") .Input("r: R") \
    .Input("type_coefficients: T") .Input("type_indices: int64") \
    .Input("dt: T") .Input("voltage_gradient_retention: T") \
    .Input("grad_threshold: T") .Input("grad_v: T") .Input("grad_asc: T") \
    .Input("grad_rise: S") .Input("grad_psc: S")
#define NEST_DEVICE_BACKWARD_OUTPUTS \
    .Output("v_grad: T") .Output("asc_grad: T") .Output("rise_grad: S") \
    .Output("psc_grad: S") .Output("currents_grad: S")
#define NEST_DEVICE_EVENT_ATTRS \
    .Attr("detach_reset: bool = true") .Attr("detach_asc_reset: bool = true") \
    .Attr("pseudo_gauss: bool = false")
#define NEST_DEVICE_EVENT_INPUTS \
    .Input("previous_asc: T") .Input("v_th: T") \
    .Input("dampening: T") .Input("gauss_std: T")
#define NEST_DEVICE_GUARD_INPUTS \
    .Input("fallback_coefficients: T") .Input("type_identity: bool")

REGISTER_OP("DpointnetNestStateForwardDeviceDispatch")
    NEST_DEVICE_TYPES .Attr("emit_pre_reset_voltage: bool = false")
    NEST_DEVICE_FORWARD_INPUTS NEST_DEVICE_GUARD_INPUTS NEST_DEVICE_FORWARD_OUTPUTS
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      return NestDeviceDispatchShape(c, false, false, false);
    });
REGISTER_OP("DpointnetNestStateHistoryForwardDeviceDispatch")
    NEST_DEVICE_TYPES .Attr("emit_pre_reset_voltage: bool = false")
    NEST_DEVICE_FORWARD_INPUTS .Input("history: S")
    NEST_DEVICE_GUARD_INPUTS NEST_DEVICE_FORWARD_OUTPUTS
    .Output("spikes: S") .Output("new_history: S")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      return NestDeviceDispatchShape(c, false, true, false);
    });
REGISTER_OP("DpointnetNestStateBackwardDeviceDispatch")
    NEST_DEVICE_TYPES NEST_DEVICE_BACKWARD_INPUTS
    NEST_DEVICE_GUARD_INPUTS NEST_DEVICE_BACKWARD_OUTPUTS
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      return NestDeviceDispatchShape(c, true, false, false);
    });
REGISTER_OP("DpointnetNestStateBackwardEventsDeviceDispatch")
    NEST_DEVICE_TYPES NEST_DEVICE_EVENT_ATTRS NEST_DEVICE_BACKWARD_INPUTS
    NEST_DEVICE_EVENT_INPUTS NEST_DEVICE_GUARD_INPUTS NEST_DEVICE_BACKWARD_OUTPUTS
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      return NestDeviceDispatchShape(c, true, false, true);
    });
REGISTER_OP("DpointnetNestStateHistoryBackwardDeviceDispatch")
    NEST_DEVICE_TYPES .Attr("pseudo_gauss: bool = false")
    NEST_DEVICE_BACKWARD_INPUTS .Input("dampening: T") .Input("gauss_std: T")
    .Input("grad_spikes: S") .Input("grad_history: S")
    NEST_DEVICE_GUARD_INPUTS NEST_DEVICE_BACKWARD_OUTPUTS .Output("old_history_grad: S")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      return NestDeviceDispatchShape(c, true, true, false);
    });
REGISTER_OP("DpointnetNestStateHistoryBackwardEventsDeviceDispatch")
    NEST_DEVICE_TYPES NEST_DEVICE_EVENT_ATTRS NEST_DEVICE_BACKWARD_INPUTS
    NEST_DEVICE_EVENT_INPUTS .Input("grad_spikes: S") .Input("grad_history: S")
    NEST_DEVICE_GUARD_INPUTS NEST_DEVICE_BACKWARD_OUTPUTS .Output("old_history_grad: S")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      return NestDeviceDispatchShape(c, true, true, true);
    });
// ``grad_history_addend`` is a second new-history cotangent summed with
// ``grad_history`` inside the kernel (direct recorded adjoint: the recurrent
// source credit of the following step).
REGISTER_OP("DpointnetNestStateHistoryBackwardEventsAddendDeviceDispatch")
    NEST_DEVICE_TYPES NEST_DEVICE_EVENT_ATTRS NEST_DEVICE_BACKWARD_INPUTS
    NEST_DEVICE_EVENT_INPUTS .Input("grad_spikes: S") .Input("grad_history: S")
    .Input("grad_history_addend: S")
    NEST_DEVICE_GUARD_INPUTS NEST_DEVICE_BACKWARD_OUTPUTS .Output("old_history_grad: S")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      return NestDeviceDispatchShape(c, true, true, true);
    });

#undef NEST_DEVICE_TYPES
#undef NEST_DEVICE_FORWARD_INPUTS
#undef NEST_DEVICE_FORWARD_OUTPUTS
#undef NEST_DEVICE_BACKWARD_INPUTS
#undef NEST_DEVICE_BACKWARD_OUTPUTS
#undef NEST_DEVICE_EVENT_ATTRS
#undef NEST_DEVICE_EVENT_INPUTS
#undef NEST_DEVICE_GUARD_INPUTS

REGISTER_OP("DpointnetNestStateForward")
    .Attr("launch_geometry_version: int = 1")
    .Attr("T: {half, float}")
    .Attr("S: {half, float}")
    .Attr("R: {int8, int16}")
    .Attr("hard_reset: bool = false")
    .Attr("emit_pre_reset_voltage: bool = false")
    .Attr("coefficients_layout: {'aos', 'soa'} = 'aos'")
    .Input("v: T")
    .Input("r: R")
    .Input("asc: T")
    .Input("psc_rise: S")
    .Input("psc: S")
    .Input("currents: S")
    .Input("coefficients: T")
    .Input("t_ref: R")
    .Input("dt: T")
    .Input("v_th: T")
    .Output("threshold_voltage: T")
    .Output("new_v: T")
    .Output("new_r: R")
    .Output("new_asc: T")
    .Output("new_rise: S")
    .Output("new_psc: S")
    .SetShapeFn([](shape_inference::InferenceContext* context) {
      TF_RETURN_IF_ERROR(NestCoefficientShape(context, 6));
      shape_inference::ShapeHandle voltage;
      TF_RETURN_IF_ERROR(context->WithRank(context->input(0), 2, &voltage));
      context->set_output(0, voltage);
      for (int output = 1; output < 6; ++output) {
        context->set_output(output, context->input(output - 1));
      }
      return OkStatus();
    });

REGISTER_OP("DpointnetNestStateHistoryForward")
    .Attr("launch_geometry_version: int = 1")
    .Attr("T: {half, float}")
    .Attr("S: {half, float}")
    .Attr("R: {int8, int16}")
    .Attr("hard_reset: bool = false")
    .Attr("emit_pre_reset_voltage: bool = false")
    .Attr("coefficients_layout: {'aos', 'soa'} = 'aos'")
    .Input("v: T")
    .Input("r: R")
    .Input("asc: T")
    .Input("psc_rise: S")
    .Input("psc: S")
    .Input("currents: S")
    .Input("coefficients: T")
    .Input("t_ref: R")
    .Input("dt: T")
    .Input("v_th: T")
    .Input("history: S")
    .Output("threshold_voltage: T")
    .Output("new_v: T")
    .Output("new_r: R")
    .Output("new_asc: T")
    .Output("new_rise: S")
    .Output("new_psc: S")
    .Output("spikes: S")
    .Output("new_history: S")
    .SetShapeFn([](shape_inference::InferenceContext* context) {
      TF_RETURN_IF_ERROR(NestCoefficientShape(context, 6));
      shape_inference::ShapeHandle voltage, history;
      TF_RETURN_IF_ERROR(context->WithRank(context->input(0), 2, &voltage));
      TF_RETURN_IF_ERROR(context->WithRank(context->input(10), 2, &history));
      shape_inference::DimensionHandle batch;
      TF_RETURN_IF_ERROR(context->Merge(
          context->Dim(voltage, 0), context->Dim(history, 0), &batch));
      context->set_output(0, voltage);
      for (int output = 1; output < 6; ++output) {
        context->set_output(output, context->input(output - 1));
      }
      context->set_output(6, voltage);
      context->set_output(7, history);
      return OkStatus();
    });

REGISTER_OP("DpointnetNestStateBackward")
    .Attr("launch_geometry_version: int = 1")
    .Attr("T: {half, float}")
    .Attr("S: {half, float}")
    .Attr("R: {int8, int16}")
    .Attr("hard_reset: bool = false")
    .Attr("coefficients_layout: {'aos', 'soa'} = 'aos'")
    .Input("threshold_voltage: T")
    .Input("r: R")
    .Input("coefficients: T")
    .Input("dt: T")
    .Input("voltage_gradient_retention: T")
    .Input("grad_threshold: T")
    .Input("grad_v: T")
    .Input("grad_asc: T")
    .Input("grad_rise: S")
    .Input("grad_psc: S")
    .Output("v_grad: T")
    .Output("asc_grad: T")
    .Output("rise_grad: S")
    .Output("psc_grad: S")
    .Output("currents_grad: S")
    .SetShapeFn([](shape_inference::InferenceContext* context) {
      TF_RETURN_IF_ERROR(NestCoefficientShape(context, 2));
      context->set_output(0, context->input(0));
      context->set_output(1, context->input(7));
      context->set_output(2, context->input(8));
      context->set_output(3, context->input(9));
      context->set_output(4, context->input(8));
      return OkStatus();
    });

REGISTER_OP("DpointnetNestStateHistoryBackward")
    .Attr("launch_geometry_version: int = 1")
    .Attr("T: {half, float}")
    .Attr("S: {half, float}")
    .Attr("R: {int8, int16}")
    .Attr("hard_reset: bool = false")
    .Attr("pseudo_gauss: bool = false")
    .Attr("coefficients_layout: {'aos', 'soa'} = 'aos'")
    .Input("threshold_voltage: T")
    .Input("r: R")
    .Input("coefficients: T")
    .Input("dt: T")
    .Input("voltage_gradient_retention: T")
    .Input("grad_threshold: T")
    .Input("grad_v: T")
    .Input("grad_asc: T")
    .Input("grad_rise: S")
    .Input("grad_psc: S")
    .Input("dampening: T")
    .Input("gauss_std: T")
    .Input("grad_spikes: S")
    .Input("grad_history: S")
    .Output("v_grad: T")
    .Output("asc_grad: T")
    .Output("rise_grad: S")
    .Output("psc_grad: S")
    .Output("currents_grad: S")
    .Output("old_history_grad: S")
    .SetShapeFn([](shape_inference::InferenceContext* context) {
      TF_RETURN_IF_ERROR(NestCoefficientShape(context, 2));
      context->set_output(0, context->input(0));
      context->set_output(1, context->input(7));
      context->set_output(2, context->input(8));
      context->set_output(3, context->input(9));
      context->set_output(4, context->input(8));
      for (int index = 10; index < 12; ++index) {
        shape_inference::ShapeHandle scalar;
        TF_RETURN_IF_ERROR(context->WithRank(context->input(index), 0, &scalar));
      }
      context->set_output(5, context->input(13));
      return OkStatus();
    });

REGISTER_OP("DpointnetNestStateBackwardEvents")
    .Attr("launch_geometry_version: int = 1")
    .Attr("T: {half, float}")
    .Attr("S: {half, float}")
    .Attr("R: {int8, int16}")
    .Attr("hard_reset: bool = false")
    .Attr("detach_reset: bool = true")
    .Attr("detach_asc_reset: bool = true")
    .Attr("pseudo_gauss: bool = false")
    .Attr("coefficients_layout: {'aos', 'soa'} = 'aos'")
    .Input("threshold_voltage: T")
    .Input("r: R")
    .Input("coefficients: T")
    .Input("dt: T")
    .Input("voltage_gradient_retention: T")
    .Input("grad_threshold: T")
    .Input("grad_v: T")
    .Input("grad_asc: T")
    .Input("grad_rise: S")
    .Input("grad_psc: S")
    .Input("previous_asc: T")
    .Input("v_th: T")
    .Input("dampening: T")
    .Input("gauss_std: T")
    .Output("v_grad: T")
    .Output("asc_grad: T")
    .Output("rise_grad: S")
    .Output("psc_grad: S")
    .Output("currents_grad: S")
    .SetShapeFn([](shape_inference::InferenceContext* context) {
      TF_RETURN_IF_ERROR(NestCoefficientShape(context, 2));
      shape_inference::ShapeHandle asc;
      TF_RETURN_IF_ERROR(context->Merge(context->input(7), context->input(10), &asc));
      for (int index = 11; index < 14; ++index) {
        shape_inference::ShapeHandle scalar;
        TF_RETURN_IF_ERROR(context->WithRank(context->input(index), 0, &scalar));
      }
      context->set_output(0, context->input(0));
      context->set_output(1, asc);
      context->set_output(2, context->input(8));
      context->set_output(3, context->input(9));
      context->set_output(4, context->input(8));
      return OkStatus();
    });

REGISTER_OP("DpointnetNestStateForwardTypeIndexed")
    .Attr("T: {half, float}")
    .Attr("S: {half, float}")
    .Attr("R: {int8, int16}")
    .Attr("hard_reset: bool = false")
    .Attr("emit_pre_reset_voltage: bool = false")
    .Input("v: T")
    .Input("r: R")
    .Input("asc: T")
    .Input("psc_rise: S")
    .Input("psc: S")
    .Input("currents: S")
    .Input("type_coefficients: T")
    .Input("type_indices: int64")
    .Input("t_ref: R")
    .Input("dt: T")
    .Input("v_th: T")
    .Output("threshold_voltage: T")
    .Output("new_v: T")
    .Output("new_r: R")
    .Output("new_asc: T")
    .Output("new_rise: S")
    .Output("new_psc: S")
    .SetShapeFn([](shape_inference::InferenceContext* context) {
      TF_RETURN_IF_ERROR(NestTypeIndexedCoefficientShape(context, 6, 7));
      shape_inference::ShapeHandle voltage;
      TF_RETURN_IF_ERROR(context->WithRank(context->input(0), 2, &voltage));
      context->set_output(0, voltage);
      for (int output = 1; output < 6; ++output) {
        context->set_output(output, context->input(output - 1));
      }
      return OkStatus();
    });

REGISTER_OP("DpointnetNestStateHistoryForwardTypeIndexed")
    .Attr("T: {half, float}")
    .Attr("S: {half, float}")
    .Attr("R: {int8, int16}")
    .Attr("hard_reset: bool = false")
    .Attr("emit_pre_reset_voltage: bool = false")
    .Input("v: T")
    .Input("r: R")
    .Input("asc: T")
    .Input("psc_rise: S")
    .Input("psc: S")
    .Input("currents: S")
    .Input("type_coefficients: T")
    .Input("type_indices: int64")
    .Input("t_ref: R")
    .Input("dt: T")
    .Input("v_th: T")
    .Input("history: S")
    .Output("threshold_voltage: T")
    .Output("new_v: T")
    .Output("new_r: R")
    .Output("new_asc: T")
    .Output("new_rise: S")
    .Output("new_psc: S")
    .Output("spikes: S")
    .Output("new_history: S")
    .SetShapeFn([](shape_inference::InferenceContext* context) {
      TF_RETURN_IF_ERROR(NestTypeIndexedCoefficientShape(context, 6, 7));
      shape_inference::ShapeHandle voltage, history;
      TF_RETURN_IF_ERROR(context->WithRank(context->input(0), 2, &voltage));
      TF_RETURN_IF_ERROR(context->WithRank(context->input(11), 2, &history));
      shape_inference::DimensionHandle batch;
      TF_RETURN_IF_ERROR(context->Merge(
          context->Dim(voltage, 0), context->Dim(history, 0), &batch));
      context->set_output(0, voltage);
      for (int output = 1; output < 6; ++output) {
        context->set_output(output, context->input(output - 1));
      }
      context->set_output(6, voltage);
      context->set_output(7, history);
      return OkStatus();
    });

REGISTER_OP("DpointnetNestStateBackwardTypeIndexed")
    .Attr("T: {half, float}")
    .Attr("S: {half, float}")
    .Attr("R: {int8, int16}")
    .Attr("hard_reset: bool = false")
    .Input("threshold_voltage: T")
    .Input("r: R")
    .Input("type_coefficients: T")
    .Input("type_indices: int64")
    .Input("dt: T")
    .Input("voltage_gradient_retention: T")
    .Input("grad_threshold: T")
    .Input("grad_v: T")
    .Input("grad_asc: T")
    .Input("grad_rise: S")
    .Input("grad_psc: S")
    .Output("v_grad: T")
    .Output("asc_grad: T")
    .Output("rise_grad: S")
    .Output("psc_grad: S")
    .Output("currents_grad: S")
    .SetShapeFn([](shape_inference::InferenceContext* context) {
      TF_RETURN_IF_ERROR(NestTypeIndexedCoefficientShape(context, 2, 3));
      context->set_output(0, context->input(0));
      context->set_output(1, context->input(8));
      context->set_output(2, context->input(9));
      context->set_output(3, context->input(10));
      context->set_output(4, context->input(9));
      return OkStatus();
    });

REGISTER_OP("DpointnetNestStateBackwardEventsTypeIndexed")
    .Attr("T: {half, float}")
    .Attr("S: {half, float}")
    .Attr("R: {int8, int16}")
    .Attr("hard_reset: bool = false")
    .Attr("detach_reset: bool = true")
    .Attr("detach_asc_reset: bool = true")
    .Attr("pseudo_gauss: bool = false")
    .Input("threshold_voltage: T")
    .Input("r: R")
    .Input("type_coefficients: T")
    .Input("type_indices: int64")
    .Input("dt: T")
    .Input("voltage_gradient_retention: T")
    .Input("grad_threshold: T")
    .Input("grad_v: T")
    .Input("grad_asc: T")
    .Input("grad_rise: S")
    .Input("grad_psc: S")
    .Input("previous_asc: T")
    .Input("v_th: T")
    .Input("dampening: T")
    .Input("gauss_std: T")
    .Output("v_grad: T")
    .Output("asc_grad: T")
    .Output("rise_grad: S")
    .Output("psc_grad: S")
    .Output("currents_grad: S")
    .SetShapeFn([](shape_inference::InferenceContext* context) {
      TF_RETURN_IF_ERROR(NestTypeIndexedCoefficientShape(context, 2, 3));
      shape_inference::ShapeHandle asc;
      TF_RETURN_IF_ERROR(context->Merge(context->input(8), context->input(11), &asc));
      for (int index = 12; index < 15; ++index) {
        shape_inference::ShapeHandle scalar;
        TF_RETURN_IF_ERROR(context->WithRank(context->input(index), 0, &scalar));
      }
      context->set_output(0, context->input(0));
      context->set_output(1, asc);
      context->set_output(2, context->input(9));
      context->set_output(3, context->input(10));
      context->set_output(4, context->input(9));
      return OkStatus();
    });


REGISTER_OP("DpointnetNestStateHistoryBackwardTypeIndexed")
    .Attr("T: {half, float}")
    .Attr("S: {half, float}")
    .Attr("R: {int8, int16}")
    .Attr("hard_reset: bool = false")
    .Attr("pseudo_gauss: bool = false")
    .Input("threshold_voltage: T")
    .Input("r: R")
    .Input("type_coefficients: T")
    .Input("type_indices: int64")
    .Input("dt: T")
    .Input("voltage_gradient_retention: T")
    .Input("grad_threshold: T")
    .Input("grad_v: T")
    .Input("grad_asc: T")
    .Input("grad_rise: S")
    .Input("grad_psc: S")
    .Input("dampening: T")
    .Input("gauss_std: T")
    .Input("grad_spikes: S")
    .Input("grad_history: S")
    .Output("v_grad: T")
    .Output("asc_grad: T")
    .Output("rise_grad: S")
    .Output("psc_grad: S")
    .Output("currents_grad: S")
    .Output("old_history_grad: S")
    .SetShapeFn([](shape_inference::InferenceContext* context) {
      TF_RETURN_IF_ERROR(NestTypeIndexedCoefficientShape(context, 2, 3));
      context->set_output(0, context->input(0));
      context->set_output(1, context->input(8));
      context->set_output(2, context->input(9));
      context->set_output(3, context->input(10));
      context->set_output(4, context->input(9));
      for (int index = 11; index < 13; ++index) {
        shape_inference::ShapeHandle scalar;
        TF_RETURN_IF_ERROR(context->WithRank(context->input(index), 0, &scalar));
      }
      context->set_output(5, context->input(14));
      return OkStatus();
    });

REGISTER_OP("DpointnetNestStateHistoryBackwardEventsTypeIndexed")
    .Attr("T: {half, float}")
    .Attr("S: {half, float}")
    .Attr("R: {int8, int16}")
    .Attr("hard_reset: bool = false")
    .Attr("detach_reset: bool = true")
    .Attr("detach_asc_reset: bool = true")
    .Attr("pseudo_gauss: bool = false")
    .Input("threshold_voltage: T")
    .Input("r: R")
    .Input("type_coefficients: T")
    .Input("type_indices: int64")
    .Input("dt: T")
    .Input("voltage_gradient_retention: T")
    .Input("grad_threshold: T")
    .Input("grad_v: T")
    .Input("grad_asc: T")
    .Input("grad_rise: S")
    .Input("grad_psc: S")
    .Input("previous_asc: T")
    .Input("v_th: T")
    .Input("dampening: T")
    .Input("gauss_std: T")
    .Input("grad_spikes: S")
    .Input("grad_history: S")
    .Output("v_grad: T")
    .Output("asc_grad: T")
    .Output("rise_grad: S")
    .Output("psc_grad: S")
    .Output("currents_grad: S")
    .Output("old_history_grad: S")
    .SetShapeFn([](shape_inference::InferenceContext* context) {
      TF_RETURN_IF_ERROR(NestTypeIndexedCoefficientShape(context, 2, 3));
      shape_inference::ShapeHandle asc;
      TF_RETURN_IF_ERROR(context->Merge(context->input(8), context->input(11), &asc));
      for (int index = 12; index < 15; ++index) {
        shape_inference::ShapeHandle scalar;
        TF_RETURN_IF_ERROR(context->WithRank(context->input(index), 0, &scalar));
      }
      context->set_output(0, context->input(0));
      context->set_output(1, asc);
      context->set_output(2, context->input(9));
      context->set_output(3, context->input(10));
      context->set_output(4, context->input(9));
      context->set_output(5, context->input(16));
      return OkStatus();
    });
REGISTER_OP("DpointnetNestStateHistoryBackwardEventsAddendTypeIndexed")
    .Attr("T: {half, float}")
    .Attr("S: {half, float}")
    .Attr("R: {int8, int16}")
    .Attr("hard_reset: bool = false")
    .Attr("detach_reset: bool = true")
    .Attr("detach_asc_reset: bool = true")
    .Attr("pseudo_gauss: bool = false")
    .Input("threshold_voltage: T")
    .Input("r: R")
    .Input("type_coefficients: T")
    .Input("type_indices: int64")
    .Input("dt: T")
    .Input("voltage_gradient_retention: T")
    .Input("grad_threshold: T")
    .Input("grad_v: T")
    .Input("grad_asc: T")
    .Input("grad_rise: S")
    .Input("grad_psc: S")
    .Input("previous_asc: T")
    .Input("v_th: T")
    .Input("dampening: T")
    .Input("gauss_std: T")
    .Input("grad_spikes: S")
    .Input("grad_history: S")
    .Input("grad_history_addend: S")
    .Output("v_grad: T")
    .Output("asc_grad: T")
    .Output("rise_grad: S")
    .Output("psc_grad: S")
    .Output("currents_grad: S")
    .Output("old_history_grad: S")
    .SetShapeFn([](shape_inference::InferenceContext* context) {
      TF_RETURN_IF_ERROR(NestTypeIndexedCoefficientShape(context, 2, 3));
      shape_inference::ShapeHandle asc;
      TF_RETURN_IF_ERROR(context->Merge(context->input(8), context->input(11), &asc));
      for (int index = 12; index < 15; ++index) {
        shape_inference::ShapeHandle scalar;
        TF_RETURN_IF_ERROR(context->WithRank(context->input(index), 0, &scalar));
      }
      context->set_output(0, context->input(0));
      context->set_output(1, asc);
      context->set_output(2, context->input(9));
      context->set_output(3, context->input(10));
      context->set_output(4, context->input(9));
      context->set_output(5, context->input(16));
      return OkStatus();
    });
REGISTER_OP("DpointnetNestStateHistoryBackwardEvents")
    .Attr("launch_geometry_version: int = 1")
    .Attr("T: {half, float}")
    .Attr("S: {half, float}")
    .Attr("R: {int8, int16}")
    .Attr("hard_reset: bool = false")
    .Attr("detach_reset: bool = true")
    .Attr("detach_asc_reset: bool = true")
    .Attr("pseudo_gauss: bool = false")
    .Attr("coefficients_layout: {'aos', 'soa'} = 'aos'")
    .Input("threshold_voltage: T")
    .Input("r: R")
    .Input("coefficients: T")
    .Input("dt: T")
    .Input("voltage_gradient_retention: T")
    .Input("grad_threshold: T")
    .Input("grad_v: T")
    .Input("grad_asc: T")
    .Input("grad_rise: S")
    .Input("grad_psc: S")
    .Input("previous_asc: T")
    .Input("v_th: T")
    .Input("dampening: T")
    .Input("gauss_std: T")
    .Input("grad_spikes: S")
    .Input("grad_history: S")
    .Output("v_grad: T")
    .Output("asc_grad: T")
    .Output("rise_grad: S")
    .Output("psc_grad: S")
    .Output("currents_grad: S")
    .Output("old_history_grad: S")
    .SetShapeFn([](shape_inference::InferenceContext* context) {
      TF_RETURN_IF_ERROR(NestCoefficientShape(context, 2));
      shape_inference::ShapeHandle asc;
      TF_RETURN_IF_ERROR(context->Merge(context->input(7), context->input(10), &asc));
      for (int index = 11; index < 14; ++index) {
        shape_inference::ShapeHandle scalar;
        TF_RETURN_IF_ERROR(context->WithRank(context->input(index), 0, &scalar));
      }
      context->set_output(0, context->input(0));
      context->set_output(1, asc);
      context->set_output(2, context->input(8));
      context->set_output(3, context->input(9));
      context->set_output(4, context->input(8));
      context->set_output(5, context->input(15));
      return OkStatus();
    });
REGISTER_OP("DpointnetNestStateHistoryBackwardEventsAddend")
    .Attr("launch_geometry_version: int = 1")
    .Attr("T: {half, float}")
    .Attr("S: {half, float}")
    .Attr("R: {int8, int16}")
    .Attr("hard_reset: bool = false")
    .Attr("detach_reset: bool = true")
    .Attr("detach_asc_reset: bool = true")
    .Attr("pseudo_gauss: bool = false")
    .Attr("coefficients_layout: {'aos', 'soa'} = 'aos'")
    .Input("threshold_voltage: T")
    .Input("r: R")
    .Input("coefficients: T")
    .Input("dt: T")
    .Input("voltage_gradient_retention: T")
    .Input("grad_threshold: T")
    .Input("grad_v: T")
    .Input("grad_asc: T")
    .Input("grad_rise: S")
    .Input("grad_psc: S")
    .Input("previous_asc: T")
    .Input("v_th: T")
    .Input("dampening: T")
    .Input("gauss_std: T")
    .Input("grad_spikes: S")
    .Input("grad_history: S")
    .Input("grad_history_addend: S")
    .Output("v_grad: T")
    .Output("asc_grad: T")
    .Output("rise_grad: S")
    .Output("psc_grad: S")
    .Output("currents_grad: S")
    .Output("old_history_grad: S")
    .SetShapeFn([](shape_inference::InferenceContext* context) {
      TF_RETURN_IF_ERROR(NestCoefficientShape(context, 2));
      shape_inference::ShapeHandle asc;
      TF_RETURN_IF_ERROR(context->Merge(context->input(7), context->input(10), &asc));
      for (int index = 11; index < 14; ++index) {
        shape_inference::ShapeHandle scalar;
        TF_RETURN_IF_ERROR(context->WithRank(context->input(index), 0, &scalar));
      }
      context->set_output(0, context->input(0));
      context->set_output(1, asc);
      context->set_output(2, context->input(8));
      context->set_output(3, context->input(9));
      context->set_output(4, context->input(8));
      context->set_output(5, context->input(15));
      return OkStatus();
    });

REGISTER_OP("DpointnetGlifStateForward")
    .Attr("T: {half, float}")
    .Attr("S: {half, float}")
    .Attr("R: {int8, int16}")
    .Attr("hard_reset: bool = false")
    .Input("prev_z: S")
    .Input("v: T")
    .Input("r: R")
    .Input("asc: T")
    .Input("psc_rise: S")
    .Input("psc: S")
    .Input("rec_inputs: S")
    .Input("syn_decay: T")
    .Input("psc_initial: T")
    .Input("asc_decay: T")
    .Input("asc_amps: T")
    .Input("decay: T")
    .Input("current_factor: T")
    .Input("t_ref_steps: R")
    .Input("dt: T")
    .Input("v_reset: T")
    .Output("new_v: T")
    .Output("new_r: R")
    .Output("new_asc: T")
    .Output("new_psc_rise: S")
    .Output("new_psc: S")
    .SetShapeFn([](shape_inference::InferenceContext* context) {
      context->set_output(0, context->input(1));
      context->set_output(1, context->input(2));
      context->set_output(2, context->input(3));
      context->set_output(3, context->input(4));
      context->set_output(4, context->input(5));
      return OkStatus();
    });

REGISTER_OP("DpointnetGlifStateBackward")
    .Attr("T: {half, float}")
    .Attr("S: {half, float}")
    .Attr("R: {int8, int16}")
    .Attr("hard_reset: bool = false")
    .Attr("detach_reset: bool = true")
    .Attr("detach_asc_reset: bool = true")
    .Input("prev_z: S")
    .Input("r: R")
    .Input("syn_decay: T")
    .Input("psc_initial: T")
    .Input("asc_decay: T")
    .Input("asc_amps: T")
    .Input("decay: T")
    .Input("current_factor: T")
    .Input("t_ref_steps: R")
    .Input("dt: T")
    .Input("voltage_gradient_retention: T")
    .Input("grad_v: T")
    .Input("grad_asc: T")
    .Input("grad_psc_rise: S")
    .Input("grad_psc: S")
    .Output("prev_z_grad: S")
    .Output("v_grad: T")
    .Output("asc_grad: T")
    .Output("psc_rise_grad: S")
    .Output("psc_grad: S")
    .Output("rec_inputs_grad: S")
    .SetShapeFn([](shape_inference::InferenceContext* context) {
      context->set_output(0, context->input(0));
      context->set_output(1, context->input(11));
      context->set_output(2, context->input(12));
      context->set_output(3, context->input(13));
      context->set_output(4, context->input(14));
      context->set_output(5, context->input(13));
      return OkStatus();
    });

REGISTER_OP("DpointnetSpikeShift")
    .Attr("T: {half, float}")
    .Attr("S: {half, float}")
    .Input("voltage: T")
    .Input("refractory: bool")
    .Input("history: S")
    .Output("spikes: S")
    .Output("new_history: S")
    .SetShapeFn([](shape_inference::InferenceContext* context) {
      shape_inference::ShapeHandle voltage;
      shape_inference::ShapeHandle refractory;
      shape_inference::ShapeHandle history;
      TF_RETURN_IF_ERROR(context->WithRank(context->input(0), 2, &voltage));
      TF_RETURN_IF_ERROR(context->WithRank(context->input(1), 2, &refractory));
      TF_RETURN_IF_ERROR(context->WithRank(context->input(2), 2, &history));
      TF_RETURN_IF_ERROR(context->Merge(voltage, refractory, &voltage));
      context->set_output(0, voltage);
      context->set_output(1, history);
      return OkStatus();
    });

REGISTER_OP("DpointnetSpikeShiftBackwardV2")
    .Attr("T: {half, float}")
    .Attr("S: {half, float}")
    .Attr("pseudo_gauss: bool = false")
    .Input("voltage: T")
    .Input("refractory: bool")
    .Input("spike_grad: S")
    .Input("history_grad: S")
    .Input("dampening: T")
    .Input("gauss_std: T")
    .Output("voltage_grad: T")
    .Output("old_history_grad: S")
    .SetShapeFn([](shape_inference::InferenceContext* context) {
      context->set_output(0, context->input(0));
      context->set_output(1, context->input(3));
      return OkStatus();
    });

REGISTER_OP("DpointnetVoltagePenaltyForward")
    .Attr("T: {half, float}")
    .Attr("mode: {'range', 'threshold'} = 'range'")
    .Input("voltage: T")
    .Output("penalty: float")
    .SetShapeFn([](shape_inference::InferenceContext* context) {
      shape_inference::ShapeHandle voltage;
      TF_RETURN_IF_ERROR(context->WithRank(context->input(0), 2, &voltage));
      context->set_output(0, context->Vector(context->Dim(voltage, 0)));
      return OkStatus();
    });

REGISTER_OP("DpointnetVoltagePenaltyBackward")
    .Attr("T: {half, float}")
    .Attr("mode: {'range', 'threshold'} = 'range'")
    .Input("voltage: T")
    .Input("grad_penalty: float")
    .Output("voltage_grad: T")
    .SetShapeFn([](shape_inference::InferenceContext* context) {
      shape_inference::ShapeHandle voltage;
      TF_RETURN_IF_ERROR(context->WithRank(context->input(0), 2, &voltage));
      shape_inference::ShapeHandle grad;
      TF_RETURN_IF_ERROR(context->WithRank(context->input(1), 1, &grad));
      shape_inference::DimensionHandle dimension;
      TF_RETURN_IF_ERROR(context->Merge(context->Dim(voltage, 0),
                                        context->Dim(grad, 0), &dimension));
      context->set_output(0, voltage);
      return OkStatus();
    });

// Bit-packed per-step records for the direct recorded adjoint: the step's
// spikes and its pre-step refractory (blocked) mask, 32 neurons per int32 word.
REGISTER_OP("DpointnetPackStepRecord")
    .Attr("S: {half, float}")
    .Attr("R: {int8, int16}")
    .Input("spikes: S")
    .Input("refractory: R")
    .Output("words: int32")
    .SetShapeFn([](shape_inference::InferenceContext* context) {
      shape_inference::ShapeHandle spikes, refractory;
      TF_RETURN_IF_ERROR(context->WithRank(context->input(0), 2, &spikes));
      TF_RETURN_IF_ERROR(context->WithRank(context->input(1), 2, &refractory));
      shape_inference::ShapeHandle merged;
      TF_RETURN_IF_ERROR(context->Merge(spikes, refractory, &merged));
      shape_inference::DimensionHandle words;
      TF_RETURN_IF_ERROR(context->Add(context->Dim(merged, 1), 31, &words));
      TF_RETURN_IF_ERROR(context->Divide(words, 32, false, &words));
      TF_RETURN_IF_ERROR(context->Multiply(words, 2, &words));
      context->set_output(0, context->Matrix(context->Dim(merged, 0), words));
      return OkStatus();
    });

REGISTER_OP("DpointnetUnpackStepRecord")
    .Attr("S: {half, float}")
    .Attr("R: {int8, int16}")
    .Attr("neurons: int >= 1")
    .Input("words: int32")
    .Output("spikes: S")
    .Output("refractory: R")
    .SetShapeFn([](shape_inference::InferenceContext* context) {
      int neurons;
      TF_RETURN_IF_ERROR(context->GetAttr("neurons", &neurons));
      shape_inference::ShapeHandle words;
      TF_RETURN_IF_ERROR(context->WithRank(context->input(0), 2, &words));
      shape_inference::DimensionHandle expected;
      TF_RETURN_IF_ERROR(context->WithValue(
          context->Dim(words, 1), 2 * ((neurons + 31) / 32), &expected));
      const auto shape = context->Matrix(context->Dim(words, 0), neurons);
      context->set_output(0, shape);
      context->set_output(1, shape);
      return OkStatus();
    });

// Direct recorded adjoint: exact ASC after one recorded step, from the pre-step
// ASC, the step's packed record and the NEST coefficient table (AoS [N, 28]).
REGISTER_OP("DpointnetAdvanceAscRecord")
    .Attr("T: {half, float}")
    .Input("asc: T")
    .Input("words: int32")
    .Input("coefficients: T")
    .Output("new_asc: T")
    .SetShapeFn([](shape_inference::InferenceContext* context) {
      shape_inference::ShapeHandle asc;
      TF_RETURN_IF_ERROR(context->WithRank(context->input(0), 2, &asc));
      context->set_output(0, asc);
      return OkStatus();
    });

// Direct recorded adjoint reverse prelude: unpack the step record, form the
// threshold voltage, add the voltage-penalty VJP of the post-reset membrane to
// the carried voltage cotangent and emit the recorded spikes.
REGISTER_OP("DpointnetDirectReverseRecord")
    .Attr("T: {half, float}")
    .Attr("S: {half, float}")
    .Attr("R: {int8, int16}")
    .Attr("mode: {'range', 'threshold'} = 'range'")
    .Input("pre_reset: T")
    .Input("row: int32")
    .Input("words: int32")
    .Input("grad_v: T")
    .Input("grad_penalty: float")
    .Input("v_th: T")
    .Input("v_reset: T")
    .Input("spike_base: S")
    .Input("trace_column: float")
    .Input("trace_indices: int64")
    .Input("trace_mask: float")
    .Input("trace_groups: int64")
    .Output("threshold: T")
    .Output("blocked: R")
    .Output("total_grad_v: T")
    .Output("spikes: S")
    .Output("spike_cotangent: S")
    .SetShapeFn([](shape_inference::InferenceContext* context) {
      // pre_reset is either this step's [batch, neurons] record or the whole
      // [time, batch, neurons] record buffer read at ``row``.
      shape_inference::ShapeHandle voltage;
      TF_RETURN_IF_ERROR(context->WithRank(context->input(3), 2, &voltage));
      for (int output = 0; output < 5; ++output) context->set_output(output, voltage);
      return OkStatus();
    });

namespace {
// Shape function of the record-writing NEST history forwards: the usual state
// outputs, spikes, new history and the packed ``[batch, 2 * ceil(neurons/32)]``
// step record; the pre-reset membrane goes to the resource row instead.
Status NestRecordForwardShape(shape_inference::InferenceContext* context, int history_index) {
  shape_inference::ShapeHandle voltage, history;
  TF_RETURN_IF_ERROR(context->WithRank(context->input(0), 2, &voltage));
  TF_RETURN_IF_ERROR(context->WithRank(context->input(history_index), 2, &history));
  shape_inference::DimensionHandle batch;
  TF_RETURN_IF_ERROR(context->Merge(context->Dim(voltage, 0), context->Dim(history, 0), &batch));
  context->set_output(0, voltage);
  for (int output = 1; output < 5; ++output) {
    context->set_output(output, context->input(output));
  }
  context->set_output(5, voltage);
  context->set_output(6, history);
  shape_inference::DimensionHandle words = context->UnknownDim();
  if (context->ValueKnown(context->Dim(voltage, 1))) {
    const int64_t neurons = context->Value(context->Dim(voltage, 1));
    words = context->MakeDim(2 * ((neurons + 31) / 32));
  }
  context->set_output(7, context->MakeShape({batch, words}));
  return OkStatus();
}
}  // namespace

// NEST history forwards that also emit the packed step record and write the
// pre-reset membrane into row ``row`` of the ``[time, batch, neurons]`` record
// scratch variable ``buffer`` (direct recorded adjoint forward).
REGISTER_OP("DpointnetNestStateHistoryForwardRecord")
    .Attr("T: {half, float}")
    .Attr("S: {half, float}")
    .Attr("R: {int8, int16}")
    .Attr("hard_reset: bool = false")
    .Attr("coefficients_layout: {'aos', 'soa'} = 'aos'")
    .Input("v: T")
    .Input("r: R")
    .Input("asc: T")
    .Input("psc_rise: S")
    .Input("psc: S")
    .Input("currents: S")
    .Input("coefficients: T")
    .Input("t_ref: R")
    .Input("dt: T")
    .Input("v_th: T")
    .Input("history: S")
    .Input("buffer: resource")
    .Input("row: int32")
    .Output("new_v: T")
    .Output("new_r: R")
    .Output("new_asc: T")
    .Output("new_rise: S")
    .Output("new_psc: S")
    .Output("spikes: S")
    .Output("new_history: S")
    .Output("words: int32")
    .SetIsStateful()
    .SetShapeFn([](shape_inference::InferenceContext* context) {
      TF_RETURN_IF_ERROR(NestCoefficientShape(context, 6));
      return NestRecordForwardShape(context, 10);
    });
REGISTER_OP("DpointnetNestStateHistoryForwardRecordTypeIndexed")
    .Attr("T: {half, float}") .Attr("S: {half, float}")
    .Attr("R: {int8, int16}") .Attr("hard_reset: bool = false")
    .Input("v: T") .Input("r: R") .Input("asc: T")
    .Input("psc_rise: S") .Input("psc: S") .Input("currents: S")
    .Input("type_coefficients: T") .Input("type_indices: int64")
    .Input("t_ref: R") .Input("dt: T") .Input("v_th: T") .Input("history: S")
    .Input("buffer: resource") .Input("row: int32")
    .Output("new_v: T") .Output("new_r: R") .Output("new_asc: T")
    .Output("new_rise: S") .Output("new_psc: S")
    .Output("spikes: S") .Output("new_history: S") .Output("words: int32")
    .SetIsStateful()
    .SetShapeFn([](shape_inference::InferenceContext* context) {
      TF_RETURN_IF_ERROR(NestTypeIndexedCoefficientShape(context, 6, 7));
      return NestRecordForwardShape(context, 11);
    });
REGISTER_OP("DpointnetNestStateHistoryForwardRecordDeviceDispatch")
    .Attr("T: {half, float}") .Attr("S: {half, float}")
    .Attr("R: {int8, int16}") .Attr("hard_reset: bool = false")
    .Input("v: T") .Input("r: R") .Input("asc: T")
    .Input("psc_rise: S") .Input("psc: S") .Input("currents: S")
    .Input("type_coefficients: T") .Input("type_indices: int64")
    .Input("t_ref: R") .Input("dt: T") .Input("v_th: T") .Input("history: S")
    .Input("buffer: resource") .Input("row: int32")
    .Input("fallback_coefficients: T") .Input("type_identity: bool")
    .Output("new_v: T") .Output("new_r: R") .Output("new_asc: T")
    .Output("new_rise: S") .Output("new_psc: S")
    .Output("spikes: S") .Output("new_history: S") .Output("words: int32")
    .SetIsStateful()
    .SetShapeFn([](shape_inference::InferenceContext* context) {
      TF_RETURN_IF_ERROR(NestTypeIndexedCoefficientShape(context, 6, 7));
      return NestRecordForwardShape(context, 11);
    });

// Same prelude reading row ``row`` of a [time, batch, neurons] record scratch
// variable directly (no full-variable read, no row copy).
REGISTER_OP("DpointnetDirectReverseRecordResource")
    .Attr("T: {half, float}")
    .Attr("S: {half, float}")
    .Attr("R: {int8, int16}")
    .Attr("mode: {'range', 'threshold'} = 'range'")
    .Input("buffer: resource")
    .Input("row: int32")
    .Input("words: int32")
    .Input("grad_v: T")
    .Input("grad_penalty: float")
    .Input("v_th: T")
    .Input("v_reset: T")
    .Input("spike_base: S")
    .Input("trace_column: float")
    .Input("trace_indices: int64")
    .Input("trace_mask: float")
    .Input("trace_groups: int64")
    .Output("threshold: T")
    .Output("blocked: R")
    .Output("total_grad_v: T")
    .Output("spikes: S")
    .Output("spike_cotangent: S")
    .SetShapeFn([](shape_inference::InferenceContext* context) {
      shape_inference::ShapeHandle voltage;
      TF_RETURN_IF_ERROR(context->WithRank(context->input(3), 2, &voltage));
      for (int output = 0; output < 5; ++output) context->set_output(output, voltage);
      return OkStatus();
    });

// Reverse delay window shift: drops the newest slot of the ``[batch,
// neurons * delay]`` history, moves the others up and unpacks the recorded
// spikes ``words`` into the oldest slot, in one pass.
REGISTER_OP("DpointnetShiftHistoryRecord")
    .Attr("S: {half, float}")
    .Attr("neurons: int >= 1")
    .Input("history: S")
    .Input("words: int32")
    .Output("shifted: S")
    .SetShapeFn([](shape_inference::InferenceContext* context) {
      shape_inference::ShapeHandle history;
      TF_RETURN_IF_ERROR(context->WithRank(context->input(0), 2, &history));
      context->set_output(0, history);
      return OkStatus();
    });

// Post-hoc statistics of the packed step records ``[time, batch, 2 * words]``:
// per-neuron spike counts over ``[begin, end)`` and masked sampled-population
// traces (``indices`` are flat [k, 2] (sample, neuron) entries, ``groups`` the
// output row of each entry).
REGISTER_OP("DpointnetRecordCounts")
    .Attr("neurons: int >= 1")
    .Attr("begin: int >= 0")
    .Attr("end: int >= 0")
    .Input("words: int32")
    .Output("counts: float")
    .SetShapeFn([](shape_inference::InferenceContext* context) {
      int neurons;
      TF_RETURN_IF_ERROR(context->GetAttr("neurons", &neurons));
      shape_inference::ShapeHandle words;
      TF_RETURN_IF_ERROR(context->WithRank(context->input(0), 3, &words));
      context->set_output(0, context->Matrix(context->Dim(words, 1), neurons));
      return OkStatus();
    });

REGISTER_OP("DpointnetRecordTraces")
    .Attr("neurons: int >= 1")
    .Attr("rows: int >= 1")
    .Attr("begin: int >= 0")
    .Attr("end: int >= 0")
    .Input("words: int32")
    .Input("indices: int64")
    .Input("mask: float")
    .Input("groups: int64")
    .Output("traces: float")
    .SetShapeFn([](shape_inference::InferenceContext* context) {
      int rows, begin, end;
      TF_RETURN_IF_ERROR(context->GetAttr("rows", &rows));
      TF_RETURN_IF_ERROR(context->GetAttr("begin", &begin));
      TF_RETURN_IF_ERROR(context->GetAttr("end", &end));
      context->set_output(0, context->Matrix(rows, end - begin));
      return OkStatus();
    });
