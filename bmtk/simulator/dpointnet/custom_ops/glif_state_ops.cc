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
