#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;

REGISTER_OP("DpointnetGlifStateForward")
    .Attr("T: {half, float}")
    .Attr("R: {int8, int16}")
    .Attr("hard_reset: bool = false")
    .Input("prev_z: T")
    .Input("v: T")
    .Input("r: R")
    .Input("asc: T")
    .Input("psc_rise: T")
    .Input("psc: T")
    .Input("rec_inputs: T")
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
    .Output("new_psc_rise: T")
    .Output("new_psc: T")
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
    .Attr("R: {int8, int16}")
    .Attr("hard_reset: bool = false")
    .Input("prev_z: T")
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
    .Input("grad_psc_rise: T")
    .Input("grad_psc: T")
    .Output("prev_z_grad: T")
    .Output("v_grad: T")
    .Output("asc_grad: T")
    .Output("psc_rise_grad: T")
    .Output("psc_grad: T")
    .Output("rec_inputs_grad: T")
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
    .Input("voltage: T")
    .Input("refractory: bool")
    .Input("history: T")
    .Output("spikes: T")
    .Output("new_history: T")
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

REGISTER_OP("DpointnetSpikeShiftBackward")
    .Attr("T: {half, float}")
    .Input("voltage: T")
    .Input("refractory: bool")
    .Input("spike_grad: T")
    .Input("history_grad: T")
    .Input("dampening: T")
    .Output("voltage_grad: T")
    .Output("old_history_grad: T")
    .SetShapeFn([](shape_inference::InferenceContext* context) {
      context->set_output(0, context->input(0));
      context->set_output(1, context->input(3));
      return OkStatus();
    });
