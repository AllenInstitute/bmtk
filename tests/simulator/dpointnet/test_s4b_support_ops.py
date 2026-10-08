import pytest

tf = pytest.importorskip("tensorflow")


def test_unity_lr_fastpath_option_is_retired():
    from bmtk.simulator.dpointnet.cell_models.glif3_cell import GLIF3Cell

    with pytest.raises(TypeError, match="use_unity_lr_scale_fastpath"):
        GLIF3Cell(None, None, use_unity_lr_scale_fastpath=False)


def test_voltage_penalty_op_shape_inference():
    from bmtk.simulator.dpointnet.custom_ops import glif_state_ops

    if glif_state_ops._OPS is None or not hasattr(
        glif_state_ops._OPS, "dpointnet_voltage_penalty_forward"
    ):
        pytest.skip("rebuilt voltage-penalty registration unavailable")
    function = tf.function(
        lambda voltage, grad: (
            glif_state_ops._OPS.dpointnet_voltage_penalty_forward(
                voltage, mode="range"
            ),
            glif_state_ops._OPS.dpointnet_voltage_penalty_backward(
                voltage, grad, mode="range"
            ),
        )
    )
    concrete = function.get_concrete_function(
        tf.TensorSpec((3, 7), tf.float32), tf.TensorSpec((3,), tf.float32)
    )
    assert [tuple(output.shape) for output in concrete.outputs] == [(3,), (3, 7)]


def test_native_voltage_penalty_option_requires_boolean():
    from bmtk.simulator.dpointnet.cell_models.glif3_cell import GLIF3Cell
    from test_nest_dynamics import make_network_inputs

    network, inputs = make_network_inputs()
    with pytest.raises(ValueError, match="use_native_voltage_penalty must be"):
        GLIF3Cell(
            network,
            inputs,
            tau_basis=[2.0],
            dynamics_mode="nest",
            use_native_voltage_penalty="true",
        )


def test_native_voltage_penalty_option_requires_rebuilt_operator(monkeypatch):
    from bmtk.simulator.dpointnet.cell_models import glif3_cell
    from test_nest_dynamics import make_network_inputs

    monkeypatch.setattr(glif3_cell, "fused_voltage_penalty_available", lambda: False)
    network, inputs = make_network_inputs()
    with pytest.raises(ValueError, match="use_native_voltage_penalty=True requires"):
        glif3_cell.GLIF3Cell(
            network,
            inputs,
            tau_basis=[2.0],
            dynamics_mode="nest",
            use_native_voltage_penalty=True,
        )
