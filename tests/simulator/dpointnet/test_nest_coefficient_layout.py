"""Raw NEST layout contract; optional registration-only library permits CPU tracing."""

import os

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")


@pytest.fixture
def layout_ops():
    from bmtk.simulator.dpointnet.custom_ops import glif_state_ops

    library = os.environ.get("DPOINTNET_NEST_LAYOUT_SHAPE_LIBRARY")
    ops = tf.load_op_library(library) if library else glif_state_ops._OPS
    if ops is None:
        pytest.skip("Rebuilt NEST operators or registration-only shape library required")
    return ops


def _shapes(backward, neurons, layout):
    coefficients = (28, neurons) if layout == "soa" else (neurons, 28)
    if backward:
        return [(2, neurons), (2, neurons), coefficients, (), (),
                (2, neurons), (2, neurons), (2, neurons * 2),
                (2, neurons * 4), (2, neurons * 4)]
    return [(2, neurons), (2, neurons), (2, neurons * 2),
            (2, neurons * 4), (2, neurons * 4), (2, neurons * 4),
            coefficients, (neurons,), (), ()]


def _specs(backward, shapes, dtype=tf.float32):
    return [
        tf.TensorSpec(shape, tf.int16 if index == 1 or
                      (not backward and index == 7) else dtype)
        for index, shape in enumerate(shapes)
    ]


def _operator(ops, backward):
    return (ops.dpointnet_nest_state_backward if backward
            else ops.dpointnet_nest_state_forward)


@pytest.mark.parametrize("backward", [False, True])
@pytest.mark.parametrize("layout", [None, "aos", "soa"])
def test_nest_coefficient_layout_shape_contract(layout_ops, backward, layout):
    kwargs = {} if layout is None else {"coefficients_layout": layout}
    shapes = _shapes(backward, 7, layout)
    concrete = tf.function(
        lambda *args: _operator(layout_ops, backward)(*args, **kwargs)
    ).get_concrete_function(*_specs(backward, shapes))
    expected = [(2, 7), (2, 14), (2, 28), (2, 28), (2, 28)] if backward else [
        (2, 7), (2, 7), (2, 7), (2, 14), (2, 28), (2, 28)]
    assert [tuple(value.shape) for value in concrete.outputs] == expected
    op_type = "DpointnetNestStateBackward" if backward else "DpointnetNestStateForward"
    operation, = [op for op in concrete.graph.get_operations() if op.type == op_type]
    assert operation.get_attr("coefficients_layout") == (layout or "aos").encode()


@pytest.mark.parametrize("backward", [False, True])
@pytest.mark.parametrize("layout", ["aos", "soa"])
@pytest.mark.parametrize("bad_shape", [(28,), (7, 27), (27, 7), (8, 28), (28, 8)])
def test_nest_coefficient_layout_shape_rejects_mismatch(
    layout_ops, backward, layout, bad_shape
):
    shapes = _shapes(backward, 7, layout)
    shapes[2 if backward else 6] = bad_shape
    with pytest.raises((ValueError, tf.errors.InvalidArgumentError)):
        tf.function(
            lambda *args: _operator(layout_ops, backward)(
                *args, coefficients_layout=layout)
        ).get_concrete_function(*_specs(backward, shapes))


@pytest.mark.parametrize("backward", [False, True])
@pytest.mark.parametrize("layout", ["aos", "soa"])
def test_nest_coefficient_layout_shape_partial(layout_ops, backward, layout):
    shapes = _shapes(backward, 7, layout)
    shapes[0] = (None, None)
    shapes[2 if backward else 6] = (28, None) if layout == "soa" else (None, 28)
    tf.function(
        lambda *args: _operator(layout_ops, backward)(
            *args, coefficients_layout=layout)
    ).get_concrete_function(*_specs(backward, shapes))


@pytest.mark.parametrize("backward", [False, True])
def test_nest_coefficient_layout_shape_rejects_fp16_soa(layout_ops, backward):
    with pytest.raises((ValueError, tf.errors.InvalidArgumentError), match="FP32"):
        tf.function(
            lambda *args: _operator(layout_ops, backward)(
                *args, coefficients_layout="soa")
        ).get_concrete_function(*_specs(backward, _shapes(backward, 7, "soa"), tf.float16))
    tf.function(
        lambda *args: _operator(layout_ops, backward)(*args)
    ).get_concrete_function(*_specs(backward, _shapes(backward, 7, "aos"), tf.float16))


@pytest.mark.parametrize("backward", [False, True])
def test_nest_coefficient_layout_shape_rejects_unknown_layout(layout_ops, backward):
    with pytest.raises((ValueError, tf.errors.InvalidArgumentError)):
        tf.function(
            lambda *args: _operator(layout_ops, backward)(
                *args, coefficients_layout="unknown")
        ).get_concrete_function(*_specs(backward, _shapes(backward, 7, "aos")))


@pytest.mark.parametrize("psc_dtype", [tf.float16, tf.float32])
@pytest.mark.parametrize("refractory_dtype", [tf.int8, tf.int16])
@pytest.mark.parametrize("hard_reset", [False, True])
@pytest.mark.parametrize("batch,neurons", [(3, 5), (2, 33), (32, 67)])
def test_nest_coefficient_layout_raw_forward_backward_bitwise(
    psc_dtype, refractory_dtype, hard_reset, batch, neurons
):
    from bmtk.simulator.dpointnet.custom_ops import glif_state_ops

    if not glif_state_ops.fused_nest_state_available():
        pytest.skip("Rebuilt NEST CUDA operators and a compatible GPU required")
    ops = glif_state_ops._OPS
    rng = np.random.default_rng(813)

    def constant(shape, low=-0.3, high=0.3, dtype=tf.float32):
        return tf.constant(rng.uniform(low, high, shape), dtype)

    with tf.device("/GPU:0"):
        coefficients = rng.uniform(0.01, 0.1, (neurons, 28)).astype(np.float32)
        coefficients[:, :4] = rng.uniform(0.7, 0.95, (neurons, 4))
        coefficients[:, 8:10] = 0.9
        coefficients[:, 10:12] *= -1
        coefficients[:, 12] = 0.95
        coefficients[:, 26] = -0.1
        coefficients[:, 27] = 0.25
        aos = tf.constant(coefficients)
        soa = tf.transpose(aos)
        long_ref = 4095 if refractory_dtype == tf.int16 else 4
        refractory = tf.constant(
            np.tile(np.resize([0, long_ref, 0, 2, 0], neurons), (batch, 1)),
            refractory_dtype,
        )
        voltage = tf.constant(
            np.tile(np.resize([0.4, 1.4, 1.4, 0.8, 1.0], neurons), (batch, 1)),
            tf.float32,
        )
        arguments = [
            voltage, refractory, constant((batch, neurons * 2)),
            constant((batch, neurons * 4), dtype=psc_dtype),
            constant((batch, neurons * 4), dtype=psc_dtype),
            constant((batch, neurons * 4), dtype=psc_dtype), aos,
            tf.fill((neurons,), tf.cast(long_ref, refractory_dtype)),
            tf.constant(0.5), tf.constant(1.0),
        ]
        expected = ops.dpointnet_nest_state_forward(*arguments, hard_reset=hard_reset)
        actual = ops.dpointnet_nest_state_forward(
            *arguments[:6], soa, *arguments[7:],
            hard_reset=hard_reset, coefficients_layout="soa")
        for left, right in zip(expected, actual):
            assert left.dtype == right.dtype
            assert left.numpy().tobytes() == right.numpy().tobytes()
        assert np.any((expected[0].numpy() > 0) & (refractory.numpy() == 0))

        # Qualify both state-dtype VJPs and FP32 derivative credit for FP16 primals.
        for derivative_dtype in dict.fromkeys([psc_dtype, tf.float32]):
            gradients = [
                constant((batch, neurons)), constant((batch, neurons)),
                constant((batch, neurons * 2)),
                constant((batch, neurons * 4), dtype=derivative_dtype),
                constant((batch, neurons * 4), dtype=derivative_dtype),
            ]
            backward = [
                expected[0], refractory, aos, arguments[8], tf.constant(0.75),
                *gradients,
            ]
            expected_gradients = ops.dpointnet_nest_state_backward(
                *backward, hard_reset=hard_reset)
            actual_gradients = ops.dpointnet_nest_state_backward(
                *backward[:2], soa, *backward[3:],
                hard_reset=hard_reset, coefficients_layout="soa")
            for left, right in zip(expected_gradients, actual_gradients):
                assert left.dtype == right.dtype
                assert left.numpy().tobytes() == right.numpy().tobytes()


@pytest.mark.parametrize("backward", [False, True])
@pytest.mark.parametrize("layout", ["aos", "soa"])
def test_nest_coefficient_layout_runtime_shape_rejects_transpose(backward, layout):
    from bmtk.simulator.dpointnet.custom_ops import glif_state_ops

    if not glif_state_ops.fused_nest_state_available():
        pytest.skip("Rebuilt NEST CUDA operators and a compatible GPU required")
    shapes = _shapes(backward, 7, layout)
    specs = _specs(backward, shapes)
    coefficient_index = 2 if backward else 6
    specs[coefficient_index] = tf.TensorSpec((None, None), tf.float32)
    concrete = tf.function(
        lambda *args: _operator(glif_state_ops._OPS, backward)(
            *args, coefficients_layout=layout)
    ).get_concrete_function(*specs)
    shapes[coefficient_index] = tuple(reversed(shapes[coefficient_index]))
    with tf.device("/GPU:0"), pytest.raises(tf.errors.InvalidArgumentError):
        concrete(*[tf.zeros(shape, spec.dtype) for shape, spec in zip(shapes, specs)])
