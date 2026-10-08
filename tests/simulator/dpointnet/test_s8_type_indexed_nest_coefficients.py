import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")


def _small_network_inputs():
    network = {
        "n_nodes": 3,
        "node_type_ids": np.array([0, 1, 0]),
        "node_params": {
            "C_m": np.array([100.0, 80.0]),
            "g": np.array([10.0, 8.0]),
            "E_L": np.array([-70.0, -65.0]),
            "V_m": np.array([-69.0, -64.0]),
            "V_th": np.array([-50.0, -45.0]),
            "V_reset": np.array([-68.0, -64.0]),
            "t_ref": np.array([2.0, 1.65]),
            "k": np.array([[0.01, 0.1], [0.02, 0.15]]),
            "asc_amps": np.array([[-10.0, -50.0], [-20.0, -40.0]]),
            "asc_init": np.array([[-1.0, -2.0], [-2.0, -1.0]]),
        },
        "synapses": {
            "indices": np.array([[0, 0], [0, 1], [1, 0], [2, 1], [2, 2]]),
            "weights": np.array([80.0, -100.0, 40.0, -60.0, 50.0]),
            "delays": np.array([1.0, 2.0, 3.0, 1.0, 2.0]),
            "dense_shape": (3, 3),
            "syn_ids": np.array([0, 1, 0, 1, 0]),
            "dynamics_params": {
                "basis_weights": [[1.0, 0.3, 0.1, 0.05], [0.1, 1.0, 0.2, 0.1]]
            },
        },
    }
    inputs = {
        "drive": {
            "n_inputs": 4,
            "indices": np.array(
                [[0, 0], [0, 1], [1, 1], [2, 0], [2, 2]], dtype=np.uint32
            ),
            "weights": np.array([1200.0, 800.0, 1000.0, 900.0, 700.0]),
            "delays": np.array([1.0, 3.0, 2.0, 1.0, 3.0]),
            "syn_ids": np.array([0, 1, 0, 1, 0]),
            "input_type": "spikes",
            "options": {"trainable": True},
        }
    }
    return network, inputs


def test_type_indexed_pack_identity_and_fallback_detection():
    from bmtk.simulator.dpointnet.custom_ops.glif_state_ops import (
        pack_type_indexed_nest_state_coefficients,
    )

    coefficients = tf.constant(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ],
        tf.float32,
    )
    table, indices, identity = pack_type_indexed_nest_state_coefficients(
        coefficients, [0, 1, 0, 1], [0, 1]
    )
    np.testing.assert_array_equal(table.numpy(), coefficients.numpy()[:2])
    np.testing.assert_array_equal(indices.numpy(), [0, 1, 0, 1])
    assert bool(identity.numpy())

    changed = tf.tensor_scatter_nd_update(coefficients, [[2, 1]], [7.0])
    _, _, identity = pack_type_indexed_nest_state_coefficients(
        changed, [0, 1, 0, 1], [0, 1]
    )
    assert not bool(identity.numpy())


def test_cell_prepare_rechecks_live_assignable_coefficients(monkeypatch):
    from bmtk.simulator.dpointnet.cell_models import glif3_cell

    monkeypatch.setattr(glif3_cell, "_resolve_fused_state", lambda *a, **k: True)
    monkeypatch.setattr(
        glif3_cell, "fused_nest_type_indexed_coefficients_available", lambda: True
    )
    network, inputs = _small_network_inputs()
    cell = glif3_cell.GLIF3Cell(
        network,
        inputs,
        dynamics_mode="nest",
        tau_basis=[2.0, 4.0, 8.0, 16.0],
        hard_reset=False,
        use_fused_state=True,
        use_type_indexed_nest_coefficients=True,
    )

    packed = cell.prepare_rollout_nest_coefficients()
    assert len(packed) == 6
    assert bool(packed[4].numpy())
    np.testing.assert_array_equal(packed[3].numpy(), [0, 1, 0])
    assert packed[2].shape == (2, 28)

    decay = cell.decay.numpy()
    decay[2] = np.nextafter(decay[2], np.float32(2.0))
    cell.decay.assign(decay)
    assert not bool(cell.prepare_rollout_nest_coefficients()[4].numpy())

    decay[0] = decay[2]
    cell.decay.assign(decay)
    assert bool(cell.prepare_rollout_nest_coefficients()[4].numpy())


def _type_indexed_specs(backward, events=False):
    if backward:
        shapes = [
            (2, 5), (2, 5), (3, 28), (5,), (), (),
            (2, 5), (2, 5), (2, 10), (2, 20), (2, 20),
        ]
        if events:
            shapes += [(2, 10), (), (), ()]
    else:
        shapes = [
            (2, 5), (2, 5), (2, 10), (2, 20), (2, 20), (2, 20),
            (3, 28), (5,), (5,), (), (),
        ]
    specs = []
    for index, shape in enumerate(shapes):
        if index == 1 or (not backward and index == 8):
            dtype = tf.int16
        elif index == (3 if backward else 7):
            dtype = tf.int64
        else:
            dtype = tf.float32
        specs.append(tf.TensorSpec(shape, dtype))
    return specs


def test_type_indexed_raw_op_shape_contract():
    from bmtk.simulator.dpointnet.custom_ops import glif_state_ops

    if glif_state_ops._OPS is None:
        pytest.skip("rebuilt NEST operators are required for shape inference")
    ops = glif_state_ops._OPS
    cases = [
        (ops.dpointnet_nest_state_forward_type_indexed, False, False),
        (ops.dpointnet_nest_state_backward_type_indexed, True, False),
        (ops.dpointnet_nest_state_backward_events_type_indexed, True, True),
    ]
    for operator, backward, events in cases:
        concrete = tf.function(lambda *args: operator(*args)).get_concrete_function(
            *_type_indexed_specs(backward, events)
        )
        expected = (
            [(2, 5), (2, 10), (2, 20), (2, 20), (2, 20)]
            if backward
            else [(2, 5), (2, 5), (2, 5), (2, 10), (2, 20), (2, 20)]
        )
        assert [tuple(value.shape) for value in concrete.outputs] == expected


@pytest.mark.parametrize("dtype,psc_dtype", [(tf.float32, tf.float32), (tf.float32, tf.float16)])
def test_type_indexed_raw_op_bitwise_when_gpu_available(dtype, psc_dtype):
    from bmtk.simulator.dpointnet.custom_ops import glif_state_ops

    if not glif_state_ops.fused_nest_type_indexed_coefficients_available():
        pytest.skip("compatible rebuilt GPU operators unavailable")
    ops = glif_state_ops._OPS
    rng = np.random.default_rng(820)
    batch, neurons = 3, 6
    type_indices = tf.constant([0, 1, 0, 2, 1, 2], tf.int64)
    type_coefficients = rng.uniform(0.01, 0.1, (3, 28)).astype(np.float32)
    type_coefficients[:, :4] = rng.uniform(0.7, 0.95, (3, 4))
    type_coefficients[:, 8:10] = 0.9
    type_coefficients[:, 10:12] *= -1
    type_coefficients[:, 12] = 0.95
    type_coefficients[:, 26] = -0.1
    type_coefficients[:, 27] = 0.25
    full_coefficients = tf.gather(tf.constant(type_coefficients, dtype), type_indices)

    def constant(shape, target=dtype):
        return tf.constant(rng.uniform(-0.2, 0.3, shape), target)

    with tf.device("/GPU:0"):
        refractory = tf.constant(np.tile([0, 1, 0, 2, 0, 1], (batch, 1)), tf.int16)
        arguments = [
            constant((batch, neurons)), refractory, constant((batch, neurons * 2)),
            constant((batch, neurons * 4), psc_dtype),
            constant((batch, neurons * 4), psc_dtype),
            constant((batch, neurons * 4), psc_dtype),
            full_coefficients,
            tf.constant([2, 3, 2, 4, 3, 4], tf.int16),
            tf.constant(0.5, dtype),
            tf.constant(1.0, dtype),
        ]
        expected = ops.dpointnet_nest_state_forward(*arguments, hard_reset=False)
        actual = ops.dpointnet_nest_state_forward_type_indexed(
            *arguments[:6], tf.constant(type_coefficients, dtype), type_indices,
            *arguments[7:], hard_reset=False)
        for left, right in zip(expected, actual):
            assert left.numpy().tobytes() == right.numpy().tobytes()

        backward_inputs = [
            expected[0],
            refractory,
            full_coefficients,
            arguments[8],
            tf.constant(0.75, dtype),
            constant((batch, neurons)),
            constant((batch, neurons)),
            constant((batch, neurons * 2)),
            constant((batch, neurons * 4), psc_dtype),
            constant((batch, neurons * 4), psc_dtype),
        ]
        expected_gradients = ops.dpointnet_nest_state_backward(*backward_inputs)
        actual_gradients = ops.dpointnet_nest_state_backward_type_indexed(
            *backward_inputs[:2],
            tf.constant(type_coefficients, dtype),
            type_indices,
            *backward_inputs[3:],
        )
        for left, right in zip(expected_gradients, actual_gradients):
            assert left.numpy().tobytes() == right.numpy().tobytes()

        event_inputs = backward_inputs + [
            arguments[2],
            arguments[9],
            tf.constant(0.3, dtype),
            tf.constant(0.5, dtype),
        ]
        expected_event_gradients = ops.dpointnet_nest_state_backward_events(
            *event_inputs, detach_reset=False, detach_asc_reset=False)
        actual_event_gradients = ops.dpointnet_nest_state_backward_events_type_indexed(
            *event_inputs[:2],
            tf.constant(type_coefficients, dtype),
            type_indices,
            *event_inputs[3:],
            detach_reset=False,
            detach_asc_reset=False,
        )
        for left, right in zip(expected_event_gradients, actual_event_gradients):
            assert left.numpy().tobytes() == right.numpy().tobytes()
