"""Default-preserving interfaces for external current and release components."""

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from test_precision_credit import build_core, make_cell
from bmtk.simulator.dpointnet.cell_models.glif3_cell import GLIF3Cell
from bmtk.simulator.dpointnet.custom_ops.glif_state_ops import fused_glif_state_available
from bmtk.simulator.dpointnet.segmented_recompute import SegmentedRecomputeRunner


@pytest.fixture(autouse=True)
def restore_policy():
    previous = tf.keras.mixed_precision.global_policy()
    yield
    tf.keras.mixed_precision.set_global_policy(previous)


def assert_equal_nested(actual, expected):
    tf.nest.assert_same_structure(actual, expected)
    for value, reference in zip(tf.nest.flatten(actual), tf.nest.flatten(expected)):
        assert value.dtype == reference.dtype
        np.testing.assert_array_equal(value.numpy(), reference.numpy())


@pytest.mark.parametrize("mode", ["legacy", "nest"])
@pytest.mark.parametrize("selective", [False, True])
@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("fused", [False, True])
def test_current_injection_preserves_base_outputs_carry_and_vjps(mode, selective, compiled, fused):
    if fused and not fused_glif_state_available():
        pytest.skip("Requires rebuilt native state operators and an authorized GPU")
    cell = make_cell(mode, selective=selective, noise=True, fused=fused)
    inputs = tf.ones((2, 2), cell.compute_dtype)
    state = cell.zero_state(2, cell.compute_dtype)
    currents = tf.constant(np.linspace(0.1, 1.5, 16).reshape(2, 8), cell.compute_dtype)
    noise_seed = tf.constant(17, tf.int32)

    def run(use_public, current, initial):
        floating = [value for value in initial if value.dtype.is_floating]
        with tf.GradientTape() as tape:
            tape.watch([current, inputs, *floating])
            result = (
                cell.advance_with_currents(inputs, initial, current, noise_seed=noise_seed)
                if use_public else GLIF3Cell._call_impl(
                    cell, inputs, initial, recorded_currents=current, noise_seed=noise_seed
                )
            )
            values = [value for value in tf.nest.flatten(result) if value.dtype.is_floating]
            loss = tf.add_n([
                tf.reduce_sum(tf.cast(value, tf.float32) * (index + 1))
                for index, value in enumerate(values)
            ])
        return result, tape.gradient(
            loss, [current, inputs, *floating], unconnected_gradients=tf.UnconnectedGradients.ZERO
        )

    evaluate = tf.function(run) if compiled else run
    actual = evaluate(True, currents, state)
    expected = evaluate(False, currents, state)
    assert_equal_nested(tuple(actual), tuple(expected))
    assert np.any(actual[1][0].numpy() != 0), "Injected-current credit must stay live."
    np.testing.assert_array_equal(actual[0][1][6], state[6] + 1)


def test_injection_bypasses_subclass_projection_override():
    class ExternalReleaseCell(GLIF3Cell):
        def _call_impl(self, inputs, states, **kwargs):
            raise AssertionError("Current injection must not re-enter the release adapter")

    network, inputs, options = make_cell("nest", selective=False, return_spec=True)
    cell = ExternalReleaseCell(network, inputs, **options)
    state = cell.zero_state(2, cell.compute_dtype)
    output, final = cell.advance_with_currents(
        tf.ones((2, 2)), state, tf.ones((2, 8))
    )
    assert len(final) == len(state)
    np.testing.assert_array_equal(final[6], 1)
    assert all(np.isfinite(value.numpy()).all() for value in tf.nest.flatten(output))


@pytest.mark.parametrize("mode", ["legacy", "nest"])
def test_current_adapter_segmented_remainder_input_carry_and_master_vjps(mode):
    class ExternalProjectionCell(GLIF3Cell):
        def _call_impl(self, inputs, states, **kwargs):
            unknown = {
                name for name, value in kwargs.items()
                if name != "noise_seed" and value is not None and value is not False
            }
            if unknown:
                raise ValueError("Unsupported external projection options: " + ", ".join(sorted(unknown)))
            seed = kwargs.get("noise_seed")
            currents, _ = self._project_step_currents(inputs, states, noise_seed=seed)
            return self.advance_with_currents(inputs, states, currents, noise_seed=seed)

    network, inputs, options = make_cell(mode, selective=False, return_spec=True)
    cell = ExternalProjectionCell(network, inputs, **options)
    core, state, sequence = build_core(cell, length=7)
    segmented = SegmentedRecomputeRunner(core, 7, 3, 2, differentiate_inputs=True)
    floating = [value for value in state if value.dtype.is_floating]
    targets = [sequence, *floating, cell.recurrent_weight_values, cell.inputs["drive"]["input_weight_values"]]

    def evaluate(replay):
        with tf.GradientTape() as tape:
            tape.watch([sequence, *floating])
            output = segmented(sequence, state) if replay else core([sequence, *state])
            loss = tf.add_n([
                tf.reduce_sum(tf.cast(value, tf.float32)) * (index + 1)
                for index, value in enumerate(output) if value.dtype.is_floating
            ])
        return output, tape.gradient(loss, targets, unconnected_gradients=tf.UnconnectedGradients.ZERO)

    expected, reference = evaluate(False)
    actual, gradients = evaluate(True)
    assert_equal_nested(tuple(actual), tuple(expected))
    for gradient, oracle in zip(gradients, reference):
        np.testing.assert_allclose(gradient, oracle, rtol=2e-5, atol=2e-5)
    assert np.any(gradients[-1].numpy() != 0), "External canonical weight credit must stay live."


@pytest.mark.parametrize("mode", ["legacy", "nest"])
def test_public_and_ordinary_paths_match_without_changing_default_projection(mode):
    cell = make_cell(mode, selective=False)
    inputs = tf.constant([[0.5, 0.7], [0.2, 1.1]])
    state = cell.zero_state(2, tf.float32)
    for _ in range(5):
        currents, _ = cell._project_step_currents(inputs, state)
        injected = cell.advance_with_currents(inputs, state, currents)
        ordinary = cell(inputs, state)
        assert_equal_nested(injected, ordinary)
        state = ordinary[1]


def test_injection_rejects_missing_currents_shape_and_unsupported_temporal_mode():
    cell = make_cell("nest", selective=False)
    state = cell.zero_state(2, tf.float32)
    inputs = tf.zeros((2, 2))
    with pytest.raises(ValueError, match="explicit synaptic currents"):
        cell.advance_with_currents(inputs, state, None)
    with pytest.raises((ValueError, tf.errors.InvalidArgumentError)):
        cell.advance_with_currents(inputs, state, tf.zeros((2, 7)))
    temporal = make_cell("nest", temporal_gradient_precision="float32")
    with pytest.raises(ValueError, match="FP32 temporal gradients require"):
        temporal.advance_with_currents(
            inputs, temporal.zero_state(2, temporal.compute_dtype), tf.zeros((2, 8))
        )


@pytest.mark.parametrize("selective", [False, True])
def test_public_shadow_creation_preserves_explicit_dtype_refresh_and_restore(tmp_path, selective):
    cell = make_cell("nest", selective=selective)
    master = tf.Variable([0.25, -0.75], trainable=False, dtype=tf.float32)
    shadow = cell.create_compute_shadow(master, name="external_csr", dtype=tf.float32)
    default = cell.create_compute_shadow(master, name="external_default")
    assert shadow.dtype == tf.float32
    assert default.dtype == cell.compute_dtype
    assert not shadow.trainable
    assert shadow is not master
    assert all(shadow is not value for value in cell.trainable_variables)
    checkpoint = tf.train.Checkpoint(master=master, shadow=shadow)
    path = checkpoint.save(str(tmp_path / "owned-shadow"))
    master.assign([0.5, -1.25])
    np.testing.assert_array_equal(shadow, [0.25, -0.75])
    shadow.assign(tf.cast(master, shadow.dtype))
    np.testing.assert_array_equal(shadow, master)
    master.assign([3.0, 4.0])
    shadow.assign([5.0, 6.0])
    checkpoint.restore(path).assert_consumed()
    np.testing.assert_array_equal(master, [0.25, -0.75])
    np.testing.assert_array_equal(shadow, master)


def test_native_options_are_actual_explicit_selections_not_mutable_policy():
    cell = make_cell("nest", selective=False)
    assert cell.acceleration_report is None
    options = cell.native_execution_options
    assert set(options) == {
        "use_fused_cuda", "use_fused_state", "use_fused_state_history",
        "use_pair_projection", "use_fused_nest_event_vjp",
    }
    assert not any(options.values())
    options["use_fused_state"] = True
    assert cell.native_execution_options["use_fused_state"] is False
