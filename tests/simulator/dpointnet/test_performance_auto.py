from types import SimpleNamespace

import numpy as np
import pytest
import tensorflow as tf

from bmtk.simulator.dpointnet import acceleration
from bmtk.simulator.dpointnet.loss_functions.target_firing_rates import TargetFiringRate


@pytest.fixture
def projection_hardware(monkeypatch):
    monkeypatch.setattr(acceleration.csr_spike_ops, "combined_sweep_available", lambda: True)
    monkeypatch.setattr(acceleration.csr_spike_ops, "fused_cuda_available", lambda: True)
    monkeypatch.setattr(acceleration.csr_spike_ops, "_gpu_compute_architecture", lambda: 86)


def context(**overrides):
    return {
        "standard_cell": True, "single_replica": True, "spike_inputs": True,
        "n_post": 66658, "source_rows_upper_bound": 415416,
        "max_edges": 24450554, "recurrent_edges": 24450554, **overrides,
    }


def select(options=None, metadata=None, **overrides):
    parameters = dict(
        compute_dtype=tf.float16, variable_dtype=tf.float32, batch_size=32,
        basis_width=4, architecture=86,
    )
    parameters.update(overrides)
    return acceleration.select_performance_projection(
        options or {}, context() if metadata is None else metadata, **parameters
    )


@pytest.mark.parametrize("batch,tile", [(1, 256), (16, 2048), (32, 2048)])
@pytest.mark.parametrize("dtype", [tf.float16, tf.float32])
def test_auto_projection_selects_qualified_batch_geometry(projection_hardware, batch, tile, dtype):
    enabled, actual_tile, reason = select(batch_size=batch, compute_dtype=dtype)
    assert enabled and actual_tile == tile
    assert "rounds once" in reason


@pytest.mark.parametrize("architecture", [61, 70, 75, 80, 89, 90, None])
def test_unqualified_architectures_keep_existing_routes(projection_hardware, architecture):
    assert not select(architecture=architecture)[0]


@pytest.mark.parametrize("batch", [0, 7, 13, 33, None, True])
def test_unknown_or_unqualified_batch_does_not_get_new_route(projection_hardware, batch):
    assert not select(batch_size=batch)[0]


@pytest.mark.parametrize("option,value", [
    ("acceleration_profile", None), ("use_combined_current_sweep", False),
    ("use_fused_cuda", False), ("dynamics_mode", "legacy"),
    ("temporal_gradient_precision", "float32"),
])
def test_auto_projection_preserves_explicit_and_scientific_boundaries(projection_hardware, option, value):
    assert not select({option: value})[0]


@pytest.mark.parametrize("field,value", [
    ("standard_cell", False), ("single_replica", False), ("spike_inputs", False),
    ("recurrent_edges", 0),
])
def test_custom_distributed_continuous_and_empty_routes_are_not_promoted(
    projection_hardware, field, value,
):
    assert not select(metadata=context(**{field: value}))[0]


@pytest.mark.parametrize("dtype,masters", [
    (tf.bfloat16, tf.float32), (tf.float64, tf.float64), (tf.float16, tf.float64),
])
def test_auto_projection_never_changes_precision_to_admit_a_route(projection_hardware, dtype, masters):
    assert not select(compute_dtype=dtype, variable_dtype=masters)[0]


def test_routing_budget_exact_boundary_and_fallback(projection_hardware):
    budget = 512 * 1024**2
    rows = budget // (17 * 4)
    assert select(metadata=context(n_post=4096, source_rows_upper_bound=rows), batch_size=1)[:2] == (
        True, 256,
    )
    assert select(metadata=context(n_post=4096, source_rows_upper_bound=rows + 1), batch_size=1)[:2] == (
        True, 2048,
    )
    assert not select(metadata=context(
        n_post=4096, source_rows_upper_bound=budget // (3 * 4) + 1,
    ), batch_size=1)[0]


def test_auto_projection_requires_new_native_tile_abi(monkeypatch):
    monkeypatch.setattr(acceleration.csr_spike_ops, "fused_cuda_available", lambda: True)
    monkeypatch.setattr(
        acceleration.csr_spike_ops, "_OPS",
        SimpleNamespace(dpointnet_combined_sweep=lambda sources: None),
    )
    assert not select()[0]
    monkeypatch.setattr(
        acceleration.csr_spike_ops, "_OPS",
        SimpleNamespace(dpointnet_combined_sweep=lambda sources, target_tile_size=2048: None),
    )
    assert select()[0]


def test_context_respects_effective_input_types_and_delay_expansion():
    from test_s8_type_indexed_nest_coefficients import _small_network_inputs
    network, inputs = _small_network_inputs()
    metadata = acceleration.performance_projection_context(
        network, inputs, dt=1.0, max_delay=5.0, standard_cell=True, single_replica=True,
    )
    assert metadata["spike_inputs"]
    assert metadata["source_rows_upper_bound"] == 6 * (3 + 4)
    inputs["drive"]["options"]["input_type"] = "current"
    assert not acceleration.performance_projection_context(
        network, inputs, dt=1.0, max_delay=5.0, standard_cell=True, single_replica=True,
    )["spike_inputs"]


@pytest.mark.parametrize("tile", [256, 512, 1024, 2048])
def test_resolver_preserves_integer_tile_overrides(projection_hardware, tile):
    options, report = acceleration.resolve_acceleration_options(
        {"acceleration_profile": "auto", "combined_current_target_tile_size": tile},
        compute_dtype=tf.float32, variable_dtype=tf.float32, batch_size=1,
        basis_width=4, canonical_gradient_boundary=False, performance_context=context(),
    )
    assert options["use_combined_current_sweep"] is True
    assert type(options["combined_current_target_tile_size"]) is int
    assert options["combined_current_target_tile_size"] == tile
    assert report["selected"]["combined_current_target_tile_size"] == tile


def test_rate_count_auto_uses_declared_single_replica_boundary():
    metadata = context()
    cell = SimpleNamespace(
        acceleration_report={
            "architecture": 86, "tensorflow": "2.21.0", "selected": {"use_fused_cuda": True},
        },
        _performance_auto_context=metadata,
    )
    rnn = SimpleNamespace(dt=1.0, seq_len=500, _cell=None)
    objective = TargetFiringRate(rnn, 5.0)
    assert not objective.compiled_spike_counts
    rnn._cell = cell
    assert objective.compiled_spike_counts
    assert not TargetFiringRate(rnn, 5.0, compiled_spike_counts=False).compiled_spike_counts
    metadata["single_replica"] = False
    assert not objective.compiled_spike_counts
    cell.acceleration_report = None
    assert not objective.compiled_spike_counts
    assert TargetFiringRate(rnn, 5.0, compiled_spike_counts=True).compiled_spike_counts
    objective.compiled_spike_counts = True
    assert objective.compiled_spike_counts
    with pytest.raises((TypeError, ValueError), match="compiled_spike_counts"):
        objective.compiled_spike_counts = "true"


@pytest.mark.parametrize("selective", [False, True])
@pytest.mark.parametrize("batch,tile", [(1, 256), (32, 2048)])
def test_high_level_auto_selects_real_native_projection_without_manual_flags(selective, batch, tile):
    from bmtk.simulator.dpointnet.custom_ops import csr_spike_ops
    if not csr_spike_ops.combined_sweep_available():
        pytest.skip("Rebuilt native combined sweep required")
    if csr_spike_ops._gpu_compute_architecture() != 86:
        pytest.skip("New automatic performance policy is qualified on SM86")
    from bmtk.simulator.dpointnet.rnn_model import RNN
    from bmtk.simulator.dpointnet.cell_models.state_rnn import _rollout_context
    from test_s8_type_indexed_nest_coefficients import _small_network_inputs

    policy = tf.keras.mixed_precision.global_policy()
    rnn = None
    try:
        network, inputs = _small_network_inputs()
        rnn = RNN(
            seq_len=7, batch_size=batch,
            precision={"mixed_precision": selective, "fp32_voltage_and_asc": selective},
            cell_params={
                "acceleration_profile": "auto", "tau_basis": [2.0, 4.0, 8.0, 16.0],
                "dynamics_mode": "nest", "train_recurrent": True,
            },
        )
        rnn._recurrent_networks["recurrent"] = SimpleNamespace(to_dict=lambda: network)
        for name, item in inputs.items():
            rnn._input_networks[name] = SimpleNamespace(
                name=name, n_spiking_nodes=item["n_inputs"], to_dict=lambda item=item: item,
            )
            rnn._inputs_order.append(name)
        rnn.build(training=False)
        cell = rnn.cell
        assert cell._use_combined_current_sweep
        assert cell.combined_current_target_tile_size == tile
        assert "CPU:" in cell.recurrent_indices.device
        assert all("CPU:" in item["input_indices"].device for item in cell.inputs.values())
        assert TargetFiringRate(rnn, 5.0).compiled_spike_counts

        @tf.function
        def evaluate():
            x = tf.ones((batch, 4), cell.compute_dtype)
            states = list(cell.zero_state(batch, cell.compute_dtype))
            states[0] = tf.ones_like(states[0])
            masters = [cell.recurrent_weight_values] + [
                item["input_weight_values"] for item in cell.inputs.values()
            ]
            with tf.GradientTape() as tape:
                tape.watch(states[0])
                with _rollout_context(cell):
                    currents = cell._project_step_currents(x, states)[0]
                loss = tf.reduce_sum(tf.cast(currents, tf.float32))
            return currents, tape.gradient(loss, [states[0], *masters])

        currents, gradients = evaluate()
        assert np.any(currents.numpy())
        for name, value in zip(("recurrent sources", "recurrent master", "input master"), gradients):
            assert value is not None, name
            assert np.isfinite(value.numpy()).all(), name
        assert np.any(gradients[1].numpy())
    finally:
        if rnn is not None and rnn._cell is not None:
            rnn._cell.close_fused_cuda()
        tf.keras.mixed_precision.set_global_policy(policy)


@pytest.mark.parametrize("selective", [False, True])
@pytest.mark.parametrize("rollout", [False, True])
def test_guarded_native_dispatch_rechecks_live_coefficients_and_restore(selective, rollout, tmp_path):
    from bmtk.simulator.dpointnet.custom_ops import glif_state_ops
    if not glif_state_ops.fused_nest_state_history_type_indexed_available():
        pytest.skip("Native type-indexed history required")
    from bmtk.simulator.dpointnet.cell_models.glif3_cell import GLIF3Cell
    from bmtk.simulator.dpointnet.cell_models.state_rnn import _rollout_context
    from bmtk.simulator.dpointnet.cell_models.state_rnn import ExplicitStateRNN
    from test_s8_type_indexed_nest_coefficients import _small_network_inputs

    policy = tf.keras.mixed_precision.global_policy()
    cells = []
    try:
        tf.keras.mixed_precision.set_global_policy("mixed_float16" if selective else "float32")
        network, inputs = _small_network_inputs()
        common = dict(
            dynamics_mode="nest", tau_basis=[2.0, 4.0, 8.0, 16.0], batch_size=2,
            acceleration_profile="auto", hard_reset=True,
            hard_reset_gradient_mode="soft_surrogate",
            _performance_auto_context=acceleration.performance_projection_context(
                network, inputs, dt=1.0, max_delay=5.0,
                standard_cell=True, single_replica=True,
            ),
        )
        cells.append(GLIF3Cell(network, inputs, **common))
        cells.append(GLIF3Cell(
            network, inputs, use_type_indexed_nest_coefficients=False, **common,
        ))
        guarded, reference = cells
        assert guarded._use_type_indexed_nest_coefficients
        assert not guarded._use_static_type_indexed_nest_dispatch

        def evaluator(cell):
            layer = ExplicitStateRNN(cell, return_sequences=True, return_state=True) if rollout else None

            @tf.function
            def evaluate(currents):
                states = cell.zero_state(2, cell.compute_dtype)
                targets = [value for value in states if value.dtype.is_floating] if rollout else [currents]
                with _rollout_context(cell), tf.GradientTape() as tape:
                    tape.watch(targets)
                    result = (
                        layer(
                            tf.cast(tf.reshape(tf.range(24) % 3, (2, 3, 4)), cell.compute_dtype),
                            initial_state=states,
                        ) if rollout else cell.advance_with_currents(
                            tf.ones((2, 4), cell.compute_dtype), states, currents,
                        )
                    )
                    values = [
                        value for value in tf.nest.flatten(result)
                        if value.dtype.is_floating
                    ]
                    loss = tf.add_n([
                        tf.reduce_sum(tf.cast(value, tf.float32) * tf.random.stateless_normal(
                            tf.shape(value), [23, index],
                        )) for index, value in enumerate(values)
                    ])
                return values, tape.gradient(
                    loss, targets, unconnected_gradients=tf.UnconnectedGradients.ZERO,
                )
            return evaluate

        compiled = [evaluator(cell) for cell in cells]
        currents = tf.constant(np.linspace(-1, 5, 24).reshape(2, 12), tf.float32)

        def compare():
            actual, expected = [function(currents) for function in compiled]
            for left, right in zip(tf.nest.flatten(actual), tf.nest.flatten(expected)):
                np.testing.assert_allclose(left, right, rtol=1e-6, atol=1e-6)
            return tf.nest.map_structure(lambda value: value.numpy(), actual)

        initial = compare()
        initial_traces = [function.experimental_get_tracing_count() for function in compiled]
        checkpoint = tf.train.Checkpoint(first=guarded.decay, second=reference.decay)
        path = checkpoint.save(str(tmp_path / "coefficients"))
        for cell in cells:
            changed = cell.decay.numpy()
            changed[2] *= 0.5
            cell.decay.assign(changed)
        assert not bool(guarded.prepare_rollout_nest_coefficients()[4].numpy())
        compare()
        for cell in cells:
            changed = cell.decay.numpy()
            changed[0] = changed[2]
            cell.decay.assign(changed)
        assert bool(guarded.prepare_rollout_nest_coefficients()[4].numpy())
        compare()
        for cell in cells:
            cell.decay.assign(tf.ones_like(cell.decay) * 0.01)
        checkpoint.restore(path).assert_consumed()
        restored = compare()
        for left, right in zip(tf.nest.flatten(initial), tf.nest.flatten(restored)):
            np.testing.assert_array_equal(left, right)
        assert [function.experimental_get_tracing_count() for function in compiled] == initial_traces
    finally:
        for cell in cells:
            cell.close_fused_cuda()
        tf.keras.mixed_precision.set_global_policy(policy)


@pytest.mark.parametrize("selective", [False, True])
@pytest.mark.parametrize("batch", [1, 2, 32])
def test_guarded_dispatch_preserves_segmented_remainder_vjps(selective, batch):
    from bmtk.simulator.dpointnet.custom_ops import glif_state_ops
    if not glif_state_ops.fused_nest_state_history_type_indexed_available():
        pytest.skip("Native type-indexed history required")
    from test_precision_credit import build_core, make_cell
    from bmtk.simulator.dpointnet.cell_models.glif3_cell import GLIF3Cell
    from bmtk.simulator.dpointnet.segmented_recompute import SegmentedRecomputeRunner

    policy = tf.keras.mixed_precision.global_policy()
    cell = None
    try:
        network, inputs, options = make_cell(
            "nest", selective=selective, fused=True, return_spec=True,
            acceleration_profile="auto",
        )
        inputs["drive"]["input_type"] = "spikes"
        options["_performance_auto_context"] = acceleration.performance_projection_context(
            network, inputs, dt=1.0, max_delay=5.0, standard_cell=True, single_replica=True,
        )
        options["batch_size"] = batch
        cell = GLIF3Cell(network, inputs, **options)
        assert cell._use_type_indexed_nest_coefficients
        assert not cell._use_static_type_indexed_nest_dispatch
        core, state, sequence = build_core(cell, length=7, batch=batch)
        segmented = SegmentedRecomputeRunner(core, 7, 3, 2, differentiate_inputs=True)
        floating = [value for value in state if value.dtype.is_floating]
        targets = [
            sequence, *floating, cell.recurrent_weight_values,
            cell.inputs["drive"]["input_weight_values"],
        ]

        @tf.function
        def evaluate(replay):
            with tf.GradientTape() as tape:
                tape.watch([sequence, *floating])
                output = segmented(sequence, state) if replay else core([sequence, *state])
                loss = tf.add_n([
                    tf.reduce_sum(tf.cast(value, tf.float32) * tf.random.stateless_normal(
                        tf.shape(value), [31, index],
                    )) for index, value in enumerate(output) if value.dtype.is_floating
                ])
            gradients = tape.gradient(
                loss, targets, unconnected_gradients=tf.UnconnectedGradients.ZERO,
            )
            return output, gradients

        expected, reference = evaluate(False)
        actual, gradients = evaluate(True)
        metadata_shapes = {
            tuple(tf.raw_ops.ReadVariableOp(
                resource=connectivity["metadata_handle"],
                dtype=tf.as_dtype(connectivity["index_dtype"]),
            ).shape)
            for connectivity in [cell.recurrent_fused_connectivity] + [
                value.get("fused_connectivity") for value in cell.inputs.values()
            ] if connectivity is not None
        }
        graph = evaluate.get_concrete_function(True).graph.as_graph_def(add_shapes=True)
        for function in graph.library.function:
            for node in function.node_def:
                if "zeros" in node.name:
                    consumers = [
                        consumer.name for consumer in function.node_def if any(
                            argument.lstrip("^").split(":")[0] == node.name
                            for argument in consumer.input
                        )
                    ]
                    returned = [
                        name for name, argument in function.ret.items()
                        if argument.split(":")[0] == node.name
                    ]
                    if not consumers and not returned:
                        continue
                    for shape in node.attr["_output_shapes"].list.shape:
                        assert tuple(value.size for value in shape.dim) not in metadata_shapes, (
                            "Immutable CSR routing acquired a dense zero cotangent", node.name,
                            consumers, returned,
                        )
        for value, oracle in zip(actual, expected):
            np.testing.assert_array_equal(value, oracle)
        for value, oracle in zip(gradients, reference):
            np.testing.assert_allclose(value, oracle, rtol=2e-5, atol=2e-5)
        assert np.any(gradients[-1].numpy())
    finally:
        if cell is not None:
            cell.close_fused_cuda()
        tf.keras.mixed_precision.set_global_policy(policy)
