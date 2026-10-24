from types import SimpleNamespace

import numpy as np
import pytest
import tensorflow as tf

from bmtk.simulator.dpointnet import acceleration
from bmtk.simulator.dpointnet.loss_functions.target_firing_rates import TargetFiringRate


FAT_BINARY_ARCHITECTURES = (61, 70, 75, 80, 86, 89, 90)


@pytest.fixture
def projection_hardware(monkeypatch):
    monkeypatch.setattr(acceleration.csr_spike_ops, "combined_sweep_available", lambda: True)
    monkeypatch.setattr(acceleration.csr_spike_ops, "fused_cuda_available", lambda: True)
    monkeypatch.setattr(acceleration.csr_spike_ops, "_gpu_compute_architecture", lambda: 86)
    monkeypatch.setattr(acceleration.csr_spike_ops, "_SM_ARCHITECTURES", FAT_BINARY_ARCHITECTURES)


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


@pytest.mark.parametrize("architecture", FAT_BINARY_ARCHITECTURES)
@pytest.mark.parametrize("batch", [1, 2, 7, 13, 16, 31, 32])
def test_every_fat_binary_architecture_and_batch_up_to_32_is_admitted(
    projection_hardware, architecture, batch,
):
    enabled, tile, _ = select(architecture=architecture, batch_size=batch)
    assert enabled and tile == (256 if batch == 1 else 2048)


@pytest.mark.parametrize("architecture", [52, 60, 72, 87, 100, 120, None])
def test_non_target_architectures_keep_existing_routes(projection_hardware, architecture):
    assert not select(architecture=architecture)[0]


def test_architecture_without_native_sass_keeps_existing_route(projection_hardware, monkeypatch):
    monkeypatch.setattr(acceleration.csr_spike_ops, "_SM_ARCHITECTURES", (86,))
    assert select(architecture=86)[0]
    assert not select(architecture=89)[0]


@pytest.mark.parametrize("batch", [0, 33, 64, None, True])
def test_unknown_or_large_batch_does_not_get_new_route(projection_hardware, batch):
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
    monkeypatch.setattr(acceleration.csr_spike_ops, "_SM_ARCHITECTURES", FAT_BINARY_ARCHITECTURES)
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
    assert metadata["source_rows_upper_bound"] == 4 * (3 + 4)
    inputs["drive"]["options"]["input_type"] = "current"
    assert not acceleration.performance_projection_context(
        network, inputs, dt=1.0, max_delay=5.0, standard_cell=True, single_replica=True,
    )["spike_inputs"]


@pytest.mark.parametrize("max_delay", [None, 0, -1, 2.5])
@pytest.mark.parametrize("dt", [1.0, 0.5])
def test_context_bounds_recurrent_history_for_capped_and_uncapped_delays(max_delay, dt):
    from bmtk.simulator.dpointnet.cell_models.glif3_cell import GLIF3Cell
    from test_s8_type_indexed_nest_coefficients import _small_network_inputs

    network, inputs = _small_network_inputs()
    network["synapses"]["delays"][-1] = 7.5
    metadata = acceleration.performance_projection_context(
        network, inputs, dt=dt, max_delay=max_delay,
        standard_cell=True, single_replica=True,
    )
    cell = GLIF3Cell(
        network, inputs, dt=dt, max_delay=max_delay, dynamics_mode="nest",
        acceleration_profile=None, use_fused_cuda=False, use_fused_state=False,
        tau_basis=[2.0, 4.0, 8.0, 16.0],
    )
    source_rows = cell.recurrent_dense_shape[1] + sum(
        item["input_dense_shape"][1] for item in cell.inputs.values()
    )
    assert metadata["source_rows_upper_bound"] >= source_rows
    limit = 7.5 if max_delay is None or max_delay <= 0 else max_delay
    assert metadata["source_rows_upper_bound"] == (
        (int(np.ceil(limit / dt)) + 1) * 3
        + (int(np.ceil(3.0 / dt)) + 1) * 4
    )


@pytest.mark.parametrize("n_inputs,enabled", [(20000, True), (100000, False)])
def test_context_bounds_external_delays_independently_of_recurrent_cap(
    projection_hardware, n_inputs, enabled,
):
    network = {
        "n_nodes": 4096,
        "synapses": {"indices": np.array([[0, 1]]), "delays": np.array([5.0])},
    }
    inputs = {
        "drive": {
            "n_inputs": n_inputs, "indices": np.array([[0, 0], [1, 1]]),
            "delays": np.array([1.0, 500.0]), "input_type": "spikes",
        },
        "background": {
            "n_inputs": 10, "indices": np.array([[0, 0]]),
            "delays": np.array([1.0]), "input_type": "poisson_spikes_internal",
        },
    }
    metadata = acceleration.performance_projection_context(
        network, inputs, dt=1.0, max_delay=5.0,
        standard_cell=True, single_replica=True,
    )
    assert metadata["source_rows_upper_bound"] == 6 * 4096 + 501 * n_inputs + 2 * 10
    assert select(metadata=metadata, batch_size=1)[:2] == (enabled, 2048)


@pytest.mark.parametrize("max_delay", [None, 0, -1])
def test_high_level_build_accepts_uncapped_delay_with_acceleration_disabled(max_delay):
    from bmtk.simulator.dpointnet.rnn_model import RNN
    from test_s8_type_indexed_nest_coefficients import _small_network_inputs

    policy = tf.keras.mixed_precision.global_policy()
    try:
        network, inputs = _small_network_inputs()
        rnn = RNN(
            seq_len=3, batch_size=1, precision={"mixed_precision": False},
            cell_params={
                "acceleration_profile": None, "max_delay": max_delay,
                "use_fused_cuda": False, "use_fused_state": False,
                "tau_basis": [2.0, 4.0, 8.0, 16.0], "dynamics_mode": "nest",
            },
        )
        rnn._recurrent_networks["recurrent"] = SimpleNamespace(to_dict=lambda: network)
        for name, item in inputs.items():
            rnn._input_networks[name] = SimpleNamespace(
                name=name, n_spiking_nodes=item["n_inputs"], to_dict=lambda item=item: item,
            )
            rnn._inputs_order.append(name)
        rnn.build(training=False)
        assert rnn.cell.max_delay == 3
        assert rnn.cell.acceleration_report["profile"] is None
    finally:
        tf.keras.mixed_precision.set_global_policy(policy)


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
    cell.acceleration_report["profile"] = "compatibility_v1"
    assert not objective.compiled_spike_counts
    cell.acceleration_report["profile"] = "auto"
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


@pytest.mark.parametrize("csr_masters", [False, None])
@pytest.mark.parametrize("selective", [False, True])
@pytest.mark.parametrize("batch", [1, 32])
def test_high_level_auto_selects_real_native_projection_without_manual_flags(selective, batch, csr_masters):
    from bmtk.simulator.dpointnet.custom_ops import csr_spike_ops
    if not csr_spike_ops.combined_sweep_available():
        pytest.skip("Rebuilt native combined sweep required")
    from bmtk.simulator.dpointnet.acceleration import JOINT_SWEEP_ARCHITECTURES
    if csr_spike_ops._gpu_compute_architecture() not in JOINT_SWEEP_ARCHITECTURES:
        pytest.skip("Automatic joint sweep requires a fat-binary target architecture")
    # None: default automatic CSR masters/uint8 (and adaptive routing from batch 8);
    # False: previous route.
    from bmtk.simulator.dpointnet.acceleration import ADAPTIVE_ROUTING_MIN_BATCH
    default = csr_masters is None
    adaptive = default and batch >= ADAPTIVE_ROUTING_MIN_BATCH
    tile = 2048 if adaptive or batch != 1 else 256
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
                **({} if default else {"use_csr_master_weights": False}),
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
        assert cell._use_cooperative_row_projection is True
        assert cell._use_fanout_row_projection is True
        assert cell._use_target_gather_projection is True
        assert cell._use_prepared_cpu_poisson == "auto"
        assert cell._declared_sequence_length == 7 and cell._declared_batch_size == batch
        assert "CPU:" in cell.recurrent_indices.device
        assert all("CPU:" in item["input_indices"].device for item in cell.inputs.values())
        assert TargetFiringRate(rnn, 5.0).compiled_spike_counts is (
            csr_spike_ops._gpu_compute_architecture() == 86
        )
        assert cell._csr_master_weights is default
        assert cell._uint8_synapse_types is default
        assert cell._adaptive_routing is adaptive
        report = rnn.acceleration_report
        assert report["csr_master_weights"] == (
            "direct_csr_fp32_master_host_lifecycle_v1" if default else "disabled"
        )
        assert report["adaptive_routing"] == (
            "adaptive_segment_ticket_v1" if adaptive else "disabled"
        )
        assert report["launch_geometry"]["valid"] and report["launch_geometry"]["tile"] == tile

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
        operator = "DpointnetCombinedSweepCsrMaster" if default else "DpointnetCombinedSweep"
        sweep = [
            node for node in evaluate.get_concrete_function().graph.as_graph_def().node
            if node.op == operator
        ]
        assert len(sweep) == 1
        if default:
            assert sweep[0].attr["uint8_types"].b
            assert sweep[0].attr["adaptive_routing"].b is adaptive
        if adaptive:
            assert len(sweep[0].attr["adaptive_traversal"].list.i) == len(cell.inputs) + 1
        assert sweep[0].attr["fanout_rows"].b
        assert sweep[0].attr["target_tile_size"].i == tile
        assert len(sweep[0].attr["cooperative_rows"].list.b) == len(cell.inputs) + 1
        # Gather applies only to sources with at most 16 incoming edges per target.
        assert "gather_sources" not in sweep[0].attr or len(sweep[0].attr["gather_sources"].list.b) == len(cell.inputs) + 1
    finally:
        if rnn is not None and rnn._cell is not None:
            rnn._cell.close_fused_cuda()
        tf.keras.mixed_precision.set_global_policy(policy)


@pytest.mark.parametrize("selective", [False, True])
@pytest.mark.parametrize("rollout", [False, True])
@pytest.mark.parametrize("weight_mode", ["per_edge", "frozen", "per_type"])
@pytest.mark.parametrize("device_dispatch", [False, True])
def test_guarded_native_dispatch_rechecks_live_coefficients_and_restore(
    selective, rollout, weight_mode, device_dispatch, tmp_path, monkeypatch,
):
    from bmtk.simulator.dpointnet.custom_ops import glif_state_ops
    if not glif_state_ops.fused_nest_state_history_type_indexed_available():
        pytest.skip("Native type-indexed history required")
    if device_dispatch:
        if not glif_state_ops.fused_nest_device_dispatch_available():
            pytest.skip("Rebuilt device-dispatch operators required")
    else:
        monkeypatch.setattr(glif_state_ops, "fused_nest_device_dispatch_available", lambda: False)
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
            train_recurrent=weight_mode != "frozen",
            train_recurrent_per_type=weight_mode == "per_type",
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
@pytest.mark.parametrize("shared_types", [False, True])
def test_guarded_dispatch_preserves_segmented_remainder_vjps(selective, batch, shared_types, tmp_path):
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
        if shared_types:
            network["node_type_ids"] = np.zeros_like(network["node_type_ids"])
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
        if not shared_types:
            return
        traces = evaluate.experimental_get_tracing_count()
        original = cell.decay.numpy()
        checkpoint = tf.train.Checkpoint(decay=cell.decay)
        path = checkpoint.save(str(tmp_path / "remainder-coefficients"))
        types = np.asarray(network["node_type_ids"])
        shared_type = next(value for value in np.unique(types) if np.sum(types == value) > 1)
        members = np.flatnonzero(types == shared_type)
        changed = original.copy()
        changed[members[0]] *= .5
        cell.decay.assign(changed)

        def compare_live(identity):
            assert bool(cell.prepare_rollout_nest_coefficients()[4].numpy()) is identity
            expected, reference = evaluate(False)
            actual, gradients = evaluate(True)
            for value, oracle in zip(actual, expected):
                np.testing.assert_array_equal(value, oracle)
            for value, oracle in zip(gradients, reference):
                np.testing.assert_allclose(value, oracle, rtol=2e-5, atol=2e-5)

        compare_live(False)
        changed[members] = changed[members[0]]
        cell.decay.assign(changed)
        compare_live(True)
        checkpoint.restore(path).assert_consumed()
        np.testing.assert_array_equal(cell.decay.numpy(), original)
        compare_live(True)
        assert evaluate.experimental_get_tracing_count() == traces
    finally:
        if cell is not None:
            cell.close_fused_cuda()
        tf.keras.mixed_precision.set_global_policy(policy)


def _resolver_hardware(monkeypatch, *, cooperative=True, fanout=True, gather=True):
    monkeypatch.setattr(acceleration.csr_spike_ops, "combined_sweep_available", lambda: True)
    monkeypatch.setattr(acceleration.csr_spike_ops, "fused_cuda_available", lambda: True)
    monkeypatch.setattr(acceleration.csr_spike_ops, "_gpu_compute_architecture", lambda: 86)
    monkeypatch.setattr(acceleration.csr_spike_ops, "_SM_ARCHITECTURES", FAT_BINARY_ARCHITECTURES)
    monkeypatch.setattr(acceleration.csr_spike_ops, "cooperative_combined_sweep_available", lambda: cooperative)
    monkeypatch.setattr(acceleration.csr_spike_ops, "fanout_combined_sweep_available", lambda: fanout)
    monkeypatch.setattr(acceleration.csr_spike_ops, "gather_combined_sweep_available", lambda: gather)
    monkeypatch.setattr(acceleration.glif_state_ops, "fused_nest_state_available", lambda: True)
    monkeypatch.setattr(acceleration.glif_state_ops, "glif_state_op_status", lambda: {})
    monkeypatch.setattr(acceleration.csr_spike_ops, "cuda_op_status", lambda: {})


def resolve(options=None, metadata=None, **overrides):
    parameters = dict(
        compute_dtype=tf.float16, variable_dtype=tf.float32, batch_size=32, basis_width=4,
        performance_context=context() if metadata is None else metadata,
    )
    parameters.update(overrides)
    return acceleration.resolve_acceleration_options({**(options or {})}, **parameters)


PROJECTION_OPTIONS = (
    "use_cooperative_row_projection", "use_fanout_row_projection", "use_target_gather_projection",
)


@pytest.mark.parametrize("boundary", [{}, {"dynamics_mode": "legacy"}, {"use_fused_cuda": False}])
def test_joint_override_recommendation_ignores_only_own_override(monkeypatch, boundary):
    _resolver_hardware(monkeypatch)
    options, report = resolve({"use_combined_current_sweep": False, **boundary})
    assert options["use_combined_current_sweep"] is False
    assert options["use_csr_master_weights"] is False
    assert all(options[name] is False for name in PROJECTION_OPTIONS)
    final = acceleration.finalize_acceleration_report(
        report, report["requested"], report["selected"],
    )
    entry = final["effective"]["use_combined_current_sweep"]
    assert entry["current_policy"] is (not boundary)
    audit = next(
        item for item in final["optimization_audit"] if item["option"] == "use_combined_current_sweep"
    )
    assert (audit["classification"] == "avoidable_execution_override") is (not boundary)


@pytest.mark.parametrize("batch", [1, 16, 32])
@pytest.mark.parametrize("dtype", [tf.float16, tf.float32])
def test_auto_selects_projection_traversals_and_prepared_poisson_with_joint_sweep(
    monkeypatch, batch, dtype,
):
    _resolver_hardware(monkeypatch)
    options, report = resolve(batch_size=batch, compute_dtype=dtype)
    assert options["use_combined_current_sweep"] is True
    for name in PROJECTION_OPTIONS:
        assert options[name] is True, name
        assert "admitted joint sweep" in report["reasons"][name]
    assert options["use_prepared_cpu_poisson"] == "auto"
    assert report["selected"]["use_prepared_cpu_poisson"] == "auto"


@pytest.mark.parametrize("name", PROJECTION_OPTIONS + ("use_prepared_cpu_poisson",))
def test_auto_keeps_explicit_false_and_true_overrides(monkeypatch, name):
    _resolver_hardware(monkeypatch)
    for value in (False, True):
        options, report = resolve({name: value})
        assert options[name] is value
        assert report["reasons"][name].startswith("explicit override")


@pytest.mark.parametrize("missing", ["cooperative", "fanout", "gather"])
def test_auto_omits_traversals_without_the_rebuilt_native_capability(monkeypatch, missing):
    _resolver_hardware(monkeypatch, **{missing: False})
    options, _ = resolve()
    expected = {
        "use_cooperative_row_projection": missing != "cooperative",
        "use_fanout_row_projection": missing != "fanout",
        "use_target_gather_projection": missing != "gather",
    }
    for name, value in expected.items():
        assert options[name] is value, name


@pytest.mark.parametrize("boundary", [
    {"options": {"use_combined_current_sweep": False}},
    {"options": {"acceleration_profile": None}},
    {"architecture": 120},
    {"batch_size": 33},
    {"metadata": None},
])
def test_traversals_follow_the_joint_sweep_boundary(monkeypatch, boundary):
    _resolver_hardware(monkeypatch)
    if "architecture" in boundary:
        monkeypatch.setattr(acceleration.csr_spike_ops, "_gpu_compute_architecture", lambda: boundary["architecture"])
    options = boundary.get("options", {})
    if options.get("acceleration_profile", "auto") is None:
        resolved, report = resolve(options)
        assert report is None and not any(name in resolved for name in PROJECTION_OPTIONS)
        return
    resolved, _ = resolve(
        options, **({"batch_size": boundary["batch_size"]} if "batch_size" in boundary else {}),
        **({"performance_context": None} if "metadata" in boundary else {}),
    )
    assert resolved.get("use_combined_current_sweep") is not True
    for name in PROJECTION_OPTIONS:
        assert resolved[name] is False, name


@pytest.mark.parametrize("options,metadata", [
    ({"use_device_poisson": True}, None),
    ({"temporal_gradient_precision": "float32"}, None),
    ({"dynamics_mode": "legacy"}, None),
    ({"use_direct_state_rnn_loop": False}, None),
    ({}, context(single_replica=False)),
    ({}, context(standard_cell=False)),
])
def test_prepared_poisson_auto_respects_its_boundaries(monkeypatch, options, metadata):
    _resolver_hardware(monkeypatch)
    resolved, report = resolve(options, metadata=metadata)
    assert resolved["use_prepared_cpu_poisson"] is False
    assert report["selected"]["use_prepared_cpu_poisson"] is False


def test_prepared_poisson_auto_requires_a_context():
    resolved, _ = resolve(metadata=None, performance_context=None)
    assert resolved["use_prepared_cpu_poisson"] is False


def test_context_records_declared_sequence_length():
    network = {"synapses": {"delays": np.array([1.0]), "indices": np.zeros((1, 2))}, "n_nodes": 3}
    inputs = {"bkg": {"delays": np.array([1.0]), "n_inputs": 2, "indices": np.zeros((2, 2)),
                      "input_type": "poisson_spikes_internal"}}
    metadata = acceleration.performance_projection_context(
        network, inputs, dt=1.0, max_delay=3, standard_cell=True, single_replica=True,
    )
    assert metadata["sequence_length"] is None
    metadata = acceleration.performance_projection_context(
        network, inputs, dt=1.0, max_delay=3, standard_cell=True, single_replica=True,
        sequence_length=np.int64(500),
    )
    assert metadata["sequence_length"] == 500 and isinstance(metadata["sequence_length"], int)


@pytest.mark.parametrize("length,batch,admitted", [
    (500, 32, True), (None, 32, True), (None, None, False), (3000, 32, False),
])
def test_prepared_poisson_static_admission_uses_declared_shapes(length, batch, admitted):
    from bmtk.simulator.dpointnet.cell_models import state_rnn

    cell = SimpleNamespace(
        inputs={"bkg": {"input_type": "poisson_spikes_internal", "input_dim": 100}},
        _declared_sequence_length=500 if length is None else None,
        _declared_batch_size=32 if batch is None else None,
    )
    sequences = tf.keras.Input(shape=(length, 0), batch_size=batch, dtype=tf.float32)
    if length is None and batch is None:
        cell._declared_sequence_length = None
    assert state_rnn._prepared_poisson_admitted(cell, sequences) is admitted
    # Entries: 500 * 32 * 100 * 8 B = 12.8 MB fits; 3000 steps do not.
    assert state_rnn._prepared_poisson_admitted(
        SimpleNamespace(inputs={}, _declared_sequence_length=1, _declared_batch_size=1), sequences,
    ) is False


@pytest.mark.parametrize("selective", [False, True])
def test_auto_prepared_poisson_runs_direct_loop_and_falls_back_silently(selective, monkeypatch):
    """Under auto the layer prepares exact counts on the direct loop and keeps the
    per-step sampler for an unrolled loop; explicit True still raises there."""
    from test_precision_credit import make_cell, build_core
    from bmtk.simulator.dpointnet.cell_models import state_rnn
    from bmtk.simulator.dpointnet.cell_models.state_rnn import ExplicitStateRNN

    cell = make_cell(
        "nest", selective=selective, noise=True, use_direct_state_rnn_loop=True,
    )
    assert cell._use_prepared_cpu_poisson is False  # bare cell: no context, no auto preparation
    cell._use_prepared_cpu_poisson = "auto"
    cell._declared_sequence_length = 7
    cell._declared_batch_size = 3
    calls = []
    original = state_rnn._prepare_cpu_poisson

    def counting(cell_, sequences, states, budget_bytes=state_rnn.PREPARED_POISSON_BUDGET_BYTES):
        calls.append(budget_bytes)
        return original(cell_, sequences, states, budget_bytes)

    monkeypatch.setattr(state_rnn, "_prepare_cpu_poisson", counting)
    try:
        core, states, sequence = build_core(cell, length=7, batch=3)
        prepared = core([sequence, *states])
        # Keras may trace the functional model more than once; every trace must
        # prepare under the automatic ceiling.
        assert calls and set(calls) == {state_rnn.PREPARED_POISSON_AUTO_CEILING_BYTES}
        cell._use_prepared_cpu_poisson = False
        core, states, sequence = build_core(cell, length=7, batch=3)
        plain = core([sequence, *states])
        for left, right in zip(tf.nest.flatten(prepared), tf.nest.flatten(plain)):
            np.testing.assert_array_equal(left, right)
        cell._use_prepared_cpu_poisson = "auto"
        width = 0
        unrolled = ExplicitStateRNN(cell, return_sequences=True, return_state=True, unroll=True)
        unrolled._autocast = False
        calls.clear()
        unrolled(tf.ones((3, 7, width), cell.compute_dtype), initial_state=states)
        assert calls == []
        cell._use_prepared_cpu_poisson = True
        explicit = ExplicitStateRNN(cell, return_sequences=True, return_state=True, unroll=True)
        explicit._autocast = False
        with pytest.raises(ValueError, match="direct NEST compute-temporal loop"):
            explicit(tf.ones((3, 7, width), cell.compute_dtype), initial_state=states)
    finally:
        cell.close_fused_cuda()
