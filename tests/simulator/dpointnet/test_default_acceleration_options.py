"""Automatic defaults: CSR masters, uint8 types, adaptive routing, packing and checkpoints."""

from types import SimpleNamespace

import numpy as np
import pytest
import tensorflow as tf

from bmtk.simulator.dpointnet import acceleration
from bmtk.simulator.dpointnet.custom_ops import csr_spike_ops as ops

FAT_BINARY_ARCHITECTURES = (61, 70, 75, 80, 86, 89, 90)
NEW_OPTIONS = ("use_csr_master_weights", "use_uint8_synapse_types", "use_adaptive_routing")


@pytest.fixture(autouse=True)
def restore_policy():
    policy = tf.keras.mixed_precision.global_policy()
    yield
    tf.keras.mixed_precision.set_global_policy(policy)


def context(**overrides):
    return {
        "standard_cell": True, "single_replica": True, "spike_inputs": True,
        "n_post": 66658, "source_rows_upper_bound": 415416,
        "max_edges": 24450554, "recurrent_edges": 24450554, "bptt_training": True,
        **overrides,
    }


@pytest.fixture
def hardware(monkeypatch):
    def install(architecture=86, masters=True, uint8=True, adaptive=True):
        for name, value in (
            ("combined_sweep_available", True), ("fused_cuda_available", True),
            ("cooperative_combined_sweep_available", True),
            ("fanout_combined_sweep_available", True),
            ("gather_combined_sweep_available", True),
            ("csr_master_weights_available", masters),
            ("uint8_synapse_types_available", uint8),
            ("adaptive_routing_available", adaptive),
        ):
            monkeypatch.setattr(ops, name, lambda value=value: value)
        monkeypatch.setattr(ops, "_gpu_compute_architecture", lambda: architecture)
        monkeypatch.setattr(ops, "_SM_ARCHITECTURES", FAT_BINARY_ARCHITECTURES)
        monkeypatch.setattr(ops, "cuda_op_status", lambda: {})
        monkeypatch.setattr(acceleration.glif_state_ops, "fused_nest_state_available", lambda: True)
        monkeypatch.setattr(acceleration.glif_state_ops, "glif_state_op_status", lambda: {})

    install()
    return install


def resolve(options=None, metadata=None, type_range=(0, 89), **overrides):
    parameters = dict(
        compute_dtype=tf.float16, variable_dtype=tf.float32, batch_size=32, basis_width=4,
        performance_context=context() if metadata is None else metadata,
        synapse_type_range=type_range,
    )
    parameters.update(overrides)
    return acceleration.resolve_acceleration_options(dict(options or {}), **parameters)


# ---------------------------------------------------------------------------
# Automatic resolution matrix
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("architecture", FAT_BINARY_ARCHITECTURES)
@pytest.mark.parametrize("batch", [1, 2, 7, 8, 16, 31, 32])
def test_auto_enables_all_three_on_every_target_and_batch_up_to_32(hardware, architecture, batch):
    hardware(architecture=architecture)
    options, report = resolve(batch_size=batch)
    assert options["use_combined_current_sweep"] is True
    adaptive = batch >= acceleration.ADAPTIVE_ROUTING_MIN_BATCH
    for name in NEW_OPTIONS:
        expected = adaptive if name == "use_adaptive_routing" else True
        assert options[name] is expected and report["selected"][name] is expected, name
        assert "requires" not in report["reasons"][name] or not expected, report["reasons"][name]
    if not adaptive:
        assert "batch >= 8" in report["reasons"]["use_adaptive_routing"]
    # Below the adaptive threshold the previous dense tile policy applies.
    assert options["combined_current_target_tile_size"] == (256 if batch == 1 else 2048)
    geometry = report["launch_geometry"]
    assert geometry["valid"] and geometry["threads"] == 512 and geometry["dynamic_shared_bytes"] == 0
    assert geometry["architecture"] == architecture


def test_adaptive_routing_min_batch_is_eight_and_explicit_true_is_honored(hardware):
    assert acceleration.ADAPTIVE_ROUTING_MIN_BATCH == 8
    for batch in range(1, 33):
        options, _ = resolve(batch_size=batch)
        assert options["use_adaptive_routing"] is (batch >= 8), batch
        explicit, report = resolve({"use_adaptive_routing": True}, batch_size=batch)
        assert explicit["use_adaptive_routing"] is True
        assert report["reasons"]["use_adaptive_routing"].startswith("explicit override")
        assert explicit["combined_current_target_tile_size"] == 2048


@pytest.mark.parametrize("batch", [33, 64, 128])
def test_batches_above_32_keep_previous_route_without_error(hardware, batch):
    options, report = resolve(batch_size=batch)
    assert options["use_combined_current_sweep"] is False
    for name in NEW_OPTIONS:
        assert options[name] is False, name
    assert "automatically admitted joint current sweep" in report["reasons"]["use_csr_master_weights"]
    assert "combined_current_target_tile_size" not in options


@pytest.mark.parametrize("architecture", [52, 60, 72, 100, 120, None])
def test_non_target_architectures_keep_previous_route(hardware, architecture):
    hardware(architecture=architecture)
    options, _ = resolve()
    assert not any(options[name] for name in NEW_OPTIONS)


@pytest.mark.parametrize("options,metadata,joint,reason", [
    ({"dynamics_mode": "legacy"}, None, False, "NEST dynamics"),
    ({"temporal_gradient_precision": "float32"}, None, False, "compute temporal gradients"),
    ({"train_recurrent": False}, None, True, "per-edge trainable recurrent weights"),
    ({"train_recurrent_per_type": True}, None, True, "per-edge trainable recurrent weights"),
    ({}, context(single_replica=False), False, "single-replica"),
    ({}, context(bptt_training=False), True, "BPTT learning rule"),
    ({"use_combined_current_sweep": False}, None, False, "joint current sweep"),
])
def test_unmet_prerequisites_silently_keep_previous_paths(hardware, options, metadata, joint, reason):
    resolved, report = resolve(options, metadata=metadata, batch_size=1)
    assert resolved["use_combined_current_sweep"] is joint
    for name in NEW_OPTIONS:
        assert resolved[name] is False, name
    assert reason in report["reasons"]["use_csr_master_weights"]
    if joint:
        assert resolved["combined_current_target_tile_size"] == 256


def test_missing_native_capabilities_disable_only_dependent_options(hardware):
    hardware(masters=False)
    options, report = resolve(batch_size=1)
    assert not any(options[name] for name in NEW_OPTIONS)
    assert "native CSR-master capability" in report["reasons"]["use_csr_master_weights"]
    assert options["combined_current_target_tile_size"] == 256
    hardware(uint8=False)
    options, report = resolve(batch_size=1)
    assert options["use_csr_master_weights"] is True
    assert options["use_uint8_synapse_types"] is options["use_adaptive_routing"] is False
    assert "native uint8 type capability" in report["reasons"]["use_uint8_synapse_types"]
    hardware(adaptive=False)
    options, report = resolve(batch_size=8)
    assert options["use_uint8_synapse_types"] is True and options["use_adaptive_routing"] is False
    assert "native adaptive routing capability" in report["reasons"]["use_adaptive_routing"]
    options, report = resolve(batch_size=1)
    assert options["combined_current_target_tile_size"] == 256


@pytest.mark.parametrize("type_range,fits", [
    ((0, 0), True), ((0, 255), True), ((0, 256), False), ((-1, 3), False), (None, False),
])
def test_uint8_types_only_when_every_type_id_fits(hardware, type_range, fits):
    options, report = resolve(type_range=type_range)
    assert options["use_csr_master_weights"] is True
    assert options["use_uint8_synapse_types"] is fits
    assert options["use_adaptive_routing"] is fits
    if not fits:
        assert "requires" in report["reasons"]["use_uint8_synapse_types"]


def test_explicit_false_matches_previous_resolution_exactly(hardware):
    previous, previous_report = resolve(
        {name: False for name in NEW_OPTIONS}, batch_size=1,
    )
    hardware(masters=False, uint8=False, adaptive=False)
    without_capability, report = resolve(batch_size=1)
    assert previous == without_capability
    assert previous_report["selected"] == report["selected"]
    assert previous["combined_current_target_tile_size"] == 256
    for name in NEW_OPTIONS:
        assert previous_report["reasons"][name].startswith("explicit override")


@pytest.mark.parametrize("options,match", [
    ({"use_csr_master_weights": False, "use_uint8_synapse_types": True},
     "use_uint8_synapse_types=True requires use_csr_master_weights"),
    ({"use_uint8_synapse_types": False, "use_adaptive_routing": True},
     "use_adaptive_routing=True requires use_uint8_synapse_types"),
])
def test_explicit_true_with_disabled_dependency_raises(hardware, options, match):
    with pytest.raises(ValueError, match=match):
        resolve(options)


def test_explicit_true_outside_auto_admission_raises_with_reason(hardware):
    with pytest.raises(ValueError, match="automatically admitted joint current sweep"):
        resolve({"use_uint8_synapse_types": True}, batch_size=33)
    with pytest.raises(ValueError, match="0..255"):
        resolve({"use_adaptive_routing": True}, type_range=(0, 300))


def test_dependent_options_stay_off_outside_automatic_admission(hardware):
    options, report = resolve(
        {"use_csr_master_weights": True, "use_combined_current_sweep": True}, batch_size=64,
    )
    assert options["use_csr_master_weights"] is True
    assert options["use_uint8_synapse_types"] is options["use_adaptive_routing"] is False
    assert "automatically admitted joint current sweep" in report["reasons"]["use_uint8_synapse_types"]
    explicit, _ = resolve(
        {**{name: True for name in NEW_OPTIONS}, "use_combined_current_sweep": True}, batch_size=64,
    )
    assert all(explicit[name] is True for name in NEW_OPTIONS)


def test_explicit_true_is_recorded_as_override(hardware):
    options, report = resolve({name: True for name in NEW_OPTIONS})
    for name in NEW_OPTIONS:
        assert options[name] is True
        assert report["reasons"][name].startswith("explicit override")


def test_disabled_profile_never_selects_new_options(hardware):
    options, report = resolve({"acceleration_profile": None})
    assert report is None and not any(name in options for name in NEW_OPTIONS)


def test_context_records_learning_rule_family():
    from test_s8_type_indexed_nest_coefficients import _small_network_inputs
    network, inputs = _small_network_inputs()
    for bptt in (None, True, False):
        metadata = acceleration.performance_projection_context(
            network, inputs, dt=1.0, max_delay=5.0, standard_cell=True, single_replica=True,
            bptt_training=bptt,
        )
        assert metadata["bptt_training"] is bptt


# ---------------------------------------------------------------------------
# Per-device launch geometry (pure functions of compute capability)
# ---------------------------------------------------------------------------

VALID_TILES = {
    61: (256, 512, 1024, 2048),
    70: (256, 512, 1024, 2048, 3072, 4096, 5120),
    75: (256, 512, 1024, 2048, 3072),
    80: (256, 512, 1024, 2048, 3072, 4096, 5120),
    86: (256, 512, 1024, 2048, 3072, 4096, 5120),
    89: (256, 512, 1024, 2048, 3072, 4096, 5120),
    90: (256, 512, 1024, 2048, 3072, 4096, 5120),
}


@pytest.mark.parametrize("architecture", FAT_BINARY_ARCHITECTURES)
def test_launch_geometry_validity_per_device(architecture):
    for tile in acceleration.COMBINED_TARGET_TILES:
        geometry = acceleration.combined_sweep_launch_geometry(architecture, tile)
        assert geometry["valid"] is (tile in VALID_TILES[architecture]), (tile, geometry)
        assert geometry["threads"] == (1024 if tile > 2048 else 512)
        assert geometry["static_shared_bytes"] <= 48 * 1024
        if geometry["valid"]:
            assert geometry["total_shared_bytes"] <= geometry["shared_per_block_optin"]
            assert geometry["resident_blocks_per_sm"] >= 1
    for adaptive in (False, True):
        for batch in (1, 2, 32):
            tile = acceleration.select_combined_target_tile(architecture, batch, adaptive=adaptive)
            assert tile == (256 if batch == 1 and not adaptive else 2048)
            assert acceleration.combined_sweep_launch_geometry(architecture, tile)["valid"]
            narrow = ops.adaptive_narrow_tile(tile)
            assert acceleration.combined_sweep_launch_geometry(architecture, narrow)["valid"]


@pytest.mark.parametrize("architecture,blocks", [
    (61, 2), (70, 2), (75, 1), (80, 4), (86, 2), (89, 2), (90, 4),
])
def test_wide_tile_residency_bound_per_device(architecture, blocks):
    geometry = acceleration.combined_sweep_launch_geometry(architecture, 2048, index_bytes=4)
    assert geometry["resident_blocks_per_sm"] == blocks


def test_int64_routing_uses_512_threads_for_wide_tiles():
    geometry = acceleration.combined_sweep_launch_geometry(86, 4096, index_bytes=8)
    assert geometry["threads"] == 512 and geometry["valid"]
    assert not acceleration.combined_sweep_launch_geometry(61, 4096, index_bytes=8)["valid"]
    assert not acceleration.combined_sweep_launch_geometry(None, 2048)["valid"]
    with pytest.raises(ValueError):
        acceleration.combined_sweep_launch_geometry(86, 999)


def test_scratch_bounds_cover_built_kernel_resource_usage():
    # Maximum static shared scratch (beyond tile accumulators) of every combined-sweep
    # kernel reported by ptxas -v for sm_61/70/75/80/86/89/90 at the reviewed source.
    measured = {(512, 4): 6320, (512, 8): 10416, (1024, 4): 12624}
    for key, value in measured.items():
        assert acceleration._SWEEP_SCRATCH_SHARED_BYTES[key] >= value
    assert 2048 * 16 + measured[(512, 8)] <= 48 * 1024


def test_fallback_selects_widest_valid_tile(monkeypatch):
    monkeypatch.setitem(acceleration.DEVICE_LIMITS, 999, {
        "shared_per_block_optin": 24 * 1024, "shared_per_sm": 48 * 1024,
        "threads_per_sm": 1024, "blocks_per_sm": 16, "reserved_shared_per_block": 0,
    })
    assert acceleration.select_combined_target_tile(999, 32, adaptive=True) == 1024
    assert acceleration.select_combined_target_tile(999, 1, adaptive=False) == 256
    # A routing-budget floor is never undercut, even by the device fallback.
    assert acceleration.select_combined_target_tile(999, 1, adaptive=False, minimum_tile=1024) == 1024
    assert acceleration.select_combined_target_tile(999, 1, adaptive=False, minimum_tile=2048) is None


def _routing_mib(tile, n_post, rows, index_bytes=4):
    return ((n_post + tile - 1) // tile + 1) * rows * index_bytes / 1024**2


def _develop_projection_tile(batch, n_post, rows):
    """Upstream develop 15d18e11 select_performance_projection tile (eligible case)."""
    tile = 256 if batch == 1 else 2048
    if _routing_mib(tile, n_post, rows) > 512:
        tile = 2048
    return tile


@pytest.mark.parametrize("rows,expected", [
    # 256-target routing fits 512 MiB at the full-V1 context: the batch-1 256 tile.
    (415416, 256),
    # Just inside and just outside the budget for the 256 table at n_post=120000.
    (285569, 256), (285570, 2048),
    # Review probe: 256 needs 2151 MiB, 2048 needs 275 MiB.
    (1200000, 2048),
])
@pytest.mark.parametrize("options", [
    {}, {"use_adaptive_routing": False}, {"use_csr_master_weights": False},
    {"use_uint8_synapse_types": False},
])
def test_batch1_tile_respects_the_routing_budget(hardware, rows, expected, options):
    n_post = 120000 if rows != 415416 else 66658
    metadata = context(n_post=n_post, source_rows_upper_bound=rows)
    assert (_routing_mib(256, n_post, rows) <= 512) is (expected == 256)
    resolved, report = resolve(options, metadata=metadata, batch_size=1)
    assert resolved["use_combined_current_sweep"] is True
    assert resolved["use_adaptive_routing"] is False
    tile = resolved["combined_current_target_tile_size"]
    assert tile == expected == _develop_projection_tile(1, n_post, rows)
    assert _routing_mib(tile, n_post, rows) <= 512
    assert report["launch_geometry"]["tile"] == tile and report["launch_geometry"]["valid"]


@pytest.mark.parametrize("architecture", FAT_BINARY_ARCHITECTURES)
@pytest.mark.parametrize("batch", [1, 2, 7, 8, 32])
@pytest.mark.parametrize("rows", [415416, 1200000])
def test_explicit_previous_path_reproduces_the_develop_tile(hardware, architecture, batch, rows):
    hardware(architecture=architecture)
    metadata = context(n_post=120000, source_rows_upper_bound=rows)
    for options in ({"use_csr_master_weights": False}, {"use_adaptive_routing": False}):
        resolved, _ = resolve(options, metadata=metadata, batch_size=batch)
        assert resolved["use_adaptive_routing"] is False
        assert resolved["combined_current_target_tile_size"] == _develop_projection_tile(
            batch, 120000, rows,
        )
    resolved, _ = resolve({}, metadata=metadata, batch_size=batch)
    adaptive = batch >= acceleration.ADAPTIVE_ROUTING_MIN_BATCH
    assert resolved["use_adaptive_routing"] is adaptive
    assert resolved["combined_current_target_tile_size"] == (
        2048 if adaptive else _develop_projection_tile(batch, 120000, rows)
    )


@pytest.mark.parametrize("architecture,tile,valid", [
    (61, 4096, False), (75, 4096, False), (75, 3072, True), (86, 5120, True),
])
def test_explicit_tile_is_validated_against_the_device(hardware, architecture, tile, valid):
    hardware(architecture=architecture)
    if valid:
        options, _ = resolve({"combined_current_target_tile_size": tile})
        assert options["combined_current_target_tile_size"] == tile
    else:
        with pytest.raises(ValueError, match=f"not launchable on SM{architecture}"):
            resolve({"combined_current_target_tile_size": tile})


# ---------------------------------------------------------------------------
# Cell-level explicit validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("options,match", [
    ({"dynamics_mode": "legacy"}, "requires NEST dynamics"),
    ({"train_recurrent_per_type": True}, "per-edge recurrent weights"),
])
def test_cell_explicit_true_reports_unsupported_prerequisites(options, match):
    from test_precision_credit import make_cell
    mode = options.pop("dynamics_mode", "nest")
    with pytest.raises(ValueError, match=match):
        make_cell(mode, use_csr_master_weights=True, **options)


@pytest.mark.parametrize("name", NEW_OPTIONS)
def test_cell_rejects_non_boolean_options(name):
    from test_precision_credit import make_cell
    with pytest.raises(ValueError, match=f"{name} must be true or false"):
        make_cell("nest", **{name: "yes"})


def test_bare_cell_without_context_keeps_canonical_masters():
    from test_precision_credit import make_cell
    cell = make_cell("nest", acceleration_profile="auto")
    assert not cell._csr_master_weights
    assert cell.acceleration_report["csr_master_weights"] == "disabled"
    assert cell.acceleration_report["adaptive_routing"] == "disabled"


# ---------------------------------------------------------------------------
# Packed spike checkpoints default
# ---------------------------------------------------------------------------

def _engine(rnn, **kwargs):
    from bmtk.simulator.dpointnet.training import TrainingEngine
    return TrainingEngine(rnn, 1, 1, training_approach="single", **kwargs)


def test_packing_defaults_on_for_glif3_and_off_for_other_cells():
    from bmtk.simulator.dpointnet.cell_models.glif3_cell import GLIF3Cell
    engine = _engine(SimpleNamespace(_cell=None, cell_cls=GLIF3Cell))
    assert engine.pack_spike_checkpoints is True
    assert engine.pack_spike_checkpoints_explicit is None
    assert _engine(SimpleNamespace(_cell=None, cell_cls=object)).pack_spike_checkpoints is False
    assert _engine(SimpleNamespace()).pack_spike_checkpoints is False


@pytest.mark.parametrize("value", [False, True])
def test_explicit_packing_choice_is_kept(value):
    from bmtk.simulator.dpointnet.cell_models.glif3_cell import GLIF3Cell
    engine = _engine(SimpleNamespace(_cell=None, cell_cls=GLIF3Cell), pack_spike_checkpoints=value)
    assert engine.pack_spike_checkpoints is value
    assert engine.pack_spike_checkpoints_explicit is value
    engine.pack_spike_checkpoints = "auto"
    assert engine.pack_spike_checkpoints is True
    with pytest.raises(ValueError):
        engine.pack_spike_checkpoints = "packed"


def test_packing_auto_inspects_the_built_cell_state():
    from test_precision_credit import make_cell
    cell = make_cell("nest")
    assert _engine(SimpleNamespace(_cell=cell)).pack_spike_checkpoints is True
    other = SimpleNamespace(zero_state=lambda *args, **kwargs: ((), ("v0",)))
    assert _engine(SimpleNamespace(_cell=other)).pack_spike_checkpoints is False


@pytest.mark.parametrize("value,expected", [("auto", True), (False, False), (True, True)])
def test_segmented_runner_receives_resolved_packing(value, expected):
    from test_precision_credit import make_cell
    cell = make_cell("nest")
    core = SimpleNamespace(layers=[], output=[
        tf.TensorSpec((None, 7, 1), tf.float32), tf.TensorSpec((None, 7, 1), tf.float32),
        tf.TensorSpec((None, 1), tf.float32),
    ])
    from bmtk.simulator.dpointnet.training import TrainingParameters
    engine = _engine(
        SimpleNamespace(_cell=cell, cell=cell, extractor_model=core),
        gradient_checkpointing=True, gradient_checkpoint_chunk_size=3,
        pack_spike_checkpoints=value,
    )
    engine._parameters = [TrainingParameters("test", batch_size=1, seq_len=7)]
    engine.prepare_gradient_checkpointing()
    assert engine._extractor_forward.pack_spike_checkpoints is expected


# ---------------------------------------------------------------------------
# Checkpoint compatibility between canonical and CSR-ordered masters
# ---------------------------------------------------------------------------

def _native_gpu():
    return bool(tf.config.get_visible_devices("GPU")) and ops.combined_sweep_available()


@pytest.fixture
def forward_free(monkeypatch):
    """Checkpoint tests never run a forward pass, so CPU CSR-master cells may be built
    even where the native reference kernels cannot load (for example TF-Keras 2)."""
    from bmtk.simulator.dpointnet.custom_ops import master_projection
    monkeypatch.setattr(master_projection, "reference_available", lambda: True)


def build_model(masters):
    """Small spike-input NEST cell with a trainable and a frozen named input plus Adam."""
    from test_precision_credit import make_cell
    from bmtk.simulator.dpointnet.cell_models.glif3_cell import GLIF3Cell
    gpu = _native_gpu()
    if masters and not gpu and tf.config.get_visible_devices("GPU"):
        pytest.skip("The CPU CSR-master reference requires no visible GPU.")
    network, inputs, arguments = make_cell(
        "nest", fused=gpu, two_inputs=True, return_spec=True, train_recurrent=True,
        batch_size=4, use_csr_master_weights=masters,
        **({"acceleration_profile": "auto", "_canonical_gradient_boundary": True} if gpu else {}),
    )
    for props in inputs.values():
        props["input_type"] = "spikes"
    inputs["background"]["options"]["trainable"] = False
    if gpu:
        arguments["_performance_auto_context"] = acceleration.performance_projection_context(
            network, inputs, dt=1.0, max_delay=5.0, standard_cell=True, single_replica=True,
        )
    cell = GLIF3Cell(network, inputs, **arguments)
    assert cell._csr_master_weights is masters
    surfaces = ["<recurrent>", *cell.inputs]
    trainable = [cell.weight_loss_values(name) for name in surfaces if cell.weight_loss_values(name).trainable]
    optimizer = tf.keras.optimizers.Adam(0.01)
    optimizer.build(trainable)
    bound = cell.bind_optimizer_checkpoint_slots(optimizer)
    if masters:
        assert bound == {"<recurrent>": 2, "drive": 2}
    return cell, optimizer, surfaces


def layout_of(cell, name):
    if not cell._csr_master_weights:
        return None
    return cell.recurrent_master_layout if name == "<recurrent>" else cell.inputs[name]["master_layout"]


def to_live(cell, name, canonical):
    layout = layout_of(cell, name)
    return canonical if layout is None else layout.host_to_live(canonical)


def to_canonical(cell, name, live):
    layout = layout_of(cell, name)
    return np.asarray(live) if layout is None else layout.host_to_canonical(np.asarray(live))


def canonical_slots(cell, optimizer, name):
    from bmtk.simulator.dpointnet.csr_master import paired_optimizer_slots
    master = cell.weight_loss_values(name)
    if not master.trainable:
        return []
    return [to_canonical(cell, name, slot.numpy()) for slot in paired_optimizer_slots(optimizer, master)]


def step(cell, optimizer, surfaces, seed):
    rng = np.random.default_rng(seed)
    pairs = []
    for name in surfaces:
        master = cell.weight_loss_values(name)
        if master.trainable:
            gradient = rng.normal(size=master.shape).astype(np.float32)
            pairs.append((tf.constant(to_live(cell, name, gradient)), master))
    optimizer.apply_gradients(pairs)
    cell.refresh_recurrent_weight_shadow()


def checkpoint(cell, optimizer):
    return tf.train.Checkpoint(cell=cell, optimizer=optimizer)


@pytest.mark.parametrize("source_masters,target_masters", [(False, True), (True, False), (True, True)])
def test_checkpoints_restore_across_master_layouts_with_adam_slots(forward_free, tmp_path, source_masters, target_masters):
    source, source_optimizer, surfaces = build_model(source_masters)
    step(source, source_optimizer, surfaces, 1)
    step(source, source_optimizer, surfaces, 2)
    path = checkpoint(source, source_optimizer).save(str(tmp_path / "source"))
    reader = tf.train.load_checkpoint(path)
    keys = reader.get_variable_to_shape_map()
    assert not any("CSR_MASTER" in key for key in keys)

    target, target_optimizer, _ = build_model(target_masters)
    checkpoint(target, target_optimizer).restore(path).assert_consumed()
    for name in surfaces:
        np.testing.assert_array_equal(
            target.canonical_weight_values(name), source.canonical_weight_values(name),
        )
        expected = canonical_slots(source, source_optimizer, name)
        actual = canonical_slots(target, target_optimizer, name)
        assert len(expected) == len(actual)
        for a, b in zip(actual, expected):
            assert np.any(b != 0)
            np.testing.assert_array_equal(a, b)
    assert int(target_optimizer.iterations.numpy()) == 2
    frozen = target.weight_loss_values("background")
    assert not frozen.trainable
    # The next identical canonical update continues identically after migration.
    step(source, source_optimizer, surfaces, 3)
    step(target, target_optimizer, surfaces, 3)
    for name in surfaces:
        np.testing.assert_array_equal(
            target.canonical_weight_values(name), source.canonical_weight_values(name),
        )
    # Re-saving from the target writes the same canonical checkpoint layout.
    second = checkpoint(target, target_optimizer).save(str(tmp_path / "target"))
    assert tf.train.load_checkpoint(second).get_variable_to_shape_map() == {
        key: value for key, value in tf.train.load_checkpoint(path).get_variable_to_shape_map().items()
    }


def test_saving_never_reorders_live_variables(forward_free, tmp_path):
    cell, optimizer, surfaces = build_model(True)
    step(cell, optimizer, surfaces, 7)
    before = {name: cell.weight_loss_values(name).numpy().copy() for name in surfaces}
    path = checkpoint(cell, optimizer).save(str(tmp_path / "live"))
    for name in surfaces:
        np.testing.assert_array_equal(cell.weight_loss_values(name), before[name])
        if not cell.weight_loss_values(name).trainable:
            # Frozen named inputs are not reachable from the cell's checkpoint graph in
            # either layout; trainable ones are reached through the optimizer.
            continue
        key = None
        stored = None
        for candidate, _ in tf.train.list_variables(path):
            value = tf.train.load_checkpoint(path).get_tensor(candidate)
            if isinstance(value, np.ndarray) and value.shape == before[name].shape and value.dtype == np.float32 and np.array_equal(
                value, to_canonical(cell, name, before[name])
            ):
                key, stored = candidate, value
                break
        assert key is not None, name
        np.testing.assert_array_equal(stored, cell.canonical_weight_values(name))


def test_unbound_optimizer_slots_are_reported_by_binding_errors(forward_free):
    from bmtk.simulator.dpointnet.csr_master import bind_optimizer_checkpoint_slots
    cell, _, _ = build_model(True)

    class Opaque:
        def _get_variable_index(self, variable):
            return 0
    with pytest.raises(ValueError, match="not a per-edge slot"):
        opaque = Opaque()
        opaque._momentums = [tf.Variable(np.zeros(1, np.float32))]
        bind_optimizer_checkpoint_slots(cell, opaque)


def test_keras_weight_store_round_trips_canonical_rows(forward_free):
    source, _, surfaces = build_model(True)
    for name in surfaces:
        master = source.weight_loss_values(name)
        master.assign(master * 1.5)
    store = {}
    source.save_own_variables(store)
    same, _, _ = build_model(True)
    same.load_own_variables(store)
    for name in surfaces:
        np.testing.assert_array_equal(same.weight_loss_values(name), source.weight_loss_values(name))
    for name, layout_name in (("<recurrent>", None), ("drive", "drive")):
        master = source.weight_loss_values(name)
        stored = [np.asarray(value) for value in store.values() if np.asarray(value).shape == master.shape]
        assert any(np.array_equal(value, source.canonical_weight_values(name)) for value in stored)
    target, _, _ = build_model(False)
    keras3 = hasattr(target, "_trainable_variables")
    if not keras3:
        # TF-Keras 2 weight lists also contain canonical models' compute shadows, so
        # cross-layout weight files are rejected by the variable count, never misread.
        with pytest.raises(ValueError, match="expected .* variables"):
            target.load_own_variables(store)
        return
    target.load_own_variables(store)
    for name in surfaces:
        np.testing.assert_array_equal(
            target.canonical_weight_values(name), source.canonical_weight_values(name),
        )
    back, _, _ = build_model(True)
    reverse = {}
    target.save_own_variables(reverse)
    back.load_own_variables(reverse)
    for name in surfaces:
        np.testing.assert_array_equal(back.weight_loss_values(name), source.weight_loss_values(name))


def test_emd_group_order_checkpoint_matches_canonical_models(tmp_path, monkeypatch):
    from test_csr_master_layout import fake_cell
    from bmtk.simulator.dpointnet.loss_functions import weight_regularization as wr
    groups = np.array([0, 1, 0, 2, 1], np.int64)
    monkeypatch.setattr(wr, "_connection_type_ids", lambda *args, **kwargs: groups)
    canonical_cell, _ = fake_cell(False, True)
    live_cell, _ = fake_cell(True, True)
    canonical, live = [
        wr.EMDWeightRegularization(SimpleNamespace(cell=cell, recurrent_network={}), cost=1.0)
        for cell in (canonical_cell, live_cell)
    ]
    path = tf.train.Checkpoint(order=live._group_order).save(str(tmp_path / "live"))
    stored = tf.train.load_checkpoint(path).get_tensor("order/.ATTRIBUTES/VARIABLE_VALUE")
    np.testing.assert_array_equal(stored, canonical._group_order.numpy().astype(np.int32))
    original = live._group_order.numpy().copy()
    live._group_order.assign(tf.zeros_like(live._group_order))
    tf.train.Checkpoint(order=live._group_order).restore(path).assert_consumed()
    np.testing.assert_array_equal(live._group_order, original)
    canonical_path = tf.train.Checkpoint(order=canonical._group_order).save(str(tmp_path / "canonical"))
    live._group_order.assign(tf.zeros_like(live._group_order))
    tf.train.Checkpoint(order=live._group_order).restore(canonical_path).assert_consumed()
    np.testing.assert_array_equal(live._group_order, original)
