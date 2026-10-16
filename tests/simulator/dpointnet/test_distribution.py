import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from bmtk.simulator.dpointnet.distribution import DataParallel
from bmtk.simulator.dpointnet.loss_functions.target_firing_rates import TargetFiringRate
from bmtk.simulator.dpointnet.state_modules.zero_state import ZeroStateModule
from bmtk.simulator.dpointnet.learning_rules import BPTTLearningRule
from types import SimpleNamespace


@pytest.mark.parametrize("workers", [None, ("/CPU:0",), ("/CPU:0", "/CPU:1")])
def test_rollout_constants_preserve_values_and_refresh_live_scalars(
    monkeypatch, workers
):
    from bmtk.simulator.dpointnet import distribution
    from bmtk.simulator.dpointnet.cell_models.glif3_cell import GLIF3Cell

    monkeypatch.setattr(distribution, "data_parallel_worker_devices", lambda: workers)
    values = {
        "synaptic_basis_weights": tf.constant([[1.0, 0.0]]),
        "_nest_type_indices": tf.constant([0, 1], tf.int64),
        "t_ref_steps": tf.constant([2, 3], tf.int32),
        "_dt": tf.constant(1.0),
        "v_th": tf.constant(1.0),
        "_dampening_factor": tf.constant(0.05),
        "_gauss_std": tf.constant(0.28),
        "_voltage_gradient_dampening": tf.constant(0.0),
        "_recurrent_dampening": tf.constant(0.7),
        "_lr_scale": tf.constant(1.0),
        "_nest_type_first_indices": tf.constant([0, 1], tf.int32),
        "syn_decay": tf.constant([[0.8, 0.9]]),
        "psc_initial": tf.constant([[0.1, 0.2]]),
        "asc_decay": tf.constant([[0.7, 0.6]]),
        "v_reset": tf.constant([-1.0, -2.0]),
        "asc_mean": tf.constant([[0.4, 0.5]]),
        "asc_refractory_decay": tf.constant([[0.6, 0.7]]),
        "psc_voltage": tf.constant([[0.8, 0.9]]),
        "rise_voltage": tf.constant([[0.2, 0.3]]),
    }
    cell = SimpleNamespace(dynamics_mode="nest", **values)
    constants = GLIF3Cell.prepare_rollout_constants(cell)
    if not workers or len(workers) < 2:
        assert constants == {}
        return
    assert constants.keys() == values.keys()
    for name, original in values.items():
        assert constants[name].dtype == original.dtype
        np.testing.assert_array_equal(constants[name].numpy(), original.numpy())
    cell._dt = tf.constant(0.5)
    refreshed = GLIF3Cell.prepare_rollout_constants(cell)
    assert constants["_dt"].numpy() == 1.0
    assert refreshed["_dt"].numpy() == 0.5
    cell.syn_decay = tf.Variable([[0.8, 0.9]])
    live = GLIF3Cell.prepare_rollout_constants(cell)
    assert live["syn_decay"] is cell.syn_decay
    cell.syn_decay.assign([[0.6, 0.7]])
    np.testing.assert_array_equal(live["syn_decay"].numpy(), cell.syn_decay.numpy())

    @tf.function
    def traced_value():
        prepared = GLIF3Cell.prepare_rollout_constants(cell)
        assert not tf.is_symbolic_tensor(prepared["_dt"])
        assert prepared["syn_decay"] is cell.syn_decay
        return prepared["_dt"]

    assert traced_value().numpy() == 0.5


@pytest.mark.parametrize(
    "options,batch,dtype,error",
    [
        ([], 2, "float32", TypeError),
        ({"strategy": "mpi"}, 2, "float32", ValueError),
        ({"strategy": "mirrored", "typo": 1}, 2, "float32", ValueError),
        ({"strategy": "mirrored"}, 2, "float16", ValueError),
        ({"strategy": "mirrored"}, True, "float32", ValueError),
        ({"strategy": "mirrored", "devices": []}, 2, "float32", ValueError),
        (
            {"strategy": "mirrored", "devices": ["/CPU:0", "/device:CPU:0"]},
            2,
            "float32",
            ValueError,
        ),
        ({"strategy": "mirrored", "devices": ["/GPU:999"]}, 2, "float32", ValueError),
        (
            {"strategy": "mirrored", "devices": ["/job:worker/device:CPU:0"]},
            2,
            "float32",
            ValueError,
        ),
        (
            {"strategy": "mirrored", "devices": ["/CPU:0"], "global_batch_size": 3},
            2,
            "float32",
            ValueError,
        ),
    ],
)
def test_invalid_distribution(options, batch, dtype, error):
    with pytest.raises(error):
        DataParallel(options, batch, dtype)


def test_batch_shape_is_not_silently_broadcast_or_dropped():
    parallel = DataParallel(
        {"strategy": "mirrored", "devices": ["/CPU:0"]}, 2, "float32"
    )
    for value in [tf.ones((3, 1)), tf.constant(1.0)]:
        with pytest.raises((ValueError, tf.errors.InvalidArgumentError)):
            parallel.distribute_batch(value)
    actual = parallel.distribute_batch(tf.constant([[1.0], [2.0]]))
    np.testing.assert_array_equal(
        parallel.strategy.experimental_local_results(actual)[0], [[1], [2]]
    )


def test_json_config_preserves_explicit_local_and_global_batches(monkeypatch):
    from bmtk.simulator.dpointnet.rnn_model import RNN, NetworkAdaptor, SimulationConfig

    options = {"strategy": "mirrored", "devices": ["/CPU:0"], "global_batch_size": 2}
    config = SimulationConfig(
        json.loads(
            json.dumps(
                {
                    "run": {
                        "seq_len": 9,
                        "batch_size": 2,
                        "dtype": "float32",
                        "distribution": options,
                    },
                    "rnn_cell_params": {"cell_model": "GLIF3Cell"},
                    "networks": {},
                    "inputs": {},
                }
            )
        )
    )
    monkeypatch.setattr(NetworkAdaptor, "from_dict", lambda _: ([], []))
    rnn = RNN.from_config(config)
    assert rnn.batch_size == 2
    assert rnn.data_parallel.global_batch_size == 2
    assert rnn.data_parallel.local_batch_size == 2
    assert config.run["distribution"] == options


@pytest.mark.parametrize("units", ["probability", "", None])
def test_invalid_rate_units(units):
    with pytest.raises(ValueError, match="rate_units"):
        TargetFiringRate(None, firing_rate=0.2, rate_units=units)


def test_rate_loss_matches_independent_value_and_gradient():
    spikes = tf.Variable(np.arange(24).reshape(2, 3, 4) / 30, dtype=tf.float32)
    with tf.GradientTape(persistent=True) as tape:
        actual = TargetFiringRate(
            SimpleNamespace(dt=1.0, seq_len=3),
            firing_rate=0.2,
            rate_units="spikes_per_step",
        )(spikes)
        expected = tf.reduce_mean(tf.square(tf.reduce_mean(spikes, axis=1) - 0.2))
    np.testing.assert_allclose(actual, expected, rtol=1e-6)
    np.testing.assert_allclose(
        tape.gradient(actual, spikes), tape.gradient(expected, spikes), rtol=1e-6
    )


def test_rate_loss_default_hz_is_unchanged():
    spikes = tf.constant([[[0.0], [1.0], [1.0]]])
    loss = TargetFiringRate(SimpleNamespace(dt=0.5, seq_len=3), firing_rate=5.0)
    expected = tf.reduce_mean(tf.square(tf.reduce_sum(spikes, axis=1) / 0.0015 - 5.0))
    np.testing.assert_array_equal(loss(spikes), expected)


def test_native_gpu_selection_uses_strategy_subset(monkeypatch):
    from bmtk.simulator.dpointnet import distribution
    from bmtk.simulator.dpointnet.custom_ops import csr_spike_ops as ops

    visible = [SimpleNamespace(name=f"physical:{index}") for index in range(4)]
    monkeypatch.setattr(tf.config, "get_visible_devices", lambda kind: visible)
    monkeypatch.setattr(
        distribution,
        "data_parallel_worker_devices",
        lambda: ("/device:GPU:1", "/device:GPU:3"),
    )
    assert ops._selected_physical_gpus() == [visible[1], visible[3]]
    monkeypatch.setattr(distribution, "data_parallel_worker_devices", lambda: ())
    assert ops._selected_physical_gpus() == []


@pytest.mark.parametrize(
    "capabilities,targets,ptx,message",
    [
        ([(7, 0), (7, 0)], (61, 70, 75), None, None),
        ([(7, 0), (8, 0)], (70, 80), None, "homogeneous"),
        ([(7, 0), (7, 0)], (80,), None, "incompatible"),
        ([(9, 0), (9, 0)], (), 90, None),
        ([(8, 9), (8, 9)], (), 90, "incompatible"),
        ([None, (7, 0)], (70,), None, "unavailable"),
    ],
)
def test_native_checks_every_selected_architecture(
    monkeypatch, capabilities, targets, ptx, message
):
    from bmtk.simulator.dpointnet.custom_ops import csr_spike_ops as ops

    devices = [SimpleNamespace(name=str(index)) for index in range(2)]
    monkeypatch.setattr(ops, "_selected_physical_gpus", lambda: devices)
    monkeypatch.setattr(
        tf.config.experimental,
        "get_device_details",
        lambda device: {"compute_capability": capabilities[int(device.name)]},
    )
    error = ops._gpu_compatibility_error(
        sm_architectures=targets,
        ptx_architecture=ptx,
        architecture_path=Path("fixture.archs"),
    )
    if message is None:
        assert error is None
    else:
        assert message in error


@pytest.mark.parametrize("local_arrays", [False, True])
def test_replica_metadata_selection_and_idempotent_cleanup(monkeypatch, local_arrays):
    from bmtk.simulator.dpointnet.custom_ops import csr_spike_ops as ops

    destroyed = []
    monkeypatch.setattr(ops, "_destroy_metadata_resource", destroyed.append)
    values = {"metadata_handle": "first", "_replica_handles": ("first", "second")}
    keys = ("incoming_pre_ids", "incoming_edge_ids", "incoming_types")
    if local_arrays:
        replicas = tuple(
            {"metadata_handle": handle, **{key: (index, key) for key in keys}}
            for index, handle in enumerate(("first", "second"))
        )
        values.update(replicas[0])
        values["_replica_values"] = replicas
    connectivity = ops.CsrConnectivity(values)
    for index, expected in enumerate(("first", "second")):
        monkeypatch.setattr(
            tf.distribute,
            "get_replica_context",
            lambda index=index: SimpleNamespace(
                replica_id_in_sync_group=tf.constant(index)
            ),
        )
        assert connectivity["metadata_handle"] == expected
        if local_arrays:
            assert [connectivity[key] for key in keys] == [(index, key) for key in keys]
    monkeypatch.setattr(tf.distribute, "get_replica_context", lambda: None)
    assert connectivity["metadata_handle"] == "first"
    connectivity.close()
    connectivity.close()
    assert destroyed == ["first", "second"]


def test_rollout_constants_restore_nested_context_and_live_values():
    from bmtk.simulator.dpointnet.cell_models.state_rnn import (
        _rollout_constant,
        _rollout_context,
    )

    class Cell:
        marker = 1
        _rollout_nest_coefficients = None

        def prepare_rollout_constants(self):
            return {"marker": self.marker}

    cell = Cell()
    with pytest.raises(RuntimeError, match="restore"):
        with _rollout_context(cell):
            cell.marker = 2
            assert _rollout_constant(cell, "marker") == 1
            with _rollout_context(cell):
                assert _rollout_constant(cell, "marker") == 2
            assert _rollout_constant(cell, "marker") == 1
            raise RuntimeError("restore")
    assert _rollout_constant(cell, "marker") == 2


def test_rollout_constants_are_isolated_between_replica_threads():
    from concurrent.futures import ThreadPoolExecutor
    import threading
    from bmtk.simulator.dpointnet.cell_models.state_rnn import (
        _rollout_constant,
        _rollout_context,
    )

    class Cell:
        marker = 0
        _rollout_nest_coefficients = None

        def prepare_rollout_constants(self):
            return {"marker": threading.get_ident()}

    cell = Cell()
    barrier = threading.Barrier(2)

    def check():
        with _rollout_context(cell):
            barrier.wait(timeout=10)
            assert _rollout_constant(cell, "marker") == threading.get_ident()

    with ThreadPoolExecutor(max_workers=2) as executor:
        list(executor.map(lambda _: check(), range(2)))
    assert _rollout_constant(cell, "marker") == 0


@pytest.mark.parametrize(
    "unsupported", ["condition", "batch", "loss", "state", "clip", "rule"]
)
def test_unqualified_training_semantics_are_rejected(unsupported):
    parallel = DataParallel(
        {"strategy": "mirrored", "devices": ["/CPU:0"]}, 2, "float32"
    )
    engine = SimpleNamespace(
        training_approach="single",
        n_parameters=1,
        parameters=[
            SimpleNamespace(
                batch_size=2,
                loss_functions={
                    "rate": TargetFiringRate(
                        SimpleNamespace(dt=1.0, seq_len=3), firing_rate=5.0
                    )
                },
            )
        ],
        learning_rule=BPTTLearningRule(),
        init_state=ZeroStateModule(SimpleNamespace(dtype=tf.float32)),
        optimizer=SimpleNamespace(),
    )
    if unsupported == "condition":
        engine.training_approach = "parallel"
    elif unsupported == "batch":
        engine.parameters[0].batch_size = 3
    elif unsupported == "loss":
        engine.parameters[0].loss_functions = {"nonlinear": lambda **kwargs: 0}
    elif unsupported == "state":
        engine.init_state = object()
    elif unsupported == "clip":
        engine.optimizer.clipnorm = 1.0
    elif unsupported == "rule":
        engine.learning_rule = object()
    with pytest.raises(ValueError):
        parallel.validate_training(engine)


@pytest.mark.parametrize("spike_input", [False, True])
def test_two_replica_glif_training_in_fresh_process(spike_input):
    environment = dict(
        os.environ, CUDA_VISIBLE_DEVICES="", BMTK_DPOINTNET_DISABLE_FUSED_CUDA="1"
    )
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).with_name("data_parallel_probe.py")),
            *(["--spike-input"] if spike_input else []),
        ],
        env=environment,
        check=True,
        timeout=180,
    )
