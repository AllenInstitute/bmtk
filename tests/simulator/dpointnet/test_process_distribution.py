import sys
from types import SimpleNamespace

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from bmtk.simulator.dpointnet.distribution import (
    create_data_parallel, data_parallel_replica_id,
)
from bmtk.simulator.dpointnet.process_distribution import ProcessDataParallel
from bmtk.utils.dpointnet_mpi import select_device


class Communicator:
    rank = 0
    size = 2
    IN_PLACE = object()

    def __init__(self):
        self.reductions = []
        self.broadcasts = []
        self.scatter_in_place = []

    def allgather(self, value):
        return [value] * self.size

    def bcast(self, value, root):
        return value

    def Scatter(self, source, destination, root):
        self.scatter_in_place.append(destination is self.IN_PLACE)
        if destination is not self.IN_PLACE:
            destination[:] = source[:len(destination)]

    def Bcast(self, array, root):
        self.broadcasts.append(array.copy())

    def Allreduce(self, source, destination, op):
        assert source.flags.c_contiguous
        self.reductions.append(source.copy())
        destination[:] = source * self.size


@pytest.fixture
def parallel(monkeypatch):
    communicator = Communicator()
    from bmtk.utils.io.ioutils import bmtk_world_comm

    monkeypatch.setattr(bmtk_world_comm, "_comm", communicator)
    monkeypatch.setenv("BMTK_DPOINTNET_MPI_LAUNCH", "cpu")
    monkeypatch.setitem(sys.modules, "mpi4py", SimpleNamespace(MPI=SimpleNamespace(
        COMM_WORLD=communicator, SUM="sum", IN_PLACE=communicator.IN_PLACE,
        Get_library_version=lambda: "test MPI",
    )))
    return create_data_parallel(
        {"strategy": "mpi", "devices": ["/CPU:0"], "global_batch_size": 4},
        2, "float32",
    )


@pytest.mark.parametrize("allocation,rank,size,expected", [
    ("2,3", 0, 2, "2"), ("2,3", 1, 2, "3"),
    (" GPU-first , GPU-second ", 1, 2, "GPU-second"),
])
def test_device_selection_respects_inherited_allocation(allocation, rank, size, expected):
    assert select_device(allocation, rank, size) == expected


@pytest.mark.parametrize("allocation,size", [
    (None, 2), ("", 2), ("-1", 1), ("0", 2), ("0,0", 2), ("0,1,2", 2),
])
def test_invalid_gpu_allocation_is_not_guessed(allocation, size):
    with pytest.raises(ValueError):
        select_device(allocation, 0, size)


def test_mpi_requires_failure_safe_launcher(monkeypatch):
    monkeypatch.delenv("BMTK_DPOINTNET_MPI_LAUNCH", raising=False)
    with pytest.raises(ValueError, match="dpointnet_mpi"):
        ProcessDataParallel({"strategy": "mpi"}, 2, "float32")


def test_mirrored_does_not_require_optional_mpi(monkeypatch):
    monkeypatch.setitem(sys.modules, "mpi4py", None)
    value = create_data_parallel(
        {"strategy": "mirrored", "devices": ["/CPU:0"]}, 2, "float32"
    )
    assert value.replicas == 1


def test_process_rank_is_scoped_to_its_strategy(parallel):
    assert parallel.replicas == 2 and parallel.global_batch_size == 4
    assert len(parallel.strategy.extended.worker_devices) == 1
    from bmtk.simulator.dpointnet.distribution import _PROCESS_REPLICA_IDS

    _PROCESS_REPLICA_IDS[parallel.strategy] = 1
    with parallel.strategy.scope():
        assert data_parallel_replica_id().numpy() == 1
    assert data_parallel_replica_id().numpy() == 0


def test_sample_and_target_scatter_preserves_structure_counts_and_dtype(parallel):
    counts = np.array([[0, 3], [2, 0], [1, 4], [5, 0]], np.int32)
    targets = np.arange(4, dtype=np.float32)[:, None]
    actual = parallel.distribute_batch(([counts], [{"target": targets}]))
    np.testing.assert_array_equal(actual[0][0], counts[:2])
    np.testing.assert_array_equal(actual[1][0]["target"], targets[:2])
    assert actual[0][0].dtype == tf.int32
    assert actual[0][0].numpy().max() > 1


@pytest.mark.parametrize("value", [
    np.ones((3, 1), np.float32), np.ones((), np.float32),
])
def test_invalid_global_sample_shape_fails(parallel, value):
    with pytest.raises(ValueError, match="sample batch"):
        parallel.distribute_batch(value)


def test_nonchief_cannot_silently_replace_global_samples(parallel):
    parallel.is_chief = False
    with pytest.raises(ValueError, match="Only the chief"):
        parallel.distribute_batch(np.ones((4, 1)))


@pytest.mark.parametrize("dtype", [np.float32, np.int32])
def test_receive_buffer_reuse_preserves_prior_tensor_snapshots(parallel, dtype):
    source = np.arange(8, dtype=dtype).reshape(4, 2)
    communicated = source
    parallel.is_chief = False
    parallel.rank = parallel.comm.rank = 1
    parallel.comm.bcast = lambda value, root: (
        None, [(communicated.shape, communicated.dtype.str)]
    )
    parallel.comm.Scatter = lambda source, destination, root: np.copyto(
        destination, communicated[2:]
    )
    first = parallel.distribute_batch(None)
    buffer = parallel._sample_buffers[0]
    communicated = source + 7
    second = parallel.distribute_batch(None)
    assert parallel._sample_buffers[0] is buffer
    np.testing.assert_array_equal(first, source[2:])
    np.testing.assert_array_equal(second, source[2:] + 7)
    communicated = np.ones((4, 3), dtype=dtype)
    changed = parallel.distribute_batch(None)
    assert parallel._sample_buffers[0] is not buffer
    np.testing.assert_array_equal(first, source[2:])
    np.testing.assert_array_equal(changed, np.ones((2, 3), dtype=dtype))


def test_chief_uses_in_place_scatter_without_a_receive_copy(parallel):
    source = np.arange(8, dtype=np.float32).reshape(4, 2)
    actual = parallel.distribute_batch(source)
    assert parallel.comm.scatter_in_place == [True]
    assert parallel._sample_buffers == []
    source.fill(-99)
    np.testing.assert_array_equal(actual, [[0, 1], [2, 3]])


def test_process_run_rejects_duplicate_inference_ownership(parallel):
    from bmtk.simulator.dpointnet.rnn_model import RNN

    with pytest.raises(ValueError, match="run inference separately"):
        RNN.run(SimpleNamespace(data_parallel=parallel, _inferences=[object()]))


def test_all_canonical_surfaces_share_one_mean_reduction(parallel):
    gradients = [tf.constant([1.0, -2.0]), None, tf.constant([[3.0]])]
    actual = parallel.average_gradients(gradients)
    assert len(parallel.comm.reductions) == parallel.gradient_collectives == 1
    assert actual[1] is None
    for index in (0, 2):
        np.testing.assert_array_equal(actual[index], gradients[index])
    np.testing.assert_array_equal(parallel.comm.reductions[0], [1.0, -2.0, 3.0])


def test_single_strided_surface_still_has_a_contiguous_transport(parallel):
    gradient = np.arange(12, dtype=np.float32).reshape(3, 4)[:, ::2]
    actual = parallel.average_gradients([gradient])
    assert len(parallel.comm.reductions) == parallel.gradient_collectives == 1
    np.testing.assert_array_equal(actual[0], gradient)
    np.testing.assert_array_equal(parallel.comm.reductions[0], gradient.ravel())


def test_single_rank_keeps_tensor_gradients_without_host_snapshots(parallel, monkeypatch):
    parallel.replicas = parallel.comm.size = 1
    gradient = tf.constant([[1.0, -2.0]])
    original = np.asarray

    def reject_gradient_snapshot(value, *args, **kwargs):
        assert value is not gradient
        return original(value, *args, **kwargs)

    monkeypatch.setattr(np, "asarray", reject_gradient_snapshot)
    actual = parallel.average_gradients([gradient, None])
    assert actual[0] is gradient and actual[1] is None
    assert parallel.gradient_collectives == 0 and parallel.comm.reductions == []
    assert parallel._gradient_buffer is None


def test_gradient_receive_buffer_reuse_preserves_snapshots_and_inputs(parallel):
    gradient = np.array([1.0, -2.0, 3.0], dtype=np.float32)
    first = parallel.average_gradients([gradient, None, tf.constant([[4.0]])])
    buffer = parallel._gradient_buffer
    second = parallel.average_gradients([gradient + 5, None, tf.constant([[9.0]])])
    assert parallel._gradient_buffer is buffer
    np.testing.assert_array_equal(first[0], [1.0, -2.0, 3.0])
    np.testing.assert_array_equal(first[2], [[4.0]])
    np.testing.assert_array_equal(second[0], [6.0, 3.0, 8.0])
    np.testing.assert_array_equal(second[2], [[9.0]])
    np.testing.assert_array_equal(gradient, [1.0, -2.0, 3.0])
    assert second[1] is None and parallel.gradient_collectives == 2
    parallel.average_gradients([tf.constant([1.0])])
    assert parallel._gradient_buffer is not buffer
    np.testing.assert_array_equal(first[0], [1.0, -2.0, 3.0])


@pytest.mark.parametrize("replicas", [1, 2])
@pytest.mark.parametrize("gradient,error", [
    (None, ValueError),
    (tf.constant([float("nan")]), ValueError),
    (tf.constant([float("inf")]), ValueError),
    (tf.constant([1.0], tf.float64), TypeError),
    (tf.IndexedSlices(tf.constant([1.0]), tf.constant([0]), [2]), TypeError),
])
def test_unqualified_or_invalid_gradient_fails_before_update(
    parallel, gradient, error, replicas
):
    parallel.replicas = parallel.comm.size = replicas
    with pytest.raises(error):
        parallel.average_gradients([gradient])
    assert not parallel.comm.reductions


def test_rank_gradient_schema_mismatch_fails(parallel):
    parallel.comm.allgather = lambda schema: [schema, [None]]
    with pytest.raises(ValueError, match="gradient schema"):
        parallel.average_gradients([tf.constant([1.0])])


def test_metrics_are_reduced_separately_from_canonical_gradients(parallel):
    values = {"condition": {"rate": tf.constant(0.2)}, "__total_loss": tf.constant(0.3)}
    actual = parallel.reduce_metrics(values)
    assert parallel.gradient_collectives == 0 and len(parallel.comm.reductions) == 1
    np.testing.assert_array_equal(actual["__total_loss"], values["__total_loss"])
    np.testing.assert_array_equal(actual["condition"]["rate"], values["condition"]["rate"])


def test_host_reduced_gradient_has_local_optimizer_destination(parallel):
    from bmtk.simulator.dpointnet.training import TrainingEngine

    with parallel.strategy.scope():
        model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(1,)),
            tf.keras.layers.Dense(1, use_bias=False, kernel_initializer="ones"),
        ])
        optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)
        optimizer.build(model.trainable_variables)
    engine = object.__new__(TrainingEngine)
    engine.rnn = SimpleNamespace(
        strategy=parallel.strategy, data_parallel=parallel, model=model,
    )
    engine._optimizer = optimizer
    engine._distributed_apply_gradients([tf.constant([[2.0]])])
    np.testing.assert_allclose(model.trainable_variables[0], [[0.8]])


def test_initial_sync_includes_untracked_frozen_masters_and_noise(parallel):
    from bmtk.simulator.dpointnet.learning_rules import BPTTLearningRule
    from bmtk.simulator.dpointnet.state_modules.zero_state import ZeroStateModule

    recurrent = tf.Variable([1.0])
    trainable = tf.Variable([2.0])
    frozen = tf.Variable([3.0], trainable=False)
    refreshes = []
    cell = SimpleNamespace(
        recurrent_weight_values=recurrent,
        inputs={"drive": {"input_weight_values": trainable},
                "background": {"input_weight_values": frozen}},
        noise_seed=tf.Variable(17, dtype=tf.int64),
        noise_stream=tf.Variable(9, dtype=tf.int64),
        refresh_recurrent_weight_shadow=lambda: refreshes.append(True),
    )
    rnn = SimpleNamespace(
        cell=cell, strategy=parallel.strategy,
        model=SimpleNamespace(variables=[recurrent, trainable]), dtype=tf.float32,
    )
    with parallel.strategy.scope():
        optimizer = tf.keras.optimizers.Adam()
        optimizer.build([])
    engine = SimpleNamespace(
        rnn=rnn, optimizer=optimizer, training_approach="single", n_parameters=1,
        parameters=[SimpleNamespace(
            batch_size=2, loss_functions={"rate": SimpleNamespace(
                data_parallel_reduction="replica_mean",
            )},
        )],
        learning_rule=BPTTLearningRule(), init_state=ZeroStateModule(rnn),
    )
    parallel.synchronize(engine)
    assert any(np.array_equal(value, [3.0]) for value in parallel.comm.broadcasts)
    assert any(np.array_equal(value, 17) for value in parallel.comm.broadcasts)
    assert any(np.array_equal(value, 9) for value in parallel.comm.broadcasts)
    count = len(parallel.comm.broadcasts)
    parallel.prepare(engine)
    assert len(parallel.comm.broadcasts) == count and refreshes == [True]
