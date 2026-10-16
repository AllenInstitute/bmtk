"""Explicit, single-host sample parallelism without changing legacy batch semantics."""

from collections.abc import Mapping
from numbers import Integral
import weakref

import tensorflow as tf

_DATA_PARALLEL_STRATEGIES = weakref.WeakSet()
_PROCESS_REPLICA_IDS = weakref.WeakKeyDictionary()


def data_parallel_replica_id():
    strategy = tf.distribute.get_strategy()
    if strategy in _PROCESS_REPLICA_IDS:
        return tf.constant(_PROCESS_REPLICA_IDS[strategy], tf.int32)
    context = tf.distribute.get_replica_context()
    return (
        tf.constant(0, tf.int32)
        if context is None
        else tf.cast(context.replica_id_in_sync_group, tf.int32)
    )


def create_data_parallel(options, local_batch_size, dtype):
    if isinstance(options, Mapping) and options.get("strategy") == "mpi":
        from .process_distribution import ProcessDataParallel

        return ProcessDataParallel(options, local_batch_size, dtype)
    return DataParallel(options, local_batch_size, dtype)


def _global_batch_size(options, local_batch_size, replicas):
    global_batch = local_batch_size * replicas
    requested = options.get("global_batch_size", global_batch)
    if (
        isinstance(requested, bool)
        or not isinstance(requested, Integral)
        or requested != global_batch
    ):
        raise ValueError(
            "global_batch_size must equal run.batch_size times replica count "
            f"({local_batch_size} * {replicas} = {global_batch})."
        )
    return global_batch


def data_parallel_worker_devices():
    strategy = tf.distribute.get_strategy()
    if strategy not in _DATA_PARALLEL_STRATEGIES:
        return ()
    return tuple(strategy.extended.worker_devices)


class DataParallel:
    """Equal local batches, one optimizer update, and explicit loss admission."""

    def __init__(self, options, local_batch_size, dtype):
        if not isinstance(options, Mapping):
            raise TypeError("run.distribution must be a mapping.")
        unknown = set(options) - {"strategy", "devices", "global_batch_size"}
        if unknown:
            raise ValueError(f"Unknown run.distribution options: {sorted(unknown)}")
        if options.get("strategy") != "mirrored":
            raise ValueError('run.distribution.strategy must be "mirrored".')
        if tf.as_dtype(dtype) != tf.float32:
            raise ValueError("Data-parallel training currently requires full FP32.")
        if (
            isinstance(local_batch_size, bool)
            or not isinstance(local_batch_size, Integral)
            or local_batch_size < 1
        ):
            raise ValueError("run.batch_size must be a positive per-replica integer.")
        devices = options.get("devices")
        if devices is not None:
            if (
                not isinstance(devices, (list, tuple))
                or not devices
                or any(not isinstance(device, str) for device in devices)
            ):
                raise ValueError(
                    "run.distribution.devices must be a nonempty device list."
                )
            logical = {device.name for device in tf.config.list_logical_devices()}
            resolved = []
            for device in devices:
                spec = tf.DeviceSpec.from_string(device)
                if (
                    spec.job is not None
                    or spec.task is not None
                    or spec.replica is not None
                ):
                    raise ValueError("Data parallelism supports local devices only.")
                if spec.device_type not in ("CPU", "GPU") or spec.device_index is None:
                    raise ValueError(
                        f"Explicit CPU/GPU device ordinal required: {device}"
                    )
                name = f"/device:{spec.device_type}:{spec.device_index}"
                if name not in logical:
                    raise ValueError(f"Requested device is not visible: {device}")
                resolved.append(name)
            if len(set(resolved)) != len(resolved):
                raise ValueError("Data-parallel devices must be distinct.")
            if len({tf.DeviceSpec.from_string(d).device_type for d in resolved}) != 1:
                raise ValueError(
                    "Data-parallel devices must have the same device type."
                )
            devices = resolved
        self.strategy = tf.distribute.MirroredStrategy(devices=devices)
        self.replicas = self.strategy.num_replicas_in_sync
        self.local_batch_size = int(local_batch_size)
        self.global_batch_size = _global_batch_size(
            options, self.local_batch_size, self.replicas
        )
        _DATA_PARALLEL_STRATEGIES.add(self.strategy)
        self.report = {
            "strategy": "mirrored",
            "devices": list(self.strategy.extended.worker_devices),
            "replicas": self.replicas,
            "local_batch_size": self.local_batch_size,
            "global_batch_size": self.global_batch_size,
            "accumulation_steps": 1,
            "input_policy": "one global batch, source-local slices placed once on each replica; generator RNG unchanged",
            "precision": "float32",
        }

    def distribute_batch(self, values):
        """Split every tensor leaf on its sample axis; never broadcast sample data."""

        def split(value):
            if isinstance(value, tf.distribute.DistributedValues):
                raise ValueError(
                    "Expected global tensors, not already distributed inputs."
                )
            value = tf.convert_to_tensor(value)
            if value.shape.rank is None or value.shape.rank < 1:
                raise ValueError(
                    "Distributed sample tensors require a batch dimension."
                )
            tf.debugging.assert_equal(
                tf.shape(value)[0],
                self.global_batch_size,
                message="Global sample batch does not match run.distribution.",
            )
            def local_value(context):
                replica = context.replica_id_in_sync_group
                with tf.device(value.device or "/CPU:0"):
                    local = value[
                        replica * self.local_batch_size : (replica + 1) * self.local_batch_size
                    ]
                # Nested temporal functions must not repeatedly fetch a peer's full batch.
                with tf.device(self.strategy.extended.worker_devices[replica]):
                    return tf.identity(local)

            return self.strategy.experimental_distribute_values_from_function(local_value)

        return tf.nest.map_structure(split, values)

    def reduce_metrics(self, values):
        def reduce(value):
            if isinstance(value, tf.distribute.DistributedValues):
                return self.strategy.reduce(
                    tf.distribute.ReduceOp.MEAN, value, axis=None
                )
            return value

        return tf.nest.map_structure(reduce, values)

    def validate_training(self, engine):
        if engine.training_approach != "single" or engine.n_parameters != 1:
            raise ValueError(
                "Data parallelism currently supports one condition with "
                'training_approach="single"; condition series/parallel/accumulation '
                "are not sample parallelism."
            )
        if engine.parameters[0].batch_size != self.local_batch_size:
            raise ValueError(
                "The training parameter batch_size must equal run.batch_size per replica."
            )
        from .learning_rules import BPTTLearningRule

        if not isinstance(engine.learning_rule, BPTTLearningRule):
            raise ValueError("Data parallelism currently supports BPTT only.")
        from .state_modules.zero_state import ZeroStateModule

        if not isinstance(engine.init_state, ZeroStateModule):
            raise ValueError(
                "Data parallelism currently requires zero_state initialization; "
                "warmup and cached-state ownership need separate qualification."
            )
        for mode in ("clipnorm", "clipvalue", "global_clipnorm"):
            if getattr(engine.optimizer, mode, None) is not None:
                raise ValueError(
                    "Data-parallel clipping must be applied after global gradient "
                    f"reduction; optimizer {mode} is not yet qualified."
                )
        losses = engine.parameters[0].loss_functions
        if not losses:
            raise ValueError("Data-parallel training requires an explicit loss.")
        for name, loss in losses.items():
            if getattr(loss, "data_parallel_reduction", None) != "replica_mean":
                raise ValueError(
                    f'Loss "{name}" has no qualified data-parallel reduction. '
                    "Nonlinear batch statistics cannot be averaged across local losses."
                )
