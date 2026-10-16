"""Single-host, one-process-per-device sample parallelism using host MPI."""

import os
import socket

import numpy as np
import tensorflow as tf

from .distribution import DataParallel, _global_batch_size, _PROCESS_REPLICA_IDS
from bmtk.utils.io.ioutils import bmtk_world_comm


class ProcessDataParallel(DataParallel):
    process_based = True

    def __init__(self, options, local_batch_size, dtype):
        mode = os.environ.get("BMTK_DPOINTNET_MPI_LAUNCH")
        if mode not in ("cpu", "gpu"):
            raise ValueError(
                "MPI training requires mpiexec -n N python -m "
                "bmtk.utils.dpointnet_mpi training_script.py [arguments]."
            )
        from mpi4py import MPI

        self.comm = bmtk_world_comm.comm
        if self.comm is None:
            raise RuntimeError("DPointNet MPI launch requires an MPI communicator.")
        self._sum = MPI.SUM
        self._in_place = MPI.IN_PLACE
        self.rank = self.comm.rank
        self.is_chief = self.rank == 0
        if len(set(self.comm.allgather(socket.gethostname()))) != 1:
            raise ValueError("MPI training currently supports one host only.")
        devices = options.get("devices", ["/GPU:0" if mode == "gpu" else "/CPU:0"])
        if (
            not isinstance(devices, (list, tuple))
            or len(devices) != 1
            or not isinstance(devices[0], str)
        ):
            raise ValueError("MPI requires exactly one local device per process.")
        expected_type = "GPU" if mode == "gpu" else "CPU"
        spec = tf.DeviceSpec.from_string(devices[0])
        if spec.device_type != expected_type or spec.device_index != 0:
            raise ValueError(f"MPI launch requires /{expected_type}:0 in each process.")
        if mode == "gpu" and len(tf.config.list_physical_devices("GPU")) != 1:
            raise ValueError("MPI training requires exactly one visible GPU per process.")
        super().__init__(
            {**options, "strategy": "mirrored", "devices": devices,
             "global_batch_size": local_batch_size},
            local_batch_size,
            dtype,
        )
        self.replicas = self.comm.size
        self.global_batch_size = _global_batch_size(
            options, self.local_batch_size, self.replicas
        )
        self._check_schema(
            (self.local_batch_size, self.global_batch_size, mode), "batch/device"
        )
        _PROCESS_REPLICA_IDS[self.strategy] = self.rank
        self._prepared_engine = None
        self._sample_specifications = None
        self._sample_buffers = []
        self._gradient_buffer = None
        self._gpu_upload = mode == "gpu"
        self.gradient_collectives = 0
        self.report.update(
            strategy="mpi", rank=self.rank, replicas=self.replicas,
            global_batch_size=self.global_batch_size,
            input_policy="chief-generated global host batch; contiguous sample/target Scatter",
            gradient_transport="one packed canonical FP32 host-MPI Allreduce per update",
            mpi_library=MPI.Get_library_version().splitlines()[0],
            callbacks="chief only; early-stop decision broadcast",
        )

    def _check_schema(self, schema, label):
        schemas = self.comm.allgather(schema)
        if any(value != schemas[0] for value in schemas):
            raise ValueError(f"MPI {label} schema differs between ranks.")

    def _local_tensor(self, value):
        if tf.is_tensor(value):
            tensor = value
        elif (
            self._gpu_upload and value.dtype == np.float32
            and value.flags.c_contiguous and value.flags.writeable
        ):
            tensor = tf.experimental.dlpack.from_dlpack(value.__dlpack__())
        else:
            tensor = tf.convert_to_tensor(value)
        with tf.device(self.strategy.extended.worker_devices[0]):
            return tf.identity(tensor)

    def distribute_batch(self, values):
        if not self.is_chief and values is not None:
            raise ValueError("Only the chief supplies the global MPI sample batch.")
        arrays = None
        payload = None
        if self.is_chief:
            arrays = [np.asarray(value) for value in tf.nest.flatten(values)]
            for value in arrays:
                if value.ndim < 1 or value.shape[0] != self.global_batch_size:
                    raise ValueError("Global MPI sample batch does not match distribution.")
                if value.dtype.kind not in "biufc":
                    raise TypeError("MPI sample leaves must have numeric or Boolean dtypes.")
            payload = (
                tf.nest.map_structure(lambda _: None, values),
                [(value.shape, value.dtype.str) for value in arrays],
            )
        structure, specifications = self.comm.bcast(payload, root=0)
        if (
            self.replicas > 1 and not self.is_chief
            and specifications != self._sample_specifications
        ):
            self._sample_buffers = [
                np.empty((self.local_batch_size, *shape[1:]), dtype=dtype)
                for shape, dtype in specifications
            ]
            self._sample_specifications = specifications
        local_values = []
        original = tf.nest.flatten(values) if self.replicas == 1 else None
        # Finish uploads before reusable receive buffers or caller arrays can change.
        with tf.experimental.async_scope():
            for index, (shape, dtype) in enumerate(specifications):
                if self.replicas == 1:
                    local = arrays[index]
                elif self.is_chief:
                    source = np.ascontiguousarray(arrays[index])
                    self.comm.Scatter(source, self._in_place, root=0)
                    local = source[:self.local_batch_size]
                else:
                    local = self._sample_buffers[index]
                    self.comm.Scatter(None, local, root=0)
                value = original[index] if original is not None and tf.is_tensor(
                    original[index]
                ) else local
                local_values.append(self._local_tensor(value))
        return tf.nest.pack_sequence_as(structure, local_values)

    def synchronize(self, engine):
        """Broadcast chief masters/optimizer/noise, including explicitly frozen inputs."""
        self.validate_training(engine)
        cell = engine.rnn.cell
        optimizer_variables = engine.optimizer.variables
        if callable(optimizer_variables):
            optimizer_variables = optimizer_variables()
        variables = (
            list(engine.rnn.model.variables)
            + [cell.recurrent_weight_values]
            + [net["input_weight_values"] for net in cell.inputs.values()]
            + [cell.noise_seed, cell.noise_stream]
            + list(optimizer_variables)
        )
        variables = list({id(value): value for value in variables}.values())
        self._check_schema(
            [(tuple(value.shape), tf.as_dtype(value.dtype).name) for value in variables],
            "master/optimizer/noise",
        )
        for variable in variables:
            array = variable.numpy() if self.is_chief else np.empty(
                tuple(variable.shape), dtype=tf.as_dtype(variable.dtype).as_numpy_dtype
            )
            if self.replicas > 1:
                self.comm.Bcast(array, root=0)
            variable.assign(array)
        engine.rnn.strategy.run(cell.refresh_recurrent_weight_shadow)
        self._prepared_engine = engine

    def prepare(self, engine):
        if self._prepared_engine is not engine:
            self.synchronize(engine)

    def average_gradients(self, gradients):
        arrays = []
        schema = []
        for gradient in gradients:
            if gradient is None:
                schema.append(None)
                continue
            if isinstance(gradient, tf.IndexedSlices):
                raise TypeError("MPI training currently requires dense canonical gradients.")
            array = gradient if self.replicas == 1 and tf.is_tensor(
                gradient
            ) else np.asarray(gradient)
            if tf.as_dtype(array.dtype) != tf.float32:
                raise TypeError("MPI canonical gradient reduction requires FP32.")
            arrays.append(array)
            schema.append(tuple(array.shape))
        self._check_schema(schema, "gradient")
        if not arrays:
            raise ValueError("MPI update has no connected trainable gradients.")
        if self.replicas == 1:
            tensors = [tf.convert_to_tensor(value) for value in arrays]
            with tf.device(self.strategy.extended.worker_devices[0]):
                finite = tf.reduce_all(tf.stack([
                    tf.reduce_all(tf.math.is_finite(value)) for value in tensors
                ]))
            if not bool(finite.numpy()):
                raise ValueError("Nonfinite local canonical MPI gradient.")
            connected = iter(tensors)
            return [None if shape is None else next(connected) for shape in schema]
        packed = (
            np.ascontiguousarray(arrays[0]).reshape(-1) if len(arrays) == 1
            else np.concatenate([value.reshape(-1) for value in arrays])
        )
        if not np.isfinite(packed).all():
            raise ValueError("Nonfinite local canonical MPI gradient.")
        if self._gradient_buffer is None or self._gradient_buffer.shape != packed.shape:
            self._gradient_buffer = np.empty_like(packed)
        self.comm.Allreduce(packed, self._gradient_buffer, op=self._sum)
        self._gradient_buffer /= np.float32(self.replicas)
        packed = self._gradient_buffer
        self.gradient_collectives += 1
        if not np.isfinite(packed).all():
            raise ValueError("Nonfinite reduced canonical MPI gradient.")
        result = []
        offset = 0
        with tf.experimental.async_scope():
            for shape in schema:
                if shape is None:
                    result.append(None)
                    continue
                size = int(np.prod(shape, dtype=np.int64))
                result.append(self._local_tensor(
                    packed[offset:offset + size].reshape(shape)
                ))
                offset += size
        return result

    def reduce_metrics(self, values):
        leaves = tf.nest.flatten(values)
        arrays = [np.asarray(value) for value in leaves]
        if any(value.shape or value.dtype.kind != "f" for value in arrays):
            raise ValueError("MPI loss metrics must be floating-point scalars.")
        self._check_schema([value.dtype.str for value in arrays], "metric")
        packed = np.asarray(arrays, dtype=np.float64)
        if self.replicas > 1:
            reduced = np.empty_like(packed)
            self.comm.Allreduce(packed, reduced, op=self._sum)
            packed = reduced / self.replicas
        return tf.nest.pack_sequence_as(
            values,
            [tf.convert_to_tensor(value, dtype=tf.convert_to_tensor(original).dtype)
             for value, original in zip(packed, leaves)],
        )

    def broadcast_stop(self, stop):
        return self.comm.bcast(bool(stop) if self.is_chief else None, root=0)
