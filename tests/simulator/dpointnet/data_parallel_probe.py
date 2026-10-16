"""Independent full-batch oracle and actual multi-replica GLIF training probe."""

import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace

import numpy as np
import tensorflow as tf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--native", action="store_true")
    parser.add_argument("--spike-input", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    steps = 17 if args.spike_input else 9
    if args.gpu:
        if len(tf.config.list_physical_devices("GPU")) != 2:
            raise AssertionError(
                "Physical probe requires exactly two allocated visible GPUs"
            )
        devices = ["/GPU:0", "/GPU:1"]
        for device in tf.config.list_physical_devices("GPU"):
            tf.config.experimental.set_memory_growth(device, True)
    else:
        physical = tf.config.list_physical_devices("CPU")[0]
        tf.config.set_logical_device_configuration(
            physical,
            [
                tf.config.LogicalDeviceConfiguration(),
                tf.config.LogicalDeviceConfiguration(),
            ],
        )
        devices = ["/CPU:0", "/CPU:1"]

    from test_precision_credit import make_cell
    from bmtk.simulator.dpointnet.rnn_model import RNN
    from bmtk.simulator.dpointnet.loss_functions.target_firing_rates import (
        TargetFiringRate,
    )
    from bmtk.simulator.dpointnet.state_modules.zero_state import ZeroStateModule

    def create(batch, selected):
        network, inputs, options = make_cell(
            "nest",
            selective=False,
            two_inputs=True,
            return_spec=True,
            batch_size=batch,
            use_fused_cuda=args.native if selected else False,
            use_fused_state=args.native if selected else False,
            use_direct_state_rnn_loop=args.native if selected else False,
            use_direct_csr_recurrent_gradient=args.native if selected else False,
            use_packed_sm120_backward=False,
            use_packed_sm120_external_backward=False,
            hard_reset=True,
            hard_reset_gradient_mode="soft_surrogate",
            detach_reset=True,
            detach_asc_reset=False,
        )
        inputs["background"]["options"]["trainable"] = False
        if args.spike_input:
            inputs["drive"]["input_type"] = "spikes"
            inputs["drive"]["weights"] *= 0.1
        options.pop("batch_size")
        rnn = RNN(
            seq_len=steps,
            batch_size=batch,
            dtype="float32",
            cell_params=options,
            distribution=(
                {"strategy": "mirrored", "devices": devices, "global_batch_size": 4}
                if selected
                else {
                    "strategy": "mirrored",
                    "devices": devices[:1],
                    "global_batch_size": 4,
                }
            ),
        )
        rnn._recurrent_networks["test"] = SimpleNamespace(
            to_dict=lambda: copy.deepcopy(network)
        )
        for name, spec in inputs.items():
            rnn._input_networks[name] = SimpleNamespace(
                name=name,
                n_spiking_nodes=2,
                to_dict=lambda spec=spec: copy.deepcopy(spec),
            )
            rnn._inputs_order.append(name)
        engine = rnn.set_training(
            rnn=rnn,
            n_epochs=1,
            steps_per_epoch=1,
            training_approach="single",
            gradient_checkpointing=selected,
            gradient_checkpoint_chunk_size=3,
        )
        parameter = engine.add_parameters("test", batch_size=batch, seq_len=steps)
        parameter.add_loss_function(
            "rate",
            TargetFiringRate(rnn, firing_rate=0.005, rate_units="spikes_per_step"),
        )
        engine.set_init_state(ZeroStateModule(rnn))
        with rnn.strategy.scope():
            engine.set_optimizer(tf.keras.optimizers.Adam(learning_rate=1e-4))
        rnn.build(training=True)
        with rnn.strategy.scope():
            rnn.extractor_model = rnn._build_extractor_model()
            engine.optimizer.build(rnn.model.trainable_variables)
        engine.prepare_gradient_checkpointing()
        return rnn

    reference = create(4, False)
    distributed = create(2, True)

    def check_shadows():
        cell = distributed.cell
        tf.debugging.assert_equal(
            cell.recurrent_weight_values_compute,
            tf.cast(cell.recurrent_weight_values, cell.compute_dtype),
        )
        if args.native:
            from bmtk.simulator.dpointnet.custom_ops.csr_spike_ops import (
                reorder_csr_values,
            )

            tf.debugging.assert_equal(
                cell.recurrent_csr_weight_values_compute,
                reorder_csr_values(
                    cell.recurrent_weight_values_compute,
                    cell.recurrent_fused_connectivity,
                ),
            )
        for net in cell.inputs.values():
            tf.debugging.assert_equal(
                net["input_weight_values_compute"],
                tf.cast(net["input_weight_values"], cell.compute_dtype),
            )
            if args.native and net["input_type"] != "current":
                tf.debugging.assert_equal(
                    net["csr_weight_values_compute"],
                    reorder_csr_values(
                        net["input_weight_values_compute"], net["fused_connectivity"]
                    ),
                )

    try:
        for expected, actual in zip(
            reference.model.trainable_variables, distributed.model.trainable_variables
        ):
            np.testing.assert_array_equal(expected.numpy(), actual.numpy())
        rng = np.random.default_rng(17)
        values = rng.uniform(0.2, 1.5, (4, steps, 4))
        if args.spike_input:
            values[:, :, :2] = rng.integers(0, 4, (4, steps, 2))
        x = tf.constant(values, tf.float32)
        local_x = distributed.data_parallel.distribute_batch([x])
        parts = distributed.strategy.experimental_local_results(local_x[0])
        np.testing.assert_array_equal(parts[0], x[:2])
        np.testing.assert_array_equal(parts[1], x[2:])
        for index, part in enumerate(parts):
            device = tf.DeviceSpec.from_string(part.device)
            if (
                device.device_type != ("GPU" if args.gpu else "CPU")
                or device.device_index != index
            ):
                raise AssertionError(
                    f"Replica {index} input is not resident locally: {part.device}"
                )
        if np.array_equal(parts[0], parts[1]):
            raise AssertionError("Replica inputs are accidentally duplicated")
        local_targets = distributed.data_parallel.distribute_batch(
            [{"sample": tf.range(4)}]
        )
        for index, target in enumerate(
            distributed.strategy.experimental_local_results(local_targets[0]["sample"])
        ):
            np.testing.assert_array_equal(target, np.arange(index * 2, index * 2 + 2))
        ref_state = reference.zero_state
        states = distributed.strategy.run(
            distributed.training_engine.init_state.get_state
        )
        constants = distributed.strategy.run(distributed.cell.prepare_rollout_constants)
        constant_devices = {}
        for name, values in constants.items():
            constant_parts = distributed.strategy.experimental_local_results(values)
            constant_devices[name] = [part.device for part in constant_parts]
            for index, part in enumerate(constant_parts):
                expected = f"/device:{'GPU' if args.gpu else 'CPU'}:{index}"
                if not part.device.endswith(expected):
                    raise AssertionError(
                        f"Rollout constant {name} is not local: {part.device}"
                    )
                np.testing.assert_array_equal(
                    part.numpy(), getattr(distributed.cell, name).numpy()
                )
        loss_fn = TargetFiringRate(
            reference, firing_rate=0.005, rate_units="spikes_per_step"
        )

        def gradient(rnn, inputs, state):
            with tf.GradientTape() as tape:
                outputs = rnn.training_engine._run_extractor(inputs, state)
                loss = loss_fn(outputs[0][0])
                scaled = tf.nn.scale_regularization_loss(loss)
            grads = tape.gradient(scaled, rnn.model.trainable_variables)
            if any(grad is None for grad in grads):
                raise AssertionError("Disconnected trainable surface")
            return loss, grads, outputs

        ref_loss, ref_grads, ref_outputs = reference.strategy.run(
            lambda: gradient(reference, x, ref_state)
        )
        actual_loss, actual_grads, actual_outputs = distributed.strategy.run(
            lambda value, state: gradient(distributed, value, state),
            args=(local_x[0], states),
        )
        averaged_loss = distributed.strategy.reduce(
            tf.distribute.ReduceOp.MEAN, actual_loss, None
        )
        np.testing.assert_allclose(averaged_loss, ref_loss, rtol=2e-4, atol=1e-7)
        worst = 0.0
        for expected, values in zip(ref_grads, actual_grads):
            actual = distributed.strategy.reduce(
                tf.distribute.ReduceOp.SUM, values, None
            )
            np.testing.assert_allclose(actual, expected, rtol=2e-4, atol=1e-6)
            worst = max(
                worst,
                float(
                    np.linalg.norm(actual.numpy() - expected.numpy())
                    / max(np.linalg.norm(expected.numpy()), 1e-12)
                ),
            )
        if worst > 2e-4:
            raise AssertionError(f"Gradient relative L2 exceeds gate: {worst}")
        gradient_surfaces = [
            {
                "name": variable.name,
                "l2": float(np.linalg.norm(gradient.numpy().astype(np.float64))),
            }
            for variable, gradient in zip(
                reference.model.trainable_variables, ref_grads
            )
        ]
        if any(not np.any(gradient.numpy() != 0) for gradient in ref_grads):
            raise AssertionError(
                f"Every trainable canonical surface needs nonzero credit: {gradient_surfaces}"
            )
        for expected, values in zip(
            tf.nest.flatten(ref_outputs), tf.nest.flatten(actual_outputs)
        ):
            actual = tf.concat(
                distributed.strategy.experimental_local_results(values), axis=0
            )
            np.testing.assert_allclose(actual, expected, rtol=2e-4, atol=1e-6)

        reference_validation = reference.training_engine._distributed_validation_step(
            [x], [[]], ref_state, "single"
        )
        validation = distributed.data_parallel.reduce_metrics(
            distributed.training_engine._distributed_validation_step(
                local_x, [[]], states, "single"
            )
        )
        for expected, actual in zip(
            tf.nest.flatten(reference_validation), tf.nest.flatten(validation)
        ):
            np.testing.assert_allclose(actual, expected, rtol=2e-4, atol=1e-7)

        losses = []
        for update in range(3):
            reference.training_engine._distributed_train_step([x], [[]], ref_state)
            result = distributed.training_engine._distributed_train_step(
                local_x, [[]], states
            )
            distributed.strategy.run(distributed.cell.refresh_recurrent_weight_shadow)
            distributed.strategy.run(check_shadows)
            reference.strategy.run(reference.cell.refresh_recurrent_weight_shadow)
            result = distributed.data_parallel.reduce_metrics(result)
            losses.append(float(result["__total_loss"].numpy()))
            if (
                int(distributed.training_engine.optimizer.iterations.numpy())
                != update + 1
            ):
                raise AssertionError("Optimizer must advance once per global update")
            for expected, actual in zip(
                reference.model.trainable_variables,
                distributed.model.trainable_variables,
            ):
                np.testing.assert_allclose(
                    actual.numpy(), expected.numpy(), rtol=2e-4, atol=1e-6
                )
                for replica in distributed.strategy.experimental_local_results(
                    resource(actual)
                ):
                    np.testing.assert_array_equal(replica.numpy(), actual.numpy())
            for expected, actual in zip(
                optimizer_variables(reference), optimizer_variables(distributed)
            ):
                np.testing.assert_allclose(
                    actual.numpy(), expected.numpy(), rtol=2e-4, atol=1e-6
                )

        masters = [distributed.cell.recurrent_weight_values] + [
            net["input_weight_values"] for net in distributed.cell.inputs.values()
        ]
        tracked = {f"master_{i}": resource(v) for i, v in enumerate(masters)}
        tracked["noise_seed"] = distributed.cell.noise_seed
        tracked["noise_stream"] = distributed.cell.noise_stream
        checkpoint = tf.train.Checkpoint(
            optimizer=distributed.training_engine.optimizer, **tracked
        )
        before = [v.numpy().copy() for v in masters]
        optimizer_before = [v.numpy().copy() for v in optimizer_variables(distributed)]
        noise_before = [
            distributed.cell.noise_seed.numpy().copy(),
            distributed.cell.noise_stream.numpy().copy(),
        ]
        with tempfile.TemporaryDirectory() as directory:
            saved = checkpoint.save(str(Path(directory) / "model"))
            for master in masters:
                master.assign(tf.zeros_like(master))
            for variable in optimizer_variables(distributed):
                variable.assign(tf.zeros_like(variable))
            distributed.cell.noise_seed.assign(-99)
            distributed.cell.noise_stream.assign(-99)
            checkpoint.restore(saved).assert_consumed()
        for expected, actual in zip(before, masters):
            np.testing.assert_array_equal(actual.numpy(), expected)
        for expected, actual in zip(optimizer_before, optimizer_variables(distributed)):
            np.testing.assert_array_equal(actual.numpy(), expected)
        np.testing.assert_array_equal(
            noise_before,
            [
                distributed.cell.noise_seed.numpy(),
                distributed.cell.noise_stream.numpy(),
            ],
        )
        distributed.strategy.run(distributed.cell.refresh_recurrent_weight_shadow)
        distributed.strategy.run(check_shadows)

        class CaptureCallbacks:
            def __init__(self):
                self.steps, self.epochs = [], []
                self.begins, self.ends = 0, 0

            def on_train_begin(self):
                self.begins += 1

            def on_epoch_start(self):
                pass

            def on_step_start(self):
                pass

            def on_step_end(self, values):
                self.steps.append(values)

            def on_epoch_end(self, values):
                self.epochs.append(values)
                return False

            def on_train_end(self, **kwargs):
                self.ends += 1

        capture = CaptureCallbacks()
        engine = distributed.training_engine
        engine.set_callbacks(capture)
        for index, name in enumerate(distributed.ordered_inputs_populations):
            inputs = x[:, :, index * 2 : (index + 1) * 2]
            module = SimpleNamespace(
                population_name=name,
                use_device_generation=False,
                create_generator=lambda seq_len, inputs=inputs: (
                    tf.data.Dataset.from_tensor_slices(
                        (inputs, {"sample": tf.range(4)})
                    ).repeat()
                ),
            )
            engine.parameters[0].add_inputs_generator(name, module)
        distributed.train()
        distributed.strategy.run(check_shadows)
        if (capture.begins, capture.ends, len(capture.steps), len(capture.epochs)) != (
            1,
            1,
            1,
            1,
        ):
            raise AssertionError("High-level training must invoke host callbacks once")
        if int(engine.optimizer.iterations.numpy()) != 4:
            raise AssertionError("High-level global update must advance Adam once")
        if any(
            isinstance(v, tf.distribute.DistributedValues)
            for v in tf.nest.flatten((capture.steps, capture.epochs))
        ):
            raise AssertionError("Callbacks received unreduced replica metrics")
        reference.training_engine._distributed_train_step([x], [[]], ref_state)
        for expected, actual in zip(
            reference.model.trainable_variables, distributed.model.trainable_variables
        ):
            np.testing.assert_allclose(
                actual.numpy(), expected.numpy(), rtol=2e-4, atol=1e-6
            )
        poisson_reports = []
        for device_poisson in (False, True):
            with distributed.strategy.scope():
                noise_cell = make_cell(
                    "nest",
                    noise=True,
                    batch_size=2,
                    use_device_poisson=device_poisson,
                    use_fused_cuda=False,
                    use_fused_state=False,
                )

            @tf.function
            def sample(step):
                return distributed.strategy.run(
                    lambda: noise_cell.sample_noise_spikes(
                        8192, tf.reshape(step, [1]), noise_cell.inputs["drive"]
                    )
                )

            first = distributed.strategy.experimental_local_results(
                sample(tf.constant(0))
            )
            replay = distributed.strategy.experimental_local_results(
                sample(tf.constant(0))
            )
            next_step = distributed.strategy.experimental_local_results(
                sample(tf.constant(1))
            )
            for counts, repeated, changed in zip(first, replay, next_step):
                np.testing.assert_array_equal(counts, repeated)
                if np.array_equal(counts, changed):
                    raise AssertionError("Poisson logical time did not advance")
                if abs(float(tf.reduce_mean(tf.cast(counts, tf.float32))) - 0.5) > 0.04:
                    raise AssertionError("Poisson sample mean outside fixed-seed gate")
                if counts.dtype != tf.int32 or np.max(counts.numpy()) <= 1:
                    raise AssertionError(
                        "Background noise must preserve Poisson counts"
                    )
            if np.array_equal(first[0], first[1]):
                raise AssertionError("Replica Poisson streams are duplicated")
            poisson_reports.append(
                {"device_poisson": device_poisson, "replay": "passed"}
            )

        report = {
            "status": "passed",
            "physical_gpu": args.gpu,
            "native": args.native,
            "trainable_input_type": "spikes" if args.spike_input else "current",
            "replicas": 2,
            "global_batch": 4,
            "local_batch": 2,
            "steps": steps,
            "worst_gradient_relative_l2": worst,
            "canonical_gradient_surfaces": gradient_surfaces,
            "accepted_updates": 3,
            "losses": losses,
            "restored_masters": len(masters),
            "high_level_updates": 1,
            "host_callback_ownership": "single",
            "per_replica_compute_shadows": "exact canonical and CSR checks after updates/restore",
            "replica_local_input_devices": [part.device for part in parts],
            "rollout_constant_devices": constant_devices,
            "validation_metrics": "global mean matches independent full batch",
            "distribution": distributed.data_parallel.report,
            "poisson": poisson_reports,
            "scope": "Tiny GLIF independent ordinary-full-batch versus distributed checkpointed BPTT; not full V1 or throughput qualification.",
        }
        if args.native and not (
            distributed.cell._use_fused_cuda and distributed.cell._use_fused_state
        ):
            raise AssertionError("Native operators were requested but not executed")
        if args.native:
            graph = engine._distributed_train_step.get_concrete_function(
                local_x, [[]], states
            ).graph.as_graph_def()
            nodes = list(graph.node)
            for function in graph.library.function:
                nodes.extend(function.node_def)
            native_ops = sorted(
                {node.op for node in nodes if node.op.startswith("Dpointnet")}
            )
            if not any("CsrSpike" in op for op in native_ops) or not any(
                op.startswith("DpointnetNestState") for op in native_ops
            ):
                raise AssertionError(
                    f"Training graph lacks native CSR/NEST-state operators: {native_ops}"
                )
            metadata = [distributed.cell.recurrent_fused_connectivity]
            metadata.extend(
                net["fused_connectivity"]
                for net in distributed.cell.inputs.values()
                if net["input_type"]
                in ("spikes", "poisson_spikes_internal", "noisy_current")
            )
            resource_devices = [
                [handle.device for handle in connectivity._handles]
                for connectivity in metadata
            ]
            for placement in resource_devices:
                if len(placement) != 2 or not all(
                    f"/device:GPU:{index}" in device
                    for index, device in enumerate(placement)
                ):
                    raise AssertionError(f"Metadata is not replica-local: {placement}")
            report["native_ops"] = native_ops
            report["metadata_devices"] = resource_devices
            array_devices = [
                [
                    [
                        values[key].device
                        for key in (
                            "incoming_pre_ids",
                            "incoming_edge_ids",
                            "incoming_types",
                        )
                    ]
                    for values in connectivity._replica_values
                ]
                for connectivity in metadata
            ]
            for connectivity in array_devices:
                for index, placement in enumerate(connectivity):
                    if not all(
                        f"/device:GPU:{index}" in device for device in placement
                    ):
                        raise AssertionError(
                            f"Routing arrays are not replica-local: {placement}"
                        )
            report["routing_array_devices"] = array_devices
            if (
                distributed.cell._use_prepacked_nest_coefficients
                and distributed.cell._use_type_indexed_nest_coefficients
            ):
                predicates = distributed.strategy.run(
                    lambda: distributed.cell.prepare_rollout_nest_coefficients()[4]
                )
                predicates = distributed.strategy.experimental_local_results(predicates)
                if not all(value.device.endswith("/device:CPU:0") for value in predicates):
                    raise AssertionError("NEST dispatch predicates must remain on the host")
                report["dispatch_predicate_devices"] = [value.device for value in predicates]
            else:
                report["dispatch_predicate_devices"] = []
                report["dispatch_predicate_scope"] = "Generic native fixture; type-indexed dispatch requires the separate full-V1 trace."
        if args.gpu:
            report["memory"] = {
                device: tf.config.experimental.get_memory_info(device.lstrip("/"))
                for device in devices
            }
            rows = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-compute-apps=gpu_uuid,pid,used_gpu_memory",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
            ).splitlines()
            report["driver_devices"] = [
                {"uuid": fields[0], "memory_mib": fields[2]}
                for row in rows
                for fields in [[field.strip() for field in row.split(",")]]
                if fields[1] == str(os.getpid())
            ]
            if len({device["uuid"] for device in report["driver_devices"]}) != 2:
                raise AssertionError(
                    "Two distinct physical GPU UUIDs were not observed"
                )
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
    finally:
        reference.cleanup()
        distributed.cleanup()


def resource(variable):
    value = getattr(variable, "value", variable)
    return variable if callable(value) else value


def optimizer_variables(rnn):
    values = rnn.training_engine.optimizer.variables
    return list(values() if callable(values) else values)


if __name__ == "__main__":
    main()
