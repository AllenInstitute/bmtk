"""Two-process independent tiny-GLIF gradient, update and ownership oracle."""

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace

from mpi4py import MPI
import numpy as np
import tensorflow as tf

from test_precision_credit import make_cell
from bmtk.simulator.dpointnet.rnn_model import RNN
from bmtk.simulator.dpointnet.loss_functions.target_firing_rates import TargetFiringRate
from bmtk.simulator.dpointnet.state_modules.zero_state import ZeroStateModule
from bmtk.simulator.dpointnet.custom_ops.csr_spike_ops import reorder_csr_values


def optimizer_variables(optimizer):
    values = optimizer.variables
    return list(values() if callable(values) else values)


def check(actual, expected):
    actual, expected = np.asarray(actual), np.asarray(expected)
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual, expected, rtol=2e-4, atol=1e-6)
    relative = float(np.linalg.norm(
        (actual - expected).ravel().astype(np.float64)
    ) / max(np.linalg.norm(expected.ravel().astype(np.float64)), 1e-12))
    assert relative <= 2e-4, relative
    return relative


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--spike-input", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    comm = MPI.COMM_WORLD
    assert comm.size == 2
    assert len(tf.config.list_physical_devices("GPU")) == int(args.gpu)
    for device in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(device, True)
    tf.random.set_seed(17)
    steps = 17 if args.spike_input else 9

    def create(batch, process):
        network, inputs, options = make_cell(
            "nest", selective=False, two_inputs=True, return_spec=True,
            batch_size=batch, use_fused_cuda=args.gpu and process,
            use_fused_state=args.gpu and process,
            use_direct_state_rnn_loop=args.gpu and process,
            use_direct_csr_recurrent_gradient=args.gpu and process,
            use_packed_sm120_backward=False, use_packed_sm120_external_backward=False,
            hard_reset=True, hard_reset_gradient_mode="soft_surrogate",
            detach_reset=True, detach_asc_reset=False,
        )
        inputs["background"]["options"]["trainable"] = False
        if args.spike_input:
            inputs["drive"]["input_type"] = "spikes"
            inputs["drive"]["weights"] *= 0.1
        options.pop("batch_size")
        rnn = RNN(
            seq_len=steps, batch_size=batch, dtype="float32", cell_params=options,
            distribution={
                "strategy": "mpi" if process else "mirrored",
                "devices": ["/GPU:0" if process and args.gpu else "/CPU:0"],
                "global_batch_size": 4,
            },
        )
        rnn._recurrent_networks["test"] = SimpleNamespace(
            to_dict=lambda: copy.deepcopy(network)
        )
        for name, spec in inputs.items():
            rnn._input_networks[name] = SimpleNamespace(
                name=name, n_spiking_nodes=2,
                to_dict=lambda spec=spec: copy.deepcopy(spec),
            )
            rnn._inputs_order.append(name)
        engine = rnn.set_training(
            rnn=rnn, n_epochs=3, steps_per_epoch=1, training_approach="single",
            gradient_checkpointing=process, gradient_checkpoint_chunk_size=3,
        )
        parameter = engine.add_parameters("test", batch_size=batch, seq_len=steps)
        parameter.add_loss_function(
            "rate", TargetFiringRate(
                rnn, firing_rate=0.005, rate_units="spikes_per_step"
            ),
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

    reference, distributed = create(4, False), create(2, True)
    dp = distributed.data_parallel
    dp.synchronize(distributed.training_engine)
    rng = np.random.default_rng(17)
    samples = rng.uniform(0.2, 1.5, (4, steps, 4)).astype(np.float32)
    if args.spike_input:
        samples[:, :, :2] = rng.integers(0, 4, (4, steps, 2))
    supplied = samples.copy()
    local = dp.distribute_batch([supplied] if dp.is_chief else None)
    supplied.fill(-99)
    selection = slice(comm.rank * 2, comm.rank * 2 + 2)
    np.testing.assert_array_equal(local[0], samples[selection])
    buffer = dp._sample_buffers[0] if not dp.is_chief else None
    supplied = samples + 7
    next_batch = dp.distribute_batch([supplied] if dp.is_chief else None)
    supplied.fill(-98)
    if not dp.is_chief:
        assert dp._sample_buffers[0] is buffer
    np.testing.assert_array_equal(next_batch[0], samples[selection] + 7)
    np.testing.assert_array_equal(local[0], samples[selection])
    targets = dp.distribute_batch(
        [{"sample": np.arange(4, dtype=np.int32)}] if dp.is_chief else None
    )
    np.testing.assert_array_equal(targets[0]["sample"], np.arange(4)[selection])
    assert local[0].device.endswith(f"/device:{'GPU' if args.gpu else 'CPU'}:0")
    ref_state = reference.strategy.run(reference.training_engine.init_state.get_state)
    state = distributed.strategy.run(distributed.training_engine.init_state.get_state)

    def gradients(rnn, value, initial):
        with tf.GradientTape() as tape:
            outputs = rnn.training_engine._run_extractor(value, initial)
            loss = rnn.training_engine.parameters[0].loss_functions["rate"](outputs[0][0])
        return loss, tape.gradient(loss, rnn.model.trainable_variables), outputs

    ref_loss, ref_grad, ref_outputs = reference.strategy.run(
        lambda: gradients(reference, tf.constant(samples), ref_state)
    )
    local_loss, local_grad, outputs = distributed.strategy.run(
        lambda: gradients(distributed, local[0], state)
    )
    worst = check(dp.reduce_metrics(local_loss), ref_loss)
    canonical = dp.average_gradients(local_grad)
    assert len(canonical) == len(ref_grad) == 2
    gradient_buffer = dp._gradient_buffer
    canonical_snapshots = [value.numpy().copy() for value in canonical]
    if args.gpu:
        assert all(value.device.endswith("/device:GPU:0") for value in canonical)
    gradient_norms = []
    for actual, expected in zip(canonical, ref_grad):
        worst = max(worst, check(actual, expected))
        assert np.any(actual.numpy() != 0)
        gradient_norms.append(float(np.linalg.norm(actual.numpy().astype(np.float64))))
    for actual, expected in zip(tf.nest.flatten(outputs), tf.nest.flatten(ref_outputs)):
        worst = max(worst, check(actual, expected[selection]))

    def refresh(rnn):
        rnn.strategy.run(rnn.cell.refresh_recurrent_weight_shadow)
        cell = rnn.cell
        np.testing.assert_array_equal(
            cell.recurrent_weight_values_compute, cell.recurrent_weight_values
        )
        if args.gpu and rnn is distributed:
            np.testing.assert_array_equal(
                cell.recurrent_csr_weight_values_compute,
                reorder_csr_values(
                    cell.recurrent_weight_values, cell.recurrent_fused_connectivity
                ),
            )
        for net in cell.inputs.values():
            np.testing.assert_array_equal(
                net["input_weight_values_compute"], net["input_weight_values"]
            )
            if args.gpu and rnn is distributed and net["input_type"] != "current":
                np.testing.assert_array_equal(
                    net["csr_weight_values_compute"],
                    reorder_csr_values(net["input_weight_values"], net["fused_connectivity"]),
                )

    before_collectives = dp.gradient_collectives
    for _ in range(3):
        reference.training_engine.train_step([tf.constant(samples)], [[]], ref_state)
        distributed.training_engine.train_step(local, [[]], state)
        assert dp._gradient_buffer is gradient_buffer
        for value, snapshot in zip(canonical, canonical_snapshots):
            np.testing.assert_array_equal(value, snapshot)
        for rnn in (reference, distributed):
            refresh(rnn)
        for actual, expected in zip(
            distributed.model.trainable_variables, reference.model.trainable_variables
        ):
            worst = max(worst, check(actual, expected))
            if actual.constraint is not None:
                np.testing.assert_array_equal(actual, actual.constraint(actual))
        for actual, expected in zip(
            optimizer_variables(distributed.training_engine.optimizer),
            optimizer_variables(reference.training_engine.optimizer),
        ):
            worst = max(worst, check(actual, expected))
    assert dp.gradient_collectives - before_collectives == 3

    masters = [distributed.cell.recurrent_weight_values] + [
        net["input_weight_values"] for net in distributed.cell.inputs.values()
    ]
    optimizer = distributed.training_engine.optimizer
    variables = masters + optimizer_variables(optimizer) + [
        distributed.cell.noise_seed, distributed.cell.noise_stream,
    ]
    saved = [value.numpy().copy() for value in variables]
    checkpoint = tf.train.Checkpoint(
        optimizer=optimizer, noise_seed=distributed.cell.noise_seed,
        noise_stream=distributed.cell.noise_stream,
        **{f"master_{index}": value for index, value in enumerate(masters)},
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.output.parent) as directory:
        path = checkpoint.save(str(Path(directory) / "checkpoint")) if dp.is_chief else None
        for value in variables:
            value.assign(tf.zeros_like(value))
        if dp.is_chief:
            checkpoint.restore(path).assert_consumed()
        dp.synchronize(distributed.training_engine)
        for actual, expected in zip(variables, saved):
            np.testing.assert_array_equal(actual, expected)
    refresh(distributed)

    from bmtk.simulator.dpointnet import training
    ownership = dict(input=0, close=0, begin=0, epoch=0, step=0, end_step=0, end_epoch=0, end=0)

    class Iterator:
        def __init__(self, *args, **kwargs):
            pass

        def next_spikes(self):
            ownership["input"] += 1
            return [tf.constant(samples)], [[]]

        def close(self):
            ownership["close"] += 1

    def callback(name, *args, **kwargs):
        ownership[name] += 1
        return name == "end_epoch"

    training.DataIterator = Iterator
    distributed.training_engine.set_callbacks(SimpleNamespace(
        on_train_begin=lambda: callback("begin"),
        on_epoch_start=lambda: callback("epoch"),
        on_step_start=lambda: callback("step"),
        on_step_end=lambda values: callback("end_step", values),
        on_epoch_end=lambda values: callback("end_epoch", values),
        on_train_end=lambda **kwargs: callback("end", **kwargs),
    ))
    count = dp.gradient_collectives
    distributed.train()
    assert dp.gradient_collectives == count + 1
    assert set(ownership.values()) == {int(dp.is_chief)}, ownership

    poisson = []
    for device_sampler in (False, True):
        with distributed.strategy.scope():
            noise_cell = make_cell(
                "nest", noise=True, batch_size=2, use_device_poisson=device_sampler,
                use_fused_cuda=False, use_fused_state=False,
            )

        @tf.function
        def sample(step):
            return distributed.strategy.run(
                lambda: noise_cell.sample_noise_spikes(
                    8192, tf.reshape(step, [1]), noise_cell.inputs["drive"]
                )
            )

        first = sample(tf.constant(3)).numpy()
        np.testing.assert_array_equal(first, sample(tf.constant(3)))
        assert first.max() > 1
        assert not np.array_equal(first, sample(tf.constant(4)))
        hashes = comm.allgather(hashlib.sha256(first.tobytes()).hexdigest())
        assert len(set(hashes)) == comm.size
        poisson.append({"device_sampler": device_sampler, "replay_and_rank_streams": "passed"})

    report = {
        "status": "passed", "rank": comm.rank, "replicas": comm.size,
        "native": args.gpu, "spike_input": args.spike_input,
        "worst_relative_l2": worst, "terminal_states": len(outputs) - 1,
        "canonical_gradient_surfaces": 2, "accepted_updates": 3,
        "reused_gradient_buffer_preserves_gpu_snapshots": "passed",
        "canonical_gradient_l2_float64": gradient_norms,
        "gradient_allreduces_for_three_updates": 3,
        "chief_restore_broadcast_all_masters_optimizer_noise": "passed",
        "canonical_and_csr_shadows": "exact", "ownership": ownership,
        "chief_only_input_callbacks_early_stop": "passed", "poisson": poisson,
        "distribution": dp.report, "input_device": local[0].device,
    }
    if args.gpu:
        assert distributed.cell._use_fused_cuda and distributed.cell._use_fused_state
        definition = distributed.training_engine._distributed_single_gradients.get_concrete_function(
            local, [[]], state
        ).graph.as_graph_def()
        text = str(definition)
        assert "DpointnetCsrSpike" in text and "DpointnetNestState" in text
        report["gpu_uuid"] = subprocess.check_output([
            "nvidia-smi", "-i", os.environ["CUDA_VISIBLE_DEVICES"],
            "--query-gpu=uuid", "--format=csv,noheader",
        ], text=True).strip()
    reference.cleanup()
    distributed.cleanup()
    rank_output = args.output.with_name(f"{args.output.stem}-rank{comm.rank}.json")
    rank_output.write_text(json.dumps(report, indent=2) + "\n")
    comm.Barrier()
    if dp.is_chief:
        print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
