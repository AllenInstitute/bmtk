from types import SimpleNamespace
import io
import logging
import os
import subprocess

import pytest

from bmtk.simulator.dpointnet import callbacks


def make_callback(monkeypatch, **kwargs):
    monkeypatch.setattr(callbacks.tf.config, "list_physical_devices", lambda kind: [])
    rnn = SimpleNamespace(
        training_engine=SimpleNamespace(n_epochs=2, steps_per_epoch=3),
        strategy=SimpleNamespace(extended=SimpleNamespace(worker_devices=[])),
    )
    callback = callbacks.Callbacks(
        rnn,
        epoch_store_weights="skip",
        losses_table_csv=False,
        performance_table_csv=False,
        **kwargs,
    )
    monkeypatch.setattr(
        callbacks.tf.config, "get_visible_devices", lambda kind: callback._gpu_devices
    )
    monkeypatch.setattr(
        callbacks.tf.config, "get_logical_device_configuration", lambda device: None
    )
    return callback


def test_memory_report_distinguishes_allocator_and_driver():
    sample = callbacks.GPUMem(
        name="GPU:0",
        tf_current=2.0,
        tf_peak=3.0,
        gpu_used=7.0,
        gpu_free=1.0,
        gpu_total=8.0,
        process_used=5.0,
        peak_scope="epoch 2",
    )
    text = callbacks.Callbacks._format_gpu_mem(sample)
    assert "TF current 2.00 GiB" in text
    assert "TF peak 3.00 GiB (epoch 2)" in text
    assert "process 5.00 GiB" in text
    assert "device used 7.00 GiB" in text
    assert "device free 1.00 GiB" in text


def test_memory_is_epoch_only_by_default(monkeypatch):
    callback = make_callback(monkeypatch)
    samples = []
    monkeypatch.setattr(callback, "_gpu_mem_usage", lambda: samples.append(True) or [])
    callback.on_epoch_start()
    samples.clear()
    callback.on_step_start()
    callback.on_step_end({"__total_loss": 1.0})
    assert samples == []
    callback.on_epoch_end({"__total_loss": 1.0})
    assert len(samples) == 1


@pytest.mark.parametrize("mode,expected", [("step", 1), ("epoch", 0), ("off", 0)])
def test_step_memory_sampling_reuses_report(monkeypatch, mode, expected):
    callback = make_callback(monkeypatch, memory_report=mode)
    callback._gpu_devices = [object()]
    samples, messages = [], []
    sample = callbacks.GPUMem(name="GPU:0", gpu_used=3.0)
    monkeypatch.setattr(
        callback, "_gpu_mem_usage", lambda: samples.append(True) or [sample]
    )
    monkeypatch.setattr(callbacks.io, "log_info", messages.append)
    callback.on_step_start()
    callback.on_step_end({"__total_loss": 1.0})
    assert len(samples) == expected
    assert sum("TF current" in text for text in messages) == expected


def test_missing_step_telemetry_does_not_repeat_old_memory(monkeypatch):
    callback = make_callback(monkeypatch, memory_report="step")
    callback._gpu_devices = [object()]
    callback._step_memory_usage = [3.0]
    records = []
    monkeypatch.setattr(
        callback, "_gpu_mem_usage", lambda: [callbacks.GPUMem(name="GPU:0")]
    )
    monkeypatch.setattr(
        callback, "_append_performance_to_csv", lambda *args: records.append(args)
    )
    callback.on_step_start()
    callback.on_step_end({"__total_loss": 1.0})
    assert not any(row[0] == "gpu_memory_usage_per_step" for row in records)
    assert ("step_gpu0_resident_used_gib", 0, 1, None) in records


def test_epoch_memory_uses_one_sample_for_csv_and_console(monkeypatch):
    callback = make_callback(monkeypatch)
    sample = callbacks.GPUMem(
        name="GPU:0", tf_current=2.0, process_used=5.0, peak_scope="epoch 2"
    )
    records, messages, samples = [], [], []
    monkeypatch.setattr(
        callback, "_gpu_mem_usage", lambda: samples.append(True) or [sample]
    )
    monkeypatch.setattr(
        callback, "_append_performance_to_csv", lambda *args: records.append(args)
    )
    monkeypatch.setattr(callbacks.io, "log_info", messages.append)
    callback.epoch_start_time = 0
    callback.epoch_num = 2
    callback.on_epoch_end({"__total_loss": 1.0})
    assert len(samples) == 1
    assert sum("TF current" in text for text in messages) == 1
    assert ("epoch_end_gpu0_process_used_gib", 2, None, 5.0) in records
    assert ("epoch_end_gpu0_tf_allocator_peak_scope", 2, None, "epoch 2") in records
    assert ("epoch_end_gpu0_resident_used_gib", 2, None, None) in records


def test_startup_peak_precedes_epoch_reset(monkeypatch):
    callback = make_callback(monkeypatch)
    callback._gpu_devices = [object()]
    callback.rnn.strategy.extended.worker_devices = ["/device:GPU:0"]
    events = []
    monkeypatch.setattr(
        callback, "_gpu_mem_usage", lambda: events.append("sample") or []
    )
    monkeypatch.setattr(
        callbacks.tf.config.experimental,
        "reset_memory_stats",
        lambda name: events.append("reset"),
    )
    callback.on_epoch_start()
    assert events == ["sample", "reset"]
    assert "first-step tracing" in callback._memory_peak_scopes["GPU:0"]
    callback.on_epoch_start()
    assert callback._memory_peak_scopes["GPU:0"] == "since start of epoch 2"

    def unavailable(name):
        raise ValueError("unavailable")

    monkeypatch.setattr(
        callbacks.tf.config.experimental, "reset_memory_stats", unavailable
    )
    callback.on_epoch_start()
    assert callback._memory_peak_scopes["GPU:0"] == "since start of epoch 2"


def test_driver_memory_uuid_remapping_and_process_filter(monkeypatch):
    callback = make_callback(monkeypatch)
    callback._gpu_devices = [object(), object()]
    callback.rnn.strategy.extended.worker_devices = ["/device:GPU:0", "/device:GPU:1"]
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-second,GPU-first")
    calls = []

    def query(arguments, **kwargs):
        calls.append(arguments)
        assert kwargs["timeout"] == 2
        if arguments[1].startswith("--query-gpu="):
            return SimpleNamespace(
                stdout="GPU-first, 4096, 4096, 8192\nGPU-second, 6144, 2048, 8192\n"
            )
        return SimpleNamespace(
            stdout=f"GPU-second, {os.getpid()}, 3072\nGPU-second, 999999, 2048\nGPU-first, {os.getpid()}, [N/A]\n"
        )

    monkeypatch.setattr(callbacks.subprocess, "run", query)
    monkeypatch.setattr(
        callbacks.tf.config.experimental,
        "get_memory_info",
        lambda name: {"current": 1024**3, "peak": 2 * 1024**3},
    )
    samples = callback._gpu_mem_usage()
    assert len(calls) == 2
    assert samples[0].gpu_used == 6.0 and samples[0].process_used == 3.0
    assert samples[1].gpu_used == 4.0 and samples[1].process_used is None
    assert samples[0].tf_peak == 2.0
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1,0")
    assert all(sample.gpu_used is None for sample in callback._gpu_mem_usage())


@pytest.mark.parametrize(
    "error", [FileNotFoundError(), subprocess.TimeoutExpired("nvidia-smi", 2)]
)
def test_missing_telemetry_does_not_break_training(monkeypatch, error):
    callback = make_callback(monkeypatch)
    callback._gpu_devices = [object()]
    callback.rnn.strategy.extended.worker_devices = ["/device:GPU:0"]

    def query(*args, **kwargs):
        raise error

    def memory_info(name):
        raise ValueError("unsupported allocator statistics")

    monkeypatch.setattr(callbacks.subprocess, "run", query)
    monkeypatch.setattr(
        callbacks.tf.config.experimental, "get_memory_info", memory_info
    )
    sample = callback._gpu_mem_usage()[0]
    assert sample.tf_current is None and sample.process_used is None
    assert "TF current n/a" in callback._format_gpu_mem(sample)
    assert "process n/a" in callback._format_gpu_mem(sample)


def test_invalid_memory_reporting_mode(monkeypatch):
    with pytest.raises(ValueError, match="memory_report"):
        make_callback(monkeypatch, memory_report="unknown")


def test_memory_off_disables_queries_and_resets(monkeypatch):
    callback = make_callback(monkeypatch, memory_report="off")

    def unexpected():
        pytest.fail("telemetry should be disabled")

    monkeypatch.setattr(callback, "_gpu_mem_usage", unexpected)
    monkeypatch.setattr(callback, "_reset_tf_memory_stats", unexpected)
    callback.on_epoch_start()
    callback.on_step_start()
    callback.on_step_end({"__total_loss": 1.0})
    callback.on_epoch_end({"__total_loss": 1.0})


@pytest.mark.parametrize("mapping", ["filtered", "virtual", "mig"])
def test_ambiguous_device_mapping_is_not_reported(monkeypatch, mapping):
    callback = make_callback(monkeypatch)
    callback._gpu_devices = [object()]
    callback.rnn.strategy.extended.worker_devices = ["/device:GPU:0"]
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-first")
    monkeypatch.setattr(
        callback,
        "_nvidia_smi_rows",
        lambda query: [["GPU-first", "4096", "4096", "8192"]],
    )
    monkeypatch.setattr(
        callbacks.tf.config.experimental,
        "get_memory_info",
        lambda name: {"current": 1024**3, "peak": 2 * 1024**3},
    )
    if mapping == "filtered":
        monkeypatch.setattr(callbacks.tf.config, "get_visible_devices", lambda kind: [])
    elif mapping == "virtual":
        monkeypatch.setattr(
            callbacks.tf.config,
            "get_logical_device_configuration",
            lambda device: [object(), object()],
        )
    else:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "MIG-first")
    sample = callback._gpu_mem_usage()[0]
    assert sample.tf_current == 1.0
    assert sample.gpu_used is None


def test_logger_preserves_lazy_configuration_and_file_output(monkeypatch, tmp_path):
    from bmtk.simulator.dpointnet.io_tools import RNNIOUtils

    logger = logging.getLogger("bmtk.simulator.dpointnet.io_tools")
    monkeypatch.setattr(logger, "handlers", [])
    monkeypatch.setattr(RNNIOUtils, "_logger", None)
    reporter = RNNIOUtils()
    reporter.log_to_console = False
    reporter.set_log_level(logging.WARNING)
    assert reporter.logger.handlers == []
    assert reporter.logger.level == logging.WARNING
    logfile = tmp_path / "run" / "log.txt"
    try:
        reporter.setup_output_dir(str(logfile.parent), str(logfile))
        assert RNNIOUtils().logger is reporter.logger
        reporter.log_warning("one file record")
        assert logfile.read_text().count("one file record") == 1
        assert len(logger.handlers) == 1
    finally:
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
            handler.close()
        logger.setLevel(logging.INFO)


def test_dpointnet_logger_does_not_duplicate_to_root(monkeypatch):
    from bmtk.simulator.dpointnet.io_tools import RNNIOUtils
    from bmtk.simulator.core.io_tools import IOUtils

    local_output, root_output = io.StringIO(), io.StringIO()
    local_handler, root_handler = logging.StreamHandler(
        local_output
    ), logging.StreamHandler(root_output)
    logger = logging.getLogger("bmtk.simulator.dpointnet.io_tools")
    monkeypatch.setattr(logger, "handlers", [local_handler])
    monkeypatch.setattr(logging.getLogger(), "handlers", [root_handler])
    shared_logger = IOUtils._logger
    reporter = RNNIOUtils()
    reporter.log_info("unique memory report")
    assert local_output.getvalue().count("unique memory report") == 1
    assert root_output.getvalue() == ""
    assert IOUtils._logger is shared_logger
