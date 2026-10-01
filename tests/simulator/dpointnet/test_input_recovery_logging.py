import io as string_io
import logging
from pathlib import Path
from queue import SimpleQueue
from types import SimpleNamespace

import numpy as np
import pytest
import tensorflow as tf

from bmtk.simulator.dpointnet.data_iterator import DataIterator
from bmtk.simulator.dpointnet.input_modules.lgn_generator import (
    LGNGenerator, _guard_generator, _RecoverableLGNIterator,
    create_grey_screen_generator,
)
from bmtk.simulator.dpointnet.io_tools import RNNIOUtils, io
from bmtk.simulator.dpointnet.state_modules.input_state import InitStateFromInputModule


@pytest.fixture
def console(tmp_path, monkeypatch):
    output = string_io.StringIO()
    logger = logging.Logger("recovery-test")
    logger.addHandler(logging.StreamHandler(output))
    file_handler = logging.FileHandler(tmp_path / "log.txt")
    logger.addHandler(file_handler)
    monkeypatch.setattr(RNNIOUtils, "_logger", logger)
    monkeypatch.setattr(io, "_diagnostic_path", None)
    yield output
    file_handler.close()


def _module(outcomes):
    outcomes = iter(outcomes)
    spikes = tf.ones((2, 3, 1))
    initial = (tf.zeros((2, 1)),)

    def next_spikes():
        result = next(outcomes)
        if isinstance(result, Exception):
            raise result
        return spikes, []

    iterator = SimpleNamespace(
        next_spikes=next_spikes, close=lambda: None, build=lambda: None,
    )
    rnn = SimpleNamespace(
        adjusted_batch_size=2, seq_len=3, dtype=tf.float32,
        parse_input_mods_from_config=lambda inputs: [],
        cell=SimpleNamespace(
            advance_noise_seed=lambda: None,
            zero_state=lambda *args, **kwargs: initial,
        ),
        state_only_model=lambda inputs: inputs[1],
        model_inputs=lambda spikes, state: (spikes, state),
    )
    module = InitStateFromInputModule(rnn, inputs=[])
    module.spikes_itrs = lambda batch_size: iterator
    return module


@pytest.mark.parametrize("error_type", [tf.errors.InvalidArgumentError, tf.errors.UnknownError])
def test_previous_state_notice_and_file_only_traceback(console, error_type):
    error = error_type(None, None, "diagnostic-only failure details")
    module = _module([None, error, None])
    previous = module.get_state(max_retries=1)
    assert module.get_state(max_retries=1) is previous
    assert module.get_state(max_retries=1) is not None
    text = console.getvalue()
    assert "continuing with the previous initial state" in text
    assert io._diagnostic_path in text
    assert "Traceback" not in text
    assert "diagnostic-only failure details" not in text
    details = Path(io._diagnostic_path).read_text()
    assert "Traceback" in details
    assert "diagnostic-only failure details" in details
    assert "attempt 1/1" in details
    assert Path(io._diagnostic_path).parent == Path(console_file_dir())


def console_file_dir():
    return next(
        Path(handler.baseFilename).parent for handler in io.logger.handlers
        if isinstance(handler, logging.FileHandler)
    )


def test_retry_success_reports_recovery_without_changing_state(console):
    module = _module([tf.errors.UnknownError(None, None, "retry detail"), None])
    state = module.get_state(max_retries=2)
    assert state is module._last_state
    assert "recovered; continuing normally" in console.getvalue()
    assert "retry detail" in Path(io._diagnostic_path).read_text()


def test_first_failure_remains_fatal(console):
    error = tf.errors.InvalidArgumentError(None, None, "fatal detail")
    module = _module([error])
    with pytest.raises(tf.errors.InvalidArgumentError) as caught:
        module.get_state(max_retries=1)
    assert caught.value is error
    assert "no previous state is available" in console.getvalue()
    assert "fatal detail" in Path(io._diagnostic_path).read_text()


def test_unexpected_errors_are_not_hidden(console):
    module = _module([ValueError("invalid configuration")])
    with pytest.raises(ValueError, match="invalid configuration"):
        module.get_state(max_retries=2)
    assert not console.getvalue()


def test_diagnostics_io_failure_is_visible(console, monkeypatch):
    def fail(*args, **kwargs):
        raise PermissionError("diagnostic directory is read-only")

    monkeypatch.setattr("bmtk.simulator.dpointnet.io_tools.tempfile.NamedTemporaryFile", fail)
    module = _module([None, tf.errors.UnknownError(None, None, "original detail")])
    previous = module.get_state(max_retries=1)
    assert module.get_state(max_retries=1) is previous
    assert "Unable to save DPointNet recovery diagnostics" in console.getvalue()
    assert "original detail" in console.getvalue()
    assert "Details:" not in console.getvalue()


def test_guard_preserves_error_identity_without_callback_traceback(capfd):
    error = tf.errors.InvalidArgumentError(None, None, "callback failure detail")
    errors = SimpleQueue()

    def generate():
        yield tf.ones((3, 1)), {"contrast": tf.ones((1,))}
        raise error

    dataset = tf.data.Dataset.from_generator(
        _guard_generator(generate, errors.put),
        output_signature=(
            tf.TensorSpec((3, 1), tf.float32),
            {"contrast": tf.TensorSpec((1,), tf.float32)},
        ),
    )
    iterator = _RecoverableLGNIterator(
        iter(dataset.batch(2, drop_remainder=True).prefetch(1)), errors,
    )
    with pytest.raises(tf.errors.InvalidArgumentError) as caught:
        next(iterator)
    assert caught.value is error
    captured = capfd.readouterr()
    assert "callback failure detail" not in captured.err
    assert "Traceback" not in captured.err


def test_guard_does_not_hide_nonrecoverable_errors():
    def generate():
        raise ValueError("bad parameters")
        yield

    with pytest.raises(ValueError, match="bad parameters"):
        next(_guard_generator(generate, SimpleQueue().put)())


def test_guard_is_opt_in():
    error = tf.errors.UnknownError(None, None, "unchanged default")

    def generate():
        raise error
        yield

    with pytest.raises(tf.errors.UnknownError) as caught:
        next(_guard_generator(generate, None)())
    assert caught.value is error


def test_grey_screen_recovery_iterator_preserves_seeded_batches():
    probabilities = tf.fill((7, 4), 0.2)
    module = object.__new__(LGNGenerator)
    module.stimulus_opts = dict(
        row_size=3, col_size=5, seed=17, contrast=0.0,
    )
    module.stimulus_type = "grey_screen"
    module.rnn = SimpleNamespace(default_seed=None)
    module._grey_screen_probabilities = probabilities
    module._generator_fn = create_grey_screen_generator
    expected = iter(module.create_generator(7).batch(3))
    actual = module.create_recoverable_iterator(7, 3)
    for _ in range(3):
        for value, reference in zip(
            tf.nest.flatten(next(actual)), tf.nest.flatten(next(expected))
        ):
            np.testing.assert_array_equal(value, reference)


@pytest.mark.parametrize("recover", [False, True])
def test_data_iterator_recovery_route_is_opt_in(recover):
    calls = []

    class Module:
        @staticmethod
        def create_recoverable_iterator(seq_len, batch_size):
            calls.append("recoverable")
            return iter([])

        @staticmethod
        def create_generator(seq_len):
            calls.append("default")
            return tf.data.Dataset.from_tensors(tf.zeros((3, 1)))

    iterator = DataIterator([Module()], 2, 3, recover_input_errors=recover)
    iterator.build()
    assert calls == ["recoverable" if recover else "default"]
    iterator.close()
