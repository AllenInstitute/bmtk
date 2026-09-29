"""Lossless resource storage, partial chunks and anonymous rollout lifetimes."""

import numpy as np
import pytest
import tensorflow as tf

from bmtk.simulator.dpointnet.temporal_adjoint import _HostCurrentTape


@pytest.mark.parametrize("dtype", [tf.float16, tf.float32])
@pytest.mark.parametrize("batch", [1, 3])
def test_cpu_tape_lossless_partial_chunks(dtype, batch):
    width, chunk = 3, 3
    boundaries = tf.constant([0, 3, 5])
    tape = _HostCurrentTape(dtype, tf.constant(batch), width, chunk, boundaries)
    first = tf.reshape(tf.cast(tf.range(3 * batch * width), dtype), [3, batch, width])
    last = tf.reshape(tf.cast(tf.range(2 * batch * width), dtype), [2, batch, width]) * -1
    flow = tape.write(tf.constant(0), first, tape.flow)
    flow = tape.write(tf.constant(1), last, flow)
    tape = tape.completed(flow)
    for index, expected in enumerate((first, last)):
        actual = tape.read(index)
        assert "CPU:0" in actual.device
        np.testing.assert_array_equal(actual.numpy().view(np.uint8), expected.numpy().view(np.uint8))
    with pytest.raises(tf.errors.InvalidArgumentError, match="not recorded"):
        tape.read(2)


def test_half_bit_patterns_are_not_numerically_converted():
    bits = np.array([0, 0x8000, 1, 0x7c00, 0xfc00, 0x7e01], np.uint16)
    values = tf.constant(bits.view(np.float16).reshape(2, 1, 3))
    tape = _HostCurrentTape(tf.float16, tf.constant(1), 3, 2, tf.constant([0, 2]))
    tape = tape.completed(tape.write(tf.constant(0), values, tape.flow))
    np.testing.assert_array_equal(tape.read(0).numpy().view(np.uint16).ravel(), bits)


def test_missing_and_duplicate_writes_are_explicit_errors():
    tape = _HostCurrentTape(tf.float16, tf.constant(1), 3, 2, tf.constant([0, 2, 4]))
    value = tf.ones([2, 1, 3], tf.float16)
    with pytest.raises(tf.errors.InvalidArgumentError, match="sequential"):
        tape.write(tf.constant(1), value, tf.constant(1))
    tape = tape.completed(tape.write(tf.constant(0), value, tape.flow))
    with pytest.raises(tf.errors.InvalidArgumentError, match="sequential"):
        tape.write(tf.constant(0), value, tf.constant(0))
    with pytest.raises(tf.errors.InvalidArgumentError, match="not recorded"):
        tape.completed(tf.constant(2)).read(1)


def test_dynamic_batch_graph_resources_are_independent_across_calls():
    @tf.function(input_signature=[tf.TensorSpec([2, None, 3], tf.float16)])
    def create(values):
        batch = tf.shape(values)[1]
        tape = _HostCurrentTape(tf.float16, batch, 3, 2, tf.constant([0, 2]))
        flow = tape.write(tf.constant(0), values, tape.flow)
        return tape.handle, flow, batch

    first = create(tf.ones([2, 1, 3], tf.float16))
    second = create(tf.ones([2, 4, 3], tf.float16) * 7)
    for (handle, flow, batch), value in ((first, 1), (second, 7)):
        tape = _HostCurrentTape(tf.float16, batch, 3, 2, tf.constant([0, 2]),
                               handle=handle, flow=flow)
        np.testing.assert_array_equal(tape.read(0), value)
        if tape.backend == "integer_table":
            assert int(tf.raw_ops.LookupTableSizeV2(table_handle=handle).numpy()) == int(batch)
        else:
            assert int(flow) == 1
