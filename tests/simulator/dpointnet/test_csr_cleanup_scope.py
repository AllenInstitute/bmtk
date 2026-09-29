import pytest
import tensorflow as tf
import uuid

from bmtk.simulator.dpointnet.custom_ops.csr_spike_ops import _destroy_metadata_resource


def test_eager_resource_cleanup_is_not_captured_by_unrelated_trace():
    handle = tf.raw_ops.VarHandleOp(
        dtype=tf.int32, shape=[3], container="csr_cleanup_test",
        shared_name="metadata_" + uuid.uuid4().hex,
    )
    tf.raw_ops.AssignVariableOp(resource=handle, value=tf.constant([1, 2, 3], tf.int32))

    @tf.function
    def unrelated(x):
        _destroy_metadata_resource(handle)
        with tf.GradientTape() as tape:
            tape.watch(x)
            value = x * x
        return tape.gradient(value, x)

    concrete = unrelated.get_concrete_function(tf.TensorSpec([], tf.float32))
    assert not any(op.type == "DestroyResourceOp" for op in concrete.graph.get_operations())
    assert float(concrete(tf.constant(3.)).numpy()) == 6.
    with pytest.raises(tf.errors.OpError):
        tf.raw_ops.ReadVariableOp(resource=handle, dtype=tf.int32)
