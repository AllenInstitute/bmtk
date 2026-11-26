import pytest


@pytest.fixture(autouse=True)
def _restore_keras_precision_policy():
    """Keep one test's global Keras precision policy from leaking into the next.

    Building a model from a configuration sets the global policy; with pytest-xdist any
    test can follow it on the same worker.
    """
    try:
        import tensorflow as tf
    except ImportError:
        yield
        return
    previous = tf.keras.mixed_precision.global_policy()
    yield
    tf.keras.mixed_precision.set_global_policy(previous)
