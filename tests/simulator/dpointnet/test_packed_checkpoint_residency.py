"""Packed spike checkpoints must stay on the GPU (no state-sized host copies)."""

import re

import numpy as np
import pytest
import tensorflow as tf

from bmtk.simulator.dpointnet.segmented_recompute import _pack_spikes, _unpack_spikes


pytestmark = pytest.mark.skipif(
    not tf.config.list_physical_devices("GPU"), reason="GPU residency check requires a GPU"
)


def _host_copies(logdir):
    from tensorflow.tsl.profiler.protobuf import xplane_pb2

    copies = []
    for path in logdir.rglob("*.xplane.pb"):
        space = xplane_pb2.XSpace()
        space.ParseFromString(path.read_bytes())
        for plane in space.planes:
            if "gpu" not in plane.name.lower():
                continue
            for line in plane.lines:
                for event in line.events:
                    name = plane.event_metadata[event.metadata_id].name
                    if name not in ("MemcpyH2D", "MemcpyD2H"):
                        continue
                    details = ""
                    for stat in list(plane.event_metadata[event.metadata_id].stats) + list(event.stats):
                        if plane.stat_metadata[stat.metadata_id].name == "memcpy_details":
                            field = stat.WhichOneof("value")
                            details = (
                                plane.stat_metadata[stat.ref_value].name
                                if field == "ref_value" else str(getattr(stat, field))
                            )
                    size = re.search(r"\bsize:(\d+)\b", details)
                    copies.append((name, int(size.group(1)) if size else -1))
    return copies


@pytest.mark.parametrize("dtype", [tf.float16, tf.float32])
def test_gpu_pack_unpack_has_no_state_sized_host_copies(tmp_path, dtype):
    batch, width = 8, 50_000
    rng = np.random.default_rng(3)
    with tf.device("/GPU:0"):
        spikes = tf.constant(rng.integers(0, 2, size=(batch, width)), dtype)

    @tf.function
    def roundtrip(value):
        packed = _pack_spikes(value)
        return packed, _unpack_spikes(packed, width, dtype)

    packed, restored = roundtrip(spikes)
    np.testing.assert_array_equal(restored.numpy(), spikes.numpy())
    tf.profiler.experimental.start(str(tmp_path))
    for _ in range(3):
        packed, restored = roundtrip(spikes)
    tf.test.experimental.sync_devices()
    tf.profiler.experimental.stop()
    threshold = batch * width // 16
    large = [copy for copy in _host_copies(tmp_path) if copy[1] >= threshold]
    assert not large, f"State-sized host copies during on-device packing: {large}"
    assert packed.dtype == tf.int64 and "GPU" in packed.device
