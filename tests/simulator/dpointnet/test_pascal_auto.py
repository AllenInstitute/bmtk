import inspect
from types import SimpleNamespace

import pytest
import tensorflow as tf

from bmtk.simulator.dpointnet.custom_ops import glif_state_ops
from test_precision_credit import make_cell


def test_occupancy_marker_rejects_old_library(monkeypatch):
    monkeypatch.setattr(glif_state_ops, "fused_nest_state_available", lambda: True)
    names = (
        "dpointnet_nest_state_forward", "dpointnet_nest_state_backward",
        "dpointnet_nest_state_backward_events", "dpointnet_nest_state_history_forward",
        "dpointnet_nest_state_history_backward", "dpointnet_nest_state_history_backward_events",
    )

    def safe(launch_geometry_version=1):
        return launch_geometry_version

    def old():
        return None

    library = SimpleNamespace(**dict.fromkeys(names, safe))
    monkeypatch.setattr(glif_state_ops, "_OPS", library)
    assert glif_state_ops.fused_pascal_nest_launch_available()
    for name in names:
        setattr(library, name, old)
        assert not glif_state_ops.fused_pascal_nest_launch_available()
        setattr(library, name, safe)


@pytest.mark.skipif(not glif_state_ops.fused_pascal_nest_launch_available(), reason="Rebuilt occupancy-aware state library unavailable")
@pytest.mark.parametrize("hard", [False, True])
@pytest.mark.parametrize("selective", [False, True])
def test_actual_auto_cell_routes_and_canonical_boundary(hard, selective):
    cell = make_cell(
        "nest", fused=True, selective=selective, acceleration_profile="auto",
        hard_reset=hard, hard_reset_gradient_mode="soft_surrogate" if hard else "exact",
        use_fused_cuda=True,
    )
    try:
        assert cell._use_fused_state
        assert cell._use_fused_cuda
        assert cell.use_fused_nest_event_vjp
        assert cell._use_type_indexed_nest_coefficients
        assert cell._use_direct_state_rnn_loop
        assert not cell._use_direct_csr_recurrent_gradient
        assert not cell.use_fused_recurrent_accumulation
        for name in ("dpointnet_nest_state_forward", "dpointnet_nest_state_history_backward_events"):
            assert "launch_geometry_version" in inspect.signature(getattr(glif_state_ops._OPS, name)).parameters
    finally:
        cell.close_fused_cuda()
        tf.keras.mixed_precision.set_global_policy("float32")
