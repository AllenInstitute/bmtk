import pytest

from bmtk.simulator.dpointnet.cell_models.nest_dynamics import resolve_hard_reset_options
from bmtk.simulator.dpointnet.cell_models.glif3_cell import GLIF3Cell
from bmtk.simulator.dpointnet.rnn_model import RNN


def test_omitted_nest_defaults_to_hard_surrogate():
    assert resolve_hard_reset_options() == (True, "soft_surrogate")
    assert resolve_hard_reset_options(dynamics_mode="legacy") == (False, "exact")


@pytest.mark.parametrize("hard", [True, False])
def test_explicit_reset_preserves_exact_gradient_compatibility(hard):
    assert resolve_hard_reset_options(hard) == (hard, "exact")


def test_explicit_soft_and_exact_hard_remain_supported():
    assert resolve_hard_reset_options(False, "exact") == (False, "exact")
    assert resolve_hard_reset_options(True, "soft_surrogate") == (True, "soft_surrogate")
    with pytest.raises(ValueError, match="requires"):
        resolve_hard_reset_options(False, "soft_surrogate")
    with pytest.raises(ValueError, match="requires"):
        resolve_hard_reset_options(True, "soft_surrogate", dynamics_mode="legacy")


@pytest.mark.parametrize("value", [0, 1, "true"])
def test_reset_boolean_validation(value):
    with pytest.raises(ValueError, match="hard_reset"):
        resolve_hard_reset_options(value)


def test_legacy_null_reset_preserves_context_dependent_reset():
    assert resolve_hard_reset_options(None, training=True) == (False, "exact")
    assert resolve_hard_reset_options(None) == (True, "exact")
    assert resolve_hard_reset_options(None, dynamics_mode="legacy") == (False, "exact")


def test_rnn_resolves_same_default_without_mutating_config():
    rnn = object.__new__(RNN)
    rnn.cell_params = {}
    rnn.cell_cls = GLIF3Cell
    rnn._model_built = False
    for training in (False, True):
        resolved = rnn._resolve_cell_params(training=training)
        assert resolved["hard_reset"] is True
        assert resolved["hard_reset_gradient_mode"] == "soft_surrogate"
    assert rnn.cell_params == {}
    rnn.cell_params = {"hard_reset": False}
    assert rnn._resolve_cell_params(training=True)["hard_reset"] is False
    rnn.cell_params = {"hard_reset": True}
    with pytest.raises(ValueError, match="GLIF training"):
        rnn._resolve_cell_params(training=True)
