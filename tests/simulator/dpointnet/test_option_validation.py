import numpy as np
import pytest

from bmtk.simulator.dpointnet._options import validate_bool_option
from bmtk.simulator.dpointnet.cell_models.glif3_cell import (
    _validate_direct_csr_gradient_option,
    _validate_fixed4_forward_option,
    _validate_fused_cuda_option,
    _validate_fused_current_accumulation_option,
    _validate_pair_projection_option,
)
from bmtk.simulator.dpointnet.custom_ops.csr_spike_ops import _validate_packed_sm120_option


PARSERS = [
    (_validate_fused_cuda_option, "use_fused_cuda", True),
    (_validate_pair_projection_option, "use_pair_projection", True),
    (_validate_fixed4_forward_option, "use_fixed4_input_forward", False),
    (_validate_fused_current_accumulation_option, "use_fused_current_accumulation", False),
    (_validate_direct_csr_gradient_option, "use_direct_csr_recurrent_gradient", False),
    (_validate_packed_sm120_option, "use_packed_sm120_backward", True),
]


@pytest.mark.parametrize("parser,name,auto", PARSERS)
@pytest.mark.parametrize("value,expected", [
    (True, True), (False, False), (np.array(True), True), (np.array(False), False),
])
def test_existing_parsers_preserve_boolean_scalar_normalization(parser, name, auto, value, expected):
    assert parser(value) is expected


@pytest.mark.parametrize("parser,name,auto", PARSERS)
@pytest.mark.parametrize("value", [
    "auto", b"auto", np.str_("auto"), np.bytes_("auto"),
    np.array("auto"), np.array(b"auto"),
])
def test_existing_parsers_preserve_auto_support(parser, name, auto, value):
    if auto:
        assert parser(value) == "auto"
    else:
        with pytest.raises(ValueError) as error:
            parser(value)
        assert str(error.value) == f"{name} must be true or false."


@pytest.mark.parametrize("parser,name,auto", PARSERS)
@pytest.mark.parametrize("value", [
    0, 1, None, "true", b"\xff", np.bool_(True), np.array(1),
    np.array([True]), np.array(["auto"]), object(),
])
def test_existing_parsers_preserve_invalid_values_and_messages(parser, name, auto, value):
    with pytest.raises(ValueError) as error:
        parser(value)
    expected = f'{name} must be true, false, or "auto".' if auto else f"{name} must be true or false."
    assert str(error.value) == expected


@pytest.mark.parametrize("value", [True, False])
def test_strict_parser_accepts_only_python_booleans(value):
    assert validate_bool_option(value, "test_flag") is value


@pytest.mark.parametrize("value", [0, 1, None, "auto", np.bool_(True), np.array(True)])
def test_strict_parser_does_not_inherit_numpy_or_auto_normalization(value):
    with pytest.raises(ValueError, match="^test_flag must be true or false\\.$"):
        validate_bool_option(value, "test_flag")
