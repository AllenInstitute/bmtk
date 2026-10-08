"""Shared option parsing without changing individual option contracts."""

import numpy as np


def validate_bool_option(value, name, *, allow_auto=False, unwrap_numpy=False):
    if unwrap_numpy and isinstance(value, np.ndarray) and value.ndim == 0:
        value = value.item()
    if value is True or value is False:
        return value
    if allow_auto:
        if isinstance(value, (bytes, np.bytes_)):
            try:
                value = value.decode("utf-8")
            except UnicodeDecodeError:
                pass
        if isinstance(value, (str, np.str_)) and value == "auto":
            return "auto"
        raise ValueError(f'{name} must be true, false, or "auto".')
    raise ValueError(f"{name} must be true or false.")
