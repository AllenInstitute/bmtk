"""CSR-ordered FP32 master layout, canonical checkpoint format and migration helpers.

With ``use_csr_master_weights`` the live recurrent and named-input FP32 masters,
their endpoint index variables and paired optimizer slots are stored in stable
CSR (source-major) order. Checkpoints, exports and loss metadata stay canonical:
every CSR-ordered checkpointed variable serializes a host-converted canonical
copy under the ordinary ``VARIABLE_VALUE`` key and converts it back on restore,
so checkpoints are interchangeable with canonical-order models.
"""

import hashlib
import types

import numpy as np
import tensorflow as tf


@tf.custom_gradient
def _permute(values, order, inverse):
    result = tf.gather(values, order)

    def gradient(dresult):
        return tf.gather(tf.convert_to_tensor(dresult), inverse), None, None

    return result, gradient


class CsrMasterLayout:
    def __init__(self, connectivity, *, host_lifecycle=False):
        n = connectivity["n_edges"]
        self.host_lifecycle = host_lifecycle
        if host_lifecycle:
            order = np.asarray(connectivity["host_edge_order"])
        else:
            order = self._read_order(connectivity, n)
        if not np.array_equal(np.sort(order), np.arange(n)):
            raise ValueError("CSR master mapping must be a reversible edge permutation.")
        self.checkpoint_key = "CSR_MASTER_V1_" + hashlib.sha256(
            order.astype("<i8", copy=False).tobytes()
        ).hexdigest()
        if host_lifecycle:
            self.csr_to_canonical = order.astype(
                np.uint32 if n <= np.iinfo(np.uint32).max else np.uint64, copy=False
            )
            self.csr_to_canonical.setflags(write=False)
        else:
            with tf.device("/CPU:0"):
                self.csr_to_canonical = tf.constant(order, tf.int64)
                self._canonical_to_csr = tf.constant(np.argsort(order), tf.int64)

    @staticmethod
    def _read_order(connectivity, n):
        metadata = connectivity.get("master_metadata")
        if metadata is None:
            metadata = tf.raw_ops.ReadVariableOp(
                resource=connectivity["metadata_handle"],
                dtype=tf.as_dtype(connectivity["index_dtype"]),
            )
        start = 2 * n + connectivity["n_sources"] + 1
        return np.asarray(metadata.numpy()[start:start + n], dtype=np.int64)

    @property
    def canonical_to_csr(self):
        if not self.host_lifecycle:
            return self._canonical_to_csr
        inverse = np.empty_like(self.csr_to_canonical)
        inverse[self.csr_to_canonical] = np.arange(inverse.size, dtype=inverse.dtype)
        return inverse

    def host_to_live(self, canonical):
        canonical = self._host_edges(canonical)
        return canonical[np.asarray(self.csr_to_canonical)]

    def host_to_canonical(self, live):
        live = self._host_edges(live)
        result = np.empty_like(live)
        result[np.asarray(self.csr_to_canonical)] = live
        return result

    def _host_edges(self, values):
        values = np.asarray(values)
        if values.ndim == 0 or values.shape[0] != len(self.csr_to_canonical):
            raise ValueError("Host edge arrays must match the layout's edge dimension.")
        return values

    def _tensor_orders(self):
        if self.host_lifecycle:
            with tf.device("/CPU:0"):
                return (
                    tf.constant(self.csr_to_canonical, tf.int64),
                    tf.constant(self.canonical_to_csr, tf.int64),
                )
        return self.csr_to_canonical, self.canonical_to_csr

    def to_live(self, canonical):
        if self.host_lifecycle and not tf.is_tensor(canonical) and not hasattr(canonical, "assign"):
            with tf.device("/CPU:0"):
                return tf.convert_to_tensor(self.host_to_live(canonical))
        order, inverse = self._tensor_orders()
        return _permute(tf.convert_to_tensor(canonical), order, inverse)

    def to_canonical(self, live):
        order, inverse = self._tensor_orders()
        return _permute(tf.convert_to_tensor(live), inverse, order)

    def assign_canonical(self, master, canonical):
        if self.host_lifecycle:
            return master.assign(self.host_to_live(np.asarray(canonical)))
        return master.assign(self.to_live(canonical))

    def canonical_indices_to_live(self, indices):
        if self.host_lifecycle:
            if not tf.executing_eagerly():
                raise ValueError("Compose loss indices on the host before tracing.")
            dtype = tf.as_dtype(getattr(indices, "dtype", np.asarray(indices).dtype))
            if dtype not in (tf.int32, tf.int64, tf.uint32):
                raise TypeError("Canonical loss indices must be integer edge-row indices.")
            rows = np.asarray(indices)
            if np.any(rows < 0) or np.any(rows >= self.csr_to_canonical.size):
                raise ValueError("Canonical loss indices are outside the edge dimension.")
            with tf.device("/CPU:0"):
                return tf.convert_to_tensor(self.canonical_to_csr[rows], dtype=dtype)
        indices = tf.convert_to_tensor(indices)
        if indices.dtype not in (tf.int32, tf.int64, tf.uint32):
            raise TypeError("Canonical loss indices must be integer edge-row indices.")
        return tf.cast(
            tf.gather(self.canonical_to_csr, tf.cast(indices, tf.int64)), indices.dtype
        )

    def convert_state(self, values, *, to_live):
        """Convert explicit per-edge masters, Adam slots and constraint arrays."""
        if self.host_lifecycle:
            transform = self.host_to_live if to_live else self.host_to_canonical
            return {name: transform(np.asarray(value)) for name, value in values.items()}
        transform = self.to_live if to_live else self.to_canonical
        return {name: transform(value) for name, value in values.items()}


CSR_MASTER_SCHEMA = "direct_csr_fp32_master_host_lifecycle_v1"


def _raw_variable(variable):
    value = getattr(variable, "value", None)
    return variable if value is None or callable(value) else value


def _convert(layout, value, *, to_live):
    if tf.executing_eagerly() and layout.host_lifecycle:
        array = np.asarray(value.numpy() if hasattr(value, "numpy") else value)
        with tf.device("/CPU:0"):
            return tf.convert_to_tensor(
                layout.host_to_live(array) if to_live else layout.host_to_canonical(array)
            )
    with tf.device("/CPU:0"):
        return layout.to_live(value) if to_live else layout.to_canonical(value)


def canonical_checkpoint(variable, layout, *, verify_unchanged=False):
    """Serialize a CSR-ordered edge variable in canonical order under ``VARIABLE_VALUE``.

    Saving converts a host copy (the live variable is never reordered); restoring
    converts the canonical rows back to the live layout. ``verify_unchanged`` is
    for immutable topology (endpoint indices): restoring different canonical rows
    raises instead of silently disagreeing with the native connectivity.
    """
    resource = _raw_variable(variable)

    def serialize(var):
        return {"VARIABLE_VALUE": _convert(layout, var.read_value(), to_live=False)}

    def restore(var, restored_tensors):
        value = restored_tensors["VARIABLE_VALUE"]
        if verify_unchanged and tf.executing_eagerly():
            current = layout.host_to_canonical(var.numpy()) if layout.host_lifecycle else (
                layout.to_canonical(var.read_value()).numpy()
            )
            if not np.array_equal(np.asarray(value), current):
                raise ValueError(
                    "Checkpoint connectivity differs from this model's CSR-ordered topology; "
                    "rebuild the model from the checkpoint's network instead."
                )
            return var.read_value()
        return var.assign(_convert(layout, value, to_live=True))

    resource._serialize_to_tensors = types.MethodType(serialize, resource)
    resource._restore_from_tensors = types.MethodType(restore, resource)
    resource._dpointnet_canonical_layout = layout
    return variable


class CanonicalShadowCheckpoint(tf.Module):
    """Checkpoint-only stand-in for a canonical-order model's compute-dtype weight shadow.

    Canonical-order models track FP16/compute shadows of their masters in checkpoints.
    CSR-master models have no shadows; these stand-ins serialize the value such a
    model would hold (the cast master in canonical or CSR row order), so checkpoint
    object graphs match in both directions. Restoring ignores the value: the masters
    are authoritative.
    """

    def __init__(self, master, layout, dtype, *, canonical_order):
        super().__init__(name=None)
        self._source = types.SimpleNamespace(
            master=master, layout=layout, dtype=tf.as_dtype(dtype), canonical=canonical_order,
        )

    def _serialize_to_tensors(self):
        source = self._source
        value = _raw_variable(source.master).read_value()
        if source.canonical:
            value = _convert(source.layout, value, to_live=False)
        with tf.device("/CPU:0"):
            return {"VARIABLE_VALUE": tf.cast(value, source.dtype)}

    def _restore_from_tensors(self, restored_tensors):
        del restored_tensors
        return tf.no_op()


class CheckpointChildrenView(tf.Module):
    """Checkpoint-only node exposing an explicit child mapping for save and restore."""

    def __init__(self, children):
        super().__init__(name=None)
        self._view = types.SimpleNamespace(children=dict(children))

    def _trackable_children(self, save_type="checkpoint", **kwargs):
        return dict(self._view.children)

    def _lookup_dependency(self, name, cached_dependencies=None):
        return self._view.children.get(name)


# Per-variable optimizer slot lists, paired with their variable by the optimizer's
# own variable index (Keras 3 and TF-Keras optimizers, including ExponentiatedAdam).
_INDEXED_SLOT_ATTRIBUTES = ("_momentums", "_velocities", "_velocity_hats", "momentums")


def _optimizer_variable_index(optimizer, variable):
    lookup = getattr(optimizer, "_get_variable_index", None)
    if lookup is not None:
        try:
            return lookup(variable)
        except (KeyError, ValueError, AttributeError):
            pass
    index_dict = getattr(optimizer, "_index_dict", None)
    var_key = getattr(optimizer, "_var_key", None)
    if index_dict is not None and var_key is not None:
        return index_dict.get(var_key(variable))
    return None


def paired_optimizer_slots(optimizer, variable):
    """Explicit per-edge optimizer slots of ``variable``, by optimizer variable identity.

    Uses the optimizer's variable index for Keras 3/TF-Keras ``_momentums``,
    ``_velocities``, ``_velocity_hats`` and SGD ``momentums`` lists, or legacy
    ``get_slot``; never infers ownership from slot shapes. Returns None when the
    optimizer's slot structure is not recognized.
    """
    optimizer = getattr(optimizer, "inner_optimizer", None) or optimizer
    if hasattr(optimizer, "get_slot_names") and hasattr(optimizer, "get_slot"):
        return [
            slot for slot in (optimizer.get_slot(variable, name) for name in optimizer.get_slot_names())
            if slot is not None
        ]
    index = _optimizer_variable_index(optimizer, variable)
    if index is None:
        return None
    slots = []
    for attribute in _INDEXED_SLOT_ATTRIBUTES:
        values = getattr(optimizer, attribute, None)
        if isinstance(values, (list, tuple)) and len(values) > index and values[index] is not None:
            slots.append(values[index])
    return slots


def bind_optimizer_checkpoint_slots(cell, optimizer):
    """Give every per-edge optimizer slot of a CSR-ordered master the canonical format.

    Call after the optimizer has created its slots (``optimizer.build``). Idempotent.
    Returns ``{surface: number_of_slots}``; raises when a trainable CSR master's
    slots cannot be paired explicitly, because their checkpoints would otherwise
    hold live CSR rows under canonical keys.
    """
    if not getattr(cell, "_csr_master_weights", False):
        return {}
    surfaces = [("<recurrent>", cell.recurrent_weight_values, cell.recurrent_master_layout)]
    surfaces += [
        (name, net["input_weight_values"], net["master_layout"]) for name, net in cell.inputs.items()
    ]
    inner = getattr(optimizer, "inner_optimizer", None) or optimizer
    bound = {}
    for name, master, layout in surfaces:
        if _optimizer_variable_index(inner, master) is None and not (
            hasattr(inner, "get_slot_names") and master.trainable
        ):
            continue
        slots = paired_optimizer_slots(optimizer, master)
        if slots is None:
            raise ValueError(
                f"Cannot pair optimizer slots of CSR-ordered master {name!r} for "
                f"{type(inner).__name__}; use Adam/SGD/ExponentiatedAdam or set "
                "use_csr_master_weights=False."
            )
        for slot in slots:
            if tuple(slot.shape) != tuple(master.shape):
                raise ValueError(f"Optimizer slot of {name!r} is not a per-edge slot.")
            if getattr(_raw_variable(slot), "_dpointnet_canonical_layout", None) is not layout:
                canonical_checkpoint(slot, layout)
        bound[name] = len(slots)
    return bound


def transfer_canonical_state(source, target, *, per_edge_slots, scalar_state=()):
    """Explicit checkpoint conversion after restoring the source representation.

    ``per_edge_slots`` maps each surface name to (source_slot, target_slot)
    pairs. Supply every optimizer slot explicitly; scalar_state covers iterations
    and other nonedge optimizer state. Save a fresh target checkpoint afterward.
    """
    surfaces = {"<recurrent>", *source.inputs}
    if surfaces != {"<recurrent>", *target.inputs} or set(per_edge_slots) != surfaces:
        raise ValueError("Migration requires identical surfaces and explicit slots for each.")
    def convert(layout, value, to_live):
        if layout is None:
            return tf.identity(value)
        if layout.host_lifecycle:
            transform = layout.host_to_live if to_live else layout.host_to_canonical
            return transform(np.asarray(value))
        return layout.to_live(value) if to_live else layout.to_canonical(value)

    assignments = []
    for name in sorted(surfaces):
        canonical = source.canonical_weight_values(name)
        source_master = (
            source.recurrent_weight_values if name == "<recurrent>"
            else source.inputs[name]["input_weight_values"]
        )
        target_master = (
            target.recurrent_weight_values if name == "<recurrent>"
            else target.inputs[name]["input_weight_values"]
        )
        if canonical.shape != target_master.shape:
            raise ValueError("Checkpoint migration edge shapes disagree.")
        source_layout = (
            source.recurrent_master_layout if name == "<recurrent>"
            else source.inputs[name]["master_layout"]
        ) if source._csr_master_weights else None
        target_layout = (
            target.recurrent_master_layout if name == "<recurrent>"
            else target.inputs[name]["master_layout"]
        ) if target._csr_master_weights else None
        source_constraint = getattr(source_master, "constraint", None)
        target_constraint = getattr(target_master, "constraint", None)
        if source_constraint is not None and target_constraint is not None:
            source_sign = source_constraint.condition
            target_sign = target_constraint.condition
            if source_layout:
                source_sign = convert(source_layout, source_sign, False)
            if target_layout:
                target_sign = convert(target_layout, target_sign, False)
            tf.debugging.assert_equal(source_sign, target_sign, message="Constraint signs differ")
        assignments.append((target_master, convert(target_layout, canonical, True)))
        for source_slot, target_slot in per_edge_slots[name]:
            if source_slot.shape != canonical.shape or target_slot.shape != canonical.shape:
                raise ValueError("Migration slots must have the corresponding edge shape.")
            value = convert(source_layout, source_slot, False)
            assignments.append((target_slot, convert(target_layout, value, True)))
    for src, dst in scalar_state:
        if src.shape != dst.shape or src.dtype != dst.dtype:
            raise ValueError("Migration scalar state shapes/dtypes disagree.")
        assignments.append((dst, tf.identity(src)))
    for variable, value in assignments:
        variable.assign(value)
    target.refresh_recurrent_weight_shadow()
