"""Private CSR-live parameter layout and explicit canonical migration boundary."""

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
    def __init__(self, connectivity):
        n = connectivity["n_edges"]
        metadata = connectivity.get("master_metadata")
        if metadata is None:
            metadata = tf.raw_ops.ReadVariableOp(
                resource=connectivity["metadata_handle"],
                dtype=tf.as_dtype(connectivity["index_dtype"]),
            )
        start = 2 * n + connectivity["n_sources"] + 1
        order = np.asarray(metadata.numpy()[start:start + n], dtype=np.int64)
        if not np.array_equal(np.sort(order), np.arange(n)):
            raise ValueError("CSR master mapping must be a reversible edge permutation.")
        self.checkpoint_key = "CSR_MASTER_V1_" + hashlib.sha256(
            order.astype("<i8", copy=False).tobytes()
        ).hexdigest()
        with tf.device("/CPU:0"):
            self.csr_to_canonical = tf.constant(order, tf.int64)
            self.canonical_to_csr = tf.constant(np.argsort(order), tf.int64)

    def to_live(self, canonical):
        return _permute(tf.convert_to_tensor(canonical), self.csr_to_canonical, self.canonical_to_csr)

    def to_canonical(self, live):
        return _permute(tf.convert_to_tensor(live), self.canonical_to_csr, self.csr_to_canonical)

    def assign_canonical(self, master, canonical):
        return master.assign(self.to_live(canonical))

    def canonical_indices_to_live(self, indices):
        indices = tf.convert_to_tensor(indices)
        if indices.dtype not in (tf.int32, tf.int64, tf.uint32):
            raise TypeError("Canonical loss indices must be integer edge-row indices.")
        return tf.cast(
            tf.gather(self.canonical_to_csr, tf.cast(indices, tf.int64)), indices.dtype
        )

    def convert_state(self, values, *, to_live):
        """Convert explicit per-edge masters, Adam slots and constraint arrays."""
        transform = self.to_live if to_live else self.to_canonical
        return {name: transform(value) for name, value in values.items()}


def guard_csr_checkpoint(master, layout=None):
    """Give CSR resources a nonlegacy checkpoint key; never reinterpret old rows."""
    resource = master.value
    if callable(resource):
        resource = master
    key = "CSR_MASTER_V1" if layout is None else layout.checkpoint_key

    def serialize(variable):
        return {key: variable.read_value()}

    def restore(variable, restored_tensors):
        return variable.assign(restored_tensors[key])

    resource._serialize_to_tensors = types.MethodType(serialize, resource)
    resource._restore_from_tensors = types.MethodType(restore, resource)


def transfer_canonical_state(source, target, *, per_edge_slots, scalar_state=()):
    """Explicit checkpoint conversion after restoring the source representation.

    ``per_edge_slots`` maps each surface name to (source_slot, target_slot)
    pairs. Supply every optimizer slot explicitly; scalar_state covers iterations
    and other nonedge optimizer state. Save a fresh target checkpoint afterward.
    """
    surfaces = {"<recurrent>", *source.inputs}
    if surfaces != {"<recurrent>", *target.inputs} or set(per_edge_slots) != surfaces:
        raise ValueError("Migration requires identical surfaces and explicit slots for each.")
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
        ) if source._experimental_csr_master else None
        target_layout = (
            target.recurrent_master_layout if name == "<recurrent>"
            else target.inputs[name]["master_layout"]
        ) if target._experimental_csr_master else None
        source_constraint = getattr(source_master, "constraint", None)
        target_constraint = getattr(target_master, "constraint", None)
        if source_constraint is not None and target_constraint is not None:
            source_sign = source_constraint.condition
            target_sign = target_constraint.condition
            if source_layout:
                source_sign = source_layout.to_canonical(source_sign)
            if target_layout:
                target_sign = target_layout.to_canonical(target_sign)
            tf.debugging.assert_equal(source_sign, target_sign, message="Constraint signs differ")
        assignments.append((target_master, target_layout.to_live(canonical)
                            if target_layout else canonical))
        for source_slot, target_slot in per_edge_slots[name]:
            if source_slot.shape != canonical.shape or target_slot.shape != canonical.shape:
                raise ValueError("Migration slots must have the corresponding edge shape.")
            value = source_layout.to_canonical(source_slot) if source_layout else source_slot
            assignments.append((target_slot, target_layout.to_live(value)
                                if target_layout else tf.identity(value)))
    for src, dst in scalar_state:
        if src.shape != dst.shape or src.dtype != dst.dtype:
            raise ValueError("Migration scalar state shapes/dtypes disagree.")
        assignments.append((dst, tf.identity(src)))
    for variable, value in assignments:
        variable.assign(value)
    target.refresh_recurrent_weight_shadow()
