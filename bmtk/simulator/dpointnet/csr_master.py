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


def _restore_uid(variable):
    candidates = [variable, _raw_variable(variable)]
    container = getattr(variable, "_distributed_container", None)
    if callable(container):
        candidates.append(container())
    uids = [getattr(candidate, "_self_update_uid", None) for candidate in candidates]
    return max([uid for uid in uids if uid is not None] or [-1])


def _restored_before_binding(variable):
    """True when a checkpoint already restored ``variable`` (TensorFlow restore uid).

    A ``tf.train.Checkpoint.restore`` issued before an object exists is applied
    when the object is created and tracked (for example optimizer slots created
    by ``optimizer.build``); those restored values are canonical rows.
    """
    return _restore_uid(variable) > -1


def canonical_checkpoint(variable, layout, *, verify_unchanged=False, restored="convert"):
    """Serialize a CSR-ordered edge variable in canonical order under ``VARIABLE_VALUE``.

    Saving converts a host copy (the live variable is never reordered); restoring
    converts the canonical rows back to the live layout. ``verify_unchanged`` is
    for immutable topology (endpoint indices): restoring different canonical rows
    raises instead of silently disagreeing with the native connectivity.

    A variable that a checkpoint restored before this call holds canonical rows:
    ``restored="convert"`` converts them to the live layout once, ``"raise"``
    raises (used when the value may have changed since the restore).
    """
    if restored not in ("convert", "raise"):
        raise ValueError('restored must be "convert" or "raise".')
    resource = _raw_variable(variable)
    if getattr(resource, "_dpointnet_canonical_layout", None) is layout:
        return variable
    first_binding = getattr(resource, "_dpointnet_canonical_layout", None) is None
    if first_binding and _restored_before_binding(variable):
        if verify_unchanged or restored == "raise":
            raise ValueError(
                "A checkpoint was restored into a CSR-ordered variable before its canonical "
                "checkpoint format was bound, and its canonical rows cannot be converted "
                "safely. Build the model and call cell.bind_optimizer_checkpoint_slots("
                "optimizer) (RNN.train and TrainingEngine do this) before the first "
                "optimizer update, then restore."
            )
        with tf.init_scope():
            resource.assign(_convert(layout, resource.read_value(), to_live=True))

    def serialize(var):
        return {"VARIABLE_VALUE": _convert(layout, var.read_value(), to_live=False)}

    def restore(var, restored_tensors):
        value = restored_tensors["VARIABLE_VALUE"]
        if verify_unchanged:
            # Immutable topology: verify and never assign, in eager and graph mode alike,
            # because native CSR metadata is not rebuilt from restored endpoints.
            message = (
                "Checkpoint connectivity differs from this model's CSR-ordered topology; "
                "rebuild the model from the checkpoint's network instead."
            )
            if tf.executing_eagerly():
                current = layout.host_to_canonical(var.numpy()) if layout.host_lifecycle else (
                    layout.to_canonical(var.read_value()).numpy()
                )
                if not np.array_equal(np.asarray(value), current):
                    raise ValueError(message)
                return var.read_value()
            current = _convert(layout, var.read_value(), to_live=False)
            value = tf.convert_to_tensor(value)
            if value.shape.rank is not None and value.shape.rank != current.shape.rank:
                same = tf.constant(False)
            else:
                same = tf.cond(
                    tf.reduce_all(tf.equal(tf.shape(value), tf.shape(current))),
                    lambda: tf.reduce_all(tf.equal(tf.cast(value, current.dtype), current)),
                    lambda: tf.constant(False),
                )
            check = tf.debugging.Assert(same, [message])
            with tf.control_dependencies([check]):
                return tf.identity(var.read_value())
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


# Complete per-variable optimizer state, by optimizer class (Keras 3 and TF-Keras 2
# "new" optimizers share class names). Each attribute is a list aligned with the
# optimizer's variable index; conditional lists (amsgrad, momentum, centered) are
# empty or absent when unused. Paired with their variable through the optimizer's
# own variable index, never by shape.
OPTIMIZER_SLOT_ATTRIBUTES = {
    "Adam": ("_momentums", "_velocities", "_velocity_hats"),
    "AdamW": ("_momentums", "_velocities", "_velocity_hats"),
    "ExponentiatedAdam": ("_momentums", "_velocities", "_velocity_hats"),
    "Nadam": ("_momentums", "_velocities"),
    "Adamax": ("_m", "_u"),
    "SGD": ("momentums",),
    "RMSprop": ("_velocities", "_momentums", "_average_gradients"),
    "Adagrad": ("_accumulators",),
    "Adadelta": ("_accumulated_grads", "_accumulated_delta_vars"),
    "Ftrl": ("_accumulators", "_linears"),
    "Lion": ("_momentums", "momentums"),
    "Lamb": ("_momentums", "_velocities"),
}
# Per-variable state of the optimizer base classes: exponential moving averages
# (use_ema) and Keras 3 gradient accumulation.
_BASE_SLOT_ATTRIBUTES = ("_model_variables_moving_average", "_accumulated_gradients")
_LEGACY = "legacy"


def _optimizer_classes():
    modules = [tf.keras.optimizers, getattr(tf.keras.optimizers, "experimental", None)]
    try:
        import keras
        modules.append(keras.optimizers)
    except ImportError:
        pass
    from .optimizers import ExponentiatedAdam
    classes = {ExponentiatedAdam: OPTIMIZER_SLOT_ATTRIBUTES["ExponentiatedAdam"]}
    for module in modules:
        for name, attributes in OPTIMIZER_SLOT_ATTRIBUTES.items():
            cls = getattr(module, name, None) if module is not None else None
            if isinstance(cls, type):
                classes.setdefault(cls, attributes)
    return classes


def _inner_optimizer(optimizer):
    while getattr(optimizer, "inner_optimizer", None) is not None:
        optimizer = optimizer.inner_optimizer
    return optimizer


def optimizer_slot_attributes(optimizer):
    """Complete per-variable slot attributes of a supported optimizer, else None.

    Loss-scale wrappers are unwrapped. Returns ``"legacy"`` for TF-Keras 2 legacy
    optimizers, whose ``get_slot_names``/``get_slot`` enumerate every slot.
    """
    inner = _inner_optimizer(optimizer)
    classes = _optimizer_classes()
    for cls in type(inner).__mro__:
        if cls in classes:
            return classes[cls] + _BASE_SLOT_ATTRIBUTES
    if hasattr(inner, "get_slot_names") and hasattr(inner, "get_slot") and hasattr(inner, "_slots"):
        return _LEGACY
    return None


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


def _indexed_slots(inner, attributes, index):
    slots = []
    for attribute in attributes:
        values = getattr(inner, attribute, None)
        if values is None:
            continue
        values = list(values)
        if not values:
            continue
        if index >= len(values) or values[index] is None:
            raise ValueError(
                f"{type(inner).__name__}.{attribute} does not hold a slot for every "
                "optimizer variable; its per-variable state cannot be paired."
            )
        slots.append(values[index])
    return slots


def _legacy_slots(inner, variable):
    try:
        return [
            slot for slot in (inner.get_slot(variable, name) for name in inner.get_slot_names())
            if slot is not None
        ]
    except KeyError:
        return []


def paired_optimizer_slots(optimizer, variable):
    """Complete per-variable optimizer state of ``variable``, by optimizer variable identity.

    Uses the explicit ``OPTIMIZER_SLOT_ATTRIBUTES`` allow-list (plus base-class EMA
    and gradient-accumulation lists) through the optimizer's variable index, or
    legacy ``get_slot``; never infers ownership from slot shapes. Returns [] when
    the optimizer does not track ``variable`` and None when the optimizer class is
    not supported.
    """
    inner = _inner_optimizer(optimizer)
    attributes = optimizer_slot_attributes(inner)
    if attributes is None:
        return None
    if attributes == _LEGACY:
        return _legacy_slots(inner, variable)
    index = _optimizer_variable_index(inner, variable)
    return [] if index is None else _indexed_slots(inner, attributes, index)


def _optimizer_state_variables(optimizer):
    """Every variable a checkpoint would save under ``optimizer`` (and its wrappers)."""
    found = {}

    def add(value):
        raw = _raw_variable(value)
        if isinstance(raw, tf.Variable):
            found.setdefault(id(raw), value)

    current = optimizer
    while current is not None:
        for value in tf.train.TrackableView(current).descendants():
            add(value)
        variables = getattr(current, "variables", None)
        try:
            variables = variables() if callable(variables) else variables
        except (TypeError, ValueError, AttributeError):
            variables = None
        for value in variables or ():
            add(value)
        legacy_slots = getattr(current, "_slots", None)
        if isinstance(legacy_slots, dict):
            for slots in legacy_slots.values():
                for value in slots.values():
                    add(value)
        current = getattr(current, "inner_optimizer", None)
    return list(found.values())


def _all_paired_slot_ids(inner, attributes):
    ids = set()
    if attributes == _LEGACY:
        for slots in inner._slots.values():
            ids.update(id(_raw_variable(slot)) for slot in slots.values())
        return ids
    for attribute in attributes:
        for value in getattr(inner, attribute, None) or ():
            if value is not None:
                ids.add(id(_raw_variable(value)))
    return ids


def bind_optimizer_checkpoint_slots(cell, optimizer, *, after_update=False):
    """Give every per-edge optimizer slot of a CSR-ordered master the canonical format.

    Call after the optimizer has created its slots (``optimizer.build``) and before
    the first optimizer update; ``RNN.train`` and ``TrainingEngine`` do this. The
    order relative to ``tf.train.Checkpoint.restore`` does not matter: slots that a
    checkpoint restored before binding hold canonical rows and are converted to the
    live layout once. ``after_update=True`` declares that updates may have run
    since such a restore; restored-but-unbound slots then raise instead.
    Idempotent. Returns ``{surface: number_of_slots}``.

    Raises ValueError for optimizers outside ``OPTIMIZER_SLOT_ATTRIBUTES`` (and
    legacy TF-Keras 2 optimizers) that track a CSR master, and for any variable
    saved under the optimizer that is shaped like a CSR master but is not paired
    with a variable, because its checkpoint would hold live CSR rows under
    canonical keys.
    """
    if not getattr(cell, "_csr_master_weights", False):
        return {}
    surfaces = [("<recurrent>", cell.recurrent_weight_values, cell.recurrent_master_layout)]
    surfaces += [
        (name, net["input_weight_values"], net["master_layout"]) for name, net in cell.inputs.items()
    ]
    inner = _inner_optimizer(optimizer)
    attributes = optimizer_slot_attributes(inner)
    unsupported = (
        f"Cannot pair the optimizer state of CSR-ordered masters for {type(inner).__name__}; "
        f"supported: {', '.join(sorted(OPTIMIZER_SLOT_ATTRIBUTES))} (Keras 3 or TF-Keras 2, "
        "optionally LossScaleOptimizer-wrapped) and TF-Keras 2 legacy optimizers. "
        "Use a supported optimizer or set use_csr_master_weights=False."
    )
    edge_shapes = {tuple(master.shape) for _, master, _ in surfaces}
    master_ids = {id(_raw_variable(master)) for _, master, _ in surfaces}
    bound = {}
    for name, master, layout in surfaces:
        if attributes is None:
            if _optimizer_variable_index(inner, master) is not None:
                raise ValueError(unsupported)
            continue
        slots = paired_optimizer_slots(inner, master)
        if not slots and (attributes == _LEGACY or _optimizer_variable_index(inner, master) is None):
            continue
        for slot in slots:
            if tuple(slot.shape) != tuple(master.shape):
                raise ValueError(f"Optimizer slot of {name!r} is not a per-edge slot.")
        restored = [
            _restore_uid(slot) for slot in slots
            if getattr(_raw_variable(slot), "_dpointnet_canonical_layout", None) is None
        ]
        if restored and max(restored) > _restore_uid(master):
            # Keras 3 checkpoints named-input masters only through the optimizer's
            # variable list, which optimizer.build replaces: a restore issued before
            # the build restores their slots but not the weights themselves.
            raise ValueError(
                f"A checkpoint restored the optimizer slots of CSR-ordered master {name!r} "
                "before the optimizer was built, but not the master itself. Build the "
                "optimizer before restoring (TrainingEngine.build_optimizer()), then "
                "restore again."
            )
        for slot in slots:
            canonical_checkpoint(slot, layout, restored="raise" if after_update else "convert")
        bound[name] = len(slots)
    # Safety net: everything shaped like a CSR master that the optimizer checkpoints
    # must be a bound slot or a slot paired with another (canonical-order) variable.
    paired = set() if attributes is None else _all_paired_slot_ids(inner, attributes)
    for value in _optimizer_state_variables(optimizer):
        raw = _raw_variable(value)
        if tuple(value.shape) not in edge_shapes or id(raw) in master_ids:
            continue
        if getattr(raw, "_dpointnet_canonical_layout", None) is not None:
            continue
        if id(raw) not in paired:
            raise ValueError(
                unsupported if attributes is None else
                f"{type(inner).__name__} saves per-edge state {getattr(value, 'name', value)!r} "
                "that is not paired with an optimizer variable; its checkpoint would hold "
                "live CSR rows. Use a supported optimizer or set use_csr_master_weights=False."
            )
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
