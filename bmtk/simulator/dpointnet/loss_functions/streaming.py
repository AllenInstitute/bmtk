"""Declared streaming support for DPointNet losses.

The checkpointed-streaming and direct-recorded-adjoint training routes never
materialize the ``[batch, time, neurons]`` spike sequence. Losses on those
routes receive a :class:`~bmtk.simulator.dpointnet.spike_summary.SpikeSummary`
(FP32 spike counts over declared windows plus declared trace plans) instead.
A loss states that it can work from such statistics by declaring the
:class:`StreamingSpikeLoss` contract; the training engine admits a loss to
the streaming routes only on that declaration, never on its class name, so a
project-local loss is treated exactly like a built-in one.

Contract (all optional overrides have defaults suitable for a count-only loss):

``supports_streaming``
    Class attribute. ``True`` declares the contract; ``False`` opts a class
    out explicitly. A loss without the attribute is *undeclared* and keeps the
    dense route.

``streaming_unmet(context)``
    Instance-level prerequisites that do not hold, as user-facing sentences
    (empty when the loss can stream in this ``context``).

``count_windows(sequence_length)``
    The ``(start, stop)`` count windows the loss reads through
    ``SpikeSummary.temporal_sum()`` / ``SpikeSummary.trim()``. The full window
    is always available.

``prepare_spike_statistics(batch_size, dtype, sequence_length, batch_start)``
    Trace plans for one parameter group, as a tuple of trace specifications
    (``SampledSpikeTraces`` or ``ProjectedSpikeTraces`` from
    ``spike_statistics``); ``()`` when the loss needs counts only. Plans are
    data interpreted by the runners, never callbacks. A loss returning plans
    reads them back with ``SpikeSummary.trace_for_loss(self.name)`` and may
    appear once per parameter group.

``streaming_plan_kinds``
    Class attribute describing the plans the loss declares (``"counts"``,
    ``"sampled traces"``, ``"projected traces"``), used for reporting and the
    reuse check; the actual plans are validated when prepared.

``build_streaming(context)``
    Optional eager hook run once before tracing (lazy target generators, tables).

``__call__(spikes, **kwargs)`` must accept a ``SpikeSummary`` (and may still
accept dense spike tensors on the dense route). On every route the engine also
passes ``voltages``, ``model_state`` and the batch labels ``y`` of the loss's
parameter group; ``SpikeSummary`` rows, trace rows and ``y`` rows refer to the
same samples (the whole group, single replica). Windows are in steps; the
context carries ``dt`` in ms.
"""

from typing import NamedTuple

PLAN_KINDS = ("counts", "sampled traces", "projected traces")
TRACE_PLAN_KINDS = PLAN_KINDS[1:]
_BUILTIN_PREFIX = __name__.rsplit(".", 1)[0] + "."


class StreamingContext(NamedTuple):
    """Engine-level facts a loss may check in ``streaming_unmet`` / use in ``build_streaming``.

    Time base: ``sequence_length``, count windows and trace windows are in **steps**;
    ``dt`` is the step in milliseconds (``steps = ms / dt``). ``batch_sizes`` are the
    per-parameter-group batch sizes; a loss is called once per group with the trace,
    count and ``y`` rows of exactly that group (single replica).
    """
    sequence_length: int
    batch_sizes: tuple
    cell: object
    neuron_partition: object
    replicas: int
    dt: float = 1.0


class StreamingSpikeLoss:
    """Mixin declaring the streaming contract with count-only defaults."""

    supports_streaming = True
    streaming_plan_kinds = ("counts",)

    def streaming_unmet(self, context):
        """Route prerequisites only (fallback sentences); configuration errors should raise
        in ``__init__`` or ``build_streaming`` instead of silently selecting the dense route."""
        return ()

    def build_streaming(self, context):
        """Eager hook, called once outside any ``tf.function`` before the first rollout
        is traced (e.g. to build target generators or constant tables)."""
        return None

    def count_windows(self, sequence_length):
        pre = getattr(self, "_pre_delay", None) or 0
        post = getattr(self, "_post_delay", None) or 0
        if pre or post:
            return ((int(pre), int(sequence_length) - int(post)),)
        return ()

    def prepare_spike_statistics(self, batch_size, dtype, sequence_length, batch_start=0):
        return ()


def streaming_basis(loss):
    """``"built-in"``, ``"declared"``, ``"opted-out"`` or ``"undeclared"`` for this loss."""
    declared = getattr(type(loss), "supports_streaming", None)
    if declared is None:
        return "undeclared"
    if declared is not True:
        return "opted-out"
    return "built-in" if type(loss).__module__.startswith(_BUILTIN_PREFIX) else "declared"


def streaming_declaration(loss):
    """Report entry for one loss: class, module, admission basis and declared plan kinds."""
    cls = type(loss)
    basis = streaming_basis(loss)
    kinds = tuple(getattr(cls, "streaming_plan_kinds", ("counts",))) if basis in ("built-in", "declared") else ()
    return {
        "name": getattr(loss, "name", cls.__name__),
        "class": cls.__name__,
        "module": cls.__module__,
        "basis": basis,
        "plans": kinds,
    }


def declares_trace_plans(loss):
    kinds = getattr(type(loss), "streaming_plan_kinds", ())
    return any(kind in TRACE_PLAN_KINDS for kind in kinds)


def streaming_unmet(loss, context):
    """Sentences describing why ``loss`` cannot stream, or an empty list."""
    basis = streaming_basis(loss)
    name = type(loss).__name__
    if basis == "undeclared":
        return [
            f"{name} does not declare streaming support (subclass "
            "bmtk.simulator.dpointnet.loss_functions.StreamingSpikeLoss to run on the streaming routes)."
        ]
    if basis == "opted-out":
        return [f"{name} opts out of streaming (supports_streaming is {type(loss).supports_streaming!r})."]
    unknown = [kind for kind in getattr(type(loss), "streaming_plan_kinds", ()) if kind not in PLAN_KINDS]
    if unknown:
        return [f"{name} declares unknown streaming plan kinds {unknown}; known kinds are {list(PLAN_KINDS)}."]
    unmet = loss.streaming_unmet(context)
    if unmet is None:
        unmet = ()
    elif isinstance(unmet, str):
        unmet = (unmet,)
    unmet = [str(sentence) for sentence in unmet]
    if declares_trace_plans(loss) and getattr(context, "neuron_partition", None) is not None:
        unmet.append(f"{name} trace plans are not neuron-partition aware.")
    return unmet


def count_windows(loss, sequence_length):
    """Validated ``(start, stop)`` windows a declared loss reads."""
    windows = getattr(loss, "count_windows", None)
    if windows is None:
        return ()
    result = []
    for window in windows(sequence_length):
        if (
            not isinstance(window, (tuple, list)) or len(window) != 2
            or any(isinstance(v, bool) or not isinstance(v, int) for v in window)
            or not 0 <= window[0] < window[1] <= sequence_length
        ):
            raise ValueError(
                f"Invalid spike-statistics loss window {tuple(window) if isinstance(window, (tuple, list)) else window} "
                f"from {type(loss).__name__}; windows must satisfy 0 <= start < stop <= {sequence_length}."
            )
        result.append((int(window[0]), int(window[1])))
    return tuple(result)


def prepare_spike_statistics(loss, batch_size, dtype, sequence_length, batch_start):
    """Normalized, type-checked trace plans of a declared loss for one batch group."""
    from ..spike_statistics import ProjectedSpikeTraces, SampledSpikeTraces

    prepare = getattr(loss, "prepare_spike_statistics", None)
    if prepare is None:
        return ()
    plans = prepare(batch_size, dtype, sequence_length, batch_start)
    if plans is None:
        return ()
    if isinstance(plans, (SampledSpikeTraces, ProjectedSpikeTraces)):
        plans = (plans,)
    plans = tuple(plans)
    for plan in plans:
        if not isinstance(plan, (SampledSpikeTraces, ProjectedSpikeTraces)):
            raise ValueError(
                f"{type(loss).__name__}.prepare_spike_statistics returned {type(plan).__name__}; trace plans "
                "must be SampledSpikeTraces or ProjectedSpikeTraces (plans are data interpreted by the runners)."
            )
        if tuple(plan.batch_range) != (batch_start, batch_start + batch_size):
            raise ValueError(
                f"{type(loss).__name__} trace plan covers batch rows {tuple(plan.batch_range)}; expected "
                f"{(batch_start, batch_start + batch_size)} for its parameter group."
            )
    return plans
