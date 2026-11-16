"""Project-local losses on the streaming routes through the declared protocol.

A loss is admitted to checkpointed streaming / the direct recorded adjoint by its
``StreamingSpikeLoss`` declaration, never by its class name. The CPU tests cover
the protocol helpers and configuration-time registration; the GPU tests run a
project-local count-window loss and a projected-trace loss, declared in this test
module, through ``"auto"`` and compare them with the checkpointed streaming
reference and the dense route.
"""

import sys
import textwrap
from types import SimpleNamespace

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from bmtk.simulator.dpointnet.loss_functions import (  # noqa: E402
    LossModules, StreamingSpikeLoss, import_loss_modules, streaming,
)
from bmtk.simulator.dpointnet.spike_summary import SpikeSummary  # noqa: E402
from bmtk.simulator.dpointnet.spike_statistics import ProjectedSpikeTraces, SampledSpikeTraces  # noqa: E402


@pytest.fixture(autouse=True)
def restore_policy():
    """``make_cell`` sets the global mixed-precision policy; leave it as found."""
    previous = tf.keras.mixed_precision.global_policy()
    yield
    tf.keras.mixed_precision.set_global_policy(previous)


class _CountWindowLoss(StreamingSpikeLoss):
    """Project-style loss: mean squared spike count over a trimmed window."""

    def __init__(self, pre_delay=1, post_delay=2, cost=1.0):
        self._pre_delay = pre_delay
        self._post_delay = post_delay
        self.cost = cost

    def __call__(self, spikes, **kwargs):
        if isinstance(spikes, SpikeSummary):
            counts = spikes.trim(self._pre_delay, self._post_delay).temporal_sum()
        else:
            window = spikes[:, self._pre_delay:spikes.shape[1] - self._post_delay]
            counts = tf.reduce_sum(tf.cast(window, tf.float32), axis=1)
        return self.cost * tf.reduce_mean(tf.square(counts))


class _ProjectedTraceLoss(StreamingSpikeLoss):
    """Project-style loss on a fixed linear readout of the spikes."""

    streaming_plan_kinds = ("projected traces",)

    def __init__(self, projection, window, cost=1.0):
        self.projection = tf.constant(projection, tf.float32)
        self.window = tuple(window)
        self.cost = cost

    def prepare_spike_statistics(self, batch_size, dtype, sequence_length, batch_start=0):
        return ProjectedSpikeTraces(self.projection, self.window, (batch_start, batch_start + batch_size))

    def __call__(self, spikes, **kwargs):
        if isinstance(spikes, SpikeSummary):
            traces = spikes.trace_for_loss(self.name)
        else:
            start, stop = self.window
            projected = tf.einsum("btn,nf->bft", tf.cast(spikes[:, start:stop], tf.float32), self.projection)
            traces = tf.reshape(projected, [-1, stop - start])
        return self.cost * tf.reduce_mean(tf.square(traces))


class _LabelWeightedCountLoss(StreamingSpikeLoss):
    """Count loss whose value depends on the batch labels ``y`` (row-aligned with counts)."""

    def __init__(self, cost=1.0):
        self.cost = cost
        self.built_with = None

    def build_streaming(self, context):
        self.built_with = context  # eager hook: runs once before tracing
        assert not tf.inside_function()

    def __call__(self, spikes, y=None, **kwargs):
        counts = spikes.temporal_sum() if isinstance(spikes, SpikeSummary) else tf.reduce_sum(
            tf.cast(spikes, tf.float32), axis=1)
        weights = tf.cast(tf.reshape(y["gain"], [-1, 1]), tf.float32)  # one gain per batch row
        return self.cost * tf.reduce_mean(tf.square(weights * counts))


class _Undeclared:
    def __call__(self, spikes, **kwargs):
        return tf.reduce_mean(tf.cast(spikes, tf.float32))


class _OptedOut(StreamingSpikeLoss):
    supports_streaming = False

    def __call__(self, spikes, **kwargs):
        return tf.reduce_mean(tf.cast(spikes, tf.float32))


def _context(length=7, partition=None):
    return streaming.StreamingContext(length, (4,), None, partition, 1, dt=1.0)


def test_declaration_bases_and_unmet_sentences():
    from bmtk.simulator.dpointnet.loss_functions import TargetFiringRate

    assert streaming.streaming_basis(TargetFiringRate(SimpleNamespace(dt=1.0, seq_len=7), firing_rate=5)) == "built-in"
    declared = _CountWindowLoss()
    assert streaming.streaming_basis(declared) == "declared"
    assert streaming.streaming_unmet(declared, _context()) == []
    entry = streaming.streaming_declaration(declared)
    assert entry["basis"] == "declared" and entry["plans"] == ("counts",) and entry["module"] == __name__
    undeclared = streaming.streaming_unmet(_Undeclared(), _context())
    assert len(undeclared) == 1 and "does not declare streaming support" in undeclared[0]
    assert streaming.streaming_declaration(_Undeclared())["basis"] == "undeclared"
    opted = streaming.streaming_unmet(_OptedOut(), _context())
    assert len(opted) == 1 and "opts out" in opted[0]
    assert streaming.streaming_declaration(_OptedOut())["basis"] == "opted-out"

    class Unknown(StreamingSpikeLoss):
        streaming_plan_kinds = ("histograms",)

    assert "unknown streaming plan kinds" in streaming.streaming_unmet(Unknown(), _context())[0]

    class Failing(StreamingSpikeLoss):
        def streaming_unmet(self, context):
            return "needs at least 10 steps" if context.sequence_length < 10 else ()

    assert streaming.streaming_unmet(Failing(), _context(7)) == ["needs at least 10 steps"]
    assert streaming.streaming_unmet(Failing(), _context(12)) == []
    # Trace plans are not neuron-partition aware; count-only losses are.
    projected = _ProjectedTraceLoss(np.ones((3, 2)), (1, 6))
    assert streaming.streaming_unmet(projected, _context(partition=object())) == [
        "_ProjectedTraceLoss trace plans are not neuron-partition aware."]
    assert streaming.streaming_unmet(_CountWindowLoss(), _context(partition=object())) == []
    assert _context().dt == 1.0


def test_count_windows_and_plan_validation():
    assert streaming.count_windows(_CountWindowLoss(1, 2), 7) == ((1, 5),)
    assert streaming.count_windows(_CountWindowLoss(0, 0), 7) == ()
    assert streaming.count_windows(_Undeclared(), 7) == ()
    with pytest.raises(ValueError, match="Invalid spike-statistics loss window"):
        streaming.count_windows(_CountWindowLoss(4, 4), 7)
    loss = _ProjectedTraceLoss(np.ones((3, 2)), (1, 6))
    plans = streaming.prepare_spike_statistics(loss, 4, tf.float16, 7, 2)
    assert len(plans) == 1 and plans[0].batch_range == (2, 6)
    assert streaming.prepare_spike_statistics(_CountWindowLoss(), 4, tf.float16, 7, 0) == ()
    assert streaming.declares_trace_plans(loss) and not streaming.declares_trace_plans(_CountWindowLoss())

    class WrongRows(_ProjectedTraceLoss):
        def prepare_spike_statistics(self, batch_size, dtype, sequence_length, batch_start=0):
            return ProjectedSpikeTraces(self.projection, self.window, (0, 1))

    with pytest.raises(ValueError, match="covers batch rows"):
        streaming.prepare_spike_statistics(WrongRows(np.ones((3, 2)), (1, 6)), 4, tf.float16, 7, 2)

    class Callback(StreamingSpikeLoss):
        streaming_plan_kinds = ("sampled traces",)

        def prepare_spike_statistics(self, batch_size, dtype, sequence_length, batch_start=0):
            return (lambda spikes: spikes,)

    with pytest.raises(ValueError, match="plans are data interpreted by the runners"):
        streaming.prepare_spike_statistics(Callback(), 4, tf.float16, 7, 0)
    assert isinstance(SampledSpikeTraces, type)


def test_engine_admits_declared_losses_without_a_gpu():
    """The engine's prerequisite list treats a declared project loss like a built-in one."""
    from bmtk.simulator.dpointnet import training
    from bmtk.simulator.dpointnet.learning_rules import BPTTLearningRule

    def engine_with(losses):
        engine = object.__new__(training.TrainingEngine)
        engine.gradient_checkpointing = True
        engine._learning_rule = BPTTLearningRule()
        engine.rnn = SimpleNamespace(cell=SimpleNamespace(temporal_gradient_precision="compute"),
                                     strategy=tf.distribute.get_strategy(), data_parallel=None)
        engine._training_approach = "single"
        parameter = SimpleNamespace(batch_size=4, seq_len=7, loss_functions=dict(losses), name="p0")
        engine._parameters = [parameter]
        return engine

    engine = engine_with({"count": _CountWindowLoss(1, 2), "proj": _ProjectedTraceLoss(np.ones((3, 2)), (1, 6))})
    assert engine._spike_statistics_unmet() == []
    assert engine._statistics_windows() == ((0, 7), (1, 5))
    entries = engine.streaming_loss_declarations()
    assert [(e["key"], e["basis"], e["plans"]) for e in entries] == [
        ("count", "declared", ("counts",)), ("proj", "declared", ("projected traces",)),
    ]
    engine = engine_with({"count": _CountWindowLoss(), "custom": _Undeclared()})
    unmet = engine._spike_statistics_unmet()
    assert len(unmet) == 1 and unmet[0].startswith("_Undeclared does not declare streaming support")
    proj = _ProjectedTraceLoss(np.ones((3, 2)), (1, 6))
    engine = engine_with({"a": proj, "b": proj})
    assert any("cannot be reused twice" in sentence for sentence in engine._spike_statistics_unmet())
    engine = engine_with({"c": _CountWindowLoss(4, 4)})
    assert any("Invalid spike-statistics loss window" in sentence for sentence in engine._spike_statistics_unmet())
    report = {"selected": {"stream_spike_statistics": True, "recorded_adjoint": "direct"},
              "record_offload": ("voltage",), "losses": entries}
    assert training.TrainingEngine.training_route_mode(report) == (
        "checkpointed streaming + direct recorded adjoint (record offload: voltage)")
    described = training.TrainingEngine.describe_route_losses(report)
    assert f"count [_CountWindowLoss: declared by {__name__}, counts]" in described
    assert "proj [_ProjectedTraceLoss: declared by" in described and "projected traces]" in described
    assert training.TrainingEngine.training_route_mode(
        {"selected": {"stream_spike_statistics": False, "recorded_adjoint": False}}) == "dense spike sequences"
    # Native library status is part of the report/checklist (a missing library or
    # .archs manifest is a named reason, not a silent slower route).
    native = training.TrainingEngine._native_library_status()
    assert set(native) == {"csr", "state", "record_store"} and all(isinstance(v, str) for v in native.values())
    report["native"] = native
    report["requested"] = {"stream_spike_statistics": "auto", "recorded_adjoint": "auto"}
    report["automatic"] = report["selected"]
    report["reasons"] = {"stream_spike_statistics": "ok", "recorded_adjoint": "ok"}
    engine = engine_with({"count": _CountWindowLoss(1, 2)})
    checklist = engine.training_route_checklist(report)
    assert any(line.startswith("native csr library: ") for line in checklist)
    assert any(line.startswith("count windows: 0-7, 1-5") for line in checklist)
    # Eager build hook runs once, outside tracing, with the context (dt included).
    labelled = _LabelWeightedCountLoss()
    engine = engine_with({"labelled": labelled})
    engine.rnn.dt = 0.5
    engine.build_streaming_losses()
    engine.build_streaming_losses()
    assert labelled.built_with.sequence_length == 7 and labelled.built_with.dt == 0.5
    assert labelled.built_with.batch_sizes == (4,)
    with pytest.raises(ValueError, match="require_route must be"):
        training.TrainingEngine.__init__(object.__new__(training.TrainingEngine), None, 1, 1, require_route="fast")


def test_import_loss_modules_registers_project_losses(tmp_path, monkeypatch):
    package = tmp_path / "proj_losses_pkg"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "losses.py").write_text(textwrap.dedent('''
        import tensorflow as tf
        from bmtk.simulator.dpointnet import StreamingSpikeLoss, register_loss_module

        @register_loss_module
        class ProjectCountLoss(StreamingSpikeLoss):
            def __init__(self, rnn=None, cost=1.0, **kwargs):
                self.cost = cost

            @staticmethod
            def module():
                return "ProjectCountLoss"

            def __call__(self, spikes, **kwargs):
                return self.cost * tf.reduce_mean(spikes.temporal_sum())
    '''))
    monkeypatch.syspath_prepend(str(tmp_path))
    modules = import_loss_modules(["proj_losses_pkg.losses"])
    assert len(modules) == 1
    cls = LossModules().get_module("ProjectCountLoss")
    assert streaming.streaming_basis(cls()) == "declared"
    assert modules[0].ProjectCountLoss is cls  # the decorator returns the class
    assert import_loss_modules(()) == () and import_loss_modules("proj_losses_pkg.losses")
    with pytest.raises(ValueError, match="training.loss_modules: could not import 'no_such_project.losses'"):
        import_loss_modules(["no_such_project.losses"])
    with pytest.raises(ValueError, match="importable module paths"):
        import_loss_modules([3])
    sys.modules.pop("proj_losses_pkg.losses", None)
    sys.modules.pop("proj_losses_pkg", None)


# --- GPU: declared project losses on the automatic route -------------------------------

def _gpu_fixture():
    from test_precision_credit import make_cell
    from test_direct_recorded_adjoint import DIRECT_OPTIONS, _engine_for, _model, fused_cuda_available

    if not fused_cuda_available():
        pytest.skip("Authorized GPU and rebuilt CUDA operators required.")
    return make_cell, DIRECT_OPTIONS, _engine_for, _model


def _project_losses(projection):
    def losses(rnn):
        from bmtk.simulator.dpointnet.loss_functions import TargetFiringRate
        return {
            "rate": TargetFiringRate(rnn, firing_rate=5, compiled_spike_counts=True),
            "window": _CountWindowLoss(1, 2, cost=0.5),
            "readout": _ProjectedTraceLoss(projection, (1, 6), cost=2.0),
            "labelled": _LabelWeightedCountLoss(cost=0.25),
        }
    return losses


_LABELS = {"gain": tf.constant(np.linspace(0.5, 2.0, 32, dtype=np.float32).reshape(32, 1))}


def _run(engine, cell, model, states, y=None):
    values = tf.zeros((32, 7, 0), cell.compute_dtype)
    labels = _LABELS if y is None else y
    losses, gradients, _ = tf.function(lambda: engine._single_gradients((values,), labels, tuple(states)))()
    return losses, [tf.identity(g) for g in tf.nest.flatten(gradients) if g is not None]


def test_declared_project_losses_take_the_direct_route_under_auto(monkeypatch):
    """Project-local declared losses: ``"auto"`` selects the direct recorded adjoint, the
    route log names each loss with its basis, and losses/gradients match the checkpointed
    streaming reference exactly and the dense route to compute-dtype rounding."""
    make_cell, DIRECT_OPTIONS, _engine_for, _model = _gpu_fixture()
    from bmtk.simulator.dpointnet.direct_recorded_adjoint import DirectRecordedAdjointRunner
    from bmtk.simulator.dpointnet.segmented_recompute import SegmentedRecomputeRunner
    from bmtk.simulator.dpointnet.spike_statistics import SpikeStatisticsRunner
    from bmtk.simulator.dpointnet import training

    messages = []
    monkeypatch.setattr(training.io, "log_info", messages.append)
    results = {}
    for route, options in (
        ("auto", {}),
        ("streaming", {"stream_spike_statistics": True, "recorded_adjoint": False}),
        ("dense", {"stream_spike_statistics": False}),
    ):
        tf.keras.utils.set_random_seed(23)
        cell = make_cell("nest", selective=True, batch_size=32, hard_reset_gradient_mode="soft_surrogate", **DIRECT_OPTIONS)
        try:
            states = list(cell.zero_state(32, cell.compute_dtype))
            states[1] = tf.tile(tf.cast([[0.9, 1.1]], states[1].dtype), [32, 1])
            model = _model(cell, tuple(states))
            neurons = int(states[1].shape[-1])
            projection = np.random.default_rng(5).normal(size=(neurons, 3))
            engine = _engine_for(cell, states, model, _project_losses(projection), **options)
            engine.prepare_gradient_checkpointing()
            report = engine.training_route_report
            if route == "auto":
                assert engine.recorded_adjoint == "direct"
                assert isinstance(engine._extractor_forward, DirectRecordedAdjointRunner)
                assert report["mode"] == "checkpointed streaming + direct recorded adjoint"
                bases = {entry["key"]: entry["basis"] for entry in report["losses"]}
                assert bases == {"rate": "built-in", "window": "declared", "readout": "declared", "labelled": "declared"}
                assert all(str(v).startswith("loaded from") for v in report["native"].values())
                labelled = engine.parameters[0].loss_functions["labelled"]
                assert labelled.built_with is not None and labelled.built_with.dt == 1.0
                assert {entry["key"]: entry["plans"] for entry in report["losses"]}["readout"] == ("projected traces",)
                line = [m for m in messages if m.startswith("Training route:")][-1]
                assert "checkpointed streaming + direct recorded adjoint" in line
                assert f"window [_CountWindowLoss: declared by {__name__}, counts]" in line
                assert "readout [_ProjectedTraceLoss: declared by" in line and "projected traces]" in line
                assert "rate [TargetFiringRate: built-in, counts]" in line
            elif route == "streaming":
                assert isinstance(engine._extractor_forward, SpikeStatisticsRunner)
            else:
                assert isinstance(engine._extractor_forward, SegmentedRecomputeRunner)
                assert report["mode"] == "dense spike sequences"
            results[route] = _run(engine, cell, model, states)
        finally:
            cell.close_fused_cuda()
    # Labels drive the value: doubling the per-row gains quadruples the labelled term only.
    tf.keras.utils.set_random_seed(23)
    cell = make_cell("nest", selective=True, batch_size=32, hard_reset_gradient_mode="soft_surrogate", **DIRECT_OPTIONS)
    try:
        states = list(cell.zero_state(32, cell.compute_dtype))
        states[1] = tf.tile(tf.cast([[0.9, 1.1]], states[1].dtype), [32, 1])
        model = _model(cell, tuple(states))
        engine = _engine_for(cell, states, model, _project_losses(projection), require_route="direct")
        engine.prepare_gradient_checkpointing()
        doubled = _run(engine, cell, model, states, y={"gain": _LABELS["gain"] * 2.0})[0]["p0"]
        np.testing.assert_allclose(doubled["labelled"], 4.0 * results["auto"][0]["p0"]["labelled"], rtol=1e-5)
        np.testing.assert_array_equal(doubled["window"], results["auto"][0]["p0"]["window"])
    finally:
        cell.close_fused_cuda()
    for name in ("rate", "window", "readout", "labelled"):
        assert float(results["auto"][0]["p0"][name]) > 0
        np.testing.assert_array_equal(results["auto"][0]["p0"][name], results["streaming"][0]["p0"][name])
        np.testing.assert_allclose(results["auto"][0]["p0"][name], results["dense"][0]["p0"][name], rtol=1e-4, atol=1e-6)
    for direct, reference in zip(results["auto"][1], results["streaming"][1]):
        np.testing.assert_allclose(direct.numpy(), reference.numpy(), rtol=2e-3, atol=1e-4)
    for direct, dense in zip(results["auto"][1], results["dense"][1]):
        np.testing.assert_allclose(direct.numpy(), dense.numpy(), rtol=2e-3, atol=1e-4)
        assert np.any(direct.numpy() != 0)


def test_undeclared_and_opted_out_losses_keep_the_dense_route_visibly(monkeypatch):
    make_cell, DIRECT_OPTIONS, _engine_for, _model = _gpu_fixture()
    from bmtk.simulator.dpointnet.segmented_recompute import SegmentedRecomputeRunner
    from bmtk.simulator.dpointnet import training

    messages = []
    monkeypatch.setattr(training.io, "log_info", messages.append)
    for loss, phrase in ((_Undeclared(), "does not declare streaming support"), (_OptedOut(), "opts out of streaming")):
        cell = make_cell("nest", selective=True, batch_size=32, hard_reset_gradient_mode="soft_surrogate", **DIRECT_OPTIONS)
        try:
            states = list(cell.zero_state(32, cell.compute_dtype))
            model = _model(cell, tuple(states))

            def losses(rnn, loss=loss):
                from bmtk.simulator.dpointnet.loss_functions import TargetFiringRate
                return {"rate": TargetFiringRate(rnn, firing_rate=5, compiled_spike_counts=True), "custom": loss}

            engine = _engine_for(cell, states, model, losses)
            engine.prepare_gradient_checkpointing()
            assert isinstance(engine._extractor_forward, SegmentedRecomputeRunner)
            report = engine.training_route_report
            assert report["mode"] == "dense spike sequences" and phrase in report["reasons"]["stream_spike_statistics"]
            custom = [entry for entry in report["losses"] if entry["key"] == "custom"][0]
            assert custom["basis"] in ("undeclared", "opted-out") and phrase in custom["unmet"][0]
            line = [m for m in messages if m.startswith("Training route:")][-1]
            assert "dense spike sequences" in line and phrase in line
            with pytest.raises(ValueError, match=phrase):
                _engine_for(cell, states, model, losses, stream_spike_statistics=True).prepare_gradient_checkpointing()
            # require_route turns the visible fallback into an error naming the reason and the libraries.
            with pytest.raises(ValueError, match="require_route='direct'.*native libraries"):
                _engine_for(cell, states, model, losses, require_route="direct").prepare_gradient_checkpointing()
            with pytest.raises(ValueError, match="require_route='streaming'"):
                _engine_for(cell, states, model, losses, require_route="streaming").prepare_gradient_checkpointing()
        finally:
            cell.close_fused_cuda()
