from types import SimpleNamespace

from bmtk.simulator.dpointnet import tf_utils


def test_explicit_memory_budget_is_not_replaced_by_growth(monkeypatch):
    gpu = SimpleNamespace(name="GPU:0")
    messages = []
    monkeypatch.setattr(tf_utils.tf.config, "list_physical_devices", lambda kind: [gpu])
    monkeypatch.setattr(tf_utils.tf.config, "get_logical_device_configuration",
                        lambda device: [SimpleNamespace(memory_limit=20480)])
    monkeypatch.setattr(tf_utils.io, "log_info", messages.append)
    monkeypatch.setattr(tf_utils.tf.config.experimental, "set_memory_growth",
                        lambda *args: (_ for _ in ()).throw(AssertionError("budget overwritten")))
    tf_utils.enable_gpu_memory_growth()
    assert len(messages) == 1 and "Preserving explicit" in messages[0]


def test_default_memory_growth_is_preserved(monkeypatch):
    gpu = SimpleNamespace(name="GPU:0")
    calls = []
    monkeypatch.setattr(tf_utils.tf.config, "list_physical_devices", lambda kind: [gpu])
    monkeypatch.setattr(tf_utils.tf.config, "get_logical_device_configuration", lambda device: None)
    monkeypatch.setattr(tf_utils.tf.config.experimental, "set_memory_growth",
                        lambda device, enabled: calls.append((device, enabled)))
    tf_utils.enable_gpu_memory_growth()
    assert calls == [(gpu, True)]
