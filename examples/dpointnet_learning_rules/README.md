# DPointNet learning-rule example

This example runs e-prop or a configurable three-factor rule on the small
all-to-all GLIF3 network. It reuses the assets in `../dpointnet_all2all`.

Select a rule in a DPointNet training configuration:

```json
"learning_rule": {
  "name": "three_factor",
  "signal": "spike",
  "surfaces": ["<recurrent>"]
}
```

Supported signals are `combined`, `spike`, and `voltage`. The default
`learning_rule` remains `bptt`.

```bash
python run_learning_rule.py eprop
python run_learning_rule.py three_factor
python run_learning_rule.py eprop --task silence
python run_learning_rule.py eprop --task rate_control
```

Use `--batch-size` to select a different batch size:

```bash
python run_learning_rule.py three_factor --batch-size 8
```

Each invocation starts from the same network and RNG configuration, trains only
the recurrent surface, and evaluates the model before and after training on the
same deterministic Poisson batch. It reports:

- mean firing rate versus the 20 Hz target,
- target-rate mean squared error,
- voltage range penalty,
- total objective,
- recurrent-weight change and Dale-sign preservation.

In all tasks, 10 Hz external Poisson input drives the network for 100 ms. The
total objective is the sum of:

- `TargetFiringRate`: mean squared error between each neuron's rate and the
  task target (omitted for `voltage_control`, 0 Hz for `silence`, and 20 Hz for
  `rate_control`);
- `VoltageRegularization`: squared distance from threshold for
  `voltage_control`, or a 0.25-scaled out-of-range penalty for the spike tasks.

The default smooth voltage-control task performs 50 SGD steps and requires at
least 1% improvement on the fixed evaluation batch. For that task, e-prop and
voltage-only three-factor are equivalent because the spike learning signal is
zero. The optional spike tasks exercise the spike-derived third factor, but
their thresholded finite-window objectives are diagnostics rather than
guaranteed convergence benchmarks.

The shared eligibility calculation applies to rules using the derivative of
the GLIF3 state with respect to a weight. Rules requiring different neuron
dynamics, synaptic state, structural plasticity, or non-weight parameters must
implement and register their own `LearningRule` class with
`dpointnet.register_learning_rule`.
