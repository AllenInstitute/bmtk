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

Only `single` training currently supports the local-rule interface. The
`series`, `parallel`, and `series_accumulate` approaches continue to use BPTT.
Rules based on loss derivatives (`eprop` and `three_factor`) are not
strict-local even though their weight update has three factors. The
`pair_stdp` and `local_rate_homeostasis` rules use only synapse-local spike or
rate traces.

## Configuration reference

All non-BPTT rules share these optional settings:

| Option | Default | Meaning |
|---|---:|---|
| `surrogate_dampening` | `null` | Override the cell spike-surrogate scale; `null` uses `rnn_cell_params.dampening_factor`. |
| `edge_chunk_size` | `65536` | Maximum edges processed in one eligibility chunk; must be positive. |
| `surfaces` | `null` | Weight surfaces to update. `null` selects every trainable surface; use `"<recurrent>"` or an input-population name to restrict it. |
| `learning_signal_clip` | `null` | Symmetric elementwise clip for the learning signal. |
| `gradient_clip_norm` | `null` | Per-surface gradient norm limit. |
| `min_weight` | `null` | Optional lower bound enforced after each update. |
| `max_weight` | `null` | Optional upper bound enforced after each update; it must not be below `min_weight`. |

Conservative configurations should set `surfaces` explicitly and leave all
clipping and bounds disabled until their units are established for the target
network.

### Loss-derived eligibility rules

`eprop` uses the combined spike and voltage loss derivatives and has no
additional options. `three_factor` adds:

| Option | Default | Choices or meaning |
|---|---:|---|
| `signal` | `"combined"` | `"combined"`, `"spike"`, or `"voltage"`. |
| `spike_signal_scale` | `1.0` | Multiplier for the spike-derived third factor. |
| `voltage_signal_scale` | `1.0` | Multiplier for the voltage-derived third factor. |

### Strict-local rules

`pair_stdp` uses delayed presynaptic and postsynaptic spike traces:

| Option | Default | Choices or meaning |
|---|---:|---|
| `tau_pre_ms` | `20.0` | Presynaptic trace time constant. |
| `tau_post_ms` | `20.0` | Postsynaptic trace time constant. |
| `a_plus` | `1.0` | Potentiation amplitude. |
| `a_minus` | `1.0` | Depression amplitude. |
| `weight_dependence` | `"additive"` | `"additive"` or `"multiplicative"`. |

`local_rate_homeostasis` requires `target_rate_hz` and adds:

| Option | Default | Choices or meaning |
|---|---:|---|
| `target_rate_hz` | required | Non-negative postsynaptic target rate. |
| `tau_pre_ms` | `20.0` | Presynaptic trace time constant. |
| `tau_rate_ms` | `100.0` | Postsynaptic rate-trace time constant. |
| `update` | `"pre_trace_times_post_rate_error"` | Also supports `"multiplicative_post_rate_scaling"`. |

### Restricted-modulator rules

`modulated_eligibility` averages a loss-derived signal within a finite set of
broadcast channels. It is restricted-global rather than strict-local:

| Option | Default | Choices or meaning |
|---|---:|---|
| `n_channels` | `1` | Positive number of broadcast channels. |
| `signal` | `"combined"` | `"combined"`, `"spike"`, or `"voltage"`. |
| `channel_projection` | `"fixed_balanced_partition"` | Also supports `"cell_class_partition"`. |

`neuron_local_three_factor` is a delayed-association rule with a target error
specific to each postsynaptic readout neuron:

| Option | Default | Choices or meaning |
|---|---:|---|
| `modulator` | `"postsynaptic_rate_error"` | Also supports `"postsynaptic_voltage_error"`. |
| `pool_a_start`, `pool_a_end` | `0`, `100` | Half-open first readout pool. |
| `pool_b_start`, `pool_b_end` | `100`, `200` | Half-open second readout pool; pools must be ordered and disjoint. |
| `cue_duration_ms` | `20.0` | Duration used to identify cue events. |
| `response_window_ms` | `50.0` | Positive post-delay learning window. |
| `high_target_rate_hz` | `20.0` | Target rate for the class-selected pool. |
| `low_target_rate_hz` | `0.0` | Target rate for the other pool. |
| `target_rate_hz` | `null` | Optional common target when class and delay targets are absent. |
| `tau_rate_ms` | `20.0` | Postsynaptic rate-trace time constant. |
| `high_target_voltage_offset` | `0.0` | High target relative to normalized threshold. |
| `low_target_voltage_offset` | `-1.0` | Low target relative to normalized threshold. |
| `cue_memory_tau_ms` | `50.0` | Cue-memory trace time constant. |
| `cue_memory_gain` | `0.0` | Cue-memory contribution; zero disables it. |
| `cue_memory_floor` | `1.0` | Baseline multiplier for the local factor. |
| `cue_memory_signal` | `"surrogate"` | `"surrogate"` or `"spike"`. |
| `cue_memory_mode` | `"postsynaptic_gate"` | Also supports `"synaptic_eligibility"`. |
| `target_scope` | `"all_readout"` | Also supports `"target_pool_only"`. |

### Delayed-association input and loss

The optional `delayed_cue_spikes` input emits balanced two-class Poisson cues
and `class_label`/`delay_ms` targets. It requires a non-empty `delays_ms` list
and defaults to `cue_duration_ms=20.0`, `background_rate_hz=5.0`,
`cue_rate_hz=80.0`, `probe_duration_ms=0.0`, `probe_rate_hz=0.0`, and
`seed=null`.

`DelayedAssociationLoss` reads those targets. Its defaults are readout pools
`[0, 100)` and `[100, 200)`, `cue_duration_ms=20.0`,
`response_window_ms=50.0`, and `temperature_hz=5.0`. Pool indices must fit the
network and should be changed explicitly for networks that do not reserve the
first 200 neurons as two readout populations.

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
