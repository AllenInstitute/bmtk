# DPointNet learning-rule example

This example compares DPointNet's local learning rules on the small all-to-all
GLIF3 network. It reuses the network and component files in
`../dpointnet_all2all` rather than copying them.

First run the update diagnostic:

```bash
python diagnose_updates.py --output-dir output/figures
```

It uses one deterministic forward pass with a dimensionless objective:

```text
rate_loss = mean(((firing_rate_hz - 20) / 20) ** 2)
voltage_loss = mean(relu(abs(voltage - 0.5) - 0.5) ** 2)
total_loss = rate_loss + voltage_loss
```

Both component coefficients are 1. For every rule it reports:

- update norm;
- cosine similarity and inner product with the exact BPTT gradient for the
  component that the rule uses;
- relative difference from e-prop.
- actual before/after forward loss after one equal-norm constrained update.

The diagnostic also verifies the expected identities and differences:

- combined three-factor is e-prop under the same learning signals;
- spike-only and voltage-only three-factor select different third factors;
- ModProp adds a distinct delayed population-feedback term.

It writes one-step sensitivity data/plots (`loss_progression.*`),
convergence-aware loss trajectories (`training_progression.*`), and
`task_and_voltage_loss.png` when `--output-dir` is supplied.

Using equal update norm is deliberate: ModProp's raw update is much larger, so
one shared optimizer learning rate would compare both direction and arbitrary
scale. The diagnostic reports the post-Dale-constraint applied norm to make
clipping visible and verifies every sign constraint. This isolates update
direction from raw magnitude.

At the checked operating point, one update of norm 0.1 changes total loss from
0.82559 to 0.82449 for e-prop, 0.82451 for spike-only three-factor, 0.82415 for
voltage-only three-factor, and 0.82555 for ModProp.

The training plot also recomputes each rule after every full-batch update, but
it does not normalize every update. Instead, each rule gets an initial learning
rate calibrated so its first applied update norm is 0.03. Subsequent
update norms follow the rule's raw signal and the plateau schedule below.

The stopping policy is deliberately conservative for the noisy spike
objectives. It fits a linear trend over 100 updates. When the projected
improvement is at most `5e-4` of the initial reference objective, the learning
rate is reduced to 30% of its previous value instead of stopping immediately.
Training terminates only when the plateau recurs after three such reductions,
with a 2,000-update safety cap. Triangles in the plot mark reductions and
squares mark termination.

Every rule reached this revised criterion:

- e-prop stopped at update 1,517: total loss 0.82559 to 0.66772 (19.12%
  reduction and its best value);
- spike-only three-factor stopped at update 1,540: total loss to 0.66793
  (19.10% reduction; best 0.66772 at update 1,237);
- voltage-only three-factor stopped at update 1,526: total loss to 0.75453
  (8.61% reduction and its best value);
- ModProp stopped at update 399: total loss increased to 0.85421, after a
  small best value of 0.82541 at update 2.

All trajectories and updates remain finite. E-prop and spike-only continue
improving until roughly update 1,200; the reductions then expose their terminal
plateaus rather than truncating a descending trend. The jagged portions reflect
the firing-rate quantization described below, rather than numerical failure.

This task is intentionally diagnostic and has two important limitations:

1. A 100 ms window quantizes each sample's firing-rate estimate in 10 Hz
   increments per spike, so the firing-rate MSE is jagged.
2. With natural scaling, the spike learning-signal norm is about 282 times the
   voltage learning-signal norm. Therefore e-prop is close to spike-only
   three-factor on this task; this is expected rather than hidden by rescaling.

Therefore, the converged plot supports e-prop and the selected three-factor
paths on this toy objective. It does not validate ModProp learning performance:
ModProp's implementation tests pass and its initial direction is positively
aligned with BPTT, but its repeated updates increase this task's loss. A
different task, filter construction, or step-control strategy is required
before claiming useful ModProp learning.

Use the training runner for before/after forward behavior:

```bash
python run_learning_rule.py eprop
python run_learning_rule.py eprop --task silence
python run_learning_rule.py eprop --task rate_control
```

Use `--batch-size` to demonstrate that local learning supports batches:

```bash
python run_learning_rule.py modprop --batch-size 8
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

The default voltage task performs 50 SGD steps at learning rate 20.0. The large
rate compensates for averaging a smooth loss over all neurons and timesteps; it
is specific to this shared-path sanity check. The runner fails unless the fixed-batch
objective falls by at least 1%. With the checked-in seed, each rule reduces the
loss from about 0.65766 to 0.64777 (about 1.50%) while preserving Dale signs.

Those identical values are expected, not an algorithm comparison: the task has
no spike loss, voltage-only three-factor equals e-prop, and ModProp's additional
term is zero. The result validates only the shared GLIF3 eligibility dynamics and voltage
learning signal. Since a voltage-only loss has zero spike learning signal,
ModProp's additional population-feedback term is inactive and ModProp reduces
to its e-prop base term on this task. Its additional delayed, type-specific
term is covered by deterministic tests in
`tests/simulator/dpointnet/test_learning_rules.py`. The optional spike tasks
exercise that term, but are diagnostics rather than guaranteed convergence
benchmarks; approximate local rules need not descend every finite-batch
thresholded spike objective.

The presets differ as follows:

- `eprop` uses the combined spike and voltage learning signal.
- `three_factor` uses the spike-derived modulatory factor only.
- `modprop` adds fixed node-type-specific temporal modulation with three taps.

The shared eligibility calculation is general for local rules whose synaptic
factor is the derivative of the GLIF3 state with respect to a weight. Rules can
customize the third factor and temporal/population modulation without
reimplementing GLIF3 synaptic dynamics. It is not universal: a rule requiring a
different synaptic state, non-GLIF3 dynamics, structural plasticity, or
non-weight parameters must implement its own state/update calculation through
the learning-rule interface.
