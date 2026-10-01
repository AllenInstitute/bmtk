# Automatic alpha-basis fitting

DPointNet's GLIF3 RNN builds missing SONATA synaptic coefficients automatically.
Existing CSV coefficients or embedded `basis_weights` remain authoritative and
are not refitted by default. No input files or installed packages are changed.

Omit `tau_basis`, `synaptic_basis_weights`, and `basis_weights_file` when importing
a new network with raw synaptic kinetics. The default tries four shared alpha
functions, then five if the error threshold is not met. Configure it in JSON:

```json
{
  "rnn_cell_params": {
    "cell_model": "GLIF3",
    "alpha_basis": {
      "min_basis": 4,
      "max_basis": 5,
      "tolerance": 0.08012288897995931,
      "n_points": 1000,
      "max_iterations": 100,
      "seed": 42,
      "force_recompute": false
    }
  }
}
```

The same dictionary is accepted inside `RNN(cell_params=...)`. Set `alpha_basis`
to `false` to disable the fallback. Normally, if `tau_basis` is supplied, missing coefficients
are fitted to that fixed basis; its time constants and dimension are not changed.
Configured coefficient files must be valid import inputs. A nonexistent
time-constant file is also an error unless force recomputation explicitly ignores
the old time constants; the default does not silently replace experiment input.

Set `alpha_basis.force_recompute=true` to explicitly discard the supplied basis
in memory and refit **both shared time constants and all coefficients** from raw
kinetics. This overrides supplied CSV/embedded coefficients, explicit
`synaptic_basis_weights`, and `tau_basis`; it is not a weight-only fit against
the old taus. The supplied coefficient files remain unchanged. Every build/rebuild
with this flag enabled performs a fresh fit, with deterministic seeds. Missing raw
kinetics or failure to meet tolerance raises without replacing the existing rows.
The option must be a JSON boolean and defaults to `false`. Diagnostics record it.

## Method and tolerance

This follows Javier Galvan's `synaptic_data/alpha_basis_calculation.ipynb`, inspected
at `JavierGalvan9/V1_GLIF_model` commit
`2c52ec10c1eee409ddf900f8a5b8460cf9c46d24`:

- A peak-normalized alpha is `alpha(t, tau) = (t/tau) * exp(1-t/tau)`.
- Each target is `alpha(t, fast) + amp_slow * alpha(t, slow)`.
- At each shared-tau candidate, solve all per-class coefficients by unconstrained
  `numpy.linalg.lstsq`. Negative coefficients are allowed. Coefficients are
  temporal shape factors, not new physical edge weights.
- Minimize the unweighted mean waveform MSE across synaptic dynamics classes.
  Use seeded differential evolution followed by multi-start L-BFGS-B refinement.
  Unlike the notebook's fixed-four override, all fitted taus remain optimized.

For arbitrary networks, log-tau search bounds come from the input kinetics:
`min(fast, slow)/10` to `max(fast, slow)*10`. The default waveform window is
`max(30 ms, 10*max(fast, slow))`; `time_range` can override it explicitly.
These generalize the notebook's V1-specific 0.1--15 ms bounds and 30 ms window.
No regularization, amplitude renormalization, nonnegative constraint, or
neuron/edge-frequency reweighting is introduced.

Acceptance uses the **maximum across classes** of
`sqrt(mean((target-fit)^2) / mean(target^2))`. This sampled relative RMS metric
prevents low-amplitude classes from disappearing inside an aggregate absolute
MSE. The smallest attempted dimension meeting tolerance is selected. If no
dimension passes, build raises with the achieved error; tolerance is never
automatically relaxed. Solver budgets are finite and do not guarantee the global
optimum. Very extreme fast/slow ratios may require increasing `n_points`.

The default is twice the archived V1 four-basis worst-class relative RMS error:
`2 * 0.040061444489979656 = 0.08012288897995931`. Calibration covered all 90 used
synapse classes from the 2010 unique target-type/synapse pairs in `core_nll_0`.
The archived taus were `[0.9895877164, 1.8801909320, 3.6289592843, 5.6534043090]` ms.
This is a waveform approximation threshold, not a bound on voltage, spike timing,
training gradients, or network behavior. A stricter application should explicitly
choose its tolerance and sampling window.

## Kinetics and execution

Synapse JSONs may contain scalar `tau_syn` (or `tau_syn_fast`), optional
`tau_syn_slow` (defaults to fast), and optional `amp_slow` (defaults to zero).
Alternatively, JSONs may provide a one-based `receptor_type`; the actual target
cell JSONs must contain matching `tau_syn_fast`, `tau_syn_slow`, and `amp_slow`
arrays. Target types are recovered from actual SONATA edge/node IDs using bounded
edge chunks, not inferred from naming or target-query strings.

One dynamics file must identify one kinetic class. If a receptor maps to different
kinetics in different targets, use distinct dynamics files for those classes.
Missing raw kinetics fail clearly; existing coefficients alone cannot recover
the original waveform. A partially supplied coefficient table requires its shared
`tau_basis`, so supplied and generated rows remain compatible.

Recurrent and external-input populations share one fitted basis and synapse
mapping. Generated rows are keyed by full dynamics-file path. CSV loading accepts
consecutive `w0` through `wN` columns and preserves unrelated metadata columns.
General CPU/CUDA paths support five functions. Explicitly enabled four-only
accelerators are still subject to their existing topology checks; do not force
them for a five-basis network.

After `rnn.build()`, inspect `rnn.alpha_basis_fit` for taus, coefficients, per-class
MSE/relative RMS errors, input file order, seed, attempted dimensions and timing.
It is `None` if supplied coefficients were used or fitting was disabled. Fitted
taus also appear in `rnn.cell_params['tau_basis']`; rebuilding the same RNN reuses
its generated rows unless `force_recompute=true`. There is no global fitting cache
or automatic disk write.

For numerical use without a model:

```python
from bmtk.simulator.dpointnet.alpha_basis import fit_alpha_basis

result = fit_alpha_basis([[1.0, 4.0, 0.3], [2.0, 8.0, 0.2]])
taus = result['tau_basis']
weights = result['weights']
```

The fitting tests cover deterministic coefficients, tolerance failures, actual
four-to-five escalation, immutable source JSONs, noncanonical target IDs,
recurrent/input basis sharing, and supplied five-column CSV model execution.