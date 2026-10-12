Hard-forward, soft-surrogate training
=====================================

New NEST GLIF configurations default to **hard-reset forward dynamics with
soft-surrogate voltage gradients**. This retains hard reset and the refractory
voltage clamp during both training and reporting, without switching reset rules
after fitting. Explicit soft-reset configurations remain supported.

The forward reset still assigns the reset voltage, and the forward refractory
steps still clamp voltage there. The backward rule instead carries voltage
credit through the unclamped membrane integration and subtractive reset.
The hard-forward values are returned directly: the implementation does not
approximate them by cancelling floating-point additions.

Pin the learning rule
------------------------

Merge this fragment into a complete GLIF training configuration:

.. code-block:: json

   {
     "rnn_cell_params": {
       "dynamics_mode": "nest",
       "hard_reset": true,
       "hard_reset_gradient_mode": "soft_surrogate",
       "detach_reset": true,
       "detach_asc_reset": false,
       "voltage_state_gradient_scale": 1.0,
       "recurrent_spike_gradient_scale": 1.0
     }
   }

``hard_reset_gradient_mode`` accepts:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Value
     - Meaning
   * - ``"exact"``
     - Preserve the existing voltage derivatives for the selected reset policy.
       High-level hard-reset training is still rejected in this mode.
   * - ``"soft_surrogate"``
     - Default when NEST reset and derivative options are both omitted. Requires
       ``hard_reset=true`` and ``dynamics_mode="nest"``.

Here ``exact`` describes the voltage reset/clamp derivative, not an exact
derivative of threshold spiking: the separately configured spike surrogate
still applies.

Setting ``detach_reset=false`` attaches the spike-event derivative of the
subtractive backward reset. With ``detach_reset=true``, the voltage-state
path is still retained, but that spike-to-reset contribution is detached.
The ASC event and recurrent spike credit retain their existing controls.
Scale one means full configured retention, not removal of physical membrane decay.

What the backward rule means
----------------------------

For normalized voltage, reset at zero and threshold at one, the surrogate
reset is differentiated as ``v_after = v_before - spike`` while the forward
value remains the hard reset. With reset events attached, this gives a
voltage derivative ``1 - surrogate_spike_derivative`` rather than zero
at a fired neuron.

During refractory steps, voltage credit follows the usual membrane candidate,
including its physical decay and continuous-current dependence. It does not
differentiate the integer refractory counter, the Boolean active mask, or
the discrete choice of refractory duration. Spikes remain suppressed during
refractory steps. This restores continuous voltage credit, not every possible
derivative of a discrete hybrid system.

These are **deliberately approximate learning derivatives**, evaluated along
the hard-forward trajectory. They are neither the ordinary hard-reset Jacobian
nor the gradients of a separately simulated soft-reset trajectory. Finite
differences of the hard-forward map will therefore not validate the surrogate
rule; use its stated Jacobian as the backward reference.

Execution and compatibility
---------------------------

* Compatible native NEST state, event-VJP and state-history routes pair the
  existing hard-forward kernel with its soft-style backward kernel. The latter
  uses the actual hard-forward states and spike eligibility, not a second
  soft-forward simulation. Packed/type-indexed coefficients, direct state loops
  and voltage tracking retain their ordinary compatibility checks.
* Automatic acceleration uses the same hardware/ABI/precision eligibility as
  ordinary NEST dynamics. Pascal supports native CSR currents and eligible
  occupancy-aware state routes; old unmarked state binaries retain a guarded
  fallback. Packed options are not automatically forced on Pascal.
* No additional state-transition kernel is introduced for the surrogate.
  The Pascal launch repair adds defaulted capability attributes to generic
  NEST operators; rebuild both fat operators from this source for those routes.
  Missing requested native capabilities still raise.
* Ordinary FP32 and selective mixed FP16, activation checkpointing and the
  FP32 temporal-gradient reference have focused CPU value/gradient tests.
  Focused Titan Xp/SM61 and RTX 8000/SM75 tests also cover native values/VJPs, independent event
  attachment, static/live type-indexed dispatch, checkpoint/Poisson replay,
  pre-reset voltage credit and FP32 temporal event VJPs.
* Generic FP32 NEST launches on Pascal require the kernel-specific occupancy
  repair in this source. Rebuild both cluster-fat operators after
  that CUDA change; an older binary can fail with ``too many resources
  requested for launch`` despite advertising SM61 compilation coverage.
* The same mode and attachment settings must be used for checkpoint replay
  and resumed training. A previously built model must be rebuilt explicitly
  before changing reset policy or enabling this mode; changing its config
  dictionary does not change an already traced graph.
* For inference at fixed weights, ``exact`` and ``soft_surrogate`` have the
  same hard-forward dynamics on either state route. The latter does
  not turn inference into soft-reset simulation.

Measured native overhead
------------------------

The focused Titan Xp and RTX 8000 tests used three excluded warmups and twenty synchronized
measurements per route. All comparisons kept the same hard-forward input,
initial state and loss. The applied-update test restored identical canonical
masters and optimizer slots outside the timed region.

.. list-table:: Titan Xp median time in milliseconds
   :header-rows: 1
   :widths: 40 20 20 20

   * - Workload
     - Native exact-hard VJP control
     - Native hard-surrogate
     - TensorFlow hard-surrogate
   * - Synthetic 66,658-neuron BS8 state, 25 steps, forward and backward
     - 18.125
     - 18.047
     - 70.045
   * - Two-neuron 25-step applied Adam update
     - 60.760
     - 61.105
     - 74.677

.. list-table:: RTX 8000 median time in milliseconds
   :header-rows: 1
   :widths: 40 20 20 20

   * - Workload
     - Native exact-hard VJP control
     - Native hard-surrogate
     - TensorFlow hard-surrogate
   * - Synthetic 66,658-neuron BS8 state, 25 steps, forward and backward
     - 11.568
     - 11.724
     - 49.539
   * - Two-neuron 25-step applied Adam update
     - 63.123
     - 61.769
     - 79.871

Relative to the existing native derivative path, the state/update medians
changed by -0.43%/+0.57% on Titan Xp and +1.35%/-2.14% on RTX 8000.
These measurements support approximately stock native speed, not an exactly
zero-overhead claim. The implementation uses the existing backward kernel,
not an extra CUDA pass. The large state benchmark used selective FP32 voltage/ASC and FP16
synaptic state, but synthetic coefficients and fixed currents: it is **not**
full V1 connectivity or a V1 training-memory/convergence qualification.
The small update retained TensorFlow current projection. These timing ratios
must not be extrapolated to complete V1 training.

Bounded V1 training startup
---------------------------

A separate RTX 8000 execution test used the actual 66,658-neuron V1
``core_nll_0`` network, batch eight (six evoked/two spontaneous), 500 steps
and checkpoint interval 25. It retained the original objectives and ExpAdam
settings (learning rate 0.01, epsilon 1e-11), selective mixed FP16,
compute temporal gradients, attached reset/ASC events and gradient retention
one. Automatic native state, event-VJP and current routes were verified.

All three requested updates were accepted and finite, changed canonical
masters and passed configured constraints. Strict full model/optimizer
checkpoint restoration passed, with zero loss-scale rejections.
TensorFlow peak memory was 6,078,557,696 bytes (5.66 GiB) on the 48-GiB GPU.
The first update took 41.09 seconds including tracing; the next two took
2.125 and 2.127 seconds. Those two startup timings are not a twenty-sample
whole-training benchmark.

This startup establishes execution and checkpoint qualification for that specific
recipe and hardware, not general convergence.

Completed V1 fitting and endpoint comparison
--------------------------------------------

A subsequent 66,658-neuron V1 run on RTX 8000 completed 20 epochs of 64 accepted
updates (1,280 total), batch 32 (24 evoked/eight spontaneous), 500-ms windows and
exact 25-ms checkpoints. It used detached reset, attached ASC, Gaussian
width/gain 0.28/0.05, full recurrent/voltage gradient retention, selective mixed
FP16 and the original Normal objectives and ExpAdam learning rate 0.01.
These scientific settings are evidence for that recipe, not library defaults.

Epoch-20 hard-trained/hard-reset and historical soft-trained/soft-reset models
were evaluated with eight grating directions and ten repeats per direction,
500-ms gray plus 2,000-ms gratings. Metrics used core neurons within 200 um;
OSI/DSI required preferred rate at least 0.5 Hz. A third arm reused the same
soft-trained endpoint with hard reset after exact input/master/source verification.

.. list-table:: Mean per-cell-class Neuropixels similarity (1-KS; descriptive)
   :header-rows: 1
   :widths: 36 16 16 16 16

   * - Training / inference
     - Spontaneous FR
     - Preferred FR
     - OSI
     - DSI
   * - Hard / hard
     - 0.817
     - 0.752
     - 0.654
     - 0.670
   * - Soft / soft
     - 0.815
     - 0.770
     - 0.635
     - 0.666
   * - Soft / hard
     - 0.815
     - 0.724
     - 0.643
     - 0.651

The distributions are broadly similar, with modest, cell-type-dependent gains
for hard-trained/hard-reset versus soft-trained/hard-reset. This is one fixed
network seed, not independent training replication or proof of superiority.
Historical training source/hardware differ; the reused third inference arm ran
on V100 while the new pair ran on GTX1080Ti. Common inference source, operators,
precision, stimulus realizations and soft-trained physical masters were verified,
not bitwise cross-GPU trajectory identity.

The paired GTX1080Ti inference smoke peaked at 5.16/5.18 GiB after selecting
native CSR currents. The old unaccelerated Pascal input projection exceeded
headroom or ran out of memory. This is an inference-memory result, not a
qualification of batch-32 training on an 11-GiB device.

The consolidated automatic Pascal route also passed actual V1 startup on
GTX1080Ti: three accepted hard-surrogate updates, finite/changed masters,
constraints and strict model/optimizer restoration at batch eight,
500 steps and 25-step checkpoints. Peak TensorFlow allocation was 6.12 GiB.
The matched soft-reset control also passed. See :doc:`dpointnet_pascal`
for complete test coverage, timing boundaries and memory scope.
This is separate from the completed batch-32, twenty-epoch training above.

Small runnable demonstration
-----------------------------

The repository example ``examples/dpointnet_nest_comparison`` includes
``config.train.hard_surrogate.json`` alongside the fixed-weight comparison.
It uses the same 300 neurons and saved inputs, trains recurrent and input
weights, and uses exact activation checkpointing.

From a writable copy of that example, after building the assets:

.. code-block:: bash

   CUDA_VISIBLE_DEVICES="" python run_dpointnet.py config.fp32.hard.json
   CUDA_VISIBLE_DEVICES="" python probe_hard_reset_gradients.py
   CUDA_VISIBLE_DEVICES="" python run_dpointnet.py config.train.hard_surrogate.json
   MPLBACKEND=Agg python plot_hard_reset_training.py

The probe compares both backward modes at fixed weights and checks identical
hard-forward states. It separately forces a spike in all neurons followed by
a refractory step, so restored voltage credit is measured directly rather
than inferred from nonzero weight gradients.

In the measured FP32 CPU demonstration, both fixed-weight modes emitted the
same 2,397 spikes. The forced-reset initial-voltage gradient norm changed
from zero to approximately 14.63. Six accepted Adam updates reduced the
30-Hz target-rate MSE from 18.11 to 4.15 Hz squared after the last epoch.
These are startup-learning results on one replayed toy input, not a convergence
or generalization result.

Outputs include ``output_gradient_probe/gradient_probe.json``,
``output_gradient_probe/hard_reset_training.png``, and trained SONATA edge
files under ``output_train_hard_surrogate/callbacks/trained_weights``.
The coefficients, target and learning rate are explicitly chosen for the demo,
not new defaults. For new experiments, assess loss, rates, finite gradients and
the behavior of exported weights in the intended hard-reset inference model.

Compare with soft-reset training
--------------------------------

The companion ``config.train.soft.json`` matches the hard-surrogate profile
except for reset/derivative policy and output paths. Both use FP32, attached
reset and ASC events, the same network and saved input, the same 30-Hz target
and optimizer, and six checkpointed updates:

.. code-block:: bash

   CUDA_VISIBLE_DEVICES="" python run_dpointnet.py config.train.hard_surrogate.json
   CUDA_VISIBLE_DEVICES="" python run_dpointnet.py config.train.soft.json
   CUDA_VISIBLE_DEVICES="" MPLBACKEND=Agg python compare_reset_training.py

.. list-table:: Measured learning on the fixed input (MSE in Hz squared)
   :header-rows: 1

   * - Training policy
     - Initial rate MSE
     - Final rate MSE in its training policy
   * - Hard forward, soft-surrogate backward
     - 18.11
     - 4.15
   * - Soft reset
     - 19.07
     - 4.85

For this replay and learning rate, hard-surrogate loss decreased smoothly,
whereas soft-reset loss oscillated before reaching a similar final value.
This short run does not establish a generally superior method.

The script also reloads each trained SONATA export under both inference reset
policies:

.. list-table:: Cross-reset inference (rate MSE in Hz squared)
   :header-rows: 1

   * - Trained weights
     - Hard-reset inference
     - Soft-reset inference
   * - Hard-surrogate training
     - 4.15
     - 7.63
   * - Soft-reset training
     - 4.15
     - 4.85

The matching hard-inference MSE values do not imply identical weights or
spike timing. Switching reset at the same trained weights gave exact
neuron/time event F1 of 0.482 for hard-trained weights and 0.231 for
soft-trained weights. This rate objective does not constrain event timing.
The cross-reset test is an explicit comparison, not an automatic inference
policy change.

Both methods restored the same voltage-gradient norm in the isolated
forced-reset/refractory probe, but their initial recurrent and input-weight
gradient cosine similarities were only about 0.628 and 0.666. Their different
forward trajectories therefore produce different learning directions despite
the shared local voltage-credit rule.

``output_training_comparison/comparison.png`` shows the learning curves,
cross-reset losses and final rasters. The adjacent ``comparison.json`` records
the matched asset hashes, probes and unshifted event-agreement metrics.
