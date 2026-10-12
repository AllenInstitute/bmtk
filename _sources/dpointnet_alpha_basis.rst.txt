.. _dpointnet_alpha_basis--automatic-alpha-basis-fitting:

Import synaptic kinetics with alpha-basis fitting
=================================================

Use this page when importing a SONATA network that has synaptic time constants
but no DPointNet basis coefficients. You do not need to refit the supplied
small example.

DPointNet represents synaptic current waveforms as a weighted sum of a small
set of shared **alpha functions**. Each function rises then decays with its
own time constant. Different synapse classes use different coefficients in
that same basis. This representation is an approximation of the original
waveforms, separate from the physical edge weights being trained.

Prepare the network
-------------------

Each edge type's dynamics JSON must identify its kinetics. For example, this
is a **synaptic dynamics file**, not a simulation-config fragment:

.. code-block:: json

   {
     "tau_syn_fast": 1.0,
     "tau_syn_slow": 4.0,
     "amp_slow": 0.3
   }

Time constants are in milliseconds. ``amp_slow`` is the slow component's
amplitude relative to the peak-normalized fast component. Scalar ``tau_syn``
is also accepted; omitted slow time constant defaults to the fast one and
omitted slow amplitude to zero.

Alternatively, a one-based ``receptor_type`` in the synapse JSON can select
``tau_syn_fast``, ``tau_syn_slow`` and ``amp_slow`` arrays in the target-cell
JSON. One dynamics file must identify one kinetic class. If the same receptor
number selects different kinetics in different targets, use distinct dynamics
files for those classes.

Recurrent and external-input edges share the fitted basis. Existing
coefficients alone cannot reconstruct missing raw kinetics.

.. _dpointnet_alpha_basis--kinetics-and-execution:

Configure fitting
-----------------

Omit ``tau_basis``, ``synaptic_basis_weights`` and ``basis_weights_file`` to
fit a new network from raw kinetics. The default tries four shared functions,
then five if needed. Merge this **JSON fragment** into the simulation config:

.. code-block:: json

   {
     "rnn_cell_params": {
       "cell_model": "GLIF3Cell",
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

The same ``alpha_basis`` dictionary can be passed inside
``RNN(cell_params=...)``. Setting it to ``false`` disables automatic fitting.
General CPU/CUDA paths support five functions; automatic acceleration accounts
for that shape. Do not force a four-function specialization on a five-function
network.

Existing coefficient files, embedded ``basis_weights`` and explicit
``synaptic_basis_weights`` are preserved by default. If only coefficients are
missing and ``tau_basis`` is supplied, the fit uses those fixed time constants.
Partially supplied coefficient tables also need the shared time constants.
Invalid/missing configured files are errors, not silently replaced inputs.

.. warning::

   ``alpha_basis.force_recompute=true`` discards supplied basis time constants
   and coefficients **in memory**, then fits both again from raw kinetics.
   Every rebuild with that flag performs a fresh fit. It does not edit the
   source files and is not a fit of weights against the old time constants.
   Use it only when intentionally replacing the basis; it defaults to false.

Inspect the result
------------------

After building:

.. code-block:: python

   rnn.build()
   fit = rnn.alpha_basis_fit
   if fit is not None:
       print("Time constants (ms):", fit["tau_basis"])
       print("Relative RMS errors by class:", fit["relative_rms"])
       print("Fit diagnostics:", fit["diagnostics"])

The result is ``None`` when existing coefficients were used or fitting was
disabled. It includes coefficients, input-file ordering and error diagnostics.
Generated time constants are also in ``rnn.cell_params["tau_basis"]``.
Rebuilding the same RNN reuses its generated rows unless force recomputation
is enabled. No fit is automatically written to disk; record the result and
settings with the model when you need reproducible reuse.

For waveform inspection without constructing a network:

.. code-block:: python

   from bmtk.simulator.dpointnet.alpha_basis import fit_alpha_basis
   import matplotlib.pyplot as plt
   import numpy as np

   # Rows: fast tau (ms), slow tau (ms), slow relative amplitude.
   kinetics = np.array([[1.0, 4.0, 0.3], [2.0, 8.0, 0.2]])
   fit = fit_alpha_basis(kinetics)
   t = np.linspace(0, fit["diagnostics"]["time_range_ms"], 1000)

   def alpha(time, tau):
       return (time / tau) * np.exp(1 - time / tau)

   target = alpha(t, kinetics[0, 0]) + kinetics[0, 2] * alpha(t, kinetics[0, 1])
   basis = alpha(t[:, None], fit["tau_basis"][None, :])
   plt.plot(t, target, label="original class 0")
   plt.plot(t, basis @ fit["weights"][0], label="fitted class 0")
   plt.xlabel("Time (ms)")
   plt.legend()
   plt.show()

.. _dpointnet_alpha_basis--method-and-tolerance:

Accuracy and fit failures
-------------------------

Acceptance uses the maximum per-class relative root-mean-square waveform
error. The default threshold is about 8.012%, based on a V1 reference fit;
it is not a universal accuracy requirement. It does not bound voltage,
spike timing or training-gradient errors. Choose a threshold and time
window appropriate for your synapses, then evaluate network behavior too.

If no attempted basis meets tolerance, model construction raises with the
achieved error. Inspect kinetics, per-class errors, basis dimension and solver
budget before changing settings. The tolerance is never automatically relaxed.
Very different fast/slow timescales may need more sampling points.

The solver fits shared time constants and per-class least-squares coefficients.
Negative coefficients are allowed as waveform shape factors, not negative
physical excitatory weights. Default bounds and the sampled time window adapt
to the input time constants. Solver details and source attribution are in
:doc:`dpointnet_development`.
