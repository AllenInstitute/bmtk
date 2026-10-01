# Startup preprocessing

The2026-09-30 follow-up reduces CPU preprocessing without changing
network values, edge ordering, precision, RNG or initialization rollouts.

`build_csr_connectivity` deduplicates compact `(post, synapse_type)` pairs using
uint64 integer keys when the declared range fits. The resulting pair IDs and
lexicographic pair table match the original two-column `np.unique(axis=0)`.
The original tuple path remains for unrepresentable ranges/strides.

Target-grouped CSR ordering likewise uses a stable packed
`(source, target, synapse_type)` key when all declared dimensions fit uint64,
with the original three-column lexsort as fallback. This retains duplicate-edge
permutations exactly. The final screen reduced CSR construction from13.53s to
12.40s; total unprofiled startup varied57.7-60.5s, so this last change is not
claimed as a large additional end-to-end gain.

`lex_sort_order_np` similarly packs nonnegative integer pairs and uses stable
argsort, preserving canonical duplicate-edge order. Negative, floating, empty
and overflowing inputs use the original lexsort. Integer arithmetic is explicitly
unsigned; no float64 key conversion or approximate ordering is used.

Measured on the66,658-neuron V1/RTX3090 batch16 engineering workload:
matched profiled setup95.631s ->62.325s (34.83% reduction); final unprofiled
repeats58.695/58.792s. These are fresh-process runs, not flushed-filesystem-cache
benchmarks. CUDA operators were already built. The first optimizer update still
takes about26.75s including tracing/compilation, versus1.61-1.65s afterward.

No cache configuration is necessary. Existing gray-screen initialization and
input generation are retained. The benchmark runner's `--profile-setup` option
emits cProfile data and synchronized phase timings, with compilation measured
separately in the first update. Profiling adds overhead; compare like with like.

Evidence and final validation status:
`/local2/results/dpointnet_rule_search/startup_20260930/REPORT.md`.
Complete combined-source suites passed1704GPU tests,945CUDA-disabled tests and
945actual Python3.8/Keras2 tests, with no failures;26finite benchmark updates applied.
This is not in the immutable `1f0e84c2` pin or installed into existing environments.
Final publication qualification is recorded at
`/local2/results/dpointnet_rule_search/startup_publish_20260930/REPORT.md`.