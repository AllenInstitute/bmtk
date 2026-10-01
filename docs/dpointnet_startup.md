# Startup preprocessing: developer notes

This note describes connectivity-import optimizations for developers maintaining
CSR construction. They reduce CPU preprocessing without changing network values,
edge ordering, precision, RNG or initialization rollouts. No user configuration
or persistent cache is required.

`build_csr_connectivity` deduplicates compact `(post, synapse_type)` pairs using
uint64 integer keys when the declared range fits. The resulting pair IDs and
lexicographic pair table match the original two-column `np.unique(axis=0)`.
The original tuple path remains for unrepresentable ranges/strides.

Target-grouped CSR ordering likewise uses a stable packed
`(source, target, synapse_type)` key when all declared dimensions fit uint64,
with the original three-column lexsort as fallback. This retains duplicate-edge
permutations exactly.

`lex_sort_order_np` similarly packs nonnegative integer pairs and uses stable
argsort, preserving canonical duplicate-edge order. Negative, floating, empty
and overflowing inputs use the original lexsort. Integer arithmetic is explicitly
unsigned; no float64 key conversion or approximate ordering is used.

## Maintenance and profiling

Preserve canonical neuron and edge identity, including duplicate-edge order.
Test packed-key paths against the original tuple/lexsort implementations and
cover overflow, signed, floating and empty inputs before changing these routines.

Existing gray-screen initialization and input generation are retained. Measure
network import separately from initialization rollouts and the first traced or
compiled optimizer update. Profiling adds overhead, and filesystem-cache state
can affect startup measurements; compare like with like. Faster CSR construction
does not imply an equivalent reduction in total setup time or steady-state
training time.