# Performance Notes

Date: 2026-06-15

## Used Fixes

### Fast `Index.head_name` and `Index.label`

`Index.head_name` and `Index.label` previously called `str(...)` on SymPy
symbols. In tensor canonicalization, these properties are read millions of
times, and `str(Symbol)` enters SymPy's printer stack. The safe fix uses
`Symbol.name` when the stored argument is a `Symbol`, preserving the existing
fallback for non-symbol values and preserving `Wild`/integer label behavior.

Measured impact on the non-diagonal decomp tests:

```text
test_decompse_non_diagonal_metric_derivative_hooks_simplify_curvature_path
before: ~7.21s
after:  ~3.44s

test_decompse_full_non_diagonal_metric_reduces_to_maxwell_form
before: ~11.44s
after:  ~4.99s
```

Microbenchmarks:

```text
idx.head_name before: ~1.04 us/access
idx.head_name after:  ~0.034 us/access

idx.label before: ~1.06 us/access
idx.label after:  ~0.055 us/access
```

### Single-read `_index_family_key`

`_index_family_key` previously read `index.head_name` twice through
`index.head` plus a direct property access. It now reads the head name once and
looks up the index type directly.

Microbenchmark:

```text
_dummy_index_key(DN("alpha")) after: ~0.196 us/call
```

### Bounded canonicalization caches

The product canonicalization hot path repeatedly canonicalizes the same
declared-symmetry and dummy-renaming subexpressions. I added bounded private LRU
caches for pure canonicalization helpers and clear them whenever index,
symmetry, metric, or saved MathGR state changes.

Measured impact on the same two non-diagonal tests after the fast index-property
fix:

```text
combined non-diagonal pair before caches: ~8.30s
combined non-diagonal pair after caches:  ~5.50s

curvature path after caches: ~2.38s
full Maxwell after caches:  ~3.11s
```

### Bounded `TSeries` helper caches

The slowest remaining example repeats the same perturbation-series helper calls
while expanding derivative bases. I added bounded private caches for `TSeries`
internals and clear them on metric/index/state changes.

Measured impact:

```text
second_order_pert zeta reduction before: ~16.9-17.2s
second_order_pert zeta reduction after:  ~15.0s
```

### Newton gauge metric-hook fast path

Profiling `examples/newton_gauge.py` showed that most Newton gauge runtime was
not tensor canonicalization. `Decomp0i` called `decomp_hook` on every node, and
`decomp_hook` ran the full wildcard `ReplaceAll` machinery even for nodes that
could not be the metric tensor `g(...)`.

The safe fix returns immediately unless `tensor_head_name(expr) == "g"`.

Measured impact:

```text
test_newton_gauge_example_ports_background_and_linear_action_cells
before: ~8.49s
after:  ~3.46s

test_newton_gauge_example_ports_displayed_quadratic_action_cell
before: ~10.73s
after:  ~4.48s
```

### Newton gauge action-density reuse

`newton_gauge.main(compute_action=True)` computed the same simplified action
density once for `results["action_density"]` and again inside each of
`action_order(0)`, `action_order(1)`, and `action_order(2)`. I factored the
order extraction into `_action_order_from_density` so `main` reuses the one
already-computed density. The public `action_order(order)` API is unchanged.

Measured impact after the metric-hook fast path:

```text
background/linear action test: ~3.44s -> ~2.37s
displayed quadratic test:      ~4.44s -> ~3.42s
```

## Tried But Not Used

### One-pass `_replace_index_keys`

I tested rewriting `_replace_index_keys` to compute each dummy key once while
building the replacement map. After the fast index-property change, this showed
no measurable improvement on the full Maxwell reduction, so it was not used.

I retested after canonicalization caches, including a replacement-map LRU cache.
It still showed no material improvement:

```text
base:              ~5.50s
one-pass replace:  ~5.51s
replacement cache: ~5.48s
both:              ~5.56s
```

### Direct coefficient extraction for `OO(2)`

For `examples/second_order_pert.py`, the main bottleneck is `OO(2)` /
`TSeries`, dominated by SymPy `series`/`nseries`, not tensor canonicalization.
Replacing `TSeries` with direct `expand(...).coeff(Eps, 2)` was much faster but
not algebraically equivalent, so it was rejected.

```text
current OO(2):                  ~15.0s, correct
direct expand coeff + Simp:     ~0.77s, not equivalent
direct expand coeff + Collect:  ~0.62s, not equivalent
```

### Skip `TSeries` on derivative bases independent of `Eps`

I tested returning a partial derivative unchanged when its base has no `Eps`.
The result was equivalent, but runtime did not improve measurably because most
expensive derivative bases still contain `Eps` or the time is elsewhere in
SymPy series expansion.

```text
current OO(2):                    ~15.14s
skip independent derivative bases: ~15.19s
```

### Skip `_prepare_series_expr` subtrees independent of `Eps`

I also tested an early return in `_prepare_series_expr` for any subtree that
does not contain the expansion symbol. It was equivalent on the second-order
perturbation example but did not improve runtime.

```text
current OO(2):                         ~15.18s
skip subtrees without Eps in prepare:   ~15.21s
visited prepare nodes: 7353
skipped prepare nodes: 3609
```

### Coefficient-only `OO`

I tested extracting the requested `Eps` coefficient directly from `TSeries`
before applying `CollectEps`. It was equivalent on the slow zeta reduction test
but gave no material improvement:

```text
current OO:      ~15.27s
coefficient-only: ~15.08s
```

### Lower-level canonicalization caches

I also tested the lower-level caches named in the performance plan after the
higher-level canonicalization caches were in place.

```text
base non-diagonal pair:         ~5.50s
cache _canonical_slot_key:      ~5.56s
cache _expr_shape_key:          ~5.43s
```

Those did not show a material win. A naive `_dummy_key_signatures` LRU cache was
not safe because the helper receives a mutable `set` of dummy keys in current
call paths.

## Verified Existing Work

`docs-local/performance.md` calls for private profiling before algorithm work.
That profiler already exists in `src/mathgr/debug.py`, with tests in
`tests/test_debug.py`. I used direct `pytest --durations` and `cProfile` for
this round because they gave enough evidence for the safe fixes above.

The MCP timeout/process issue from the performance plan is also already covered:
structured tools default `timeout_seconds` to `0` and therefore run in-process
by default, while explicit positive timeouts still use a worker process. Current
tests cover both the default and explicit-timeout paths.

## Not Implemented

The contraction-graph canonicalizer, compact domain-specific sort keys, and
non-expansive `Simp` mode from `docs-local/performance.md` are not used here.
They may be worthwhile later, but they are not safe/simple changes: each changes
core algebraic canonicalization or expansion behavior and needs a broader design
plus more equivalence tests.

I also did not refactor `_canonicalize_additive_tensor_terms` into bucketed
prekeys in this pass. After profiling the current code, the observed safe wins
were repeated metadata access, repeated symmetry/dummy canonicalization, and
example-level repeated replacement/recomputation. Bucketed additive
canonicalization remains a larger algorithmic change.

## Remaining Bottlenecks

After the index-property fix, the non-diagonal tensor tests are much faster.
After canonicalization caches, the remaining broad-suite slow cases are
example-level perturbation reductions, especially Newton gauge and second-order
perturbation examples. Profiling `second_order_pert` shows:

```text
action_density(simplify=True): ~1.8-2.0s
OO(2, op=Simp)(expr):         ~15s after helper caches
```

The next safe optimization should target `TSeries`/`OO` only with strong
algebraic equivalence tests. Naive coefficient extraction is not safe.

The non-diagonal decomp tests now include a broad 30-second runtime budget
around the core simplification path. Current runtime is about 2-3 seconds for
each, so the budget is meant to catch catastrophic regressions without becoming
a microbenchmark.

## Current Verification

Fresh verification after the first safe fixes:

```text
./tools/test.sh
fast bucket: 310 passed, 9 deselected in 12.25s
slow bucket: 8 passed in 16.23s

uv run pytest -q -n 4 -m slow --durations=20
slowest:
  second_order_pert zeta reduction: ~15.12s
  newton_gauge quadratic action:    ~11.26s
  newton_gauge background/linear:   ~8.69s
  non-diagonal Maxwell:             ~3.15s
  non-diagonal curvature:           ~2.45s
```

After the Newton gauge fixes:

```text
./tools/test.sh
fast bucket: 312 passed, 9 deselected in 9.60s
slow bucket: 8 passed in 16.19s

uv run pytest -q -n 4 -m slow --durations=20
slowest:
  second_order_pert zeta reduction: ~15.24s
  newton_gauge quadratic action:    ~3.70s
  non-diagonal Maxwell:             ~3.13s
  newton_gauge background/linear:   ~2.47s
  non-diagonal curvature:           ~2.44s
```
