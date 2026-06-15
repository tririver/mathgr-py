# MathGR-Py Manual

`mathgr` is a Python/SymPy port of MathGR for symbolic tensor calculus,
general relativity, ADM and FRW decompositions, perturbation series,
integration by parts, and TeX export.

MathGR-Py defaults to UTF-8 Greek names for common physics scalars. Use `φ`,
`δφ`, `φ0`, `ζ`, `ε`, `η`, `α`, and `β` directly in Python and MCP
expressions. Index constructors stay ASCII (`UP`, `DN`, `U1`, `D1`, ...), and
their labels may be UTF-8, for example `D1("α")`.

This manual describes the Python API exposed by the package. It assumes:

```python
import sympy as sp
from mathgr import *
```

For scripts, explicit imports are usually better:

```python
from mathgr import tensor, declare_idx, Dta, Pd, Simp
import mathgr.gr as gr
import mathgr.frwadm as frwadm
```

## Install And Run

Install from the repository:

```bash
uv sync
```

Install editable into another environment:

```bash
python -m pip install -e .
```

Run the default local test workflow: fast tests serial, slow symbolic bucket in
parallel.

```bash
tools/test.sh
```

Override slow-test workers with `PYTEST_SLOW_WORKERS=8 tools/test.sh`.
Use `uv run pytest -q` for a full serial release check.

Run the MCP server:

```bash
uv run mathgr-mcp
```

Run examples:

```bash
uv run python examples/general_presentation.py
uv run python examples/decomp_example.py
```

## Core Concepts

MathGR-Py expressions are SymPy expressions with custom tensor heads and index
objects. Tensor heads are callable. Indices carry variance, label, dimension,
color, and dual-family metadata.

Most unsupported signatures intentionally remain as symbolic tensor calls
instead of raising. This mirrors upstream MathGR behavior and lets expressions
stay manipulable until a later rule or simplifier handles them.

Several modules keep global state:

- declared index families
- declared tensor symmetries
- metric registrations
- simplification hooks
- decomposition hooks
- TeX hooks and template mode
- current default GR metric

Use `WithMetric(...)` for temporary GR metric scopes. When mutating global hook
lists in tests or notebooks, save and restore them with `try/finally`.

## Indices

### Built-In Families

The root package exports four default index families:

```python
UP("a")    # implicit upper index
DN("a")    # implicit lower index
UE(0)      # explicit upper index, often time
DE(0)      # explicit lower index, often time
```

Built-in label pools:

```python
LatinIdx
GreekIdx
LatinCapitalIdx
```

Default symbolic dimension:

```python
DefaultDim
```

### Declare A New Index Family

Preferred Python API:

```python
u, d = declare_idx("U", "D", dim=3, index_set=LatinIdx, color="Black")

i = u("i")
j = d("j")
```

Compatibility API:

```python
u, d = DeclareIdx(("U", "D"), 3, LatinIdx, "Black")
```

Explicit index families:

```python
eu, ed = DeclareExplicitIdx(("EU", "ED"), "Gray")
```

Explicit indices are excluded from dummy and free-index discovery. Deltas with
matching explicit labels evaluate by label value:

```python
Dta(UE(1), DE(1))   # 1
Dta(UE(1), DE(2))   # 0
```

### Index Metadata

Metadata helpers accept either an index family or an index instance:

```python
IdxDual(UP)       # DN
IdxSet(DN)        # label pool
IdxColor(UP)      # color name
Dim(DN("a"))      # dimension expression
```

Predicates:

```python
IdxHeadPtn(UP)
IdxPtn(UP("a"))
IdxUpPtn(UP("a"))
IdxDnPtn(DN("a"))
```

Only registered implicit index families count for these predicates. Explicit
indices do not count as implicit tensor indices.

### Fresh Labels

```python
Uniq(3)       # list of fresh labels
Uq(3)         # tuple of fresh labels
UniqueIdx()   # 50 fresh labels
```

These helpers are useful for temporary dummy pools and generated expressions.

## Tensor Heads And Tensor Calls

Create a tensor head:

```python
f = tensor("f")
expr = f(UP("a"), DN("b"))
```

Inspect tensor calls:

```python
from mathgr.tensor import tensor_head_name, tensor_args

tensor_head_name(expr)  # "f"
tensor_args(expr)       # tuple of arguments
```

Tensor calls are SymPy expressions, so normal SymPy operators work:

```python
expr = 2 * f(UP("a")) + sp.Symbol("x") * f(UP("a"))
sp.expand(expr)
```

## Deltas And Levi-Civita

### Delta

```python
Dta(UP("a"), DN("b"))
```

`Dta` canonicalizes its two index arguments and contracts in `Simp`.
Unsupported arities stay inert:

```python
Dta(UP("a"))       # inert tensor-like expression
Dta(UP("a"), UP("b"))
```

Same-label same-family traces reduce to dimensions when appropriate:

```python
Dta(UP("a"), UP("a"))   # DefaultDim for built-in UP
```

### Generalized Delta

```python
DtaGen(UP("a"), UP("b"), DN("c"), DN("d"))
```

`DtaGen` expands as an antisymmetric determinant-like sum. It expects an even
number of indices. Odd counts raise `ValueError`.

Override the delta builder:

```python
metric = tensor("g")
DtaGen(UP("a"), UP("b"), DN("c"), DN("d"), DtaGenDta=metric)
```

### Levi-Civita

```python
LeviCivita(DN("a"), DN("b"), DN("c"))
```

`LeviCivita` becomes active when all arguments are indices from the same family
and the family has integer dimension equal to the arity. Repeated indices give
zero. Other cases stay symbolic.

## Index Discovery

```python
idx(expr)      # all implicit index labels, sorted
free(expr)     # labels that occur once
dummy(expr)    # labels that occur exactly twice
rmE(indices)   # remove explicit indices from a list
```

These functions count labels, not variance. Explicit indices are ignored.

## Partial Derivatives

### Basic Derivatives

```python
Pd(expr, DN("i"))
```

`Pd` applies add, product, and supported power rules. Constants, numbers,
declared dimensions, and deltas differentiate to zero. Unsupported arity stays
inert.

Curried form:

```python
P(DN("i"), DN("j"))(expr)
```

Internal derivative form:

```python
vars_ = PdVars(DN("i"), DN("j"))
PdT(expr, vars_)
```

`PdVars` is orderless. `PdT` expands over addition, product, powers, and nested
`PdT` where supported. Empty variable lists return the original expression.

### Derivative Tensor Storage

```python
stored = pd2pdts(Pd(f(UP("a")), DN("i")))
restored = pdts2pd(stored)
```

Direct storage constructor:

```python
Pdts(2, f, DN("i"), DN("j"))
```

Query helpers:

```python
from mathgr.tensor import is_pdt, pdt_parts

is_pdt(expr)
pdt_parts(expr)
```

`pdt_parts` raises `TypeError` for non-`PdT` expressions.

### Inverse Laplacian Form

```python
Pm2(expr, DN)
Pm2(expr, index_type=DN)
```

`Pm2` represents the inverse-Laplacian-like operator used by perturbation and
IBP helpers. It expands low-degree powers of sums, factors constants when their
derivative is zero, and commutes with `PdT` where implemented.

Query helpers:

```python
from mathgr.tensor import is_pm2, pm2_parts

is_pm2(expr)
pm2_parts(expr)
```

`pm2_parts` raises `TypeError` for non-`Pm2` expressions.

## Tensor Symmetries

Declare slot symmetries:

```python
S = tensor("S")
DeclareSym(S, (DN, DN), Symmetric((1, 2)))
```

Supported symmetry specs:

```python
Symmetric((1, 2))
Antisymmetric((1, 2))
PermutationSymmetry((2, 3, 1))
Cycles(((1, 2, 3),))
Symmetric("All")
Antisymmetric("All")
```

Slot numbers are 1-based over implicit slots. Explicit indices are skipped.
Symmetric or antisymmetric slots must have compatible signature entries.

Inspect and delete:

```python
ShowSym(S, (DN, DN))
DeleteSym(S, (DN, DN))
```

Apply explicit symmetrization to an expression:

```python
Sym(expr)
Sym(expr, [DN("a"), DN("b")])
AntiSym(expr)
AntiSym(expr, [DN("a"), DN("b")])
```

When no index list is supplied, free non-explicit indices are used.

## Simplification

Main simplifier:

```python
Simp(expr)
```

Fast mode:

```python
Simp(expr, Method="Fast")
```

`Simp` expands expressions, contracts deltas, contracts registered metric
products, contracts Levi-Civita products, canonicalizes declared symmetries,
renames dummy labels, canonicalizes tensor products, and applies hooks.

Dummy-name options:

```python
Simp(expr, Dummy=("p", "q", "r"))
SimpUq(expr)
```

`SimpUq` uses fresh labels from `UniqueIdx()`.

### Hooks

Global hook list:

```python
SimpHook.append({old: new})
```

Per-call hooks:

```python
Simp(expr, hooks=[{old: new}, callable_hook])
Simp(expr, Hooks=[(old, new)])
```

Hook entries can be callables, dictionaries, `(old, new)` pairs, or sequences of
pairs. SymPy `Wild` patterns are supported. Hooks iterate up to a fixed limit.

Term selector:

```python
import mathgr.tensor as tensor_module

old_select = tensor_module.SimpSelect
try:
    tensor_module.SimpSelect = lambda terms: [t for t in terms if t != 0]
    Simp(expr)
finally:
    tensor_module.SimpSelect = old_select
```

## Rewrite Rules

Simple rule:

```python
Rule(lhs, rhs)
```

Delayed rule:

```python
RuleDelayed(lhs, lambda **match: rhs)
```

Single-pass replacement:

```python
ReplaceAll(expr, {lhs: rhs})
ReplaceAll(expr, [Rule(lhs, rhs)])
```

Repeated replacement:

```python
ReplaceRepeated(expr, rules)
ReplaceRepeated(expr, rules, max_iter=20)
```

Wild helpers:

```python
i = LabelWild("i")
wi = IndexWild(DN, "i")
```

Tensor-specific replacement:

```python
TensorReplace(expr, rules)
TensorReplace(rules)(expr)
```

`ReplaceAll` is one pass. For chained rules such as `{x: y, y: z}`, use
`ReplaceRepeated`.

## General Relativity

GR helpers live at the package root and in `mathgr.gr`.

Default exported objects:

```python
g          # default metric head
Metric     # current default metric
IdxOfMetric
V          # scalar potential tensor head
LapseN
ShiftN
```

### Metric Registration

```python
g4 = tensor("g4")
u4, d4 = declare_idx("U4", "D4", dim=4)

UseMetric(g4, (u4, d4))
```

`UseMetric(metric, indices=(UP, DN), SetAsDefault=True)` declares metric
symmetries, registers metric contractions, and optionally changes the global
current metric.

Temporary metric scope:

```python
WithMetric(g4, (u4, d4), lambda: R())
```

You can also pass an already built expression:

```python
WithMetric(g4, (u4, d4), expr)
```

For heavy expressions, prefer a callback so expansion happens while the metric
scope is active.

### Metric Slots

`UG(label)` and `DG(label)` are metric-contraction slots. They are placeholders
that `MetricContract` turns into concrete dummy indices and metric factors.

```python
MetricContract(R(DG(1), DG(1)))
```

This is the pattern behind:

```python
RicciScalar()
```

### Connections, Curvature, And Covariant Derivatives

Christoffel:

```python
Affine(UP("a"), DN("b"), DN("c"))
```

Only the upper/lower/lower signature expands for the current metric. Other
signatures remain symbolic.

Covariant derivative:

```python
CovD(expr, DN("a"))
```

Scalars use partial derivatives. Tensor expressions add connection terms. Upper
derivative indices are raised with the current metric where supported.

Riemann, Ricci, scalar curvature:

```python
R()                                      # Ricci scalar
R(DN("a"), DN("b"))                      # Ricci tensor
R(DN("a"), DN("b"), DN("c"), DN("d"))    # lower Riemann
RicciScalar()
Rsimp()
Rsimp(DN("a"), DN("b"))
```

Lower Riemann tensors canonicalize antisymmetry, pair exchange, and Bianchi
identity forms.

Einstein tensor:

```python
G(DN("a"), DN("b"))
```

Down/down expands as `R_ij - g_ij R/2`. Unsupported variances intentionally stay
symbolic or raise only where the implementation needs a metric operation.

### Scalar Field Helpers

```python
φ = sp.Symbol("φ")

X(φ)                 # kinetic term
Dsquare(φ)           # box operator
T(φ)(DN("a"), DN("b"))  # stress tensor
V(φ)                 # potential head
```

### Generic ADM Helpers In `mathgr.gr`

```python
K(DN("i"), DN("j"))
KK()
RADM()
```

These use the current metric plus `LapseN` and `ShiftN`. For concrete ADM and
FRW workflows, prefer `mathgr.adm` or `mathgr.frwadm`.

### Bianchi Identity Example

```python
u4, d4 = declare_idx("U4", "D4", dim=4)
g4 = tensor("g4")
UseMetric(g4, (u4, d4))

expr = (
    CovD(R(d4("a"), d4("b"), d4("c"), d4("d")), d4("e"))
    + CovD(R(d4("a"), d4("b"), d4("d"), d4("e")), d4("c"))
    + CovD(R(d4("a"), d4("b"), d4("e"), d4("c")), d4("d"))
)

Simp(expr)   # 0
```

## Decomposition

Decomposition helpers split total indices into explicit sectors.

Index families:

```python
UTot, DTot    # total family
U1, D1        # first sector
U2, D2        # second sector
DimTot, Dim1, Dim2
```

Generic API:

```python
Decomp(expr, sectors, indices=None, hooks=None)
```

Convenience splitters:

```python
Decomp0i(expr, indices=None, hooks=None)
Decomp01i(expr, indices=None, hooks=None)
Decomp0123(expr, indices=None, hooks=None)
Decomp1i(expr, indices=None, hooks=None)
Decomp123(expr, indices=None, hooks=None)
DecompSe(expr, indices=None, hooks=None)
```

Example:

```python
f = tensor("f")
expr = f(UTot("mu"), DTot("mu"))

Decomp0i(expr)
# f(UE(0), DE(0)) + f(UP('mu'), DN('mu'))
```

Automatic decomposition applies to total dummy labels that occur exactly twice.
Free total labels require a list selector:

```python
Decomp0i(f(UTot("A")), indices=["A"])
```

Passing a scalar string or an index object as `indices` is intentionally a
no-op.

Hooks:

```python
Decomp0i(expr, hooks=[{old: new}, callable_hook])
DecompHook.append((old, new))
```

Hook formats match `Simp` hooks: callables, dictionaries, pairs, and lists of
pairs. SymPy `Wild` patterns are supported. Hooks run after each label split.

## ADM Module

ADM helpers live in `mathgr.adm`.

```python
import mathgr.adm as adm
```

Module exports:

```python
adm.a
adm.Sqrth
adm.ScriptCapitalN
adm.ScriptCapitalNVector
adm.LapseN
adm.Sqrtg
adm.g
adm.h
adm.ShiftN
adm.Simp
adm.DecompG2H
```

Important import side effect: importing `mathgr.adm` calls `UseMetric(adm.h)`,
so the global GR metric becomes `adm.h`.

Shift:

```python
adm.ShiftN(DN("i"))   # ScriptCapitalN(DN("i"))
```

ADM simplifier:

```python
adm.Simp(expr)
```

This wraps tensor `Simp` with ADM rules, including `DefaultDim -> 3`.

Four-metric decomposition:

```python
adm.DecompG2H(lambda: R())
adm.DecompG2H(expr)
```

`DecompG2H` evaluates under the four-metric, contracts metric slots, splits
`0+i`, and replaces metric components by ADM `h`, lapse, and shift pieces.

## FRW ADM Module

FRW ADM helpers live in `mathgr.frwadm`.

```python
import mathgr.frwadm as frw
```

Symbols and tensor heads:

```python
frw.k
frw.a
frw.H
frw.α
frw.β
frw.ζ
frw.ε
frw.η
frw.η2
frw.η3
frw.Mp
frw.b
frw.g
```

Lapse and determinant:

```python
frw.LapseN   # 1 + Eps*α
frw.Sqrtg    # LapseN*exp(3*Eps*ζ)*a**3
```

Shift:

```python
frw.ShiftN(DN("i"))
# Eps*Pd(β, DN("i")) + Eps*b(DN("i"))
```

Spatial metric:

```python
frw.h(DN("i"), DN("j"))
frw.h(UP("i"), UP("j"))
frw.h(UP("i"), DN("j"))
```

Extrinsic and spatial curvature:

```python
frw.K(DN("i"), DN("j"))
frw.KK()
frw.RADM()
```

Unsupported signatures remain symbolic.

FRW four-to-three split:

```python
frw.DecompG2H(lambda: X(φ))
```

FRW simplifier:

```python
frw.Simp(expr)
```

Rules include:

- background spatial derivatives of `a`, `H`, `ε`, and `η` vanish
- time derivatives of `a`, `H`, `ε`, `η`, and higher slow-roll symbols
  use the module rules
- `DefaultDim -> 3`
- `Mp` derivative vanishes
- transverse shift simplifications for `b`

Fourier-space two-point simplifier:

```python
frw.Fourier2(expr)
```

It maps repeated spatial derivatives to `-k**2`, paired gradients to `k**2`,
gradient-vector products to momentum factors, and transverse shift products to
zero where supported.

## Perturbation And Utility Helpers

Utilities live at the package root and in `mathgr.util`.

### Perturbation Symbol And Series

```python
Eps
```

`Eps` is the default perturbation symbol and is treated as a constant by tensor
partial derivatives.

Tensor-aware series:

```python
TSeries(expr, (Eps, 0, 3))
```

Keep series through order `n`:

```python
SS(2)(expr)
SS(2, vars=[x], op=Simp)(expr)
```

Extract order `n`:

```python
OO(2)(expr)
OO(2, vars=[x], op=Simp)(expr)
```

Collect:

```python
CollectEps()(expr)
CollectEps(vars=[x, y], op=Simp)(expr)
```

### Tensor Powers

```python
TPower(expr, 3)
TPower(expr, -1)
```

`TPower` builds repeated products directly. It preserves raw repeated dummy
labels; call `Simp` if you need canonical dummy labels later.

### Momentum-Space Conversion

```python
LocalToK(expr)
LocalToK(expr, index_type=DN, Momentum=k)
```

Momentum labels:

```python
k(1)(DN("i"))
k(2)(DN("j"))
```

`LocalToK` maps local differentiated fields into momentum-labeled tensor
fields. It handles scalar and indexed tensor variables.

### Solve For Compound Expressions

```python
SolveExpr(equations, expressions)
```

Unlike plain `sympy.solve`, `SolveExpr` is designed for solving for compound
expressions, not only symbols.

### Tensor Replacement Helper

```python
TReplace(expr, rules)
TReplace(rules)(expr)
```

Rules may be dictionaries, pairs, lists of pairs, or callable delayed
replacements. SymPy `Wild` patterns are supported.

## Integration By Parts

IBP helpers live at the package root and in `mathgr.ibp`.

Main transforms:

```python
Ibp(expr)
Ibp(expr, Rank=custom_rank, Rule=custom_rule, Level=1)
Ibp2(expr)
IbpNB(expr)
```

`Ibp` applies ranked integration-by-parts rules. `Ibp2` starts with a deeper
search level. `IbpNB` runs IBP and drops boundary holders.

Variation helper:

```python
IbpVariation(expr, target)
```

This moves derivatives off `target` and drops boundary terms.

One-step public rules:

```python
IbpRules(expr)
Pm2Rules(expr)
Pm2Simp(expr)
```

Ranked replacement search:

```python
TrySimp(expr, rule, Rank=None, Level=1)
TrySimp2(expr, rule, Rank=None, Level=2)
```

Rank helpers:

```python
IbpVar(target)
IbpCountLeaf(expr)
IbpCountTerm(expr)
IbpCountPt2(expr)
IbpCountPd2(expr)
IbpStd2(expr)
IbpReduceOrder(vars)
IbpRuleWithForbiddenPattern(rule, pattern)
IbpRuleWithForbiddenPatterrn(rule, pattern)  # compatibility typo alias
```

Boundary holders:

```python
PdHold(expr, DN("i"))
IdHold(expr)
```

`PdHold` normalizes its dummy label. Boundary holders remain visible unless you
use `IbpNB` or an explicit drop rule.

Global search preferences:

```python
TrySimpPreferredPattern
TrySimpPreferredPatternStrength
```

Save and restore these globals when temporarily biasing a simplification search.

## TeX Export

TeX helpers live at the package root and in `mathgr.typeset`.

Return a TeX string:

```python
ToTeXString(expr)
```

Print TeX:

```python
ToTeX(expr)
```

Decorate or postprocess a raw TeX string:

```python
DecorateTeXString(text)
```

By default, output is wrapped in a small LaTeX document:

```python
import mathgr.typeset as typeset

old = typeset.ToTeXTemplate
try:
    typeset.ToTeXTemplate = False
    fragment = typeset.ToTeXString(expr)
finally:
    typeset.ToTeXTemplate = old
```

Hook list:

```python
ToTeXHook.append({old: new})
ToTeXHook.append(callable_hook)
```

Hooks accept the same broad formats as `Simp` hooks. TeX rendering supports
tensor calls, deltas, Levi-Civita, partial derivatives, `Pm2`, lower Riemann,
and covariant derivatives of lower Riemann forms.

## MCP Server

MathGR-Py includes an MCP server for coding agents.

Command:

```bash
uv run mathgr-mcp
```

Registered tools:

```text
mathgr_capabilities
mathgr_manual
mathgr_parse
mathgr_compute
mathgr_inspect
mathgr_tex
mathgr_context_get
mathgr_context_clear
mathgr_context_save
mathgr_context_load
mathgr_script
mathgr_run_python
mathgr_eval
```

`mathgr_compute` is the first-choice tool for almost all MathGR calculations.
It is usually easier than raw Python because it accepts Python-like expression
strings or multi-line notebook blocks, auto-declares ordinary scalar names,
tensor heads, and index families, and persists assignments in a context. For
most derivations, do not predefine scalar symbols, tensor heads, or temporary
names just to make them exist; write the calculation directly, using multi-line
`mathgr_compute` assignments only when a derived expression should be reused.
Use `mathgr_parse`, `mathgr_inspect`, and `mathgr_script` only for debugging or
reproduction. Use `mathgr_run_python` / `mathgr_eval` only as last-resort
debugging escape hatches when `mathgr_compute` cannot express the workflow.
For ordinary calls, pass only the expression string and omit optional fields
such as `context`, `output`, and `timeout_seconds`. JSON objects are only the MCP
transport format; examples and traces should prefer `mathgr_compute("...")`.
Structured MCP tools default to `timeout_seconds=0`, which runs in-process for
low latency. Set a positive timeout only for risky or potentially long symbolic
calls that need subprocess cancellation.

Example:

```text
mathgr_compute("Simp(Dta(U('α'), D('β')) * f(U('β')) + x)")
mathgr_compute("Pd(δφ, D1('α'))")
```

Auto declarations:

```python
Dim = sp.Symbol("Dim")
U, D = declare_idx("U", "D", dim=Dim)
f = tensor("f")
x = sp.Symbol("x")
```

Dimension override:

```json
{"index_dims": {"U/D": 3}}
```

`mathgr_parse(expr, ...)`

: Debugging aid only. Dry-run parser that returns inferred index families,
  tensor heads, symbols, diagnostics, and generated Python.

`mathgr_compute(expr, ...)`

: First-choice tool. Evaluates a Python-like MathGR expression or restricted
  multi-line block exactly as written, with auto-declared symbols, tensor heads,
  and index families. Put ordinary MathGR calls directly in the expression:

```python
Simp(Dta(U('α'), D('β')) * f(U('β')))
Simp(lhs - rhs)
Decomp0i(f(DTot('a')) * f(UTot('a')))
Ibp(y * Pd(x, D('i')))
OO(2)((1 + Eps*x)**3)
```

### MCP Cookbook

Perturbation expansion:

```text
mathgr_compute("""
L2 = Simp(OO(2)(TSeries(L, (Eps, 0, 3))))
result = L2
""")
```

Derivative conventions:

| notation | MCP expression |
| --- | --- |
| `dot(f)` | `Pd(f, DE(0))` |
| `partial_i f` | `Pd(f, DN("i"))` |
| `partial_i partial_j f` | `Pd(Pd(f, DN("i")), DN("j"))` |

When calling the MCP tool with a multi-line block, pass the block body as
`expr`. Do not include Python triple-quote delimiters inside the MCP `expr`
value. Triple quotes are only a convenient way to show multi-line examples in
this manual.

Flat-gauge ADM scalar-field setup:

```text
mathgr_compute("""
N = 1 + Eps*α
Ni = Eps*Pd(β, DN("i"))
φ = φ0 + Eps*δφ
φdot = Pd(φ, DE(0))
gradφ = Pd(φ, DN("i"))
result = Ni + φdot
""")
```

For the full ADM scalar-field density, use the FRW ADM helpers as the starting
point and then extract the desired perturbative order:

```text
mathgr_compute("""
φ = φ0 + Eps*δφ
i = sp.Wild("i")
flat_rules = {b(DN(i)): 0, Pd(φ0, DN(i)): 0}
L = frwadm.Sqrtg * (frwadm.RADM()/2 + frwadm.DecompG2H(lambda: X(φ)) - V(φ))
L_flat = TReplace(flat_rules)(L.xreplace({ζ: 0}))
L2 = frwadm.Simp(OO(2)(TSeries(L_flat, (Eps, 0, 3))))
result = L2
""", timeout_seconds=30)
```

Constraint extraction by integration by parts:

```text
mathgr_compute("""
beta_terms = Simp(IbpVariation(L2, β))
constraint_beta = Simp(beta_terms.coeff(β))
result = constraint_beta
""")
```

Use `IbpVariation(expr, β)` to move derivatives off `β`, then take the
coefficient of `β`. Use `IbpNB` when you want boundary holders dropped.

TeX output:

```text
mathgr_tex("L2", fragment=True)
mathgr_compute("Simp(lhs - rhs)")
```

Use raw MCP TeX for exact, checkable output tied to the expression tree. For a
paper or note, hand-clean notation after the algebra is verified; keep the
machine expression for zero-checks such as `Simp(lhs - rhs)`.

For normal multi-step calculations, omit `context` and use the default context.
Assignments in a compute block persist there:

```python
mathgr_compute("""
trace = Dta(U('a'), D('a'))
simplified = Simp(Dta(U('a'), D('b')) * f(U('b')))
result = trace
""")
mathgr_compute("simplified")
mathgr_context_get()
```

`result = ...` controls the returned value but is not persisted as context
state. Use a regular assignment, or `store_as`, for durable values.

Named contexts are auto-created on first use, but they are intended for explicit
forks only, such as preserving two incompatible calculation branches. Do not
create new contexts for ordinary retries or probes:

```python
mathgr_compute(
    """
trace = Dta(U('a'), D('a'))
result = trace
""",
    context="demo",
    index_dims={"U/D": 3},
)
mathgr_compute("trace", context="demo")
```

Top-level `UseMetric(...)` and `DeclareSym(...)` calls in a block persist as
structured context declarations:

```python
mathgr_compute("""
gMcp = tensor("gMcp")
UseMetric(gMcp, (U, D))
F = tensor("F")
DeclareSym(F, (D, D), Antisymmetric((1, 2)))
result = Simp(F(D('a'), D('a')))
""")
```

Compute blocks intentionally reject imports, loops, function/class definitions,
`with`, private attributes, and unsafe builtins. Module aliases such as `sp`,
`mathgr`, `adm`, `frwadm`, `gr`, `decomp`, and `typeset` are already preloaded.
Restricted expression-only lambdas are allowed for local hooks and callbacks.
If a scalar name collides with a preloaded API name, pass `symbols=[...]` to make
that name a SymPy symbol for the call/context.

`mathgr_inspect(expr, ...)`

: Debugging aid only. Returns `idx`, `free`, `dummy`, tensor heads,
  derivative-node count, and `Pm2` count.

`mathgr_tex(expr, fragment=True, ...)`

: Renders an expression to TeX.

`mathgr_context_get(context="default", name=None)`

: Lists stored declarations and expression source strings. With `name`, returns
  the stored source definition for that name only. It does not evaluate values;
  use `mathgr_compute("name")` for the default context, or pass `context` only
  when reading a named branch.

`mathgr_context_clear(context="default")`

: Deletes a context from MCP memory.

`mathgr_context_save(context="default", path=None, overwrite=False)`

: Saves a context as JSON. With no `path`, uses
  `.mathgr/contexts/<context>.json`.

`mathgr_context_load(path=None, context=None, overwrite=False)`

: Loads a saved JSON context into MCP memory. Context JSON stores declarations
  and expression source strings, not Python objects.

`mathgr_script(expr_or_context, operation=None, ...)`

: Debugging/reproduction aid only. Exports reproducible Python for a structured
  calculation.

`mathgr_manual(section=None, query=None)`

: Reads this manual through MCP, so agents can discover usage after install even
  when the repository files are not otherwise in context.

`mathgr_capabilities` returns grouped public APIs.

`mathgr_run_python(code, timeout_seconds=180.0)` and legacy `mathgr_eval` are
last-resort debugging escape hatches. They run a trusted snippet in a child
process. The namespace preloads:

```python
import sympy as sp
import mathgr
from mathgr import *
```

Set `result = ...` to return a value:

```python
code = """
f = tensor("f")
result = Simp(Dta(UP("a"), DN("b")) * f(UP("b")))
"""
```

Eval limits:

- maximum snippet length is 20,000 characters
- timeout is at least 0.1 seconds
- `open(...)` is blocked
- imports are limited to `json`, `sympy`, `mathgr`, and MathGR submodules
- MathGR global state is snapshotted and restored after each eval

Raw Python tools are for workflows that `mathgr_compute` cannot express. They
are for trusted symbolic snippets and are not a general Python sandbox.

See the repository `README.md` for Codex and Claude Code installation recipes.

## Examples In Repository

`examples/general_presentation.py`

: Core GR presentation, Ricci scalar TeX, Bianchi checks.

`examples/decomp_example.py`

: Total-to-sector decomposition, hooks, metric decomposition.

`examples/newton_gauge.py`

: Newtonian gauge metric rules, perturbative action extraction.

`examples/second_order_pert.py`

: FRW ADM second-order action density in ζ gauge.

`examples/third_order_pert.py`

: FRW ADM third-order perturbation flow.

`examples/third_order_pert_pm2.py`

: Third-order perturbation with `Pm2` constraint kernel.

`examples/zeta_gauge_action_from_delta_phi.py`

: Constraints, `Pm2`, and selected cubic TeX output.

`examples/gauge_trans_df_to_zeta.py`

: Gauge transform expansion with `Eps` and `CollectEps`.

`examples/second_order_gw_pert.py`

: Gravitational-wave perturbation metric setup and action pieces.

`examples/equilateral.py`

: Plain SymPy perturbation expansion with `Eps`.

## Public Root API

The package root exports these names.

Tensor and indices:

```text
UP DN UE DE
LatinIdx GreekIdx LatinCapitalIdx DefaultDim
declare_idx DeclareIdx DeclareExplicitIdx
IdxDual IdxSet IdxColor Dim
IdxHeadPtn IdxPtn IdxUpPtn IdxDnPtn
IdxList IdxUpList IdxDnList
Uniq Uq UniqueIdx
tensor Dta DtaGen LeviCivita
idx free dummy rmE
```

Derivatives and simplification:

```text
Pd P PdT PdVars Pdts pd2pdts pdts2pd
Pm2
Simp SimpHook SimpInto1 SimpSelect SimpUq
TensorReplace
```

Symmetry:

```text
DeclareSym ShowSym DeleteSym
Symmetric Antisymmetric PermutationSymmetry Cycles
Sym AntiSym
```

Rewrite:

```text
Rule RuleDelayed ReplaceAll ReplaceRepeated LabelWild IndexWild
```

GR:

```text
g Metric IdxOfMetric V LapseN ShiftN
UseMetric WithMetric UG DG MetricContract
Affine CovD R G RicciScalar Rsimp
X Dsquare T K KK RADM
```

Decomposition:

```text
UTot DTot U1 D1 U2 D2 DimTot Dim1 Dim2
Decomp Decomp0i Decomp01i Decomp0123 Decomp1i Decomp123 DecompSe
DecompHook
```

Utilities and perturbation:

```text
Eps TSeries TPower SS OO CollectEps LocalToK SolveExpr TReplace k
```

IBP:

```text
Ibp Ibp2 IbpNB IbpVariation IbpRules
Pm2Rules Pm2Simp TrySimp TrySimp2
IbpVar IbpCountLeaf IbpCountTerm IbpCountPt2 IbpCountPd2 IbpStd2
IbpReduceOrder IbpRuleWithForbiddenPattern IbpRuleWithForbiddenPatterrn
PdHold IdHold TrySimpPreferredPattern TrySimpPreferredPatternStrength
```

FRW ADM root aliases:

```text
H DecompG2H Fourier2 FRWK FRWKK FRWRADM Mp Sqrtg FRWShiftN FRWLapseN
a α b β ε η η2 η3 h ζ
```

TeX:

```text
ToTeXString ToTeX DecorateTeXString ToTeXHook ToTeXTemplate
```

## Common Gotchas

- `UseMetric(..., SetAsDefault=True)` mutates the current GR metric.
- Importing `mathgr.adm` mutates the current GR metric to `adm.h`.
- Unsupported signatures usually stay symbolic by design.
- `idx`, `free`, and `dummy` count labels, not variance.
- Explicit indices are skipped by most implicit-index logic.
- Declared symmetry slots are 1-based over implicit slots.
- `ReplaceAll` is one pass; use `ReplaceRepeated` for chains.
- `SimpHook`, `SimpSelect`, `DecompHook`, `ToTeXHook`, and IBP preference
  globals are mutable process state.
- `TPower` and `TSeries` may preserve raw repeated dummy labels until `Simp`.
- `Method="Fast"` skips heavier tensor-product symmetry reduction.
- MCP raw Python escape hatches restore MathGR globals after each run, but they
  still run trusted Python snippets and should not be treated as a hardened
  sandbox.
