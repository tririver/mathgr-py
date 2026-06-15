# MathGR-Py: A Python/SymPy Port of MathGR for Tensor Calculus and General Relativity

`mathgr` is a Python/SymPy port of
[`tririver/MathGR`](https://github.com/tririver/MathGR), a symbolic toolkit for
tensor calculus, general relativity, ADM/FRW decompositions, perturbation
series, integration by parts, and TeX export.

This port keeps the MathGR-style API where practical, but runs as normal Python.
Wolfram is not needed at runtime. The test suite translates upstream MathGR
tests and notebook cells to pytest, with an optional Wolfram oracle test for
checking the original package.

## Why Python and SymPy?

Wolfram Language is concise and pleasant for human symbolic work. This port
exists because Python is easier for AI coding agents to inspect, run, test,
modify, and expose through MCP tools. The initial motivation was better agent
interaction with MathGR; it is also useful for humans who prefer a Python/SymPy
workflow.

## Install

### 1. Install MCP

For Codex or Claude Code, you usually do not need to install the MCP server by
hand. Start your agent in this repository and ask:

```text
Read the Agent MCP Install Recipe at the end of README.md and install MathGR MCP
for your own future sessions.
```

The agent should follow that recipe, update its own MCP configuration, then tell
you to start a new Codex or Claude Code session. New sessions can then call the
`mathgr` MCP tools immediately.

### 2. Install Standalone

From this repository:

```bash
uv sync
```

Use it in another editable environment:

```bash
python -m pip install -e .
```

Requirements:

- Python 3.12 or newer
- SymPy 1.13 or newer

## First Session

Most users can start from the package root import:

```python
import sympy as sp
from mathgr import *
```

For scripts and libraries, prefer explicit imports from `mathgr`, `mathgr.tensor`,
`mathgr.gr`, `mathgr.decomp`, `mathgr.frwadm`, `mathgr.util`, `mathgr.ibp`, and
`mathgr.typeset`.

### 1. Declare Indices And Tensors

```python
from mathgr import LatinIdx, declare_idx, tensor, Dta, Simp

u, d = declare_idx("U", "D", dim=3, index_set=LatinIdx)
f = tensor("f")

expr = Dta(u("a"), d("b")) * f(u("b"))
print(Simp(expr))
```

Output:

```text
f(U('a'))
```

`declare_idx` creates dual index families. `tensor("f")` creates a callable
tensor head. `Dta` is the delta tensor. `Simp` applies MathGR-style
simplification: delta contraction, dummy-index normalization, declared tensor
symmetries, metric contractions, and supported hook rules.

### 2. Add Tensor Symmetries

```python
from mathgr import DeclareSym, Symmetric

S = tensor("S")
DeclareSym(S, (d, d), Symmetric((1, 2)))

print(Simp(S(d("b"), d("a")) - S(d("a"), d("b"))))
```

Output:

```text
0
```

Use `Antisymmetric`, `PermutationSymmetry`, or `Cycles` for other slot
symmetries. Use `ShowSym(head, signature)` and `DeleteSym(head, signature)` to
inspect or remove declarations.

### 3. Differentiate Symbolically

```python
from mathgr import Pd

h = tensor("h")
expr = f(u("a")) * h(d("b"))

print(Pd(expr, d("c")))
```

Output is an unevaluated tensor derivative with product rule applied:

```text
f(U('a'))*_PdT(h(D('b')), _PdVars(D('c'))) + h(D('b'))*_PdT(f(U('a')), _PdVars(D('c')))
```

Shortcuts:

- `Pd(expr, index)` differentiates by one index.
- `P(i, j)(expr)` is curried partial differentiation.
- `PdT(expr, PdVars(...))` is the internal derivative form.
- `Pm2(expr, index_type)` represents inverse Laplacian behavior used by the
  perturbation utilities.

### 4. Use A Metric And GR Identities

```python
from mathgr import LatinIdx, declare_idx, tensor, UseMetric, R, CovD, Simp

u4, d4 = declare_idx("U4", "D4", dim=4, index_set=LatinIdx)
g4 = tensor("g4")
UseMetric(g4, (u4, d4))

second_bianchi = (
    CovD(R(d4("a"), d4("b"), d4("c"), d4("d")), d4("e"))
    + CovD(R(d4("a"), d4("b"), d4("d"), d4("e")), d4("c"))
    + CovD(R(d4("a"), d4("b"), d4("e"), d4("c")), d4("d"))
)

print(Simp(second_bianchi))
```

Output:

```text
0
```

Core GR helpers include:

- `UseMetric(metric, (up, down))`
- `WithMetric(metric, (up, down), callback)`
- `MetricContract(expr)`
- `Affine`, `CovD`, `R`, `G`, `RicciScalar`, `Rsimp`
- scalar-field helpers `X`, `Dsquare`, `T`
- ADM helpers `K`, `KK`, `RADM`

### 5. Split Time And Space Indices

The decomposition module has total indices `UTot`/`DTot` and sector splitters.
`Decomp0i` splits one total dummy contraction into explicit time plus spatial
parts:

```python
from mathgr import Decomp0i, Simp, tensor
from mathgr.decomp import UTot, DTot

f = tensor("f")
expr = Decomp0i(f(UTot("mu"), DTot("mu")))

print(expr)
print(Simp(expr))
```

Output:

```text
f(UE(0), DE(0)) + f(UP('mu'), DN('mu'))
f(UE(0), DE(0)) + f(UP('a'), DN('a'))
```

Other splitters:

- `Decomp01i`
- `Decomp0123`
- `Decomp1i`
- `Decomp123`
- `DecompSe`
- generic `Decomp(expr, sectors, indices=None, hooks=None)`

### 6. Expand Perturbation Series

```python
import sympy as sp
from mathgr import Eps, SS, OO

x = sp.symbols("x")
expr = (1 + Eps * x) ** 3

print(SS(2)(expr))
print(OO(2)(expr))
```

Output:

```text
3*Eps**2*x**2 + 3*Eps*x + 1
3*x**2
```

Useful perturbation utilities:

- `Eps`: default perturbation symbol
- `TSeries(expr, (Eps, 0, n))`: tensor-aware series
- `SS(n)`: keep series through order `n`
- `OO(n)`: extract order `n`
- `CollectEps`, `LocalToK`, `SolveExpr`, `TReplace`

### 7. Export TeX

```python
import mathgr.typeset as typeset
from mathgr import DN, UP, Pd, tensor

previous = typeset.ToTeXTemplate
try:
    typeset.ToTeXTemplate = False
    f = tensor("f")
    print(typeset.ToTeXString(f(UP("a"), DN("b"))))
    print(typeset.ToTeXString(Pd(f(DN("i")), DN("j"))))
finally:
    typeset.ToTeXTemplate = previous
```

Output:

```text
f^{a}_{b}
\partial_{j} f_{i}
```

Set `mathgr.typeset.ToTeXTemplate = True` to wrap output in a small LaTeX
document.

## Run Examples

Examples are ordinary Python modules:

```bash
uv run python -m examples.general_presentation
uv run python -m examples.decomp_example
uv run python -m examples.second_order_pert
uv run python -m examples.zeta_gauge_action_from_delta_phi
```

Larger notebook ports may be expensive if you enable full action computation.
Many expose flags such as `main(compute_action=False)`.

## Agent MCP Install Recipe

This section is written for Codex, Claude Code, or another coding agent. If a
human asks you to install MathGR MCP, execute these steps from this repository.

1. Resolve the absolute repo path:

```bash
REPO="$(pwd)"
```

Use `$REPO` for all commands below.

2. Install runtime dependencies and smoke-check the stdio MCP command:

```bash
uv --directory "$REPO" sync
timeout 2s uv --directory "$REPO" run mathgr-mcp
```

Exit code `0` or `124` from `timeout` is acceptable. Any Python traceback is not
acceptable; debug before continuing.

3. Install for Claude Code if `claude` is available:

```bash
claude mcp remove mathgr || true
claude mcp add --scope user mathgr -- uv --directory "$REPO" run mathgr-mcp
claude mcp get mathgr
```

This user-scope install makes MathGR MCP available to new Claude Code sessions
on this machine. Claude Code can also use this repo's project-scoped `.mcp.json`
when started in the repo and approved by the user.

4. Install for Codex if `codex` is available:

```bash
codex mcp remove mathgr || true
codex mcp add mathgr -- uv --directory "$REPO" run mathgr-mcp
codex mcp get mathgr
```

This user-level install makes MathGR MCP available to new Codex sessions on this
machine. Codex can also use this repo's project-scoped `.codex/config.toml` after
the repo is trusted.

5. Tell the human exactly what changed and what to do next:

- Start a new Codex or Claude Code session so the MCP server is loaded at
  startup.
- In Codex, `/mcp` shows active MCP servers.
- In Claude Code, `/mcp` shows connection state and tool count.
- Ask the agent to call `mathgr_capabilities` to verify MathGR tools are live.

Available MCP tools:

- `mathgr_capabilities`: grouped public API names
- `mathgr_topic`: quick reference for `quickstart`, `tensor`, `gr`, `decomp`,
  `perturbation`, `ibp`, or `typeset`
- `mathgr_eval`: run a small trusted MathGR snippet; set `result = ...` to
  return a value. Optional `timeout_seconds` defaults to 10 seconds.

Example `mathgr_eval` code:

```python
f = tensor("f")
expr = Dta(UP("a"), DN("b")) * f(UP("b"))
result = Simp(expr)
```

Security note: `mathgr_eval` is intended for trusted symbolic snippets from the
agent session, not as a sandbox for hostile code. It runs snippets in a child
process with a timeout, restores MathGR global state after each call, and blocks
common filesystem/shell/import paths, but Python execution is not a security
boundary.
