# MathGR-Py: A MCP/Python Port of MathGR for Tensor Calculus and General Relativity

`mathgr` is a Python/SymPy and MCP port of
[`tririver/MathGR`](https://github.com/tririver/MathGR), a symbolic toolkit for
tensor calculus, general relativity, ADM/FRW decompositions, perturbation
series, integration by parts, and TeX export.

The package has two user-facing interfaces:

- MCP interface: designed for Codex, Claude Code, and other coding agents. It
  accepts Python-like expression strings, auto-declares tensors and indices, and
  exposes tools such as `mathgr_compute`, `mathgr_parse`, and `mathgr_tex`.
- Python interface: normal Python/SymPy API for scripts, notebooks, tests, and
  direct human use.

Wolfram is not needed at runtime. The test suite translates upstream MathGR
tests and notebook cells to pytest, with an optional Wolfram oracle test for
checking the original package.

## Why Python and SymPy?

Wolfram Language is concise and pleasant for human symbolic work. This port
exists because Python is easier for AI coding agents to inspect, run, test,
modify, and expose through MCP tools. The initial motivation was better agent
interaction with MathGR; it is also useful for humans who prefer a Python/SymPy
workflow.

## MCP Interface

Use the MCP interface when you want a coding agent to call MathGR directly after
startup. The MCP surface is intentionally different from raw Python: agents send
Python-like expression strings and the server auto-declares common objects.

### MCP Install

For Codex or Claude Code, you usually do not need to install the MCP server by
hand. Start your agent in this repository and ask:

```text
Read the Agent MCP Install Recipe at the end of README.md and install MathGR MCP
for your own future sessions.
```

The agent should follow that recipe, update its own MCP configuration, then tell
you to start a new Codex or Claude Code session. New sessions can call the
`mathgr` MCP tools immediately.

Manual smoke check from this repository:

```bash
uv sync
timeout 2s uv run mathgr-mcp
```

Exit code `0` or `124` from `timeout` is acceptable. A Python traceback is not.

### MCP Usage

Core tools:

- `mathgr_manual`: read the full manual or a named section
- `mathgr_compute`: first-choice tool for almost all MathGR calculations;
  auto-declare and evaluate a Python-like MathGR expression
- `mathgr_tex`: render an expression to TeX
- `mathgr_context_create`, `mathgr_context_update`, `mathgr_context_get`,
  `mathgr_context_clear`: keep reusable declarations and named expressions in
  the MCP server process
- `mathgr_parse`: only for debugging; dry-run a Python-like expression and show
  inferred declarations plus reproducible Python
- `mathgr_inspect`: only for debugging; list indices, free/dummy labels, tensor
  heads, derivative nodes, and `Pm2` nodes
- `mathgr_script`: only for debugging/reproduction; export Python for a
  structured calculation
- `mathgr_capabilities`: compact API group list
- `mathgr_run_python` / `mathgr_eval`: last-resort debugging escape hatch for
  raw trusted Python

Agents should use `mathgr_compute` first for almost every calculation. The
compute tool is easier than raw Python because it auto-declares index families,
tensor heads, and scalar symbols. Use `mathgr_parse`, `mathgr_inspect`, and
`mathgr_script` only for debugging. Use `mathgr_run_python` / `mathgr_eval` only
when `mathgr_compute` cannot express the workflow.

Example:

```text
mathgr_compute("Simp(Dta(U('a'), D('b')) * f(U('b')))")
```

The MCP server infers:

```python
Dim = sp.Symbol("Dim")
U, D = declare_idx("U", "D", dim=Dim)
f = tensor("f")
```

Typical MCP calls:

```text
mathgr_compute("Simp(Dta(U('a'), D('b')) * f(U('b')))")
mathgr_compute("Decomp0i(f(DTot('a')) * f(UTot('a')))")
mathgr_compute("Ibp(y * Pd(x, D('i')))")
mathgr_compute("OO(2)((1 + Eps*x)**3)")
mathgr_compute("Simp(lhs - rhs)")
mathgr_tex("Pd(f(U('a')), D('i'))")
```

Override dimensions when needed:

```json
{"index_dims": {"U/D": 3}}
```

Use contexts for multi-step calculations:

```text
mathgr_context_create("demo")
mathgr_context_update(
  "demo",
  declarations={"index_dims": {"U/D": 3}},
  expressions={"trace": "Dta(U('a'), D('a'))"}
)
mathgr_compute("trace", context="demo")
```

Result:

```text
3
```

Security note: raw Python tools are intended for trusted symbolic snippets from
the agent session, not as a sandbox for hostile code. They run snippets in a
child process with a timeout, restore MathGR global state after each call, and
block common filesystem/shell/import paths, but Python execution is not a
security boundary.

## Python Interface

Use the Python interface for scripts, notebooks, tests, and direct work in a
Python process.

### Python Install

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

### Python Usage

Most notebooks can start from the package root import:

```python
import sympy as sp
from mathgr import *
```

For scripts and libraries, prefer explicit imports from `mathgr`,
`mathgr.tensor`, `mathgr.gr`, `mathgr.decomp`, `mathgr.frwadm`, `mathgr.util`,
`mathgr.ibp`, and `mathgr.typeset`.

Declare indices and tensors:

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

Add tensor symmetries:

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

Differentiate symbolically:

```python
from mathgr import Pd

h = tensor("h")
expr = f(u("a")) * h(d("b"))

print(Pd(expr, d("c")))
```

`Pd` applies product, sum, and supported power rules and stores derivatives as
`PdT(expr, PdVars(...))`.

Use a metric and GR identities:

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

Split time and space indices:

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

Expand perturbation series:

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

Export TeX:

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

Run examples:

```bash
uv run python -m examples.general_presentation
uv run python -m examples.decomp_example
uv run python -m examples.second_order_pert
uv run python -m examples.zeta_gauge_action_from_delta_phi
```

Larger notebook ports may be expensive if you enable full action computation.
Many expose flags such as `main(compute_action=False)`.

See [doc/manual.md](doc/manual.md) for a more complete API manual.

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
