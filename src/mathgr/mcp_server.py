from __future__ import annotations

from contextlib import redirect_stdout
from dataclasses import dataclass
from importlib import import_module
from io import StringIO
import json
from multiprocessing import get_context
from queue import Empty
import sys
import traceback
from typing import Any

import sympy as sp

import mathgr
from .mcp_structured import (
    clear_mathgr_context,
    compute_mathgr,
    get_mathgr_context,
    get_mathgr_manual,
    inspect_mathgr,
    load_mathgr_context,
    parse_mathgr,
    save_mathgr_context,
    script_mathgr,
    tex_mathgr,
)


SERVER_NAME = "mathgr"
MAX_CODE_CHARS = 20_000
DEFAULT_TIMEOUT_SECONDS = 10.0
ALLOWED_IMPORTS = {
    "json",
    "mathgr",
    "mathgr.adm",
    "mathgr.decomp",
    "mathgr.frwadm",
    "mathgr.gr",
    "mathgr.ibp",
    "mathgr.rewrite",
    "mathgr.tensor",
    "mathgr.typeset",
    "mathgr.util",
    "sympy",
}


CAPABILITIES = {
    "tensor": [
        "declare_idx",
        "tensor",
        "Dta",
        "DtaGen",
        "LeviCivita",
        "Pd",
        "P",
        "PdT",
        "PdVars",
        "Pm2",
        "Simp",
        "SimpUq",
        "DeclareSym",
        "Symmetric",
        "Antisymmetric",
        "PermutationSymmetry",
        "Cycles",
    ],
    "gr": [
        "UseMetric",
        "WithMetric",
        "MetricContract",
        "Affine",
        "CovD",
        "R",
        "G",
        "RicciScalar",
        "Rsimp",
        "X",
        "Dsquare",
        "T",
        "K",
        "KK",
        "RADM",
    ],
    "decomp": [
        "UTot",
        "DTot",
        "U1",
        "D1",
        "U2",
        "D2",
        "Decomp",
        "Decomp0i",
        "Decomp01i",
        "Decomp0123",
        "Decomp1i",
        "Decomp123",
        "DecompSe",
    ],
    "perturbation": [
        "Eps",
        "TSeries",
        "SS",
        "OO",
        "CollectEps",
        "LocalToK",
        "SolveExpr",
        "TReplace",
    ],
    "ibp": [
        "Ibp",
        "Ibp2",
        "IbpNB",
        "IbpVariation",
        "IbpRules",
        "Pm2Rules",
        "Pm2Simp",
        "TrySimp",
        "TrySimp2",
    ],
    "typeset": [
        "ToTeXString",
        "ToTeX",
        "DecorateTeXString",
    ],
    "mcp_primary": [
        "mathgr_compute",
        "mathgr_tex",
    ],
    "mcp_debugging": [
        "mathgr_parse",
        "mathgr_inspect",
        "mathgr_script",
    ],
    "mcp_context": [
        "mathgr_context_get",
        "mathgr_context_clear",
        "mathgr_context_save",
        "mathgr_context_load",
    ],
    "mcp_docs": [
        "mathgr_manual",
        "mathgr_capabilities",
    ],
    "mcp_escape_hatch": [
        "mathgr_run_python",
        "mathgr_eval",
    ],
}

@dataclass(frozen=True)
class EvalResult:
    ok: bool
    result: str
    stdout: str
    stderr: str

    def as_dict(self) -> dict[str, str | bool]:
        return {
            "ok": self.ok,
            "result": self.result,
            "stdout": self.stdout,
            "stderr": self.stderr,
        }


def list_mathgr_capabilities() -> dict[str, list[str]]:
    """Return grouped public MathGR APIs useful to coding agents."""
    return {group: list(names) for group, names in CAPABILITIES.items()}


def evaluate_mathgr(code: str, timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS) -> dict[str, str | bool]:
    """Execute a trusted MathGR snippet and return `result`, stdout, stderr.

    The snippet runs in a child process with an isolated namespace preloaded
    with `sympy as sp` and all names exported from `mathgr`. Set a variable
    named `result` to return a value. JSON-serializable results are encoded as
    JSON; other results use `str(...)`.
    """
    if len(code) > MAX_CODE_CHARS:
        return EvalResult(
            ok=False,
            result="",
            stdout="",
            stderr=f"Snippet too large: {len(code)} chars > {MAX_CODE_CHARS}.",
        ).as_dict()

    timeout_seconds = max(0.1, float(timeout_seconds))
    context = get_context("fork") if sys.platform != "win32" else get_context("spawn")
    queue = context.Queue(maxsize=1)
    process = context.Process(target=_evaluate_mathgr_worker, args=(str(code), queue))
    process.start()
    process.join(timeout_seconds)
    if process.is_alive():
        process.terminate()
        process.join(1)
        return EvalResult(
            ok=False,
            result="",
            stdout="",
            stderr=f"mathgr_eval timed out after {timeout_seconds:g} seconds.",
        ).as_dict()
    try:
        response = queue.get_nowait()
    except Empty:
        return EvalResult(
            ok=False,
            result="",
            stdout="",
            stderr=f"mathgr_eval worker exited with code {process.exitcode} without a result.",
        ).as_dict()
    return response


def _evaluate_mathgr_worker(code: str, queue) -> None:
    queue.put(_evaluate_mathgr_in_process(code))


def _evaluate_mathgr_in_process(code: str) -> dict[str, str | bool]:
    stdout = StringIO()
    namespace = _eval_namespace()
    state = _snapshot_mathgr_state()
    try:
        compiled = compile(str(code), "<mathgr-mcp>", "exec")
        with redirect_stdout(stdout):
            exec(compiled, namespace, namespace)
        result = _serialize_result(namespace.get("result", ""))
    except Exception:
        response = EvalResult(
            ok=False,
            result="",
            stdout=stdout.getvalue(),
            stderr=traceback.format_exc(),
        )
    else:
        response = EvalResult(
            ok=True,
            result=result,
            stdout=stdout.getvalue(),
            stderr="",
        )
    finally:
        _restore_mathgr_state(state)

    return response.as_dict()


def _eval_namespace() -> dict[str, Any]:
    def unavailable_open(*_args, **_kwargs):
        raise RuntimeError("filesystem access is not available in mathgr_eval")

    namespace: dict[str, Any] = {
        "__builtins__": {
            "abs": abs,
            "all": all,
            "any": any,
            "bool": bool,
            "dict": dict,
            "enumerate": enumerate,
            "float": float,
            "int": int,
            "len": len,
            "list": list,
            "max": max,
            "min": min,
            "pow": pow,
            "print": print,
            "range": range,
            "repr": repr,
            "round": round,
            "set": set,
            "sorted": sorted,
            "str": str,
            "sum": sum,
            "tuple": tuple,
            "Exception": Exception,
            "ValueError": ValueError,
            "open": unavailable_open,
            "__import__": _safe_import,
        },
        "sp": sp,
        "mathgr": mathgr,
    }
    namespace.update({name: getattr(mathgr, name) for name in mathgr.__all__ if hasattr(mathgr, name)})
    namespace["UseMetric"] = _use_metric_wrapper(namespace)
    return namespace


def _use_metric_wrapper(namespace):
    def wrapped_use_metric(*args, **kwargs):
        value = mathgr.UseMetric(*args, **kwargs)
        namespace["Metric"] = mathgr.Metric
        namespace["IdxOfMetric"] = mathgr.IdxOfMetric
        return value

    return wrapped_use_metric


def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    if level != 0:
        raise ImportError("relative imports are not available in mathgr_eval")
    if name == "json":
        return json
    if name == "sympy":
        return sp
    if name == "mathgr" and _mathgr_fromlist_allowed(fromlist):
        return mathgr
    if name in ALLOWED_IMPORTS:
        return __import__(name, globals, locals, fromlist, level)
    raise ImportError(f"module {name!r} is not available in mathgr_eval")


def _mathgr_fromlist_allowed(fromlist) -> bool:
    for item in fromlist or ():
        if item == "*":
            continue
        if item in mathgr.__all__:
            continue
        if f"mathgr.{item}" in ALLOWED_IMPORTS:
            continue
        return False
    return True


def _serialize_result(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple, str, int, float, bool)) or value is None:
        try:
            return json.dumps(value, sort_keys=True)
        except TypeError:
            pass
    return str(value)


def _snapshot_mathgr_state() -> dict[str, Any]:
    tensor_module = import_module("mathgr.tensor")
    decomp_module = import_module("mathgr.decomp")
    gr_module = import_module("mathgr.gr")
    typeset_module = import_module("mathgr.typeset")
    return {
        "index_types": dict(tensor_module._INDEX_TYPES),
        "constants": set(tensor_module._CONSTANTS),
        "metrics": dict(tensor_module._METRICS),
        "metric_heads": set(tensor_module._METRIC_HEADS),
        "metric_index_pairs": {key: list(value) for key, value in tensor_module._METRIC_INDEX_PAIRS.items()},
        "symmetries": {key: list(value) for key, value in tensor_module._SYMMETRIES.items()},
        "uniq_counter_value": tensor_module._UNIQ_COUNTER_VALUE,
        "idx_list": list(tensor_module.IdxList),
        "idx_up_list": list(tensor_module.IdxUpList),
        "idx_dn_list": list(tensor_module.IdxDnList),
        "simp_hook": list(tensor_module.SimpHook),
        "simp_into1": tuple(tensor_module.SimpInto1),
        "simp_select": tensor_module.SimpSelect,
        "decomp_hook": list(decomp_module.DecompHook),
        "metric": gr_module.Metric,
        "idx_of_metric": tuple(gr_module.IdxOfMetric),
        "tex_hook": list(typeset_module.ToTeXHook),
        "tex_template": typeset_module.ToTeXTemplate,
    }


def _restore_mathgr_state(state: dict[str, Any]) -> None:
    tensor_module = import_module("mathgr.tensor")
    decomp_module = import_module("mathgr.decomp")
    gr_module = import_module("mathgr.gr")
    typeset_module = import_module("mathgr.typeset")

    tensor_module._INDEX_TYPES.clear()
    tensor_module._INDEX_TYPES.update(state["index_types"])
    tensor_module._CONSTANTS.clear()
    tensor_module._CONSTANTS.update(state["constants"])
    tensor_module._METRICS.clear()
    tensor_module._METRICS.update(state["metrics"])
    tensor_module._METRIC_HEADS.clear()
    tensor_module._METRIC_HEADS.update(state["metric_heads"])
    tensor_module._METRIC_INDEX_PAIRS.clear()
    tensor_module._METRIC_INDEX_PAIRS.update({key: list(value) for key, value in state["metric_index_pairs"].items()})
    tensor_module._SYMMETRIES.clear()
    tensor_module._SYMMETRIES.update({key: list(value) for key, value in state["symmetries"].items()})
    tensor_module._UNIQ_COUNTER_VALUE = state["uniq_counter_value"]

    tensor_module.IdxList[:] = state["idx_list"]
    tensor_module.IdxUpList[:] = state["idx_up_list"]
    tensor_module.IdxDnList[:] = state["idx_dn_list"]
    tensor_module.SimpHook[:] = state["simp_hook"]
    tensor_module.SimpInto1 = state["simp_into1"]
    tensor_module.SimpSelect = state["simp_select"]
    decomp_module.DecompHook[:] = state["decomp_hook"]
    gr_module.Metric = state["metric"]
    gr_module.IdxOfMetric = state["idx_of_metric"]
    typeset_module.ToTeXHook[:] = state["tex_hook"]
    typeset_module.ToTeXTemplate = state["tex_template"]


def create_mcp():
    """Create the MathGR MCP app."""
    from mcp.server.fastmcp import FastMCP

    mcp = FastMCP(
        SERVER_NAME,
        instructions=(
            "MathGR symbolic tensor/GR toolkit. PRIMARY RULE: use mathgr_compute "
            "as the first-choice tool for almost all MathGR calculations. It accepts "
            "single expressions and multi-line notebook blocks with assignments. "
            "Put ordinary MathGR calls such as Simp, Decomp0i, Ibp, OO(2), ToTeXString "
            "inside mathgr_compute. Use mathgr_parse, mathgr_inspect, and "
            "mathgr_script only for debugging/reproduction. Use mathgr_run_python "
            "or mathgr_eval only as last-resort debugging escape hatches when "
            "mathgr_compute cannot express the workflow. Use mathgr_manual for docs."
        ),
    )

    @mcp.tool()
    def mathgr_capabilities() -> dict[str, list[str]]:
        """List MathGR APIs; use `mcp_primary` first, especially `mathgr_compute`."""
        return list_mathgr_capabilities()

    @mcp.tool()
    def mathgr_eval(code: str, timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS) -> dict[str, str | bool]:
        """Legacy debugging escape hatch. Do not use for normal calculations; prefer mathgr_compute."""
        return evaluate_mathgr(code, timeout_seconds=timeout_seconds)

    @mcp.tool()
    def mathgr_run_python(code: str, timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS) -> dict[str, str | bool]:
        """Last-resort debugging escape hatch for trusted Python. Prefer mathgr_compute."""
        return evaluate_mathgr(code, timeout_seconds=timeout_seconds)

    @mcp.tool()
    def mathgr_manual(section: str | None = None, query: str | None = None) -> dict[str, Any]:
        """Read the MathGR manual. For calculations, use mathgr_compute first."""
        return get_mathgr_manual(section=section, query=query)

    @mcp.tool()
    def mathgr_parse(
        expr: str,
        context: str | None = None,
        auto_declare: bool = True,
        declarations: dict[str, Any] | None = None,
        index_dims: dict[str, Any] | None = None,
        index_sets: dict[str, str] | None = None,
        tensors: list[str] | None = None,
        symbols: list[str] | None = None,
        metric: dict[str, Any] | None = None,
        symmetries: list[dict[str, Any]] | None = None,
        output: list[str] | None = None,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> dict[str, Any]:
        """Debugging aid only: dry-run auto declarations/Python. For actual math, use mathgr_compute."""
        return parse_mathgr(
            expr,
            context=context,
            auto_declare=auto_declare,
            declarations=declarations,
            index_dims=index_dims,
            index_sets=index_sets,
            tensors=tensors,
            symbols=symbols,
            metric=metric,
            symmetries=symmetries,
            output=output,
            timeout_seconds=timeout_seconds,
        )

    @mcp.tool()
    def mathgr_compute(
        expr: str,
        context: str | None = None,
        auto_declare: bool = True,
        declarations: dict[str, Any] | None = None,
        index_dims: dict[str, Any] | None = None,
        index_sets: dict[str, str] | None = None,
        tensors: list[str] | None = None,
        symbols: list[str] | None = None,
        metric: dict[str, Any] | None = None,
        symmetries: list[dict[str, Any]] | None = None,
        output: list[str] | None = None,
        store_as: str | None = None,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> dict[str, Any]:
        """PRIMARY calculation tool: evaluate expressions or multi-line notebook blocks; persists assignments in context."""
        return compute_mathgr(
            expr,
            context=context,
            auto_declare=auto_declare,
            declarations=declarations,
            index_dims=index_dims,
            index_sets=index_sets,
            tensors=tensors,
            symbols=symbols,
            metric=metric,
            symmetries=symmetries,
            output=output,
            store_as=store_as,
            timeout_seconds=timeout_seconds,
        )

    @mcp.tool()
    def mathgr_inspect(
        expr: str,
        context: str | None = None,
        auto_declare: bool = True,
        declarations: dict[str, Any] | None = None,
        index_dims: dict[str, Any] | None = None,
        index_sets: dict[str, str] | None = None,
        tensors: list[str] | None = None,
        symbols: list[str] | None = None,
        metric: dict[str, Any] | None = None,
        symmetries: list[dict[str, Any]] | None = None,
        output: list[str] | None = None,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> dict[str, Any]:
        """Debugging aid only: inspect indices/tensor heads/Pd/Pm2. For actual math, use mathgr_compute."""
        return inspect_mathgr(
            expr,
            context=context,
            auto_declare=auto_declare,
            declarations=declarations,
            index_dims=index_dims,
            index_sets=index_sets,
            tensors=tensors,
            symbols=symbols,
            metric=metric,
            symmetries=symmetries,
            output=output,
            timeout_seconds=timeout_seconds,
        )

    @mcp.tool()
    def mathgr_tex(
        expr: str,
        fragment: bool = True,
        context: str | None = None,
        auto_declare: bool = True,
        declarations: dict[str, Any] | None = None,
        index_dims: dict[str, Any] | None = None,
        output: list[str] | None = None,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> dict[str, Any]:
        """Render an expression to TeX."""
        return tex_mathgr(
            expr,
            fragment=fragment,
            context=context,
            auto_declare=auto_declare,
            declarations=declarations,
            index_dims=index_dims,
            output=output,
            timeout_seconds=timeout_seconds,
        )

    @mcp.tool()
    def mathgr_context_get(
        context: str = "default",
        name: str | None = None,
        output: list[str] | None = None,
    ) -> dict[str, Any]:
        """List stored context declarations/expressions, or one stored source definition by name."""
        return get_mathgr_context(context, name=name, output=output)

    @mcp.tool()
    def mathgr_context_clear(context: str = "default") -> dict[str, Any]:
        """Clear a named MathGR context."""
        return clear_mathgr_context(context)

    @mcp.tool()
    def mathgr_context_save(
        context: str = "default",
        path: str | None = None,
        overwrite: bool = False,
    ) -> dict[str, Any]:
        """Save context declarations and expression source strings to JSON."""
        return save_mathgr_context(context, path=path, overwrite=overwrite)

    @mcp.tool()
    def mathgr_context_load(
        path: str | None = None,
        context: str | None = None,
        overwrite: bool = False,
    ) -> dict[str, Any]:
        """Load a saved JSON context into MCP memory."""
        return load_mathgr_context(path=path, context=context, overwrite=overwrite)

    @mcp.tool()
    def mathgr_script(
        expr_or_context: str | None = None,
        operation: str | None = None,
        context: str | None = None,
        declarations: dict[str, Any] | None = None,
        index_dims: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Debugging/reproduction aid only: export Python. For actual math, use mathgr_compute."""
        return script_mathgr(
            expr_or_context,
            operation=operation,
            context=context,
            declarations=declarations,
            index_dims=index_dims,
        )

    return mcp


def main() -> None:
    """Run the MathGR MCP server over stdio."""
    create_mcp().run()


if __name__ == "__main__":
    main()
