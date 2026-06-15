from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from itertools import count
from functools import wraps
from importlib import import_module
from pathlib import Path
import time
from typing import Any

import sympy as sp

import mathgr

tensor_module = import_module("mathgr.tensor")


DEFAULT_OUTPUT = ("str", "tex", "diagnostics", "python")
DEFAULT_DIM_SYMBOL = "Dim"
DEFAULT_CONTEXT_NAME = "default"
CONTEXT_SCHEMA_VERSION = 1
_CONTEXT_COUNTER = count(1)
_CONTEXTS: dict[str, dict[str, Any]] = {}

_BUILTIN_INDEX_FAMILIES = {
    "UP",
    "DN",
    "UE",
    "DE",
    "UTot",
    "DTot",
    "U1",
    "D1",
    "U2",
    "D2",
}

_SYMPY_NAMES = {
    "Abs": sp.Abs,
    "Eq": sp.Eq,
    "Function": sp.Function,
    "I": sp.I,
    "Integer": sp.Integer,
    "Rational": sp.Rational,
    "Symbol": sp.Symbol,
    "cos": sp.cos,
    "cosh": sp.cosh,
    "diff": sp.diff,
    "exp": sp.exp,
    "factor": sp.factor,
    "pi": sp.pi,
    "sin": sp.sin,
    "sinh": sp.sinh,
    "sqrt": sp.sqrt,
    "symbols": sp.symbols,
}

_SAFE_EXPR_ATTRIBUTES = {
    "coeff",
    "expand",
    "factor",
    "subs",
    "xreplace",
}

_MODULE_ALIASES = {
    "adm": "mathgr.adm",
    "decomp": "mathgr.decomp",
    "frwadm": "mathgr.frwadm",
    "gr": "mathgr.gr",
    "ibp": "mathgr.ibp",
    "rewrite": "mathgr.rewrite",
    "tensor": "mathgr.tensor",
    "typeset": "mathgr.typeset",
    "util": "mathgr.util",
}

_MODULE_ATTR_NAMES = {"sp", "mathgr", *_MODULE_ALIASES}

_BLOCKED_CALL_NAMES = {
    "__import__",
    "compile",
    "delattr",
    "dir",
    "eval",
    "exec",
    "getattr",
    "globals",
    "input",
    "locals",
    "open",
    "setattr",
    "vars",
}


@dataclass(frozen=True)
class _Prepared:
    namespace: dict[str, Any]
    auto_declarations: dict[str, Any]
    python_prefix: list[str]
    diagnostics: list[str]


@dataclass(frozen=True)
class _BlockAssignment:
    name: str
    source: str


@dataclass(frozen=True)
class _ComputeBlock:
    assignments: list[_BlockAssignment]
    expr_sources: list[str]
    result_source: str | None
    statement_sources: list[str]


class _ExpressionScanner(ast.NodeVisitor):
    def __init__(self, known_names: set[str], context_names: set[str]):
        self.known_names = known_names
        self.context_names = context_names
        self.call_names: set[str] = set()
        self.index_names: set[str] = set()
        self.tensor_names: set[str] = set()
        self.bare_names: set[str] = set()
        self._call_func_stack: list[str] = []

    def visit_Call(self, node: ast.Call) -> Any:
        func_name = node.func.id if isinstance(node.func, ast.Name) else None
        if func_name is not None:
            self.call_names.add(func_name)
            if self._looks_like_index_call(func_name, node):
                self.index_names.add(func_name)
            elif func_name not in self.known_names and func_name not in self.context_names:
                self.tensor_names.add(func_name)
            self._call_func_stack.append(func_name)
            self.visit(node.func)
            self._call_func_stack.pop()
        else:
            self.visit(node.func)
        for arg in node.args:
            self.visit(arg)
        for keyword in node.keywords:
            self.visit(keyword.value)

    def visit_Name(self, node: ast.Name) -> Any:
        if self._call_func_stack and self._call_func_stack[-1] == node.id:
            return
        if node.id not in self.known_names and node.id not in self.context_names:
            self.bare_names.add(node.id)

    def _looks_like_index_call(self, name: str, node: ast.Call) -> bool:
        if name in self.known_names or name in self.context_names:
            return False
        if len(node.args) != 1 or node.keywords:
            return False
        arg = node.args[0]
        if not (isinstance(arg, ast.Constant) and isinstance(arg.value, (str, int))):
            return False
        return _index_pair_for_name(name) is not None


def get_mathgr_manual(section: str | None = None, query: str | None = None) -> dict[str, Any]:
    """Return the local MathGR manual, optionally narrowed by section/query."""
    manual_path = Path(__file__).resolve().parents[2] / "doc" / "manual.md"
    if manual_path.exists():
        content = manual_path.read_text(encoding="utf-8")
    else:
        content = _fallback_manual_text()

    selected = _select_manual_section(content, section) if section else content
    if query:
        selected = _select_query_lines(selected, query)
    return {
        "ok": True,
        "section": section or "",
        "query": query or "",
        "content": selected,
        "path": str(manual_path) if manual_path.exists() else "",
    }


def parse_mathgr(
    expr: str,
    *,
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
    timeout_seconds: float = 10.0,
) -> dict[str, Any]:
    """Infer declarations for a Python-like MathGR expression without evaluating it."""
    started = time.perf_counter()
    try:
        prepared = _prepare_runtime(
            [expr],
            context=context,
            auto_declare=auto_declare,
            declarations=declarations,
            index_dims=index_dims,
            index_sets=index_sets,
            tensors=tensors,
            symbols=symbols,
            metric=metric,
            symmetries=symmetries,
        )
    except Exception as exc:
        return _error_response(exc, started)
    return {
        "ok": True,
        "result": "",
        "auto_declarations": prepared.auto_declarations,
        "python": _script_for_expression(prepared, expr, operation=None),
        "diagnostics": prepared.diagnostics,
        "time_seconds": _elapsed(started),
    }


def compute_mathgr(
    expr: str,
    *,
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
    timeout_seconds: float = 10.0,
) -> dict[str, Any]:
    """Auto-declare and evaluate a Python-like MathGR expression without implicit transforms."""
    started = time.perf_counter()
    try:
        expr_text = str(expr).strip()
        context_name = _ensure_context(context)
        block = _parse_compute_block(expr_text)
        prepare_exprs = block.expr_sources if block else [expr_text]
        prepared = _prepare_runtime(
            prepare_exprs,
            context=context_name,
            auto_declare=auto_declare,
            declarations=declarations,
            index_dims=index_dims,
            index_sets=index_sets,
            tensors=tensors,
            symbols=symbols,
            metric=metric,
            symmetries=symmetries,
        )
        if block:
            value = _eval_block(block, prepared.namespace, context_name)
        else:
            value = _eval_expr(expr_text, prepared.namespace)
        _persist_runtime_declarations(
            context_name,
            declarations=declarations,
            index_dims=index_dims,
            index_sets=index_sets,
            tensors=tensors,
            symbols=symbols,
            metric=metric,
            symmetries=symmetries,
        )
        if store_as:
            _store_expression(context_name, store_as, str(value))
        response = _operation_response(value, prepared, expr_text, started, output, operation="compute")
        response["context"] = context_name
        return response
    except Exception as exc:
        return _error_response(exc, started)


def simplify_mathgr(
    expr: str,
    *,
    context: str | None = None,
    auto_declare: bool = True,
    declarations: dict[str, Any] | None = None,
    index_dims: dict[str, Any] | None = None,
    index_sets: dict[str, str] | None = None,
    tensors: list[str] | None = None,
    symbols: list[str] | None = None,
    metric: dict[str, Any] | None = None,
    symmetries: list[dict[str, Any]] | None = None,
    hooks: list[Any] | None = None,
    method: str = "Hybrid",
    dummy: list[str] | None = None,
    output: list[str] | None = None,
    store_as: str | None = None,
    timeout_seconds: float = 10.0,
) -> dict[str, Any]:
    """Auto-declare a Python-like expression, run `Simp`, and return diagnostics."""
    started = time.perf_counter()
    try:
        prepared = _prepare_runtime(
            [expr, *_hook_exprs(hooks)],
            context=context,
            auto_declare=auto_declare,
            declarations=declarations,
            index_dims=index_dims,
            index_sets=index_sets,
            tensors=tensors,
            symbols=symbols,
            metric=metric,
            symmetries=symmetries,
        )
        value = _eval_expr(expr, prepared.namespace)
        hook_values = _coerce_rule_items(hooks, prepared.namespace)
        options: dict[str, Any] = {"Method": method}
        if dummy is not None:
            options["Dummy"] = tuple(dummy)
        if hook_values:
            options["hooks"] = hook_values
        simplified = mathgr.Simp(value, **options)
        if store_as:
            _store_expression(context, store_as, str(simplified))
        return _operation_response(
            simplified,
            prepared,
            expr,
            started,
            output,
            operation="simplify",
            changed=simplified != value,
        )
    except Exception as exc:
        return _error_response(exc, started)


def compare_mathgr(
    lhs: str,
    rhs: str = "0",
    *,
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
    timeout_seconds: float = 10.0,
) -> dict[str, Any]:
    """Simplify `lhs - rhs` and report whether the identity is proven."""
    started = time.perf_counter()
    try:
        prepared = _prepare_runtime(
            [lhs, rhs],
            context=context,
            auto_declare=auto_declare,
            declarations=declarations,
            index_dims=index_dims,
            index_sets=index_sets,
            tensors=tensors,
            symbols=symbols,
            metric=metric,
            symmetries=symmetries,
        )
        residual = mathgr.Simp(_eval_expr(lhs, prepared.namespace) - _eval_expr(rhs, prepared.namespace))
        response = _operation_response(residual, prepared, f"({lhs}) - ({rhs})", started, output, operation="compare")
        response["equal"] = residual == 0
        response["residual"] = str(residual)
        return response
    except Exception as exc:
        return _error_response(exc, started)


def inspect_mathgr(
    expr: str,
    *,
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
    timeout_seconds: float = 10.0,
) -> dict[str, Any]:
    """Return index and tensor-head diagnostics for an expression."""
    started = time.perf_counter()
    try:
        prepared = _prepare_runtime(
            [expr],
            context=context,
            auto_declare=auto_declare,
            declarations=declarations,
            index_dims=index_dims,
            index_sets=index_sets,
            tensors=tensors,
            symbols=symbols,
            metric=metric,
            symmetries=symmetries,
        )
        value = _eval_expr(expr, prepared.namespace)
        response = _operation_response(value, prepared, expr, started, output, operation="inspect")
        response["tensor_heads"] = _tensor_heads(value)
        response["pdt_count"] = _count_nodes(value, tensor_module.is_pdt)
        response["pm2_count"] = _count_nodes(value, tensor_module.is_pm2)
        return response
    except Exception as exc:
        return _error_response(exc, started)


def derivative_mathgr(
    expr: str,
    indices: list[str],
    *,
    context: str | None = None,
    simplify: bool = False,
    auto_declare: bool = True,
    declarations: dict[str, Any] | None = None,
    index_dims: dict[str, Any] | None = None,
    output: list[str] | None = None,
    timeout_seconds: float = 10.0,
) -> dict[str, Any]:
    """Apply `Pd` over one or more index expressions."""
    started = time.perf_counter()
    try:
        prepared = _prepare_runtime(
            [expr, *indices],
            context=context,
            auto_declare=auto_declare,
            declarations=declarations,
            index_dims=index_dims,
        )
        value = _eval_expr(expr, prepared.namespace)
        for index_expr in indices:
            value = mathgr.Pd(value, _eval_expr(index_expr, prepared.namespace))
        if simplify:
            value = mathgr.Simp(value)
        return _operation_response(value, prepared, expr, started, output, operation="derivative")
    except Exception as exc:
        return _error_response(exc, started)


def rewrite_mathgr(
    expr: str,
    rules: list[Any],
    *,
    repeated: bool = False,
    method: str = "ReplaceAll",
    context: str | None = None,
    auto_declare: bool = True,
    declarations: dict[str, Any] | None = None,
    index_dims: dict[str, Any] | None = None,
    output: list[str] | None = None,
    timeout_seconds: float = 10.0,
) -> dict[str, Any]:
    """Apply structured replacement rules to an expression."""
    started = time.perf_counter()
    try:
        prepared = _prepare_runtime(
            [expr, *_rule_exprs(rules)],
            context=context,
            auto_declare=auto_declare,
            declarations=declarations,
            index_dims=index_dims,
        )
        value = _eval_expr(expr, prepared.namespace)
        rule_values = _coerce_rule_items(rules, prepared.namespace)
        if method == "TReplace":
            rewritten = mathgr.TReplace(value, rule_values)
        elif repeated:
            rewritten = mathgr.ReplaceRepeated(value, rule_values)
        else:
            rewritten = mathgr.ReplaceAll(value, rule_values)
        return _operation_response(rewritten, prepared, expr, started, output, operation="rewrite")
    except Exception as exc:
        return _error_response(exc, started)


def decompose_mathgr(
    expr: str,
    *,
    preset: str = "0i",
    indices: list[str] | None = None,
    hooks: list[Any] | None = None,
    context: str | None = None,
    auto_declare: bool = True,
    declarations: dict[str, Any] | None = None,
    output: list[str] | None = None,
    timeout_seconds: float = 10.0,
) -> dict[str, Any]:
    """Apply one of the public decomposition presets."""
    started = time.perf_counter()
    try:
        prepared = _prepare_runtime(
            [expr, *_hook_exprs(hooks)],
            context=context,
            auto_declare=auto_declare,
            declarations=declarations,
        )
        value = _eval_expr(expr, prepared.namespace)
        hook_values = _coerce_rule_items(hooks, prepared.namespace)
        decomposer = _decomposer(preset)
        decomposed = decomposer(value, indices=indices, hooks=hook_values or None)
        return _operation_response(decomposed, prepared, expr, started, output, operation="decompose")
    except Exception as exc:
        return _error_response(exc, started)


def series_mathgr(
    expr: str,
    *,
    order: int,
    symbol: str = "Eps",
    mode: str = "keep",
    context: str | None = None,
    auto_declare: bool = True,
    declarations: dict[str, Any] | None = None,
    output: list[str] | None = None,
    timeout_seconds: float = 10.0,
) -> dict[str, Any]:
    """Apply perturbation-series helpers: keep, coefficient, series, or collect."""
    started = time.perf_counter()
    try:
        prepared = _prepare_runtime([expr, symbol], context=context, auto_declare=auto_declare, declarations=declarations)
        value = _eval_expr(expr, prepared.namespace)
        variable = _eval_expr(symbol, prepared.namespace)
        if mode == "coefficient":
            if variable == mathgr.Eps:
                result = mathgr.OO(order)(value)
            else:
                result = sp.expand(mathgr.TSeries(value, (variable, 0, order))).coeff(variable, order)
        elif mode == "series":
            result = mathgr.TSeries(value, (variable, 0, order))
        elif mode == "collect":
            result = mathgr.CollectEps()(value) if variable == mathgr.Eps else sp.collect(sp.expand(value), variable)
        else:
            if variable == mathgr.Eps:
                result = mathgr.SS(order)(value)
            else:
                result = mathgr.TSeries(value, (variable, 0, order))
        return _operation_response(result, prepared, expr, started, output, operation="series")
    except Exception as exc:
        return _error_response(exc, started)


def ibp_mathgr(
    expr: str,
    *,
    mode: str = "Ibp",
    target: str | None = None,
    level: int = 1,
    context: str | None = None,
    auto_declare: bool = True,
    declarations: dict[str, Any] | None = None,
    index_dims: dict[str, Any] | None = None,
    output: list[str] | None = None,
    timeout_seconds: float = 10.0,
) -> dict[str, Any]:
    """Apply public IBP helpers."""
    started = time.perf_counter()
    try:
        exprs = [expr] + ([target] if target else [])
        prepared = _prepare_runtime(
            exprs,
            context=context,
            auto_declare=auto_declare,
            declarations=declarations,
            index_dims=index_dims,
        )
        value = _eval_expr(expr, prepared.namespace)
        if mode == "Ibp2":
            result = mathgr.Ibp2(value, Level=level)
        elif mode == "IbpNB":
            result = mathgr.IbpNB(value, Level=level)
        elif mode == "IbpVariation":
            if target is None:
                raise ValueError("target is required for IbpVariation")
            result = mathgr.IbpVariation(value, _eval_expr(target, prepared.namespace))
        elif mode == "Pm2Simp":
            result = mathgr.Pm2Simp(value)
        else:
            result = mathgr.Ibp(value, Level=level)
        return _operation_response(result, prepared, expr, started, output, operation="ibp")
    except Exception as exc:
        return _error_response(exc, started)


def transform_mathgr(
    expr: str,
    *,
    transforms: list[str],
    context: str | None = None,
    auto_declare: bool = True,
    declarations: dict[str, Any] | None = None,
    index_dims: dict[str, Any] | None = None,
    output: list[str] | None = None,
    timeout_seconds: float = 10.0,
) -> dict[str, Any]:
    """Apply a small pipeline of named public MathGR transforms."""
    started = time.perf_counter()
    try:
        prepared = _prepare_runtime(
            [expr],
            context=context,
            auto_declare=auto_declare,
            declarations=declarations,
            index_dims=index_dims,
        )
        value = _eval_expr(expr, prepared.namespace)
        for name in transforms:
            value = _apply_transform(value, name)
        return _operation_response(value, prepared, expr, started, output, operation="transform")
    except Exception as exc:
        return _error_response(exc, started)


def tex_mathgr(
    expr: str,
    *,
    fragment: bool = True,
    context: str | None = None,
    auto_declare: bool = True,
    declarations: dict[str, Any] | None = None,
    index_dims: dict[str, Any] | None = None,
    output: list[str] | None = None,
    timeout_seconds: float = 10.0,
) -> dict[str, Any]:
    """Render an expression to TeX."""
    started = time.perf_counter()
    try:
        prepared = _prepare_runtime(
            [expr],
            context=context,
            auto_declare=auto_declare,
            declarations=declarations,
            index_dims=index_dims,
        )
        value = _eval_expr(expr, prepared.namespace)
        tex = _tex(value, fragment=fragment)
        response = _operation_response(value, prepared, expr, started, output, operation="tex")
        response["tex"] = tex
        return response
    except Exception as exc:
        return _error_response(exc, started)


def script_mathgr(
    expr_or_context: str | None = None,
    *,
    operation: str | None = None,
    context: str | None = None,
    declarations: dict[str, Any] | None = None,
    index_dims: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Export reproducible Python for an expression operation or context."""
    started = time.perf_counter()
    try:
        exprs = [expr_or_context] if expr_or_context else []
        prepared = _prepare_runtime(exprs, context=context, declarations=declarations, index_dims=index_dims)
        python = "\n".join(prepared.python_prefix)
        if expr_or_context:
            python = _script_for_expression(prepared, expr_or_context, operation=operation)
        return {
            "ok": True,
            "python": python,
            "auto_declarations": prepared.auto_declarations,
            "diagnostics": prepared.diagnostics,
            "time_seconds": _elapsed(started),
        }
    except Exception as exc:
        return _error_response(exc, started)


def create_mathgr_context(name: str | None = None, defaults: dict[str, Any] | None = None) -> dict[str, Any]:
    context_name = name or f"ctx_{next(_CONTEXT_COUNTER)}"
    _CONTEXTS[context_name] = _new_context_state(defaults=defaults)
    return {"ok": True, "context": context_name}


def update_mathgr_context(
    context: str,
    *,
    declarations: dict[str, Any] | None = None,
    expressions: dict[str, str] | None = None,
    metric: dict[str, Any] | None = None,
    symmetries: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    state = _require_context(context)
    if declarations:
        _merge_context_declarations(state["declarations"], declarations)
    if metric:
        state["declarations"]["metric"] = dict(metric)
    if symmetries:
        state["declarations"].setdefault("symmetries", []).extend(symmetries)
    if expressions:
        state["expressions"].update({str(key): str(value) for key, value in expressions.items()})
    return {"ok": True, "context": context, "summary": _context_summary(context, state)}


def get_mathgr_context(context: str | None = None, name: str | None = None, output: list[str] | None = None) -> dict[str, Any]:
    context_name = _ensure_context(context)
    state = _require_context(context_name)
    if name is not None:
        expressions = {name: state["expressions"][name]} if name in state["expressions"] else {}
    else:
        expressions = dict(state["expressions"])
    return {
        "ok": True,
        "context": context_name,
        "declarations": dict(state["declarations"]),
        "expressions": expressions,
    }


def clear_mathgr_context(context: str | None = None) -> dict[str, Any]:
    context_name = _context_name(context)
    existed = context_name in _CONTEXTS
    _CONTEXTS.pop(context_name, None)
    return {"ok": True, "context": context_name, "cleared": existed}


def save_mathgr_context(
    context: str | None = None,
    *,
    path: str | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    context_name = _ensure_context(context)
    state = _require_context(context_name)
    target = _context_path(context_name, path)
    if target.exists() and not overwrite:
        return _error_response(FileExistsError(f"Context file exists: {target}"), time.perf_counter())
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": CONTEXT_SCHEMA_VERSION,
        "context": context_name,
        "defaults": _json_ready(state["defaults"]),
        "declarations": _json_ready(state["declarations"]),
        "expressions": dict(state["expressions"]),
    }
    target.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    return {"ok": True, "context": context_name, "path": str(target)}


def load_mathgr_context(
    *,
    path: str | None = None,
    context: str | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    source = _context_path(_context_name(context), path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
        if payload.get("schema_version") != CONTEXT_SCHEMA_VERSION:
            raise ValueError(f"Unsupported MathGR context schema: {payload.get('schema_version')!r}")
        context_name = context or str(payload.get("context") or DEFAULT_CONTEXT_NAME)
        if context_name in _CONTEXTS and not overwrite:
            raise FileExistsError(f"Context already exists: {context_name}")
        _CONTEXTS[context_name] = {
            "defaults": dict(payload.get("defaults") or {}),
            "declarations": dict(payload.get("declarations") or {}),
            "expressions": {str(key): str(value) for key, value in (payload.get("expressions") or {}).items()},
        }
    except Exception as exc:
        return _error_response(exc, time.perf_counter())
    return {"ok": True, "context": context_name, "path": str(source), "summary": _context_summary(context_name, _CONTEXTS[context_name])}


def _prepare_runtime(
    exprs: list[str],
    *,
    context: str | None = None,
    auto_declare: bool = True,
    declarations: dict[str, Any] | None = None,
    index_dims: dict[str, Any] | None = None,
    index_sets: dict[str, str] | None = None,
    tensors: list[str] | None = None,
    symbols: list[str] | None = None,
    metric: dict[str, Any] | None = None,
    symmetries: list[dict[str, Any]] | None = None,
) -> _Prepared:
    context_name = _context_name(context)
    use_context = context is not None or context_name in _CONTEXTS
    context_state = _CONTEXTS.get(context_name) if use_context else None
    if context and context_state is None:
        raise ValueError(f"Unknown MathGR context: {context_name}")

    merged = _merged_declarations(
        context_state.get("declarations") if context_state else None,
        declarations,
        index_dims=index_dims,
        index_sets=index_sets,
        tensors=tensors,
        symbols=symbols,
        metric=metric,
        symmetries=symmetries,
    )
    context_exprs = context_state.get("expressions", {}) if context_state else {}
    all_exprs = list(context_exprs.values()) + [expr for expr in exprs if expr]
    trees = [_parse_safe_expr(expr) for expr in all_exprs]

    known_names = _known_names()
    context_names = set(context_exprs)
    scanner = _ExpressionScanner(known_names, context_names)
    for tree in trees:
        scanner.visit(tree)

    index_pairs = _declared_index_pairs(merged)
    if auto_declare:
        for name in scanner.index_names:
            pair = _index_pair_for_name(name)
            if pair and pair[0] not in _BUILTIN_INDEX_FAMILIES and pair[1] not in _BUILTIN_INDEX_FAMILIES:
                index_pairs.setdefault(_pair_key(pair), {"up": pair[0], "down": pair[1]})

    explicit_tensors = set(merged.get("tensors", ()))
    tensor_names = set(explicit_tensors)
    if auto_declare:
        tensor_names.update(scanner.tensor_names)

    symbol_names = set(merged.get("symbols", ()))
    if auto_declare:
        symbol_names.update(scanner.bare_names)

    namespace: dict[str, Any] = _base_namespace()
    python_prefix = ["from mathgr import *", "import sympy as sp"]
    diagnostics: list[str] = []
    auto_indices = []
    auto_symbols = set(symbol_names)

    for spec in sorted(index_pairs.values(), key=lambda item: item["up"]):
        up_name = str(spec["up"])
        down_name = str(spec["down"])
        dim_value = _dimension_for_pair(spec, merged)
        index_set_name = _index_set_for_pair(spec, merged)
        dim_expr, dim_code, dim_symbol = _dimension_expr(dim_value)
        if dim_symbol:
            namespace[dim_symbol] = dim_expr
            auto_symbols.add(dim_symbol)
            line = f"{dim_symbol} = sp.Symbol('{dim_symbol}')"
            if line not in python_prefix:
                python_prefix.append(line)
        index_set = getattr(mathgr, index_set_name)
        namespace[up_name], namespace[down_name] = mathgr.declare_idx(
            up_name,
            down_name,
            dim=dim_expr,
            index_set=index_set,
        )
        python_prefix.append(
            f"{up_name}, {down_name} = declare_idx('{up_name}', '{down_name}', dim={dim_code}, index_set={index_set_name})"
        )
        auto_indices.append({"up": up_name, "down": down_name, "dim": str(dim_expr)})

    for name in sorted(tensor_names):
        namespace[name] = mathgr.tensor(name)
        python_prefix.append(f"{name} = tensor('{name}')")

    for name in sorted(symbol_names):
        if name not in namespace:
            namespace[name] = sp.Symbol(name)
            auto_symbols.add(name)
            python_prefix.append(f"{name} = sp.Symbol('{name}')")

    _apply_context_declaration_heads(namespace, python_prefix, context_exprs, merged)
    _apply_metric_declaration(namespace, python_prefix, merged.get("metric"))
    _apply_symmetry_declarations(namespace, python_prefix, merged.get("symmetries", ()))

    for name, expression in context_exprs.items():
        namespace[name] = _eval_expr(expression, namespace)
        python_prefix.append(f"{name} = {expression}")

    return _Prepared(
        namespace=namespace,
        auto_declarations={
            "indices": auto_indices,
            "tensors": sorted(tensor_names),
            "symbols": sorted(auto_symbols),
        },
        python_prefix=python_prefix,
        diagnostics=diagnostics,
    )


def _base_namespace() -> dict[str, Any]:
    namespace: dict[str, Any] = {"sp": sp, "mathgr": mathgr}
    namespace.update({alias: import_module(module_name) for alias, module_name in _MODULE_ALIASES.items()})
    namespace.update(_SYMPY_NAMES)
    namespace.update({name: getattr(mathgr, name) for name in mathgr.__all__ if hasattr(mathgr, name)})
    return namespace


def _known_names() -> set[str]:
    return set(_base_namespace())


def _parse_safe_expr(expr: str) -> ast.Expression:
    tree = ast.parse(str(expr), mode="eval")
    _validate_safe_expr_tree(tree)
    return tree


def _validate_safe_expr_node(node: ast.AST) -> None:
    tree = ast.Expression(body=node)
    ast.fix_missing_locations(tree)
    _validate_safe_expr_tree(tree)


def _validate_safe_expr_tree(tree: ast.AST) -> None:
    for node in ast.walk(tree):
        if isinstance(
            node,
            (
                ast.Expression,
                ast.BinOp,
                ast.UnaryOp,
                ast.Call,
                ast.Name,
                ast.Load,
                ast.Constant,
                ast.Tuple,
                ast.List,
                ast.Dict,
                ast.keyword,
                ast.Add,
                ast.Sub,
                ast.Mult,
                ast.Div,
                ast.Pow,
                ast.Mod,
                ast.USub,
                ast.UAdd,
                ast.Attribute,
            ),
        ):
            if isinstance(node, ast.Attribute) and not _safe_attribute(node):
                raise ValueError(
                    "Only safe module attributes and expression methods are allowed in MCP expressions."
                )
            if isinstance(node, ast.Call) and _blocked_call(node):
                raise ValueError(f"Call {node.func.id!r} is not available in MCP expressions.")
            continue
        raise ValueError(f"Unsupported syntax in MCP expression: {type(node).__name__}")


def _safe_attribute(node: ast.Attribute) -> bool:
    if node.attr.startswith("_"):
        return False
    if isinstance(node.value, ast.Name) and node.value.id in _MODULE_ATTR_NAMES:
        return True
    return node.attr in _SAFE_EXPR_ATTRIBUTES and not node.attr.startswith("_")


def _blocked_call(node: ast.Call) -> bool:
    return isinstance(node.func, ast.Name) and node.func.id in _BLOCKED_CALL_NAMES


def _eval_expr(expr: str, namespace: dict[str, Any]):
    tree = _parse_safe_expr(expr)
    return eval(compile(tree, "<mathgr-mcp-expr>", "eval"), {"__builtins__": {}}, namespace)


def _parse_compute_block(source: str) -> _ComputeBlock | None:
    text = str(source)
    try:
        _parse_safe_expr(text)
        return None
    except SyntaxError:
        pass

    tree = ast.parse(text, mode="exec")
    if len(tree.body) == 1 and isinstance(tree.body[0], ast.Expr):
        raise ValueError("Unsupported syntax in MCP expression.")

    assignments: list[_BlockAssignment] = []
    expr_sources: list[str] = []
    result_source: str | None = None
    statement_sources: list[str] = []

    for stmt in tree.body:
        if isinstance(stmt, ast.Assign):
            if len(stmt.targets) != 1 or not isinstance(stmt.targets[0], ast.Name):
                raise ValueError("Only simple name assignments are supported in MCP compute blocks.")
            _validate_safe_expr_node(stmt.value)
            name = stmt.targets[0].id
            value_source = _source_segment(text, stmt.value)
            assignment = _BlockAssignment(name=name, source=value_source)
            assignments.append(assignment)
            expr_sources.append(value_source)
            statement_sources.append(f"{name} = {value_source}")
            if name == "result":
                result_source = value_source
            continue
        if isinstance(stmt, ast.Expr):
            _validate_safe_expr_node(stmt.value)
            stmt_source = _source_segment(text, stmt.value)
            if not _is_declaration_call(stmt.value):
                result_source = stmt_source
            expr_sources.append(stmt_source)
            statement_sources.append(stmt_source)
            continue
        raise ValueError(f"Unsupported syntax in MCP compute block: {type(stmt).__name__}")

    if result_source is None and assignments:
        result_source = assignments[-1].name
    return _ComputeBlock(
        assignments=assignments,
        expr_sources=expr_sources,
        result_source=result_source,
        statement_sources=statement_sources,
    )


def _source_segment(source: str, node: ast.AST) -> str:
    return (ast.get_source_segment(source, node) or ast.unparse(node)).strip()


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        return node.func.id
    return ""


def _is_declaration_call(node: ast.AST) -> bool:
    return _call_name(node) in {"UseMetric", "DeclareSym"}


def _eval_block(block: _ComputeBlock, namespace: dict[str, Any], context: str) -> Any:
    pending_expressions: list[_BlockAssignment] = []
    for statement in block.statement_sources:
        tree = ast.parse(statement, mode="exec")
        stmt = tree.body[0]
        if isinstance(stmt, ast.Assign):
            name = stmt.targets[0].id
            value_source = _source_segment(statement, stmt.value)
            namespace[name] = _eval_expr(value_source, namespace)
            pending_expressions.append(_BlockAssignment(name=name, source=value_source))
            continue
        if isinstance(stmt, ast.Expr):
            expr_source = _source_segment(statement, stmt.value)
            if _is_declaration_call(stmt.value):
                _eval_declaration_call(stmt.value, expr_source, namespace, context)
            else:
                namespace["_"] = _eval_expr(expr_source, namespace)

    for assignment in pending_expressions:
        _store_expression(context, assignment.name, assignment.source)
    if any(assignment.name == "result" for assignment in pending_expressions):
        return namespace["result"]
    if block.result_source is None:
        return namespace.get("_", "")
    if block.result_source in namespace:
        return namespace[block.result_source]
    return _eval_expr(block.result_source, namespace)


def _eval_declaration_call(node: ast.Call, source: str, namespace: dict[str, Any], context: str) -> None:
    _eval_expr(source, namespace)
    namespace["Metric"] = mathgr.Metric
    namespace["IdxOfMetric"] = mathgr.IdxOfMetric
    if _call_name(node) == "UseMetric":
        _persist_metric_call(node, namespace, context)
    elif _call_name(node) == "DeclareSym":
        _persist_symmetry_call(node, context)


def _persist_metric_call(node: ast.Call, namespace: dict[str, Any], context: str) -> None:
    if len(node.args) < 1:
        return
    head_value = _eval_expr(ast.unparse(node.args[0]), namespace)
    head = getattr(head_value, "name", None) or (
        node.args[0].id if isinstance(node.args[0], ast.Name) else str(head_value)
    )
    pair = "U/D"
    if len(node.args) >= 2:
        pair_names = _names_tuple(node.args[1])
        if len(pair_names) == 2:
            pair = _pair_key(pair_names)
    _require_context(context)["declarations"]["metric"] = {"head": str(head), "indices": pair}


def _persist_symmetry_call(node: ast.Call, context: str) -> None:
    if len(node.args) < 3:
        return
    head = node.args[0].id if isinstance(node.args[0], ast.Name) else ast.unparse(node.args[0])
    signature = list(_names_tuple(node.args[1]))
    kind, slots = _symmetry_spec(node.args[2])
    item = {"head": head, "signature": signature, "symmetry": kind, "slots": list(slots)}
    symmetries = _require_context(context)["declarations"].setdefault("symmetries", [])
    if item not in symmetries:
        symmetries.append(item)


def _names_tuple(node: ast.AST) -> tuple[str, ...]:
    if isinstance(node, (ast.Tuple, ast.List)):
        values = []
        for item in node.elts:
            if isinstance(item, ast.Name):
                values.append(item.id)
            else:
                values.append(ast.unparse(item))
        return tuple(values)
    return ()


def _symmetry_spec(node: ast.AST) -> tuple[str, tuple[Any, ...]]:
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        kind = node.func.id
        if node.args:
            slots_node = node.args[0]
            if isinstance(slots_node, (ast.Tuple, ast.List)):
                return kind, tuple(int(item.value) for item in slots_node.elts if isinstance(item, ast.Constant))
            if isinstance(slots_node, ast.Constant) and slots_node.value == "All":
                return kind, ("All",)
        return kind, (1, 2)
    return "Symmetric", (1, 2)


def _index_pair_for_name(name: str) -> tuple[str, str] | None:
    if name in _BUILTIN_INDEX_FAMILIES:
        return None
    if name.startswith("U") and len(name) >= 1:
        return name, "D" + name[1:]
    if name.startswith("D") and len(name) >= 1:
        return "U" + name[1:], name
    if name.startswith("u") and len(name) >= 1:
        return name, "d" + name[1:]
    if name.startswith("d") and len(name) >= 1:
        return "u" + name[1:], name
    return None


def _pair_key(pair: tuple[str, str] | list[str] | str) -> str:
    if isinstance(pair, str):
        return pair
    return f"{pair[0]}/{pair[1]}"


def _split_pair_key(key: str) -> tuple[str, str]:
    parts = str(key).split("/")
    if len(parts) != 2:
        raise ValueError(f"Index-family key must look like 'U/D': {key!r}")
    return parts[0], parts[1]


def _declared_index_pairs(declarations: dict[str, Any]) -> dict[str, dict[str, Any]]:
    pairs: dict[str, dict[str, Any]] = {}
    for key, dim in declarations.get("index_dims", {}).items():
        up, down = _split_pair_key(key)
        if up in _BUILTIN_INDEX_FAMILIES or down in _BUILTIN_INDEX_FAMILIES:
            continue
        pairs[key] = {"up": up, "down": down, "dim": dim}
    for item in declarations.get("indices", ()):
        up = item["up"]
        down = item["down"]
        if up in _BUILTIN_INDEX_FAMILIES or down in _BUILTIN_INDEX_FAMILIES:
            continue
        pairs[_pair_key((up, down))] = dict(item)
    return pairs


def _dimension_for_pair(spec: dict[str, Any], declarations: dict[str, Any]):
    key = _pair_key((spec["up"], spec["down"]))
    if "dim" in spec:
        return spec["dim"]
    return declarations.get("index_dims", {}).get(key, DEFAULT_DIM_SYMBOL)


def _index_set_for_pair(spec: dict[str, Any], declarations: dict[str, Any]) -> str:
    key = _pair_key((spec["up"], spec["down"]))
    return declarations.get("index_sets", {}).get(key, spec.get("index_set", "LatinIdx"))


def _dimension_expr(value: Any) -> tuple[Any, str, str | None]:
    if isinstance(value, int):
        return sp.Integer(value), str(value), None
    if isinstance(value, str):
        if value.isdigit():
            return sp.Integer(value), value, None
        if value == DEFAULT_DIM_SYMBOL:
            return sp.Symbol(value), value, value
        if value in _known_names():
            return _base_namespace()[value], value, None
        return sp.Symbol(value), value, value
    return sp.sympify(value), repr(value), None


def _merged_declarations(*declaration_dicts, **overrides) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for declarations in declaration_dicts:
        if declarations:
            _merge_context_declarations(merged, declarations)
    for key, value in overrides.items():
        if value is None:
            continue
        if key in {"index_dims", "index_sets"}:
            merged.setdefault(key, {}).update(value)
        elif key in {"tensors", "symbols", "symmetries"}:
            merged.setdefault(key, []).extend(value)
        else:
            merged[key] = value
    return merged


def _merge_context_declarations(target: dict[str, Any], incoming: dict[str, Any]) -> None:
    for key, value in incoming.items():
        if key in {"index_dims", "index_sets"}:
            target.setdefault(key, {}).update(value)
        elif key in {"tensors", "symbols", "indices", "symmetries"}:
            target.setdefault(key, []).extend(value)
        else:
            target[key] = value


def _persist_runtime_declarations(
    context: str,
    *,
    declarations: dict[str, Any] | None = None,
    index_dims: dict[str, Any] | None = None,
    index_sets: dict[str, str] | None = None,
    tensors: list[str] | None = None,
    symbols: list[str] | None = None,
    metric: dict[str, Any] | None = None,
    symmetries: list[dict[str, Any]] | None = None,
) -> None:
    state = _require_context(context)
    merged = _merged_declarations(
        declarations,
        index_dims=index_dims,
        index_sets=index_sets,
        tensors=tensors,
        symbols=symbols,
        metric=metric,
        symmetries=symmetries,
    )
    if merged:
        _merge_context_declarations(state["declarations"], merged)


def _apply_metric_declaration(namespace: dict[str, Any], python_prefix: list[str], metric: dict[str, Any] | None) -> None:
    if not metric:
        return
    head = str(metric["head"])
    pair = metric.get("indices", "U/D")
    up_name, down_name = _split_pair_key(pair)
    if head not in namespace:
        namespace[head] = mathgr.tensor(head)
        python_prefix.append(f"{head} = tensor('{head}')")
    mathgr.UseMetric(namespace[head], (namespace[up_name], namespace[down_name]))
    namespace["Metric"] = mathgr.Metric
    namespace["IdxOfMetric"] = mathgr.IdxOfMetric
    python_prefix.append(f"UseMetric({head}, ({up_name}, {down_name}))")


def _apply_symmetry_declarations(namespace: dict[str, Any], python_prefix: list[str], symmetries) -> None:
    for item in symmetries or ():
        head = str(item["head"])
        if head not in namespace:
            namespace[head] = mathgr.tensor(head)
            python_prefix.append(f"{head} = tensor('{head}')")
        signature = tuple(namespace[name] if isinstance(name, str) else name for name in item["signature"])
        kind = item.get("symmetry", "Symmetric")
        slots = tuple(item.get("slots", (1, 2)))
        symmetry = mathgr.Antisymmetric(slots) if kind == "Antisymmetric" else mathgr.Symmetric(slots)
        mathgr.DeclareSym(namespace[head], signature, symmetry)
        python_prefix.append(f"DeclareSym({head}, {tuple(item['signature'])!r}, {kind}({slots!r}))")


def _apply_context_declaration_heads(
    namespace: dict[str, Any],
    python_prefix: list[str],
    context_exprs: dict[str, str],
    declarations: dict[str, Any],
) -> None:
    heads: set[str] = set()
    metric = declarations.get("metric")
    if metric:
        heads.add(str(metric["head"]))
    for item in declarations.get("symmetries", ()):
        heads.add(str(item["head"]))
    for head in sorted(heads):
        if head in context_exprs:
            namespace[head] = _eval_expr(context_exprs[head], namespace)
            python_prefix.append(f"{head} = {context_exprs[head]}")


def _operation_response(
    value,
    prepared: _Prepared,
    expr: str,
    started: float,
    output: list[str] | None,
    *,
    operation: str | None = None,
    changed: bool | None = None,
) -> dict[str, Any]:
    requested = set(output or DEFAULT_OUTPUT)
    response: dict[str, Any] = {
        "ok": True,
        "result": str(value),
        "auto_declarations": prepared.auto_declarations,
        "time_seconds": _elapsed(started),
    }
    if changed is not None:
        response["changed"] = changed
    tensor_head_value = _is_tensor_head(value)
    if "tex" in requested:
        response["tex"] = str(value) if tensor_head_value else _tex(value, fragment=True)
    if "idx" in requested or "indices" in requested:
        response["idx"] = [] if tensor_head_value else _labels(mathgr.idx(value))
    response["free"] = [] if tensor_head_value else _labels(mathgr.free(value))
    response["dummy"] = [] if tensor_head_value else _labels(mathgr.dummy(value))
    if "python" in requested:
        response["python"] = _script_for_expression(prepared, expr, operation=operation)
    if "diagnostics" in requested:
        response["diagnostics"] = prepared.diagnostics
    return response


def _error_response(exc: Exception, started: float) -> dict[str, Any]:
    return {
        "ok": False,
        "result": "",
        "stderr": f"{type(exc).__name__}: {exc}",
        "diagnostics": [str(exc)],
        "time_seconds": _elapsed(started),
    }


def _elapsed(started: float) -> float:
    return round(time.perf_counter() - started, 6)


def _labels(values) -> list[Any]:
    return [value if isinstance(value, int) else str(value) for value in values]


def _is_tensor_head(value) -> bool:
    return type(value).__name__ == "TensorHead" and hasattr(value, "name")


def _tex(value, *, fragment: bool = True) -> str:
    import mathgr.typeset as typeset

    previous = typeset.ToTeXTemplate
    try:
        typeset.ToTeXTemplate = not fragment
        return typeset.ToTeXString(value)
    finally:
        typeset.ToTeXTemplate = previous


def _script_for_expression(prepared: _Prepared, expr: str | None, *, operation: str | None) -> str:
    lines = list(prepared.python_prefix)
    if expr:
        if "\n" in str(expr):
            lines.extend(line for line in str(expr).strip().splitlines() if line.strip())
            return "\n".join(lines)
        if operation == "simplify":
            lines.append(f"result = Simp({expr})")
        elif operation == "compare":
            lines.append(f"result = Simp({expr})")
        elif operation:
            lines.append(f"# operation: {operation}")
            lines.append(f"result = {expr}")
        else:
            lines.append(f"result = {expr}")
    return "\n".join(lines)


def _tensor_heads(expr) -> list[str]:
    heads: set[str] = set()

    def visit(node):
        head = tensor_module.tensor_head_name(node)
        if head is not None:
            heads.add(head)
        for arg in getattr(node, "args", ()):
            visit(arg)

    visit(expr)
    return sorted(heads)


def _count_nodes(expr, predicate) -> int:
    count_value = 1 if predicate(expr) else 0
    return count_value + sum(_count_nodes(arg, predicate) for arg in getattr(expr, "args", ()))


def _rule_exprs(rules: list[Any] | None) -> list[str]:
    exprs: list[str] = []
    for rule in rules or ():
        if isinstance(rule, dict):
            exprs.extend([rule["lhs"], rule["rhs"]])
        elif isinstance(rule, (list, tuple)) and len(rule) == 2:
            exprs.extend([str(rule[0]), str(rule[1])])
    return exprs


def _hook_exprs(hooks: list[Any] | None) -> list[str]:
    return _rule_exprs(hooks)


def _coerce_rule_items(rules: list[Any] | None, namespace: dict[str, Any]) -> list[Any]:
    values = []
    for rule in rules or ():
        if isinstance(rule, dict):
            values.append((_eval_expr(rule["lhs"], namespace), _eval_expr(rule["rhs"], namespace)))
        elif isinstance(rule, (list, tuple)) and len(rule) == 2:
            values.append((_eval_expr(str(rule[0]), namespace), _eval_expr(str(rule[1]), namespace)))
        else:
            values.append(rule)
    return values


def _decomposer(preset: str):
    table = {
        "0i": mathgr.Decomp0i,
        "01i": mathgr.Decomp01i,
        "0123": mathgr.Decomp0123,
        "1i": mathgr.Decomp1i,
        "123": mathgr.Decomp123,
        "Se": mathgr.DecompSe,
        "se": mathgr.DecompSe,
    }
    if preset not in table:
        raise ValueError(f"Unknown decomposition preset: {preset}")
    return table[preset]


def _apply_transform(value, name: str):
    if name == "Simp":
        return mathgr.Simp(value)
    if name == "SimpUq":
        return mathgr.SimpUq(value)
    if name == "MetricContract":
        return mathgr.MetricContract(value)
    if name == "pd2pdts":
        return mathgr.pd2pdts(value)
    if name == "pdts2pd":
        return mathgr.pdts2pd(value)
    if name == "LocalToK":
        return mathgr.LocalToK(value)
    if name == "Pm2Simp":
        return mathgr.Pm2Simp(value)
    if name == "adm.Simp":
        import mathgr.adm as adm

        return adm.Simp(value)
    if name == "frwadm.Simp":
        import mathgr.frwadm as frwadm

        return frwadm.Simp(value)
    if name == "Fourier2":
        import mathgr.frwadm as frwadm

        return frwadm.Fourier2(value)
    raise ValueError(f"Unknown transform: {name}")


def _store_expression(context: str | None, name: str, expr: str) -> None:
    if context is None:
        return
    _require_context(context)["expressions"][name] = expr


def _context_name(context: str | None = None) -> str:
    return str(context or DEFAULT_CONTEXT_NAME)


def _new_context_state(defaults: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "defaults": dict(defaults or {}),
        "declarations": {},
        "expressions": {},
    }


def _ensure_context(context: str | None = None) -> str:
    context_name = _context_name(context)
    _CONTEXTS.setdefault(context_name, _new_context_state())
    return context_name


def _require_context(context: str) -> dict[str, Any]:
    if context not in _CONTEXTS:
        raise ValueError(f"Unknown MathGR context: {context}")
    return _CONTEXTS[context]


def _context_path(context: str, path: str | None) -> Path:
    if path:
        return Path(path)
    return Path(".mathgr") / "contexts" / f"{context}.json"


def _json_ready(value):
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _context_summary(context: str, state: dict[str, Any]) -> dict[str, Any]:
    return {
        "context": context,
        "declaration_keys": sorted(state["declarations"]),
        "expressions": sorted(state["expressions"]),
    }


def _select_manual_section(content: str, section: str) -> str:
    target = section.strip().lower()
    lines = content.splitlines()
    start = None
    level = None
    for pos, line in enumerate(lines):
        if not line.startswith("#"):
            continue
        title = line.lstrip("#").strip().lower()
        if title == target:
            start = pos
            level = len(line) - len(line.lstrip("#"))
            break
    if start is None:
        return f"Section not found: {section}"
    end = len(lines)
    for pos in range(start + 1, len(lines)):
        line = lines[pos]
        if line.startswith("#") and (len(line) - len(line.lstrip("#"))) <= level:
            end = pos
            break
    return "\n".join(lines[start:end]).strip() + "\n"


def _select_query_lines(content: str, query: str) -> str:
    terms = [term.lower() for term in query.split() if term]
    if not terms:
        return content
    lines = content.splitlines()
    matches = []
    for pos, line in enumerate(lines):
        lower = line.lower()
        if all(term in lower for term in terms):
            start = max(0, pos - 2)
            end = min(len(lines), pos + 3)
            matches.extend(lines[start:end] + [""])
    return "\n".join(matches).strip() + "\n" if matches else ""


def _fallback_manual_text() -> str:
    return """# MathGR MCP

Use `mathgr_compute` first. It evaluates Python-like MathGR expressions or
multi-line notebook blocks, auto-declares common tensor/index names, and
persists assignments in the default or named context. Use `mathgr_context_get`
to list context source definitions, `mathgr_context_save/load` for JSON
persistence, and `mathgr_run_python` only when structured tools cannot express
the calculation.
"""


def _snapshot_mathgr_state() -> dict[str, Any]:
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
    sp.core.cache.clear_cache()


def _isolated_mathgr_state(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        state = _snapshot_mathgr_state()
        try:
            sp.core.cache.clear_cache()
            return func(*args, **kwargs)
        finally:
            _restore_mathgr_state(state)

    return wrapper


parse_mathgr = _isolated_mathgr_state(parse_mathgr)
compute_mathgr = _isolated_mathgr_state(compute_mathgr)
simplify_mathgr = _isolated_mathgr_state(simplify_mathgr)
compare_mathgr = _isolated_mathgr_state(compare_mathgr)
inspect_mathgr = _isolated_mathgr_state(inspect_mathgr)
derivative_mathgr = _isolated_mathgr_state(derivative_mathgr)
rewrite_mathgr = _isolated_mathgr_state(rewrite_mathgr)
decompose_mathgr = _isolated_mathgr_state(decompose_mathgr)
series_mathgr = _isolated_mathgr_state(series_mathgr)
ibp_mathgr = _isolated_mathgr_state(ibp_mathgr)
transform_mathgr = _isolated_mathgr_state(transform_mathgr)
tex_mathgr = _isolated_mathgr_state(tex_mathgr)
script_mathgr = _isolated_mathgr_state(script_mathgr)
