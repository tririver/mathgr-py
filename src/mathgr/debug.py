from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps
from importlib import import_module
import os
import sys
import time
from typing import Any

import sympy as sp


_TENSOR_HELPERS = (
    "_apply_simp_hooks",
    "_selected_simp_terms",
    "_split_indexed_powers",
    "_contract_levicivita",
    "pdts2pd",
    "_contract_deltas",
    "_contract_metric_products",
    "_canonicalize_additive_tensor_terms",
    "_canonicalize_tensor_product",
    "_canonicalize_mul_symmetry_dummies_for_add",
    "_canonicalize_mul_symmetry_dummies",
    "_canonicalize_large_mul_symmetry_dummies",
    "_canonicalize_dummy",
    "_canonicalize_dummy_structural",
    "_canonicalize_dummy_by_renaming",
    "_canonicalize_dummy_by_tied_renaming",
    "_canonicalize_declared_symmetries",
    "_dummy_key_signatures",
    "_factor_symmetry_variants",
)

_SYMPY_HELPERS = (
    ("expand", "sp.expand"),
    ("default_sort_key", "sp.default_sort_key"),
)


@dataclass(frozen=True)
class SimplifyProfilePhase:
    name: str
    elapsed_seconds: float
    input_terms: int | None
    output_terms: int | None
    input_ops: int | None
    output_ops: int | None


@dataclass(frozen=True)
class SimplifyProfile:
    result: Any
    total_seconds: float
    phases: tuple[SimplifyProfilePhase, ...]
    counters: dict[str, int]
    timings: dict[str, float]


def simplify_profile(expr, *, hooks=(), dummy_pool=None, method=None) -> SimplifyProfile:
    """Run `Simp` once and return timing/counter data for current hot paths.

    This is a debug helper: it temporarily wraps selected MathGR/SymPy helper
    functions, restores them immediately after the call, and does not change
    simplification semantics.
    """

    tensor_module = import_module("mathgr.tensor")
    options: dict[str, Any] = {}
    if hooks:
        options["hooks"] = hooks
    if dummy_pool is not None:
        options["Dummy"] = dummy_pool
    if method is not None:
        options["Method"] = method

    input_terms = _term_count(expr)
    input_ops = _op_count(expr)
    input_phase = SimplifyProfilePhase(
        name="input",
        elapsed_seconds=0.0,
        input_terms=None,
        output_terms=input_terms,
        input_ops=None,
        output_ops=input_ops,
    )

    with _profiled_helpers(tensor_module) as recorder:
        started = time.perf_counter()
        result = tensor_module.Simp(expr, **options)
        total_seconds = time.perf_counter() - started

    total_phase = SimplifyProfilePhase(
        name="total_simp",
        elapsed_seconds=total_seconds,
        input_terms=input_terms,
        output_terms=_term_count(result),
        input_ops=input_ops,
        output_ops=_op_count(result),
    )
    profile = SimplifyProfile(
        result=result,
        total_seconds=total_seconds,
        phases=(input_phase, total_phase),
        counters=dict(recorder.counters),
        timings=dict(recorder.timings),
    )
    if os.environ.get("MATHGR_TRACE_SIMPLIFY"):
        _print_trace(profile)
    return profile


class _ProfileRecorder:
    def __init__(self, labels: tuple[str, ...]):
        self.counters = {label: 0 for label in labels}
        self.timings = {label: 0.0 for label in labels}

    def wrap(self, label: str, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            started = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                self.counters[label] = self.counters.get(label, 0) + 1
                self.timings[label] = self.timings.get(label, 0.0) + (time.perf_counter() - started)

        return wrapper


@contextmanager
def _profiled_helpers(tensor_module):
    labels = _TENSOR_HELPERS + tuple(label for _name, label in _SYMPY_HELPERS)
    recorder = _ProfileRecorder(labels)
    originals = []
    try:
        for name in _TENSOR_HELPERS:
            if hasattr(tensor_module, name):
                original = getattr(tensor_module, name)
                originals.append((tensor_module, name, original))
                setattr(tensor_module, name, recorder.wrap(name, original))
        for name, label in _SYMPY_HELPERS:
            original = getattr(sp, name)
            originals.append((sp, name, original))
            setattr(sp, name, recorder.wrap(label, original))
        yield recorder
    finally:
        for module, name, original in reversed(originals):
            setattr(module, name, original)


def _term_count(expr) -> int:
    expr = sp.sympify(expr)
    return len(sp.Add.make_args(expr)) if isinstance(expr, sp.Add) else 1


def _op_count(expr) -> int:
    expr = sp.sympify(expr)
    return int(sp.count_ops(expr, visual=False))


def _print_trace(profile: SimplifyProfile) -> None:
    print("MATHGR simplify profile", file=sys.stderr)
    for phase in profile.phases:
        print(
            f"phase {phase.name} elapsed={phase.elapsed_seconds:.6f}s "
            f"terms={phase.output_terms} ops={phase.output_ops}",
            file=sys.stderr,
        )
    hot = sorted(profile.timings.items(), key=lambda item: item[1], reverse=True)
    for label, elapsed in hot:
        calls = profile.counters.get(label, 0)
        if calls:
            print(f"counter {label} calls={calls} elapsed={elapsed:.6f}s", file=sys.stderr)
