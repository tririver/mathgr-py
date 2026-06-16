from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from importlib import import_module
from typing import Any

import sympy as sp


@dataclass(frozen=True)
class MathGRState:
    values: dict[str, Any]


def snapshot_state() -> MathGRState:
    tensor_module = import_module("mathgr.tensor")
    decomp_module = import_module("mathgr.decomp")
    gr_module = import_module("mathgr.gr")
    typeset_module = import_module("mathgr.typeset")
    return MathGRState(
        {
            "index_types": dict(tensor_module._INDEX_TYPES),
            "constants": set(tensor_module._CONSTANTS),
            "metrics": dict(tensor_module._METRICS),
            "metric_heads": set(tensor_module._METRIC_HEADS),
            "metric_index_pairs": {key: list(value) for key, value in tensor_module._METRIC_INDEX_PAIRS.items()},
            "symmetries": {key: list(value) for key, value in tensor_module._SYMMETRIES.items()},
            "riemann_like_heads": set(tensor_module._RIEMANN_LIKE_HEADS),
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
    )


def restore_state(state: MathGRState, *, clear_sympy_cache: bool = False) -> None:
    values = state.values
    tensor_module = import_module("mathgr.tensor")
    decomp_module = import_module("mathgr.decomp")
    gr_module = import_module("mathgr.gr")
    util_module = import_module("mathgr.util")
    typeset_module = import_module("mathgr.typeset")

    tensor_module._INDEX_TYPES.clear()
    tensor_module._INDEX_TYPES.update(values["index_types"])
    tensor_module._CONSTANTS.clear()
    tensor_module._CONSTANTS.update(values["constants"])
    tensor_module._METRICS.clear()
    tensor_module._METRICS.update(values["metrics"])
    tensor_module._METRIC_HEADS.clear()
    tensor_module._METRIC_HEADS.update(values["metric_heads"])
    tensor_module._METRIC_INDEX_PAIRS.clear()
    tensor_module._METRIC_INDEX_PAIRS.update({key: list(value) for key, value in values["metric_index_pairs"].items()})
    tensor_module._SYMMETRIES.clear()
    tensor_module._SYMMETRIES.update({key: list(value) for key, value in values["symmetries"].items()})
    tensor_module._RIEMANN_LIKE_HEADS.clear()
    tensor_module._RIEMANN_LIKE_HEADS.update(values["riemann_like_heads"])
    tensor_module._UNIQ_COUNTER_VALUE = values["uniq_counter_value"]

    tensor_module.IdxList[:] = values["idx_list"]
    tensor_module.IdxUpList[:] = values["idx_up_list"]
    tensor_module.IdxDnList[:] = values["idx_dn_list"]
    tensor_module.SimpHook[:] = values["simp_hook"]
    tensor_module.SimpInto1 = values["simp_into1"]
    tensor_module.SimpSelect = values["simp_select"]
    decomp_module.DecompHook[:] = values["decomp_hook"]
    gr_module.Metric = values["metric"]
    gr_module.IdxOfMetric = values["idx_of_metric"]
    typeset_module.ToTeXHook[:] = values["tex_hook"]
    typeset_module.ToTeXTemplate = values["tex_template"]
    tensor_module._clear_canonicalization_caches()
    util_module._clear_series_caches()

    if clear_sympy_cache:
        sp.core.cache.clear_cache()


@contextmanager
def isolated_state(*, clear_sympy_cache: bool = True):
    state = snapshot_state()
    try:
        if clear_sympy_cache:
            sp.core.cache.clear_cache()
        yield
    finally:
        restore_state(state, clear_sympy_cache=clear_sympy_cache)
