import pytest
from importlib import import_module

decomp_module = import_module("mathgr.decomp")
gr_module = import_module("mathgr.gr")
tensor_module = import_module("mathgr.tensor")
typeset_module = import_module("mathgr.typeset")


@pytest.fixture(autouse=True)
def preserve_mathgr_state():
    state = {
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
    }
    try:
        yield
    finally:
        tensor_module._INDEX_TYPES.clear()
        tensor_module._INDEX_TYPES.update(state["index_types"])
        tensor_module._CONSTANTS.clear()
        tensor_module._CONSTANTS.update(state["constants"])
        tensor_module._METRICS.clear()
        tensor_module._METRICS.update(state["metrics"])
        tensor_module._METRIC_HEADS.clear()
        tensor_module._METRIC_HEADS.update(state["metric_heads"])
        tensor_module._METRIC_INDEX_PAIRS.clear()
        tensor_module._METRIC_INDEX_PAIRS.update(
            {key: list(value) for key, value in state["metric_index_pairs"].items()}
        )
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
