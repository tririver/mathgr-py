import importlib

from mathgr.decomp import DTot
import mathgr.gr as gr_module
from mathgr.gr import UseMetric
from mathgr.tensor import DE, DN, UE, UP, Dta, Simp, tensor

tensor_module = importlib.import_module("mathgr.tensor")


def test_generic_adm_definitions_and_metric_decomposition_match_upstream():
    original_metric = gr_module.Metric
    original_indices = gr_module.IdxOfMetric
    original_metric_heads = set(tensor_module._METRIC_HEADS)
    original_metrics = dict(tensor_module._METRICS)
    original_metric_index_pairs = {
        head: list(pairs) for head, pairs in tensor_module._METRIC_INDEX_PAIRS.items()
    }

    try:
        adm = importlib.import_module("mathgr.adm")

        g = tensor("g")

        assert gr_module.Metric is adm.h
        assert adm.LapseN == adm.ScriptCapitalN
        assert adm.ShiftN(DN("i")) == adm.ScriptCapitalNVector(DN("i"))
        assert adm.Sqrtg == adm.ScriptCapitalN * adm.Sqrth * adm.a**3
        assert adm.Simp(Dta(DN("i"), DN("i"))) == 3

        assert Simp(adm.DecompG2H(g(DTot("i"), DTot("i")))) == Simp(
            -adm.LapseN**2
            + adm.h(UP("a"), UP("b")) * adm.ShiftN(DN("a")) * adm.ShiftN(DN("b"))
            + adm.h(DN("i"), DN("i"))
        )
        assert adm.DecompG2H(g(DN("i"), DN("j"))) == adm.h(DN("i"), DN("j"))
        assert adm.DecompG2H(g(DE(0), DN("i"))) == adm.ShiftN(DN("i"))
        assert adm.DecompG2H(g(UE(0), UE(0))) == -adm.LapseN**-2
    finally:
        UseMetric(original_metric, original_indices)
        tensor_module._METRIC_HEADS.clear()
        tensor_module._METRIC_HEADS.update(original_metric_heads)
        tensor_module._METRICS.clear()
        tensor_module._METRICS.update(original_metrics)
        tensor_module._METRIC_INDEX_PAIRS.clear()
        tensor_module._METRIC_INDEX_PAIRS.update(original_metric_index_pairs)


def test_generic_adm_shiftn_unsupported_signatures_remain_symbolic_like_mathematica():
    original_metric = gr_module.Metric
    original_indices = gr_module.IdxOfMetric
    original_metric_heads = set(tensor_module._METRIC_HEADS)
    original_metrics = dict(tensor_module._METRICS)
    original_metric_index_pairs = {
        head: list(pairs) for head, pairs in tensor_module._METRIC_INDEX_PAIRS.items()
    }

    try:
        adm = importlib.import_module("mathgr.adm")
        shift_head = tensor("ShiftN")

        assert adm.ShiftN(UP("a")) == shift_head(UP("a"))
        assert adm.ShiftN() == shift_head()
    finally:
        UseMetric(original_metric, original_indices)
        tensor_module._METRIC_HEADS.clear()
        tensor_module._METRIC_HEADS.update(original_metric_heads)
        tensor_module._METRICS.clear()
        tensor_module._METRICS.update(original_metrics)
        tensor_module._METRIC_INDEX_PAIRS.clear()
        tensor_module._METRIC_INDEX_PAIRS.update(original_metric_index_pairs)
