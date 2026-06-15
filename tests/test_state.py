import importlib

import mathgr
from mathgr.gr import UseMetric
from mathgr.tensor import SimpHook, Uq, declare_idx, tensor


def test_isolated_state_restores_uniq_counter():
    from mathgr.state import isolated_state

    with isolated_state(clear_sympy_cache=False):
        first = Uq(1)

    with isolated_state(clear_sympy_cache=False):
        second = Uq(1)

    assert second == first


def test_isolated_state_restores_metric_registry_and_hooks():
    from mathgr.state import isolated_state

    tensor_module = importlib.import_module("mathgr.tensor")
    decomp_module = importlib.import_module("mathgr.decomp")

    original_metric_heads = set(tensor_module._METRIC_HEADS)
    original_simp_hooks = list(tensor_module.SimpHook)
    original_decomp_hooks = list(decomp_module.DecompHook)

    with isolated_state(clear_sympy_cache=False):
        up, down = declare_idx("StateU", "StateD", dim=4, index_set=mathgr.LatinIdx)
        metric = tensor("gStateIsolation")
        UseMetric(metric, (up, down))
        SimpHook.append((metric(up("a"), up("b")), 0))
        decomp_module.DecompHook.append((metric(mathgr.D1("a"), mathgr.U1("a")), 0))

    assert set(tensor_module._METRIC_HEADS) == original_metric_heads
    assert tensor_module.SimpHook == original_simp_hooks
    assert decomp_module.DecompHook == original_decomp_hooks


def test_mcp_modules_use_state_backend():
    import mathgr.mcp_server as mcp_server
    import mathgr.mcp_structured as mcp_structured
    tensor_module = importlib.import_module("mathgr.tensor")
    from mathgr.state import MathGRState

    assert not hasattr(mcp_server, "_snapshot_mathgr_state")
    assert not hasattr(mcp_server, "_restore_mathgr_state")
    assert not hasattr(mcp_structured, "_snapshot_mathgr_state")
    assert not hasattr(mcp_structured, "_restore_mathgr_state")
    assert isinstance(tensor_module._snapshot_tensor_registry_state(), MathGRState)
