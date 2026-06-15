import importlib

import mathgr

tensor_module = importlib.import_module("mathgr.tensor")


def test_package_root_exports_every_advertised_public_name():
    missing = [name for name in mathgr.__all__ if not hasattr(mathgr, name)]

    assert missing == []


def test_package_root_exposes_tensor_simpselect_hook_like_upstream_public_state():
    assert mathgr.SimpSelect is tensor_module.SimpSelect
