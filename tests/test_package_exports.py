import importlib

import mathgr

tensor_module = importlib.import_module("mathgr.tensor")


def test_package_root_exports_every_advertised_public_name():
    missing = [name for name in mathgr.__all__ if not hasattr(mathgr, name)]

    assert missing == []


def test_package_root_exposes_tensor_simpselect_hook_like_upstream_public_state():
    assert mathgr.SimpSelect is tensor_module.SimpSelect


def test_package_root_exports_utf8_frw_symbols_without_ascii_aliases():
    frwadm = importlib.import_module("mathgr.frwadm")

    for name in ("α", "β", "ζ", "ε", "η", "η2", "η3"):
        value = getattr(mathgr, name)
        assert str(value) == name
        assert name in mathgr.__all__
        assert getattr(frwadm, name) is value

    for name in ("alpha", "beta", "zeta", "epsilon", "eta", "eta2", "eta3"):
        assert name not in mathgr.__all__
        assert not hasattr(mathgr, name)
        assert not hasattr(frwadm, name)
