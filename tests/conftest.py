import pytest

from mathgr.state import restore_state, snapshot_state


@pytest.fixture(autouse=True)
def preserve_mathgr_state():
    state = snapshot_state()
    try:
        yield
    finally:
        restore_state(state)
