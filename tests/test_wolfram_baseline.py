from pathlib import Path
import subprocess

import pytest


@pytest.mark.wolfram
def test_upstream_mathgr_suite_passes_from_wolfram():
    wolfram_script = Path("/usr/local/bin/WolframScript")
    upstream = Path("/private/tmp/MathGR/resources/test.m")

    if not wolfram_script.exists():
        pytest.skip("/usr/local/bin/WolframScript is required for MathGR parity tests")
    if not upstream.exists():
        pytest.skip("Clone tririver/MathGR to /private/tmp/MathGR before running oracle tests")

    result = subprocess.run(
        [str(wolfram_script), "-script", str(upstream)],
        cwd="/private/tmp",
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "Number of tests: 73" in result.stdout
    assert "All tests passed." in result.stdout
