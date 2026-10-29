import json
import subprocess
import sys
from fractions import Fraction

import pytest

from .conftest import FIXTURES, ROOT


def test_cli_returns_an_inspectable_forecast():
    result = subprocess.run([
        sys.executable, "-m", "world_model", "--model", str(FIXTURES / "binary-world-v1.json"),
        "--readings", str(ROOT / "examples/readings.json"), "--tick", "2",
    ], cwd=ROOT, capture_output=True, text=True, check=True)
    payload = json.loads(result.stdout)
    assert payload["posterior"]["degraded"] == pytest.approx(float(Fraction(196631, 287932)), abs=1e-12, rel=0)
    assert payload["predictive"]["degraded"] == pytest.approx(float(Fraction(1117462, 1799575)), abs=1e-12, rel=0)
    assert len(payload["readings"]) == 3
    assert len(payload["model_hash"]) == 64
    assert result.stderr == ""


def test_cli_reports_invalid_data_without_a_partial_result(tmp_path):
    readings = tmp_path / "invalid.json"
    readings.write_text('[{"tick":0,"source_id":"cheap","value":"false"}]')
    result = subprocess.run([
        sys.executable, "-m", "world_model", "--model", str(FIXTURES / "binary-world-v1.json"),
        "--readings", str(readings), "--tick", "0",
    ], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 2
    assert result.stdout == ""
    assert "world-model:" in result.stderr
    assert "Traceback" not in result.stderr
