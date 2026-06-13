import subprocess
import sys


def test_extract_directions_help():
    result = subprocess.run(
        [sys.executable, "scripts/extract_directions.py", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "experiment-config" in result.stdout


def test_extract_directions_list_strategies():
    result = subprocess.run(
        [sys.executable, "scripts/extract_directions.py", "--list-strategies"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "arditi" in result.stdout
    assert "caa" in result.stdout


def test_run_monitored_finetune_help():
    result = subprocess.run(
        [sys.executable, "scripts/run_monitored_finetune.py", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "experiment-config" in result.stdout
