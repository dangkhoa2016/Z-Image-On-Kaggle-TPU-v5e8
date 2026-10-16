"""CPU-only tests for verbose subprocess logging and fail-closed behavior."""
import ast
import json
import subprocess
import sys
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[1] / "notebooks/kaggle-tpu-v5e8-production.ipynb"


def _load_runner(tmp_path):
    notebook = json.loads(SOURCE.read_text())
    source = "".join(notebook["cells"][3]["source"])
    fn = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == "run_logged")
    import time
    namespace = {"LOG_DIR": tmp_path, "time": time}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "log-runner", "exec"), namespace)
    return namespace["run_logged"]


def test_success_keeps_full_log_and_compact_status(tmp_path, capsys):
    runner = _load_runner(tmp_path)
    code = "import sys;print('OK_SENTINEL');print('WARNING_DETAIL',file=sys.stderr)"
    runner([sys.executable, "-c", code], "success")
    output = capsys.readouterr().out
    assert "COMMAND_PASS success" in output
    assert "WARNING_DETAIL" not in output
    assert "WARNING_DETAIL" in (tmp_path / "success.log").read_text()
    assert "OK_SENTINEL" in (tmp_path / "success.log").read_text()


def test_failure_shows_tail_and_fails_closed(tmp_path, capsys):
    runner = _load_runner(tmp_path)
    code = "import sys;print('FAILURE_SENTINEL',file=sys.stderr);sys.exit(7)"
    try:
        runner([sys.executable, "-c", code], "failure")
    except subprocess.CalledProcessError as exc:
        assert exc.returncode == 7
    else:
        raise AssertionError("A failing subprocess must fail the notebook")
    assert "COMMAND_FAIL failure" in capsys.readouterr().out
    assert "FAILURE_SENTINEL" in (tmp_path / "failure.log").read_text()


def test_notebook_sources_and_markdown_validate():
    notebook = json.loads(SOURCE.read_text())
    assert notebook["nbformat"] == 4
    assert len([cell for cell in notebook["cells"] if cell["cell_type"] == "code"]) == 24
    assert all("outputs" not in cell for cell in notebook["cells"] if cell["cell_type"] == "markdown")
