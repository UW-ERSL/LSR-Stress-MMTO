"""
test_layout.py
==============
Every module of the package lsr_mmto and of the app (gui) imports, so a file moved or renamed
without its imports is caught here, before a run or the app trips over it. Also: the old
folders are gone, a script outside the repository folder imports lsr_mmto, and a run saved
before the move (tests/data/old_layout_run, its part under 0-Common/Models) still opens and
draws on its part.

    python tests/test_layout.py        (or: python -m pytest tests)
"""
import importlib
import json
import os
import pkgutil
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import lsr_mmto  # noqa: E402


def modules(package):
    pkg = importlib.import_module(package)
    yield package
    for info in pkgutil.walk_packages(pkg.__path__, package + "."):
        yield info.name


def test_every_module_imports():
    failed = {}
    for package in ("lsr_mmto", "gui"):
        for name in modules(package):
            try:
                importlib.import_module(name)
            except Exception as e:                                          # noqa: BLE001
                failed[name] = f"{type(e).__name__}: {e}"
    assert not failed, failed


def test_old_folders_gone():
    for old in ("0-Common", "1-PureStructural", "utils", "comparison"):
        assert not os.path.exists(os.path.join(ROOT, old)), f"{old} should be gone: its files live in lsr_mmto"


def test_script_imports_from_another_folder():
    """A script anywhere imports lsr_mmto: installed (pip install -e .), or with the repository on PYTHONPATH."""
    env = dict(os.environ)
    env["PYTHONPATH"] = ROOT + os.pathsep + env.get("PYTHONPATH", "")
    with tempfile.TemporaryDirectory() as d:
        done = subprocess.run([sys.executable, "-c", "from lsr_mmto import Project, run, face, box; print('ok')"],
                              cwd=d, env=env, capture_output=True, text=True)
    assert done.returncode == 0 and done.stdout.strip() == "ok", done.stderr


def test_run_saved_before_the_move_opens():
    """The record names the repository folder as it was then (<REPO> in the stored copy) and the
    part under 0-Common/Models; opened now, the part is found in data/models."""
    from lsr_mmto import Project
    from lsr_mmto.results import records

    def put(o):
        if isinstance(o, dict):
            return {k: put(v) for k, v in o.items()}
        if isinstance(o, list):
            return [put(v) for v in o]
        return o.replace("<REPO>", ROOT) if isinstance(o, str) else o

    with tempfile.TemporaryDirectory() as d:
        stored = os.path.join(ROOT, "tests", "data", "old_layout_run")
        shutil.copy(os.path.join(stored, "run_data.npz"), d)
        for f in ("run_data.config.json", "U-bracket.mmto"):
            with open(os.path.join(stored, f)) as fh, open(os.path.join(d, f), "w") as out:
                json.dump(put(json.load(fh)), out)
        run = records.load(os.path.join(d, "run_data"))
        p = Project.from_record(run.config["arguments_resolved"]["to_problem"])
        assert os.path.isfile(p.stl_path()), p.stl_path()
        assert int(p.build(verbose=False)[0].num_elems) == len(run.x_cont)
        assert os.path.isfile(Project.load(os.path.join(d, "U-bracket.mmto")).stl_path())


if __name__ == "__main__":
    test_every_module_imports()
    test_old_folders_gone()
    test_script_imports_from_another_folder()
    test_run_saved_before_the_move_opens()
    print("all passed")
