"""
test_layout.py
==============
Every module of the package lsr_mmto and of the app (gui) imports, so a file moved or renamed
without its imports is caught here, before a run or the app trips over it. Also: the old
folders are gone, and a script outside the repository folder imports lsr_mmto.

    python tests/test_layout.py        (or: python -m pytest tests)
"""
import importlib
import os
import pkgutil
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


if __name__ == "__main__":
    test_every_module_imports()
    test_old_folders_gone()
    test_script_imports_from_another_folder()
    print("all passed")
