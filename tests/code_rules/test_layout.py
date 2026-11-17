"""
test_layout.py
==============
Every module of the package lsr_mmto and of the app (gui) imports, so a file moved or renamed
without its imports is caught here, before a run or the app trips over it. Also: the old
folders are gone, a script outside the repository folder imports lsr_mmto, and PyTO-AD's settings that
every run in the program shares (its SIMP exponents, its p-norm exponent) are named only inside a physics'
own folder, while no code reads the mesh's shared dof table (each solver keeps its own).

    python tests/code_rules/test_layout.py        (or: python -m pytest tests)
"""
import glob
import importlib
import os
import re
import pkgutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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


SHARED = re.compile(r"SIMP_(STR?UCTURAL|THERMAL)|_PNORM_EXPONENT|pNorm_exponent")      # PyTO-AD's run-wide settings
SHARED_TABLE = re.compile(r"\.edofMat\b(?!\s*=[^=])")                 # mesh.edofMat read, not set


def shared_settings_used(path, text):
    """[path:line] of each line of code (not a comment) that names one of PyTO-AD's run-wide settings outside
    a physics' own folder, or reads the mesh's shared dof table anywhere."""
    parts = os.path.relpath(path, ROOT).split(os.sep)
    in_physics = len(parts) > 3 and parts[:2] == ["lsr_mmto", "physics"]     # a physics' own folder, not the list
    out = []
    for i, line in enumerate(text.splitlines(), 1):
        code = line.split("#")[0]
        if (SHARED.search(code) and not in_physics) or SHARED_TABLE.search(code):
            out.append(f"{os.path.relpath(path, ROOT)}:{i}")
    return out


def test_shared_settings_stay_in_the_physics_folders():
    found = []
    for folder in ("lsr_mmto", "gui"):
        for path in sorted(glob.glob(os.path.join(ROOT, folder, "**", "*.py"), recursive=True)):
            with open(path, encoding="utf-8") as fh:
                found += shared_settings_used(path, fh.read())
    assert not found, found
    core = os.path.join(ROOT, "lsr_mmto", "optimize", "a.py")              # the check catches each kind
    assert shared_settings_used(os.path.join(ROOT, "lsr_mmto", "physics", "a.py"), "get_pNorm_exponent()")
    physics = os.path.join(ROOT, "lsr_mmto", "physics", "x", "a.py")
    assert shared_settings_used(core, "ad_mm._PNORM_EXPONENT = 6")
    assert not shared_settings_used(physics, "x._SIMP_STRUCTURAL_PENALTY")
    assert shared_settings_used(physics, "edof = mesh.edofMat[0]") and not shared_settings_used(core, "mesh.edofMat = t")
