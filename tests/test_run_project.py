"""
test_run_project.py
===================
Running a Project: its settings cover every setting of run_topopt; the same problem run
as a CustomProblem and as a Project gives the same numbers; the run record keeps the
settings as they were given, says when a setting was switched off during the run, and
stores the project so it can be rebuilt.

Two 4-iteration runs of a coarse U-bracket, about a minute in all.

    python tests/test_run_project.py        (or: python -m pytest tests)
"""
import inspect
import json
import os
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from utils import project_paths  # noqa: E402

project_paths.bootstrap()

import MainPureStructural as M  # noqa: E402  # type: ignore
import run_io  # noqa: E402  # type: ignore
from ExamplesPureStructural import CustomProblem  # noqa: E402  # type: ignore
from project import (DEFAULT_SETTINGS, NOT_STORED, Project, run_arguments)  # noqa: E402  # type: ignore

UBRACKET_PYTO = os.path.join(ROOT, "0-Common", "Models", "UBracket", "UBracket.pyto")
N_DOF, ITERS, SMOOTHING_OFF = 3000, 4, 2


def ubracket_project():
    """The tutorial's custom U-bracket, as a Project."""
    p = Project.load(UBRACKET_PYTO)
    p.n_dof, p.single_layer = N_DOF, True
    p.database = "6Materials_AF_3Attr"
    p.objective("FailureFactorPNorm()")
    p.constraint("Mass()", "<=", 0.0035)
    p.settings["common"].update(maxIterations=ITERS, relative_filter_radius=3.5, pnorm_exponent=16)
    p.settings["advanced"]["z_smoothing_disable_after_iter"] = SMOOTHING_OFF
    return p


_RUNS = {}


def _run(to_problem, args):
    """run_topopt with the history and the run record captured; nothing written to disk."""
    captured = {}
    real_init, real_post = M.init_run_history, M.run_postprocessing

    def init():
        captured["history"] = real_init()
        return captured["history"]

    def post(**kw):
        captured["config"] = kw["run_config"]

    M.init_run_history, M.run_postprocessing = init, post
    saved = run_io.WRITE_FILES
    run_io.WRITE_FILES = False
    try:
        M.run_topopt(**dict(args, to_problem=to_problem))
    finally:
        M.init_run_history, M.run_postprocessing = real_init, real_post
        run_io.WRITE_FILES = saved
    return captured


def runs():
    if not _RUNS:
        p = ubracket_project()
        with tempfile.TemporaryDirectory() as d:
            args = run_arguments(p, save_dir=d, verbose=False)
            _RUNS["project"] = (p, _run(p, args))
            custom = CustomProblem(pyto=UBRACKET_PYTO, database="6Materials_AF_3Attr", objective="FAILURE_FACTOR",
                                   constraints=[("MASS", 0.0035)], n_dof=N_DOF, single_layer=True,
                                   relative_filter_radius=3.5, pnorm_exponent=16)
            _RUNS["custom"] = (custom, _run(custom, args))
    return _RUNS


def test_settings_cover_run_topopt():
    """Every run_topopt argument is a stored setting, or explicitly not stored. Fails when one is added."""
    stored = set(DEFAULT_SETTINGS["common"]) | set(DEFAULT_SETTINGS["advanced"]) | set(DEFAULT_SETTINGS["vae"])
    covered = stored | set(NOT_STORED) | {"n_dof_desired"}
    arguments = set(inspect.signature(M.run_topopt).parameters)
    assert arguments - covered == set(), f"not in the project: {sorted(arguments - covered)}"
    assert covered - arguments == set(), f"not run_topopt arguments: {sorted(covered - arguments)}"


def test_unknown_and_missing_settings():
    p = ubracket_project()
    p.settings["advanced"]["gama_max"] = 5.0                       # a typo
    p.settings["common"]["beta_schedule_scheme"] = "FAST"
    errors = "\n".join(p.validate())
    assert "unknown setting 'gama_max'" in errors and "'FAST' is not one of" in errors, errors
    q = Project.load(UBRACKET_PYTO)                                 # a GUI file: no settings at all
    q.database = "6Materials_AF_3Attr"
    filled = q.complete_settings()
    assert "advanced.gamma_max" in filled and "vae.latentDim" in filled
    assert q.settings["vae"]["learningRate"] == 1e-4                # the database's known value


def test_same_run_two_ways():
    """CustomProblem and Project on the same problem: the same histories.

    Not bit for bit: the same problem run twice in one process occasionally differs in
    the last digits (about 1e-11 here, from the multithreaded solver), so the
    comparison allows 1e-9.
    """
    r = runs()
    hp, hc = r["project"][1]["history"], r["custom"][1]["history"]
    assert hp.keys() == hc.keys()
    for k in hp:
        a, b = np.asarray(hp[k], dtype=float), np.asarray(hc[k], dtype=float)
        assert a.shape == b.shape and np.allclose(a, b, rtol=1e-9, atol=0, equal_nan=True), k
    assert len(hp["objective"]) >= ITERS


def test_record_keeps_settings_as_given():
    """z smoothing was on at the start and switched off at iteration 2: the record says both."""
    cfg = runs()["project"][1]["config"]
    assert cfg["arguments_resolved"]["use_z_smoothing"] is True
    assert cfg["penalization_zsmoothing"]["use_z_smoothing"] is True
    assert cfg["during_run"]["z_smoothing_off_at_iter"] == SMOOTHING_OFF
    fixed = cfg["fixed_in_code"]
    assert fixed["mma_move_limit"] == 0.05 and fixed["dynamic_gamma"]["window"] == 10
    assert fixed["stress_relaxation_exponent"] == 0.5


def test_record_rebuilds_the_project():
    p, out = runs()["project"]
    rec = json.loads(json.dumps(out["config"]["arguments_resolved"]["to_problem"]))
    q = Project.from_record(rec)
    assert q.to_dict() == p.to_dict()
    from utils import render

    class FakeRun:
        config = {"arguments_resolved": {"to_problem": rec}}
    assert render._custom_problem(FakeRun()).to_dict() == p.to_dict()


if __name__ == "__main__":
    failed = 0
    for name, fn in [(k, v) for k, v in list(globals().items()) if k.startswith("test_")]:
        try:
            fn()
            print(f"ok   {name}")
        except Exception as e:                                     # noqa: BLE001
            failed += 1
            print(f"FAIL {name}: {type(e).__name__}: {e}")
    print("all passed" if not failed else f"{failed} failed")
    sys.exit(1 if failed else 0)
