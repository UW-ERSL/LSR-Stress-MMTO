"""
test_run_project.py
===================
Running a Project: its settings cover every setting of run_topopt; the run record keeps
the settings as they were given, says when a setting was switched off during the run, and
stores the project so it can be rebuilt. Runs made before the examples became project
files are rebuilt from the project files.

One 4-iteration run of a coarse U-bracket, under a minute.

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


def test_run_finishes():
    assert len(runs()["project"][1]["history"]["objective"]) >= ITERS


def test_record_keeps_settings_as_given():
    """z smoothing was on at the start and switched off at iteration 2: the record says both."""
    cfg = runs()["project"][1]["config"]
    assert cfg["arguments_resolved"]["use_z_smoothing"] is True
    assert cfg["penalization_zsmoothing"]["use_z_smoothing"] is True
    assert cfg["during_run"]["z_smoothing_off_at_iter"] == SMOOTHING_OFF
    fixed = cfg["fixed_in_code"]
    assert fixed["mma_move_limit"] == 0.05 and fixed["dynamic_gamma"]["window"] == 10
    assert fixed["stress_relaxation_exponent"] == 0.5


def test_record_has_provenance():
    """The run record says which code made it."""
    import re
    prov = runs()["project"][1]["config"]["provenance"]
    for repo in ("repository", "pyto_ad"):
        commit = prov[repo]["commit"]
        assert commit == "unknown" or re.fullmatch(r"[0-9a-f]{40}", commit), (repo, commit)
    assert prov["packages"]["numpy"] == np.__version__ and prov["python"]
    with tempfile.TemporaryDirectory() as d:                        # not a git checkout
        assert run_io.git_state(d)["commit"] == "unknown"


def test_record_rebuilds_the_project():
    p, out = runs()["project"]
    rec = json.loads(json.dumps(out["config"]["arguments_resolved"]["to_problem"]))
    q = Project.from_record(rec)
    assert q.to_dict() == p.to_dict()
    from utils import render

    class FakeRun:
        config = {"arguments_resolved": {"to_problem": rec}}
    assert render._problem(FakeRun()).to_dict() == p.to_dict()


def test_old_runs_rebuild():
    """A run made under an old example name, or from a GUI file before projects, gets
    the same mesh as before."""
    from utils import render
    from pyto_project import problem_from_pyto  # type: ignore

    class Old:
        config, example, descriptor = {}, "UBracket_Stress_FF", "old run"
    assert render._problem(Old()).name == "U-bracket"
    assert render.project_name("AirplaneBracket_Stress") == "Airplane bracket"
    assert render.MIRROR_AXES[render.project_name("Beam_Liao_Pareto_FF")] == (0, 1, 2)

    class OldGui:
        example, descriptor = "custom_UBracket", "old GUI run"
        config = {"arguments_resolved": {"to_problem": {"pyto": UBRACKET_PYTO, "database": "6Materials_AF_3Attr",
                                                        "n_dof": N_DOF, "single_layer": True}},
                  "topopt_params": {"nDOFDesired": N_DOF}}
    OldGui.n_elems = int(problem_from_pyto(UBRACKET_PYTO, nDOFDesired=N_DOF, single_layer=True)[0].num_elems)
    render._MESH_CACHE.clear()
    assert int(render.mesh_for(OldGui()).num_elems) == OldGui.n_elems


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
