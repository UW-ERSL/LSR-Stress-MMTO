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
        captured["display"] = (kw["material_colors"], kw["material_labels"])

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


def test_progress_and_stop():
    """progress() hears every iteration; should_stop() ends the run there, and the design reached so far
    is still post-processed (the GUI's Run and Stop)."""
    p = ubracket_project()
    p.settings["common"]["maxIterations"] = 20
    seen = []
    with tempfile.TemporaryDirectory() as d:
        args = run_arguments(p, save_dir=d, verbose=False)
        args.update(progress=seen.append, should_stop=lambda: len(seen) >= 3)
        captured = _run(p, args)
    assert [s["iteration"] for s in seen] == [0, 1, 2]
    s = seen[-1]
    n = len(s["x_phys"])
    assert s["z"].shape[0] == n and s["z_real"].shape[1] == s["z"].shape[1] and len(s["constraints"]) == 1
    assert captured["config"]["during_run"]["stopped_at_iter"] == 3
    assert "progress" not in json.dumps(captured["config"]["run"], default=str)    # watching is not recorded


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
    units = runs()["project"][1]["config"]["units"]
    assert units == {"system": "MKS", "length": "m", "force": "N", "stress": "Pa", "density": "kg/m^3",
                     "mass": "kg", "cost per mass": "USD/kg"}, units       # the U-bracket GUI file says MKS
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


def test_record_stores_display():
    """The run takes its colours and labels from the database; the record stores them. A
    record written before that takes them from its database's Display sheet."""
    from databases import default_colours  # type: ignore
    from utils import records
    colours, labels = runs()["project"][1]["display"]
    assert colours == default_colours(6) and labels == list("ABCDEF")     # 6Materials: no Display sheet
    assert runs()["project"][1]["config"]["materials"]["material_labels"] == labels

    n = 8
    fields = {"material_indices": np.arange(n) % 3, "x_bin": np.ones(n), "x_cont": np.ones(n),
              "von_mises": np.zeros(n), "z_elem": np.zeros((n, 2))}
    with tempfile.TemporaryDirectory() as d:
        saved = run_io.WRITE_FILES
        run_io.WRITE_FILES = True
        try:
            run_io.save_run_data(d, {}, fields, None, {}, "test", np.zeros((3, 2)), ["#010203", "#040506", "#070809"],
                                 ["m1", "m2", "m3"], 2, n, 9, material_labels=["one", "two", "three"])
        finally:
            run_io.WRITE_FILES = saved
        run = records.load(os.path.join(d, "run_data.npz"))
    assert run.material_colors == ["#010203", "#040506", "#070809"] and run.material_labels == ["one", "two", "three"]

    old = records.load(os.path.join(ROOT, "comparison", "reference", "main", "L-bracket", "run_data.npz"))
    assert old.material_colors == ["#ff0000", "#4b00f1", "#23ff24"], old.material_colors   # BM1's Display sheet
    assert old.material_labels == old.material_names == ["Mat1_Red", "Mat2_Blue", "Mate3_Green"]


def test_vae_training_guards():
    """Training never writes into the shipped folder; a network being replaced is kept as
    <name>.previous.nt; a network that reproduces the database less closely than the project's
    network_error_warning is kept and the run warns, by property, and goes on (on_network_error,
    the GUI's question, can end it); the setting moves the limit."""
    import contextlib
    import io
    import shutil
    from databases import DATABASE_DIR  # type: ignore
    p = ubracket_project()                                              # 6Materials_AF_3Attr, shipped
    shipped = os.path.join(ROOT, DATABASE_DIR, "6Materials_AF_3Attr")
    with tempfile.TemporaryDirectory() as d:
        p.settings["vae"]["use_pretrained_vae"] = False                 # the project check refuses it
        assert any("save it over the shipped 6Materials_AF_3Attr.nt" in e for e in p.validate())
        p.settings["vae"]["use_pretrained_vae"] = True                  # and so does the run, called directly
        args = run_arguments(p, save_dir=d, verbose=False)
        args.update(use_pretrained_vae=False, saveNet=os.path.join(ROOT, DATABASE_DIR, "new.nt"))
        try:
            _run(p, args)
        except ValueError as e:
            assert "would overwrite the shipped new.nt" in str(e), e
        else:
            raise AssertionError("a run trained into the shipped folder")
        assert not os.path.exists(os.path.join(ROOT, DATABASE_DIR, "new.nt"))
        p.settings["vae"]["use_pretrained_vae"] = False
        for ext in (".xlsx", ".nt"):
            shutil.copy(shipped + ext, os.path.join(d, "ggd" + ext))
        old = open(os.path.join(d, "ggd.nt"), "rb").read()
        p.database = os.path.join(d, "ggd.xlsx")
        p.settings["vae"]["numEpochs"] = 3                              # a network that has learned nothing
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            captured = _run(p, run_arguments(p, save_dir=d, verbose=False))
        assert "[WARNING] The material network reproduces the database less closely than the 5% warning " \
               "limit: " in out.getvalue() and "Young's modulus" in out.getvalue(), out.getvalue()[-2000:]
        errors = captured["config"]["network_errors_percent"]
        assert set(errors) == {"Density", "Youngs_Modulus", "Yield_Strength"} and max(errors.values()) > 5
        assert open(os.path.join(d, "ggd.previous.nt"), "rb").read() == old
        assert open(os.path.join(d, "ggd.nt"), "rb").read() != old      # the poor one is kept, as trained

        p.settings["vae"]["use_pretrained_vae"] = True                  # that poor network, read from its file
        from materialEncoder import NetworkRunCancelled  # type: ignore
        asked = []
        args = run_arguments(p, save_dir=d, verbose=False, on_network_error=lambda w: asked.append(w) and False)
        try:
            _run(p, args)
            raise AssertionError("the run went on after on_network_error said no")
        except NetworkRunCancelled:
            pass
        assert len(asked) == 1 and "Yield strength" in asked[0] and "%" in asked[0], asked

        p.settings["vae"]["network_error_warning"] = 1000.0             # the limit is the project's
        asked.clear()
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            _run(p, run_arguments(p, save_dir=d, verbose=False, on_network_error=asked.append))
        assert asked == [] and "[WARNING] The material network" not in out.getvalue()


def test_network_check_leaves_the_encoder_as_it_was():
    """Checking a network read from its file leaves its encoder giving each material's own latent point
    (not a random sample around it): the same points every time, each nearest its own material."""
    import contextlib
    import io
    import torch
    from databases import DATABASE_DIR, DATABASE_VAE, VAEParams  # type: ignore
    from materialEncoder import MaterialEncoder  # type: ignore
    vae = VAEParams()
    for k, v in DATABASE_VAE["BM1_3Mats_2Attr"].items():
        setattr(vae, k, v)
    enc = MaterialEncoder(vae)
    xlsx = os.path.join(ROOT, DATABASE_DIR, "BM1_3Mats_2Attr.xlsx")
    with contextlib.redirect_stdout(io.StringIO()):
        enc.readExcel(xlsx)
        enc.loadAutoencoderFromFile(xlsx[:-5] + ".nt")
    enc.largestEncodingErrorPercent()
    assert not enc.vaeNet.encoder.isTraining
    with torch.no_grad():
        a, b = enc.vaeNet.encoder(enc.scaledMaterialData), enc.vaeNet.encoder(enc.scaledMaterialData)
    assert torch.equal(a, b)
    assert enc.getClosestRealMaterialIndex(a).tolist() == list(range(len(a)))


def test_shipped_networks_are_usable():
    """Every shipped network reproduces its database well within the default warning limit."""
    import contextlib
    import io
    from databases import DATABASE_DIR, DATABASE_VAE, VAEParams  # type: ignore
    from materialEncoder import MaterialEncoder  # type: ignore
    limit = DEFAULT_SETTINGS["vae"]["network_error_warning"]
    for name, settings in DATABASE_VAE.items():
        vae = VAEParams()
        for k, v in settings.items():
            setattr(vae, k, v)
        enc = MaterialEncoder(vae)
        xlsx = os.path.join(ROOT, DATABASE_DIR, f"{name}.xlsx")
        with contextlib.redirect_stdout(io.StringIO()):
            enc.readExcel(xlsx)
            enc.loadAutoencoderFromFile(xlsx[:-5] + ".nt")
        assert enc.largestEncodingErrorPercent() < limit / 2, name


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



def test_training_stop():
    """The material network's training: unchanged when nothing asks it to stop (the same seed gives the
    same network, to the last bit, with or without a should_stop that says no); asked to stop, it raises
    TrainingStopped and saves no network."""
    import shutil
    import torch
    from databases import DATABASE_DIR, DATABASE_VAE, VAEParams  # type: ignore
    from materialEncoder import MaterialEncoder, TrainingStopped  # type: ignore
    with tempfile.TemporaryDirectory() as d:
        xlsx = shutil.copy(os.path.join(ROOT, DATABASE_DIR, "BM1_3Mats_2Attr.xlsx"), d)

        def trained(should_stop, name):
            vae = VAEParams()
            for k, v in DATABASE_VAE["BM1_3Mats_2Attr"].items():
                setattr(vae, k, v)
            torch.manual_seed(0)
            np.random.seed(0)
            enc = MaterialEncoder(vae)
            enc.readExcel(xlsx)
            path = os.path.join(d, name)
            enc.trainAutoencoder(400, vae.klFactor, path, vae.learningRate, 1e-9, should_stop=should_stop)
            return torch.load(path)

        plain, asked = trained(None, "plain.nt"), trained(lambda: False, "asked.nt")
        assert plain.keys() == asked.keys()
        assert all(torch.equal(plain[k], asked[k]) for k in plain)              # bit for bit
        try:
            trained(lambda: True, "stopped.nt")
            raise AssertionError("the training did not stop")
        except TrainingStopped as e:
            assert e.epoch == 0
        assert not os.path.exists(os.path.join(d, "stopped.nt"))                # nothing saved

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
