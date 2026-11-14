"""
test_thermal_physics.py
=======================
The thermal physics (lsr_mmto/physics/thermal/) keeps working: it is registered as its line in
available_physics.MODULES would register it, its solver gives the right answer on a bar, and short
U-bracket runs go through with it and its example database, with no core file changed. Copy this file
for a physics of your own.

    python tests/test_thermal_physics.py        (or: python -m pytest tests)
"""
import contextlib
import os
import sys
import tempfile

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from lsr_mmto.physics import available_physics  # noqa: E402
from lsr_mmto.physics.thermal import thermal_module as THERMAL  # noqa: E402

available_physics._checked(THERMAL)                 # the defaults of the entries it leaves out, as registering gives

UBRACKET = os.path.join(ROOT, "data", "models", "UBracket")
BAR = os.path.join(ROOT, "data", "models", "RectangularBeam")
EXAMPLE_DATABASE = os.path.join(ROOT, "lsr_mmto", "physics", "thermal", "example_materials.xlsx")
EXAMPLE_VAE = dict(latentDim=2, vae_hiddenDim=50, learningRate=5e-4, klFactor=1e-4, maxAttributeErrorPercent=0.1,
                   numEpochs=100000)                # the settings its network (example_materials.nt) was trained with


@contextlib.contextmanager
def registered(module):
    """The core's lists with module added, as its line in available_physics.MODULES adds it when they are
    made (its quantities, penalties, kinds, settings, material properties and unit kinds), and taken out
    again afterwards. A module already in MODULES is left as it is: nothing added, nothing taken out. The
    kinds of support and load are made again from MODULES each way (boundary_conditions.bc_kinds_of, as at
    start-up), since a kind may be shared: one tagged with several physics stays when one of them goes."""
    if module in available_physics.MODULES:
        yield
        return
    from lsr_mmto.design import latent_penalties
    from lsr_mmto.formulation import quantities
    from lsr_mmto.materials import databases
    from lsr_mmto.problem import boundary_conditions, settings
    available_physics._checked(module)
    added_q = [quantities.Quantity(physics=[module.NAME], **q) for q in module.QUANTITIES]
    added_p = [latent_penalties.Penalty(physics=[module.NAME], **p) for p in module.PENALTIES]
    added_s = [settings.Setting(physics=module.NAME, **a) for a in module.SETTINGS]
    added_u = [k for k in module.UNIT_KINDS if k not in databases.UNITS]
    databases._add_unit_kinds(module)                     # first: an attribute's kind must be a unit kind
    added_a = {k: a for k, a in databases.attributes_of([module]).items() if k not in databases.ATTRIBUTES}
    available_physics.MODULES.append(module)
    available_physics.PHYSICS[module.NAME] = module
    quantities.QUANTITIES.extend(added_q)
    latent_penalties.PENALTIES.extend(added_p)
    _remake_kinds()
    settings.SETTINGS.extend(added_s)
    settings._make_views()
    databases.ATTRIBUTES.update(added_a)
    databases.SHOWN.update({k: a.shown for k, a in added_a.items()})
    aliases = {n.lower(): a.name for a in added_a.values() for n in (a.name,) + a.aliases}
    databases._BY_ALIAS.update(aliases)
    try:
        yield
    finally:
        available_physics.MODULES.remove(module)
        del available_physics.PHYSICS[module.NAME]
        for q in added_q:
            quantities.QUANTITIES.remove(q)
        for p in added_p:
            latent_penalties.PENALTIES.remove(p)
        _remake_kinds()
        for s in added_s:
            settings.SETTINGS.remove(s)
        settings._make_views()
        for k in added_a:
            del databases.ATTRIBUTES[k], databases.SHOWN[k]
        for n in aliases:
            del databases._BY_ALIAS[n]
        for k in added_u:
            del databases.UNITS[k]
            for table in databases.UNIT_SYSTEMS.values():
                del table[k]


def _remake_kinds():
    """The core's kinds of support and load made again from available_physics.MODULES, in place (the core
    reads the same dict), as start-up makes them. Made in full before the old ones are cleared, so an error
    leaves them as they were."""
    from lsr_mmto.problem import boundary_conditions
    kinds = boundary_conditions.bc_kinds_of(available_physics.MODULES)
    boundary_conditions.BC_KINDS.clear()
    boundary_conditions.BC_KINDS.update(kinds)


@contextlib.contextmanager
def added(entries, *new, then=None):
    """The core's list entries with new added at its end, in place (the core reads the same list), and taken
    out again afterwards; then(), when given, is called after each change (settings._make_views)."""
    entries.extend(new)
    if then is not None:
        then()
    try:
        yield
    finally:
        for e in new:
            entries.remove(e)
        if then is not None:
            then()


def _project(name, stl, base_dir, n_dof):
    from lsr_mmto.problem.project import Project
    p = Project(name, stl=stl, n_dof=n_dof, single_layer=True, base_dir=base_dir)
    p.physics = "thermal"
    p.database = EXAMPLE_DATABASE
    p.settings["vae"].update(EXAMPLE_VAE)
    p.objective("ThermalCompliance()")
    p.constraint("VolumeFraction()", "<=", 0.5)
    return p


def _held_and_heated(name, n_dof=3000):
    """A U-bracket held at 0 on its bottom face with 1 of heat put in at its top right corner."""
    from lsr_mmto.problem import selections as S
    p = _project(name, "UBracket.STL", UBRACKET, n_dof)
    p.add_selection("bottom", S.face("ymin"))
    p.add_selection("top_right", S.box([0.75, 0.9, -1], [1.0, 1.0, 1]))
    p.bcs += [{"kind": "fixed_temperature", "on": "bottom", "value": 0.0},
              {"kind": "heat_source", "on": "top_right", "total": 1.0}]
    return p


def test_a_project_sees_only_its_own_physics_settings():
    """The Settings panel shows the settings every project has and those of the project's own physics, each
    group once. Thermal has none of its own, so one is given to it here, added to the core's settings as
    registering adds a front page's."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from gui import settings_panel
    from lsr_mmto.problem import settings
    test_setting = settings.Setting(physics="thermal", section="advanced", key="thermal_test_scale", default=1.0,
                                    label="Test scale", help="A setting for the tests only.", group="Material model")
    with registered(THERMAL), added(settings.SETTINGS, test_setting, then=settings._make_views):
        structural, thermal = (set(settings_panel.all_setting_keys(name)) for name in ("structural", "thermal"))
        assert ("common", "pnorm_exponent") in structural and ("common", "pnorm_exponent") not in thermal
        assert ("advanced", "thermal_test_scale") in thermal
        assert ("advanced", "thermal_test_scale") not in structural
        assert ("common", "maxIterations") in structural & thermal                  # every project has it
        for name in ("structural", "thermal", None):
            groups = [(tab, group) for tab, group, _ in settings_panel.setting_groups(name)]
            assert len(groups) == len(set(groups)), groups                        # each group once


def test_thermal_solver_gives_a_straight_line_along_a_bar():
    """T held at 0 at one end of a bar and 1 at the other, solid everywhere, one conductivity: T rises in a
    straight line from end to end, at every node. Trilinear elements give this exactly."""
    from lsr_mmto.problem import selections as S
    from lsr_mmto.pyto_ad_imports import linear_solvers
    with registered(THERMAL):
        p = _project("bar", "RectangularBeam.STL", BAR, 2000)
        p.add_selection("left", S.face("xmin"))
        p.add_selection("right", S.face("xmax"))
        p.bcs += [{"kind": "fixed_temperature", "on": "left", "value": 0.0},
                  {"kind": "fixed_temperature", "on": "right", "value": 1.0}]    # held only: no source needed
        p.complete_settings()
        assert p.validate() == []
        mesh, mat_prop, bcs, _, _, _ = p.build(verbose=False)
        fe = THERMAL.make_solver(mesh, mat_prop, bcs, linear_solvers.Solvers.PARDISO, None, None)
        fe.set_conductivity(torch.full((mesh.num_elems,), 3.0, dtype=torch.float64))
        T = fe.solve(torch.ones(mesh.num_elems, dtype=torch.float64), None).detach().numpy()
        x = np.asarray(mesh.node_xyz, dtype=float)[:, 0]
        expected = (x - x.min()) / (x.max() - x.min())
        assert np.max(np.abs(T - expected)) < 1e-8


def test_thermal_physics_runs_with_no_core_file_changed():
    from lsr_mmto.problem import selections as S
    from lsr_mmto.problem.project import run
    from lsr_mmto.results import load_results
    with registered(THERMAL):
        p = _held_and_heated("thermal")
        p.settings["common"]["maxIterations"] = 2
        p.complete_settings()
        assert p.validate() == []

        q = _project("thermal", "UBracket.STL", UBRACKET, 3000)
        q.add_selection("bottom", S.face("ymin"))
        q.add_bc("fixed", "bottom", dofs="xyz")                         # a structural kind: not thermal's
        assert any("not a kind of thermal support or load" in e for e in q.validate())

        mesh, _, bcs, _, _, _ = p.build(verbose=False)
        assert list(bcs) == ["T"] and len(bcs["T"].fixed_dofs) > 0
        assert np.isclose(bcs["T"].force.sum(), 1.0)                 # the source, shared by its nodes

        with tempfile.TemporaryDirectory() as d:
            run(p, save_dir=d, verbose=False)
            r = load_results.load(os.path.join(d, "run_data.npz"))
        assert list(r.results) == ["temperature"] and r.results["temperature"].shape == (len(mesh.node_xyz),)
        assert r.result_info["temperature"]["view"] == "Temperature"
        assert r.result_info["temperature"]["where"] == "node"
        assert len(r.history["objective"]) >= 2                     # MMA evaluates the start too
        assert np.all(np.isfinite(r.history["objective"])) and r.history["objective"][0] > 0
        assert len(r.history["max_temperature"]) >= 2
        assert "max_failure_factor" not in r.history                 # structural's keys are structural's


def test_a_thermal_summary_shows_its_own_row_and_units():
    """A run in mmKS: its Summary has thermal's own row, the max temperature in K, and its Units row the units
    of the thermal kinds; nothing of structural's (no failure factor, no force or stress)."""
    from lsr_mmto.problem.project import run
    from lsr_mmto.results import load_results, tables
    with registered(THERMAL):
        p = _held_and_heated("units")
        p.units = "mmKS"
        p.settings["common"]["maxIterations"] = 2
        with tempfile.TemporaryDirectory() as d:
            run(p, save_dir=d, verbose=False)
            r = load_results.load(os.path.join(d, "run_data.npz"))
            rows = tables.physics_rows(r)
            summary = {row[0]: row[1:] for t in tables.summary(r) if isinstance(t, tables.Table) for row in t.rows}
    assert [(label, unit) for label, _, unit in rows] == [("Max temperature", "K")]
    assert float(rows[0][1]) == float(f"{r.history['max_temperature'][-1]:.4g}")
    assert summary["Max temperature"] == [rows[0][1], "K"] and "Max failure factor" not in summary
    assert summary["Units"] == ["mmKS: mm, K, mW, mW/(mm*K), t"]


def test_a_thermal_run_with_held_temperatures_only():
    """The physics says what a project must have (NEEDS): thermal needs a support and no load, so a run with
    two held temperatures and no source goes through, and its save gives no RuntimeWarning (no load nodes,
    so no load centroid in the record); the structural physics still needs a load."""
    import warnings
    from lsr_mmto.problem import selections as S
    from lsr_mmto.problem.project import Project, run
    from lsr_mmto.results import load_results
    with registered(THERMAL):
        p = _project("held", "UBracket.STL", UBRACKET, 3000)
        p.add_selection("bottom", S.face("ymin"))
        p.add_selection("top_right", S.box([0.75, 0.9, -1], [1.0, 1.0, 1]))
        p.bcs += [{"kind": "fixed_temperature", "on": "bottom", "value": 0.0},
                  {"kind": "fixed_temperature", "on": "top_right", "value": 1.0}]
        p.objective("Mass()")                                       # f . T is zero with no source
        p.settings["common"]["maxIterations"] = 2
        assert p.validate() == []
        with tempfile.TemporaryDirectory() as d, warnings.catch_warnings(record=True) as heard:
            warnings.simplefilter("always")
            run(p, save_dir=d, verbose=False)
            r = load_results.load(os.path.join(d, "run_data.npz"))
        assert np.all(np.isfinite(r.history["objective"]))
        runtime = [f"{w.filename}:{w.lineno} {w.message}" for w in heard if issubclass(w.category, RuntimeWarning)]
        assert runtime == [], runtime
        summary = r.config["fea"]["loading_and_bc"]["T"]
        assert summary["load_node_count"] == 0 and "load_centroid_xyz" not in summary
    s = Project("structural", stl="UBracket.STL", n_dof=3000, single_layer=True, base_dir=UBRACKET)
    s.add_selection("bottom", S.face("ymin"))
    s.add_bc("fixed", "bottom", dofs="xyz")
    assert "no loads" in s.validate()


def test_the_saved_run_keeps_the_properties_the_physics_reads(monkeypatch):
    """A run saves each element's value of every property its physics reads (PROPERTIES): the conductivity,
    and nothing of the structural physics, which its database does not have."""
    from lsr_mmto.optimize import save_results
    from lsr_mmto.problem.project import run
    saved = {}
    real = save_results.save_run_data

    def keep(**kw):
        saved.update(kw["fields"])
        return real(**kw)
    monkeypatch.setattr(save_results, "save_run_data", keep)
    with registered(THERMAL):
        p = _held_and_heated("properties")
        p.settings["common"]["maxIterations"] = 2
        with tempfile.TemporaryDirectory() as d:
            run(p, save_dir=d, verbose=False)
    assert "property_Conductivity" in saved and "property_Youngs_Modulus" not in saved
    assert not any(k.lower().startswith("youngs") for k in saved)


def test_thermal_is_analysed_with_no_file_of_it_changed():
    """Analyse (optimize/solid_part_analysis.py) on a thermal project: its analyse and final_summary run on
    the solid part in one material, and the analysis gives its objective and its RESULT_FIELDS."""
    from lsr_mmto.optimize.solid_part_analysis import analyse_solid_part
    with registered(THERMAL):
        r = analyse_solid_part(_held_and_heated("analysed"), 2)
    assert r.analysed_material and list(r.results) == list(THERMAL.RESULT_FIELDS) == ["temperature"]
    assert r.results["temperature"].ndim == 1 and np.all(np.isfinite(r.results["temperature"]))
    assert r.result_info["temperature"]["view"] == "Temperature"
    assert np.isfinite(r.final_objective) and r.final_objective > 0             # f . T of the source
    assert np.isclose(r.utilisation(0), 1.0 / 0.5)                            # every element full: volume 1


def test_result_fields_say_where_and_shape():
    """A RESULT_FIELDS entry may say where its values live and their shape; one that does not is a number at
    each element. The final check refuses values of another length or shape than declared."""
    import pytest
    from types import SimpleNamespace
    from lsr_mmto.optimize.final_design_analysis import checked_results
    from lsr_mmto.results.load_results import result_info, result_shape
    scalar = dict(view="Value", title="Value", measures=None)
    info = result_info(scalar)
    assert info["where"] == "element" and info["shape"] == "scalar" and result_shape(info, 10, 4) == (10,)
    assert result_shape(result_info(THERMAL.RESULT_FIELDS["temperature"]), 10, 4) == (4,)
    vector = dict(view="Displacement", title="Displacement", measures="length", where="node", shape="vector")
    assert result_shape(result_info(vector), 10, 4) == (4, 3)
    with pytest.raises(ValueError, match="where 'nodes'"):
        result_info(dict(vector, where="nodes"))
    physics = SimpleNamespace(NAME="test", RESULT_FIELDS={"value": scalar, "displacement": vector})
    good = {"value": np.zeros(10), "displacement": np.zeros((4, 3))}
    assert checked_results(physics, good, 10, 4) is good
    for wrong in (np.zeros(12), np.zeros((10, 3)), np.zeros((4, 2))):
        with pytest.raises(ValueError, match=r"declares a vector at each node: shape \(4, 3\)"):
            checked_results(physics, dict(good, displacement=wrong), 10, 4)
    with pytest.raises(ValueError, match="declares a scalar at each element"):
        checked_results(physics, dict(good, value=np.zeros(4)), 10, 4)


def test_a_result_the_physics_does_not_return_stops_the_save_with_a_message(monkeypatch):
    """final_summary returns the value of every RESULT_FIELDS entry; one left out is a clear error, not a
    result silently missing or taken from an earlier solve."""
    import pytest
    from lsr_mmto.physics.thermal import thermal_results
    from lsr_mmto.problem.project import run
    real = thermal_results.final_summary
    monkeypatch.setattr(THERMAL, "final_summary", lambda *a, **k: (real(*a, **k), {})[1])
    with registered(THERMAL):
        p = _held_and_heated("missing result")
        p.settings["common"]["maxIterations"] = 2
        with tempfile.TemporaryDirectory() as d, pytest.raises(ValueError, match="returned no temperature"):
            run(p, save_dir=d, verbose=False)


def test_a_large_thermal_problem_keeps_the_direct_solver(monkeypatch, capsys):
    """Above the direct solver's limit the loop uses DPCG only for a physics with a deflation space; thermal
    has none, so it keeps PARDISO. The limit is lowered here so the test stays small."""
    import lsr_mmto.pyto_ad_imports as imports
    from lsr_mmto.problem.project import run
    monkeypatch.setattr(imports, "DIRECT_SOLVER_DOF_CUTOFF", 100)
    with registered(THERMAL):
        p = _held_and_heated("large")
        p.settings["common"]["maxIterations"] = 2
        with tempfile.TemporaryDirectory() as d:
            run(p, save_dir=d, verbose=True)
    out = capsys.readouterr().out
    assert "Solver: PARDISO" in out and "Solver: DPCG" not in out


def test_a_large_structural_problem_still_uses_dpcg(monkeypatch, capsys):
    """The structural physics builds its deflation space, so above the limit it runs DPCG as before."""
    import lsr_mmto.pyto_ad_imports as imports
    from lsr_mmto.problem import selections as S
    from lsr_mmto.problem.project import Project, run
    monkeypatch.setattr(imports, "DIRECT_SOLVER_DOF_CUTOFF", 100)
    p = Project("structural", stl="UBracket.STL", n_dof=3000, single_layer=True, base_dir=UBRACKET)
    p.add_selection("bottom", S.face("ymin"))
    p.add_selection("top_right", S.box([0.75, 0.9, -1], [1.0, 1.0, 1]))
    p.add_bc("fixed", "bottom", dofs="xyz")
    p.add_bc("force", "top_right", total=[-1.0, 0.0, 0.0])
    p.database = "6Materials_AF_3Attr"
    p.objective("Compliance()")
    p.constraint("VolumeFraction()", "<=", 0.5)
    p.settings["common"]["maxIterations"] = 2
    with tempfile.TemporaryDirectory() as d:
        run(p, save_dir=d, verbose=True)
    assert "Solver: DPCG" in capsys.readouterr().out


def _test_penalty(weight, scale="first value"):
    """A penalty of thermal for these tests: greyness of the densities plus the latent points' distance to
    0.1, so its gradient has an x block and a z block."""
    return dict(name="ThermalTestPenalty", shown="Test penalty", help="A penalty for the tests only.",
                fn=lambda m: torch.mean(m.x_phys * (1 - m.x_phys)) + torch.mean((m.z - 0.1) ** 2),
                weight=weight, scale=scale, key="thermal_test_penalty")


def test_a_physics_penalty_is_added_with_its_gradient():
    """A penalty of thermal is added after the two every physics has, divided by its first
    value and times its weight, and the gradient of the objective with it agrees with central differences
    (the two every physics has are switched off, so the objective is mass plus this penalty)."""
    from lsr_mmto.design import latent_penalties
    from lsr_mmto.optimize import optimization_loop as M
    from lsr_mmto.problem.project import _prepared, _run_arguments
    penalty = latent_penalties.Penalty(physics=["thermal"], **_test_penalty(lambda state, it: 5.0))
    with registered(THERMAL), added(latent_penalties.PENALTIES, penalty), tempfile.TemporaryDirectory() as d:
        p = _held_and_heated("penalty")
        p.objective("Mass()")
        p.settings["advanced"].update(use_penalization=False, use_z_smoothing=False)
        state = M.setup(**_run_arguments(_prepared(p), save_dir=d, verbose=False))
        assert [q.name for q in state.penalties] == ["LatentSmoothing", "DistancePenalty", "ThermalTestPenalty"]
        n = state.num_elems
        rng = np.random.default_rng(0)
        zeta = state.zeta0.flatten().copy()
        zeta[:n] = rng.uniform(0.2, 0.8, n)
        zeta[n:] = rng.uniform(state.lower.ravel()[n:], state.upper.ravel()[n:]) * 0.5
        f, g, _, _ = M.evaluate(state, zeta)              # the first: mass 1 (its first value), penalty 5
        assert np.isclose(f.item(), 6.0)
        assert state.history["thermal_test_penalty"] == [5.0]
        assert state.history["Pz_n"] == state.history["Pn"] == state.history["P"] == [0.0]

        everything = state.penalties
        state.penalties = everything[:2]                  # the same design without it: the mass's gradient alone
        g_mass = M.evaluate(state, zeta)[1]
        state.penalties = everything
        h = 1e-5
        for block in (rng.choice(n, 3, replace=False), n + rng.choice(len(zeta) - n, 3, replace=False)):
            for i in block:
                plus, minus = zeta.copy(), zeta.copy()
                plus[i] += h
                minus[i] -= h
                fd = (M.evaluate(state, plus)[0].item() - M.evaluate(state, minus)[0].item()) / (2 * h)
                assert abs(fd - g[i, 0]) <= 1e-5 * abs(g[i, 0]), (i, fd, g[i, 0])
            share = np.abs(g[block, 0] - g_mass[block, 0]).sum() / np.abs(g[block, 0]).sum()
            assert share > 0.1, share                     # the penalty's part of the gradient is not lost in it


def test_a_physics_penalty_keeps_its_history_and_its_schedule():
    """A short thermal run with the test penalty on for its first iteration only: the saved run's history has
    its key, 0.5 at the first iteration (its first value, times 0.5) and 0 from then on; the history keys of
    the two every physics has are kept."""
    from lsr_mmto.design import latent_penalties
    from lsr_mmto.problem.project import run
    from lsr_mmto.results import load_results
    penalty = latent_penalties.Penalty(physics=["thermal"],
                                       **_test_penalty(lambda state, it: 0.5 if state.iteration < 1 else 0))
    with registered(THERMAL), added(latent_penalties.PENALTIES, penalty):
        p = _held_and_heated("penalty")
        p.settings["common"]["maxIterations"] = 2
        with tempfile.TemporaryDirectory() as d:
            run(p, save_dir=d, verbose=False)
            r = load_results.load(os.path.join(d, "run_data.npz"))
    values = list(r.history["thermal_test_penalty"])
    assert len(values) >= 2 and values[0] == 0.5 and all(v == 0 for v in values[1:]), values
    assert all(len(r.history[k]) == len(values) for k in ("P", "Pn", "Pz_n"))


def test_a_quantity_of_another_physics_is_refused():
    """The project's check refuses a quantity of another physics, as the objective or as a constraint (the
    structural Compliance in a thermal project), and keeps the quantities every physics has and thermal's
    own; a structural project may still name Compliance."""
    with registered(THERMAL):
        p = _held_and_heated("other physics")
        p.complete_settings()
        assert p.validate() == []                           # ThermalCompliance and VolumeFraction
        p.objective("Compliance()")
        p.constraint("Compliance()", "<=", 2.0)
        errors = [e for e in p.validate() if "quantity" in e]
        assert errors == ["objective: Compliance is not a thermal quantity",
                          "constraint 2: Compliance is not a thermal quantity"], errors


def test_a_penalty_must_state_its_scale():
    """A penalty entry without its scale, or with one that is not "raw" or "first value", is refused when the
    core's lists are made (at start-up, where each front page's entry becomes a Penalty), and nothing of it
    is added."""
    import pytest
    from lsr_mmto.design import latent_penalties
    entry = _test_penalty(lambda state, it: 0.5)
    del entry["scale"]
    with pytest.raises(TypeError, match="scale"):
        latent_penalties.Penalty(physics=["thermal"], **entry)
    with pytest.raises(ValueError, match="scale must be one of 'raw', 'first value'"):
        latent_penalties.Penalty(physics=["thermal"], **_test_penalty(lambda state, it: 0.5, scale="normalised"))
    assert [q.name for q in latent_penalties.PENALTIES] == ["LatentSmoothing", "DistancePenalty"]


def test_the_mesh_is_sized_by_the_physics_unknowns_per_node():
    """Mesh dofs counts every unknown at every node: thermal (one unknown per node) gets about three times
    the elements the structural physics (three per node) gets for the same number."""
    from lsr_mmto.problem.mesh import make_mesh
    with registered(THERMAL):
        p = _project("mesh size", "UBracket.STL", UBRACKET, 3000)
        assert p.unknowns_per_node() == 1
        one = make_mesh(p.stl_path(), p.n_dof, p.single_layer, p.unknowns_per_node())
        three = make_mesh(p.stl_path(), p.n_dof, p.single_layer, 3)
    assert 2.0 < one.num_elems / three.num_elems < 4.5, (one.num_elems, three.num_elems)


def _another_physics(**entries):
    """A made-up second physics: what the core's lists read of a front page, empty unless given."""
    import types
    base = dict(NAME="another", FIELDS={"u": 3}, PROPERTIES=[], NEW_PROPERTIES=[], QUANTITIES=[],
                SUPPORT_AND_LOAD_KINDS={})
    return types.SimpleNamespace(**dict(base, **entries))


def test_one_name_one_meaning_across_physics():
    """The core's lists (material properties, quantities, kinds of support and load) join the core's entries and
    every physics': an entry a second physics gives exactly as the structural one does is one entry, tagged
    with both; a different entry under the same name, or a property no one defines, stops LSR MMTO with a
    message naming the places. Thermal, which brings its own conductivity, joins with its own entries."""
    import pytest
    from lsr_mmto.formulation.quantities import quantities_of
    from lsr_mmto.materials.databases import attributes_of
    from lsr_mmto.physics.structural import structural_module as STRUCTURAL
    from lsr_mmto.problem.boundary_conditions import bc_kinds_of
    yield_strength = next(p for p in STRUCTURAL.NEW_PROPERTIES if p["name"] == "Yield_Strength")
    compliance = next(q for q in STRUCTURAL.QUANTITIES if q["name"] == "Compliance")
    same = _another_physics(NEW_PROPERTIES=[yield_strength], PROPERTIES=["Yield_Strength"], QUANTITIES=[compliance],
                            SUPPORT_AND_LOAD_KINDS={"force": STRUCTURAL.SUPPORT_AND_LOAD_KINDS["force"]})
    modules = [STRUCTURAL, same]
    attributes = attributes_of(modules)
    assert list(attributes).count("Yield_Strength") == 1 and attributes["Yield_Strength"].physics == ("structural",
                                                                                                     "another")
    assert attributes["Density"].physics == () and attributes["Youngs_Modulus"].physics == ("structural",)
    quantities = [q for q in quantities_of(modules) if q.name == "Compliance"]
    assert len(quantities) == 1 and quantities[0].physics == {"structural", "another"}
    assert bc_kinds_of(modules)["force"].physics == ("structural", "another")

    other = dict(yield_strength, low=1)                             # the same name, another meaning
    with pytest.raises(ValueError, match="one in the structural physics, one in the another physics"):
        attributes_of([STRUCTURAL, _another_physics(NEW_PROPERTIES=[other])])
    with pytest.raises(ValueError, match="'Mass': one in the core, one in the another physics"):
        quantities_of([STRUCTURAL, _another_physics(QUANTITIES=[dict(compliance, name="Mass")])])
    with pytest.raises(ValueError, match="'force': one in the structural physics, one in the another physics"):
        bc_kinds_of([STRUCTURAL, _another_physics(SUPPORT_AND_LOAD_KINDS={
            "force": dict(STRUCTURAL.SUPPORT_AND_LOAD_KINDS["force"], shown="Push")})])
    with pytest.raises(ValueError, match="the another physics' PROPERTIES names the material property "
                                         "'Conductivity', which neither the core nor any physics defines"):
        quantities_of([STRUCTURAL, _another_physics(PROPERTIES=["Conductivity"])])
    with pytest.raises(ValueError, match="the quantity Heat names the material property 'Conductivity'"):
        quantities_of([STRUCTURAL, _another_physics(QUANTITIES=[dict(compliance, name="Heat",
                                                                     needs=["Conductivity"])])])
    with registered(THERMAL):
        assert len(quantities_of([STRUCTURAL, THERMAL])) == len(quantities_of([STRUCTURAL])) + len(THERMAL.QUANTITIES)
        assert list(attributes_of([STRUCTURAL, THERMAL])) == list(attributes_of([STRUCTURAL])) + ["Conductivity"]
        assert attributes_of([STRUCTURAL, THERMAL])["Conductivity"].physics == ("thermal",)
