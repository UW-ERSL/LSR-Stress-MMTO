"""
test_project.py
===============
The Project: saving and loading, the checks made before a run, the translation of
objectives and constraints, and the boundary conditions build() puts on the mesh.
The meshes are coarse U-brackets, so this runs in well under a minute.

    python tests/test_project.py        (or: python -m pytest tests)
"""
import json
import os
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from lsr_mmto.problem import selections as S  # noqa: E402
from lsr_mmto.formulation.formulation_text import to_params_form  # noqa: E402
from lsr_mmto.problem.project import Project  # noqa: E402

UBRACKET = os.path.join(ROOT, "data", "models", "UBracket")
N_DOF = 3000


def coded_ubracket(base_dir=UBRACKET):
    """A U-bracket set up entirely in code, using every kind of boundary condition."""
    p = Project("ubracket_coded", stl="UBracket.STL", n_dof=N_DOF, single_layer=True, base_dir=base_dir)
    p.comment = "test"
    p.add_selection("bottom", S.face("ymin"))
    p.add_selection("top_right", S.box([0.75, 0.9, -1], [1.0, 1.0, 1]))
    p.add_support("bottom", dofs="xyz")
    p.add_support(S.nearest([0.0, 0.5, 0.0]), dofs="z", value=0.001)
    p.add_force("top_right", total=[-1.0, 0.0, 0.0])
    p.add_traction(S.box([0.0, 0.9, -1], [0.25, 1.0, 1]), pressure=2.0, direction=[0, -1, 0], area="bbox")
    p.keep_solid("bottom")
    p.database = "6Materials_AF_3Attr"
    p.objective("FailureFactorPNorm()")
    p.constraint("Mass()", "<=", 0.0035)
    p.settings["common"].update(relative_filter_radius=3.5, pnorm_exponent=16)
    p.settings["vae"]["latentDim"] = 2
    return p


def test_round_trip():
    try:
        coded_ubracket().save(os.path.join(tempfile.gettempdir(), "x.pyto"))
    except ValueError as e:
        assert "saved as a .mmto file" in str(e)
    else:
        raise AssertionError("a project was saved as .pyto")
    with tempfile.TemporaryDirectory() as d:
        p = coded_ubracket()
        a, b = os.path.join(d, "a.mmto"), os.path.join(d, "b.mmto")
        p.save(a)
        q = Project.load(a)
        assert q.to_dict() == p.to_dict()
        q.save(b)
        assert open(a).read() == open(b).read()
        assert os.path.isfile(q.stl_path())             # the STL path still resolves after the move


def test_expressions():
    assert to_params_form("Mass()", "constraint", 4e7) == ("Mass", None, 4e7)
    assert to_params_form("FailureFactorPNorm()", "objective") == ("FailureFactorPNorm", None)
    assert to_params_form("FailureFactorPNorm()", "constraint", 1) == ("FailureFactorPNorm", None, 1.0)
    assert to_params_form("MaterialVolumeFraction(2, tau=0.5)", "constraint", 0.15) == \
        ("MaterialVolumeFraction", {"material_id": 2, "tau": 0.5}, 0.15)
    for bad, role, says in [("Compliance() + 0.1*Mass()", "objective", "one quantity"),
                            ("Volume()", "constraint", "unknown quantity"),
                            ("VolumeFraction()", "objective", "cannot be the objective"),
                            ("Mass(3)", "constraint", "takes no arguments")]:
        try:
            to_params_form(bad, role, 1.0)
        except ValueError as e:
            assert says in str(e), (bad, str(e))
        else:
            raise AssertionError(f"{bad!r} was accepted")


def test_validate():
    assert coded_ubracket().validate() == []
    p = coded_ubracket()
    p.bcs.append({"kind": "fixed", "on": "nowhere", "dofs": "xyz"})
    p.bcs.append({"kind": "fixed", "on": "bottom", "dofs": "xw"})
    p.bcs.append({"kind": "traction", "on": "bottom", "pressure": 1, "direction": [1, 1, 0], "area": "bbox"})
    p.constraint("Mass()", ">=", 1.0)
    p.keep_solid_names.append("ghost")
    p.database = "NoSuchDatabase.xlsx"
    errors = "\n".join(p.validate())
    for says in ["no selection named 'nowhere'", "dofs must be letters", "needs a direction along x, y or z",
                 "only '<=' for now", "keep_solid: no selection named 'ghost'", "does not exist"]:
        assert says in errors, (says, errors)
    empty = Project("empty", stl="UBracket.STL", base_dir=UBRACKET)
    errors = "\n".join(empty.validate())
    for says in ["no supports", "no loads", "no material database", "no objective"]:
        assert says in errors, (says, errors)


def test_build_coded():
    p = coded_ubracket()
    mesh, mat_prop, bcs, body_force, to_params, vae_params = p.build(verbose=False)
    bc = bcs["u"]
    P = np.asarray(mesh.node_xyz)
    bottom = S.face("ymin").evaluate(mesh, "nodes")
    side = S.nearest([0.0, 0.5, 0.0]).evaluate(mesh, "nodes")
    # supports: all three dofs of the bottom face; z of one node on the left side, imposed 0.001
    expected = set((3 * bottom[:, None] + np.arange(3)).ravel().tolist()) | {3 * int(side[0]) + 2}
    assert set(bc.fixed_dofs.tolist()) == expected
    imposed = dict(zip(bc.fixed_dofs.tolist(), bc.dirichlet_values.tolist()))
    assert imposed[3 * int(side[0]) + 2] == 0.001
    assert sum(v != 0 for v in imposed.values()) == 1
    # loads: the force adds up to its total; the traction to pressure * bbox area
    F = bc.force.reshape(-1, 3)
    right = S.box([0.75, 0.9, -1], [1.0, 1.0, 1]).evaluate(mesh, "nodes")
    assert np.allclose(F[right].sum(0), [-1, 0, 0])
    left = S.box([0.0, 0.9, -1], [0.25, 1.0, 1]).evaluate(mesh, "nodes")
    area = np.ptp(P[left, 0]) * np.ptp(P[left, 2])
    assert np.allclose(F[left].sum(0), [0, -2.0 * area, 0])
    assert np.allclose(F.sum(0), [-1, -2.0 * area, 0])
    # formulation, settings, keep-solid
    assert to_params.Objective == ("FailureFactorPNorm", None)
    assert to_params.Constraints == [("Mass", None, 0.0035)]
    assert to_params.RelativeFilterRadius == 3.5 and to_params.ExtrudeZ is True
    assert vae_params.latentDim == 2 and vae_params.learningRate == 1e-4      # database default kept
    assert np.array_equal(to_params.ElemsToKeep, S.face("ymin").evaluate(mesh, "elements"))


def test_conflicting_values_are_refused():
    """A dof given 0 by one support and 0.001 by another is an error, not a silent pick."""
    p = coded_ubracket()
    p.add_support(S.nearest([0.0, 0.0, 0.0]), dofs="z", value=0.001)     # (0, 0, 0) is on the bottom face
    try:
        p.build(verbose=False)
    except ValueError as e:
        assert "already given the value" in str(e)
    else:
        raise AssertionError("conflicting imposed values were accepted")


def test_free_rigid_body_motion_is_refused():
    """Held at one node and pushed sideways at the top, the bracket is free to rotate: build stops,
    before the VAE and the run."""
    p = Project("pinned", stl="UBracket.STL", n_dof=N_DOF, single_layer=True, base_dir=UBRACKET)
    p.add_support(S.nearest([0.0, 0.0, 0.0]), dofs="xyz")
    p.add_force(S.box([0.75, 0.9, -1], [1.0, 1.0, 1]), total=[-1.0, 0.0, 0.0])
    p.database = "6Materials_AF_3Attr"
    p.objective("FailureFactorPNorm()")
    try:
        p.build(verbose=False)
    except ValueError as e:
        assert "free to move or rotate" in str(e), str(e)
    else:
        raise AssertionError("an under-supported problem was built")


def test_script_round_trip():
    """Every project in projects/ exported to a script, the script run: the same project."""
    import glob
    for path in sorted(glob.glob(os.path.join(ROOT, "projects", "*.mmto"))):
        p = Project.load(path)
        script = os.path.join(os.path.dirname(path), "_round_trip_test.py")
        try:
            p.to_script(script)
            code = open(script).read()
            assert code.rstrip().splitlines()[-1].startswith("p.save(")
            body = "\n".join(code.rstrip().splitlines()[:-1])           # everything but the save
            ns = {"__file__": script, "__name__": "round_trip"}
            exec(compile(body, script, "exec"), ns)
            q = ns["p"]
            p.complete_settings()
            q.complete_settings()
            assert q.to_dict() == p.to_dict(), os.path.basename(path)
            assert q.units == p.units, os.path.basename(path)
        finally:
            if os.path.exists(script):
                os.remove(script)


def test_window_picks_become_the_project():
    """Supports and loads picked on the part in the window (faces, a clicked node, a box) become
    selections and boundary conditions with the picking tools' own rules: the same fixed dofs and
    force vector as those rules gave (tests/data/gui_picks). Saved and opened again, the project is
    the same, and its script shows the whole problem."""
    ref_dir = os.path.join(ROOT, "tests", "data", "gui_picks")
    picks = json.load(open(os.path.join(ref_dir, "picks.json")))
    ref = np.load(os.path.join(ref_dir, "reference.npz"))
    with tempfile.TemporaryDirectory() as d:
        p = Project("u", stl=os.path.join(UBRACKET, "UBracket.STL"), n_dof=N_DOF, single_layer=True, base_dir=d)
        assert p.adopt_picks(picks) == ["gui_support1", "gui_support2", "gui_load1"]
        assert p.selections["gui_support2"].TYPE == "halfspaces"
        p.database = "6Materials_AF_3Attr"
        p.objective("Compliance()")
        mesh, _, bcs, _, _, _ = p.build(verbose=False)
        bc = bcs["u"]
        assert mesh.num_elems == int(ref["num_elems"])
        assert np.array_equal(np.sort(bc.fixed_dofs), ref["fixed_dofs"])
        assert np.array_equal(bc.force, ref["force"])
        assert not np.any(bc.dirichlet_values)
        p.save(os.path.join(d, "u.mmto"))
        q = Project.load(os.path.join(d, "u.mmto"))
        assert q.to_dict() == p.to_dict()
        text = open(q.to_script(os.path.join(d, "u.py"))).read()
        assert "gui_support2" in text and "halfspaces(" in text
    try:
        p.to_script(os.path.join(ROOT, "elsewhere.py"))
    except ValueError as e:
        assert "next to the project" in str(e)
    else:
        raise AssertionError("a script away from the project was accepted")


def test_only_mmto_files_load():
    """A project is a .mmto file with an mmto section; anything else is refused with the reason."""
    with tempfile.TemporaryDirectory() as d:
        other = os.path.join(d, "u.pyto")
        json.dump({"stl_file_path": "UBracket.STL"}, open(other, "w"))
        try:
            Project.load(other)
        except ValueError as e:
            assert "a project is a .mmto file" in str(e), e
        else:
            raise AssertionError("a .pyto file was loaded")
        plain = os.path.join(d, "plain.mmto")
        json.dump({"stl_file_path": "UBracket.STL"}, open(plain, "w"))
        try:
            Project.load(plain)
        except ValueError as e:
            assert "no mmto section" in str(e), e
        else:
            raise AssertionError("a .mmto file with no mmto section was loaded")


def test_gui_sections_survive_a_save():
    """Saving from code keeps what the window wrote in the same file (its settings section)."""
    original_path = os.path.join(ROOT, "projects", "L-bracket.mmto")
    with tempfile.TemporaryDirectory() as d:
        p = Project.load(original_path)
        out = os.path.join(d, "u.mmto")
        p.save(out)
        saved = json.load(open(out))
        original = json.load(open(original_path))
        assert set(saved) == set(original)
        for k in original:
            if k not in ("stl_file_path", "mmto"):
                assert saved[k] == original[k], k
        assert saved["settings"]["unit_system"] == "mmKS"


def test_schema_upgrades():
    """A file of an older schema is brought up to date on loading; a newer one is refused."""
    from lsr_mmto.problem import project as P
    with tempfile.TemporaryDirectory() as d:
        p = coded_ubracket()
        path = os.path.join(d, "u.mmto")
        p.save(path)
        data = json.load(open(path))
        old = dict(data, mmto=dict(data["mmto"], schema_version=0))
        old["mmto"]["remark"] = old["mmto"].pop("comment")               # pretend version 0 called it remark
        json.dump(old, open(path, "w"))
        try:
            Project.load(path)
        except ValueError as e:
            assert "has no upgrade" in str(e)
        else:
            raise AssertionError("a version 0 file loaded with no upgrade")
        P._UPGRADES[0] = lambda m: dict({k: v for k, v in m.items() if k != "remark"}, comment=m["remark"])
        try:
            q = Project.load(path)
        finally:
            del P._UPGRADES[0]
        assert q.to_dict() == p.to_dict()                                # saved again, it is version 1
        json.dump(dict(data, mmto=dict(data["mmto"], schema_version=P.SCHEMA_VERSION + 1)), open(path, "w"))
        try:
            Project.load(path)
        except ValueError as e:
            assert "newer version" in str(e)
        else:
            raise AssertionError("a file from a newer version was accepted")


def test_bc_kinds():
    """Every kind of boundary condition is a support or a load, and is documented."""
    from lsr_mmto.problem.boundary_conditions import BC_KINDS
    for name, kind in BC_KINDS.items():
        assert kind.role in ("support", "load") and kind.doc, name


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
