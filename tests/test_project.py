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
from utils import project_paths  # noqa: E402

project_paths.bootstrap()

import selections as S  # noqa: E402  # type: ignore
from project import Project, to_params_form  # noqa: E402  # type: ignore
from PyTOImports import TO_QOI  # noqa: E402  # type: ignore

UBRACKET = os.path.join(ROOT, "0-Common", "Models", "UBracket")
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
    with tempfile.TemporaryDirectory() as d:
        p = coded_ubracket()
        a, b = os.path.join(d, "a.pyto"), os.path.join(d, "b.pyto")
        p.save(a)
        q = Project.load(a)
        assert q.to_dict() == p.to_dict()
        q.save(b)
        assert open(a).read() == open(b).read()
        assert os.path.isfile(q.stl_path())             # the STL path still resolves after the move


def test_expressions():
    assert to_params_form("Mass()", "constraint", 4e7) == (TO_QOI.MASS, None, 4e7)
    assert to_params_form("FailureFactorPNorm()", "objective") == (TO_QOI.PNORM_STRESS, "FAILURE_FACTOR")
    assert to_params_form("FailureFactorPNorm()", "constraint", 1) == (TO_QOI.STRESS_FAILURE_FACTOR, None, 1.0)
    assert to_params_form("MaterialVolumeFraction(2, tau=0.5)", "constraint", 0.15) == \
        (TO_QOI.VOLUME_FRACTION, {"material_id": 2, "tau": 0.5}, 0.15)
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
    mesh, mat_prop, bc, body_force, to_params, vae_params = p.build(verbose=False)
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
    assert to_params.Objective == (TO_QOI.PNORM_STRESS, "FAILURE_FACTOR")
    assert to_params.Constraints == [(TO_QOI.MASS, None, 0.0035)]
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


def test_build_gui_project_matches_problem_from_pyto():
    """A project made in the PyTO-AD GUI gives exactly the boundary conditions it gave before."""
    from pyto_project import problem_from_pyto  # type: ignore
    path = os.path.join(UBRACKET, "UBracket.pyto")
    mesh0, _, bc0, _ = problem_from_pyto(path, nDOFDesired=N_DOF, single_layer=True)
    p = Project.load(path)
    p.n_dof, p.single_layer = N_DOF, True
    p.database = "6Materials_AF_3Attr"
    p.objective("FailureFactorPNorm()")
    mesh, _, bc, _, _, _ = p.build(verbose=False)
    assert mesh.num_elems == mesh0.num_elems
    assert np.array_equal(bc.fixed_dofs, bc0.fixed_dofs)
    assert np.array_equal(bc.force, bc0.force)
    assert np.array_equal(bc.dirichlet_values, bc0.dirichlet_values)


def test_script_round_trip():
    """Every project in projects/ exported to a script, the script run: the same project."""
    import glob
    for path in sorted(glob.glob(os.path.join(ROOT, "projects", "*.pyto"))):
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


def _gui_copy(d, structural_bc=None):
    """A copy of the U-bracket GUI project in d, with other GUI picks when given."""
    import json
    import shutil
    shutil.copy(os.path.join(UBRACKET, "UBracket.STL"), d)
    data = json.load(open(os.path.join(UBRACKET, "UBracket.pyto")))
    if structural_bc is not None:
        data["structuralBC"] = structural_bc
    path = os.path.join(d, "UBracket.pyto")
    json.dump(data, open(path, "w"), indent=2)
    return path


def test_gui_picks_become_the_project():
    """Opening a GUI project turns its picks into selections and bcs; saved and opened
    again, nothing is converted twice, and the script shows the whole problem."""
    with tempfile.TemporaryDirectory() as d:
        p = Project.load(_gui_copy(d))
        assert p.bcs == [{"kind": "fixed", "on": "gui_support1", "dofs": "xyz"},
                         {"kind": "force", "on": "gui_load1", "total": [-1.0, 0.0, 0.0]}]
        assert p.selections["gui_support1"].TYPE == "halfspaces"
        sbc = p._gui["structuralBC"]
        assert all(v == [] for v in sbc.values() if isinstance(v, list))
        assert p._gui["settings"] == json.load(open(os.path.join(UBRACKET, "UBracket.pyto")))["settings"]
        p.database = "6Materials_AF_3Attr"
        p.objective("FailureFactorPNorm()")
        p.save(os.path.join(d, "UBracket.pyto"))
        q = Project.load(os.path.join(d, "UBracket.pyto"))
        assert q.to_dict() == p.to_dict()
        text = open(q.to_script(os.path.join(d, "u.py"))).read()
        assert "gui_support1" in text and "halfspaces(" in text
    try:
        p.to_script(os.path.join(ROOT, "elsewhere.py"))
    except ValueError as e:
        assert "next to the project" in str(e)
    else:
        raise AssertionError("a script away from the project was accepted")


def test_gui_picks_of_every_kind_match_the_gui():
    """Faces, picked nodes and boxes, alone and together, in both GUI layouts: the same
    fixed dofs and force vector as the GUI's own rules (pyto_project.gui_bcs_on_mesh)."""
    from pyto_project import gui_bcs_on_mesh, make_mesh, read_project  # type: ignore
    box = json.load(open(os.path.join(UBRACKET, "UBracket.pyto")))["structuralBC"]["point_supports"][0]["regions"]
    new = {"fixed_faces_indices": [0, 1, 2], "fixed_faces_z_indices": [5],
           "load_faces_indices": [[8, 9]], "load_forces": [[0.0, -2.0, 0.0]],
           "point_supports": [{"type": "Fixed Y", "points": [[0.5, 0.0, 0.0], [0.6, 0.02, 0.01]],
                               "regions": box},
                              {"type": "Fixed XYZ", "points": [], "regions": box}],   # holds it: well posed
           "point_loads": [{"points": [[0.98, 1.0, 0.0]], "regions": [], "force": [-1.0, 0.0, 0.0]}],
           "constraint_counts": {"fixed_triangles": 4, "loaded_triangles": 2}}
    old = {"constraint_data": [{"type": "Fixed X", "triangles": [0, 1], "points": [[0.5, 0.0, 0.0]]},
                               {"type": "Fixed XYZ", "triangles": [], "points": [], "regions": box}],
           "force_data": [{"type": "force_xyz", "triangles": [], "points": [], "regions": box,
                           "force_x": 0.0, "force_y": 3.0, "force_z": 0.0}]}
    for layout, sbc in (("new", new), ("old", None)):
        with tempfile.TemporaryDirectory() as d:
            path = _gui_copy(d, sbc)
            if layout == "old":
                data = json.load(open(path))
                del data["structuralBC"]
                data.update(old)
                json.dump(data, open(path, "w"))
            gui = read_project(path)
            mesh0 = make_mesh(gui["stl_file_path"], N_DOF, single_layer=True)
            fixed0, force0 = gui_bcs_on_mesh(gui, mesh0)
            p = Project.load(path)
            p.n_dof, p.single_layer = N_DOF, True
            p.database = "6Materials_AF_3Attr"
            p.objective("Compliance()")
            mesh, _, bc, _, _, _ = p.build(verbose=False)
            assert np.array_equal(np.sort(bc.fixed_dofs), fixed0), layout
            assert np.array_equal(bc.force, force0), layout
            assert not np.any(bc.dirichlet_values), layout


def test_gui_sections_survive_a_save():
    """Saving from code keeps what the GUI wrote in the same file, except its supports and
    loads, which are now in the mmto section."""
    with tempfile.TemporaryDirectory() as d:
        p = Project.load(os.path.join(UBRACKET, "UBracket.pyto"))
        p.database = "6Materials_AF_3Attr"
        p.objective("FailureFactorPNorm()")
        out = os.path.join(d, "u.pyto")
        p.save(out)
        saved = json.load(open(out))
        original = json.load(open(os.path.join(UBRACKET, "UBracket.pyto")))
        for k in original:
            if k not in ("stl_file_path", "structuralBC"):
                assert saved[k] == original[k], k
        assert saved["structuralBC"]["point_supports"] == [] and saved["structuralBC"]["point_loads"] == []
        assert [bc["on"] for bc in saved["mmto"]["bcs"]] == ["gui_support1", "gui_load1"]


def test_schema_upgrades():
    """A file of an older schema is brought up to date on loading; a newer one is refused."""
    import project as P  # type: ignore
    with tempfile.TemporaryDirectory() as d:
        p = coded_ubracket()
        path = os.path.join(d, "u.pyto")
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
    from project import BC_KINDS  # type: ignore
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
