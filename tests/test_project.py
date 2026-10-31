"""
test_project.py
===============
The Project: saving and loading, the checks made before a run, the translation of
objectives and constraints, and the boundary conditions build() puts on the mesh.
The meshes are coarse U-brackets, so this runs in well under a minute.

    python tests/test_project.py        (or: python -m pytest tests)
"""
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


def test_gui_sections_survive_a_save():
    """Saving from code keeps what the GUI wrote in the same file."""
    import json
    with tempfile.TemporaryDirectory() as d:
        p = Project.load(os.path.join(UBRACKET, "UBracket.pyto"))
        p.database = "6Materials_AF_3Attr"
        p.objective("FailureFactorPNorm()")
        out = os.path.join(d, "u.pyto")
        p.save(out)
        saved = json.load(open(out))
        original = json.load(open(os.path.join(UBRACKET, "UBracket.pyto")))
        for k in original:
            if k != "stl_file_path":
                assert saved[k] == original[k], k


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
