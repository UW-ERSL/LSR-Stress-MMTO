"""
test_thermoelastic_physics.py
=============================
The thermoelastic physics (lsr_mmto/physics/thermoelastic/) gives the right answers: a bar heated evenly
and held only against rigid motion grows freely by its thermal strain. Its solver is checked on its own
(make_solver, then analyse), so no material database is needed; its tiny example, which
tests/runs/test_every_physics_runs.py runs, writes a database of one material.

    python -m pytest tests/physics/test_thermoelastic_physics.py
"""
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from lsr_mmto.physics.thermoelastic import thermoelastic_module as THERMOELASTIC  # noqa: E402

BAR = os.path.join(ROOT, "data", "models", "RectangularBeam")


def _bar(name, n_dof=2000):
    """A thermoelastic project on the bar, its six faces as selections, nothing held yet."""
    from lsr_mmto.problem import selections as S
    from lsr_mmto.problem.project import Project
    p = Project(name, stl="RectangularBeam.STL", n_dof=n_dof, single_layer=True, base_dir=BAR)
    p.physics = "thermoelastic"
    for side in ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax"):
        p.add_selection(side, S.face(side))
    return p


def tiny_example(folder):
    """The smallest thermoelastic project that runs: the bar on a coarse mesh, held on its xmin face at the
    reference temperature, with 1 of heat put in and a pull down on its xmax face, its thermoelastic compliance
    made least with half its volume, 2 iterations. Thermoelastic has no example database with a network yet, so
    its material is one written in folder (aluminium's numbers, in MKS) and the project is of that one material."""
    from openpyxl import Workbook
    wb = Workbook()
    wb.active.append(["Attributes", "Density", "Youngs_Modulus", "Poissons_Ratio", "Thermal_Expansion", "Conductivity"])
    wb.active.append(["Material/Units", "kg/m^3", "Pa", None, "1/K", "W/(m*K)"])
    wb.active.append(["Aluminium", 2700.0, 69e9, 0.33, 23e-6, 167.0])
    database = os.path.join(folder, "one_material.xlsx")
    wb.save(database)
    p = _bar("tiny thermoelastic", n_dof=1500)
    p.database, p.material_source = database, "one_database_material"
    p.add_bc("fixed", "xmin", dofs="xyz")
    p.add_bc("force", "xmax", total=[0, -1, 0])
    p.bcs += [{"kind": "fixed_temperature", "on": "xmin", "value": 293.15},
              {"kind": "heat_source", "on": "xmax", "total": 1.0}]
    p.objective("ThermoelasticCompliance()")
    p.constraint("VolumeFraction()", "<=", 0.5)
    p.settings["common"]["maxIterations"] = 2
    return p


def _solved(p, E, nu, alpha, conductivity, T_ref):
    """The bar's mesh and a model m after analyse: one value of each property in every element, every element
    solid. m holds what analyse reads (fe, x_phys, props, settings) and what it sets (T, f_th, u)."""
    from lsr_mmto.pyto_ad_imports import linear_solvers, mat_lib
    p.complete_settings()
    mesh = p.mesh()
    bcs, _ = p.supports_and_loads(mesh)
    assert THERMOELASTIC.supports_and_loads_error(mesh, bcs) is None
    mat_prop = mat_lib.create_material_with_defaults(          # placeholder, as Project.build gives it
        "CustomMaterial", youngs_modulus=1.0, poissons_ratio=0.3, mass_density=1.0, yield_strength=1.0)
    fe = THERMOELASTIC.make_solver(mesh, mat_prop, bcs, linear_solvers.Solvers.PARDISO, None, None)
    n = mesh.num_elems
    each = lambda v: torch.full((n,), v, dtype=torch.float64)  # noqa: E731
    m = SimpleNamespace(fe=fe, x_phys=each(1.0),
                        props={"Youngs_Modulus": each(E), "Conductivity": each(conductivity),
                               "Thermal_Expansion": each(alpha)},
                        settings=SimpleNamespace(reference_temperature=T_ref, te_use_constant_poissons_ratio=False,
                                                 te_constant_poissons_ratio=nu))
    THERMOELASTIC.analyse(m)
    return mesh, m


def test_a_bar_heated_evenly_grows_freely():
    """The bar held at T_ref + dT on two opposite faces with no heat source, so dT everywhere; held only in x on
    xmin, y on ymin and z on zmin, so nothing stops it growing. Each node moves by alpha * dT times its
    distance from those three faces, in each direction: the thermal strain with no stress. Trilinear elements
    give this linear field exactly. A bar that shrinks has H's sign wrong; one that grows (1 - 2 nu) times
    too little is missing the 1 / (1 - 2 nu)."""
    T_ref, dT, alpha, nu = 293.15, 100.0, 1e-5, 0.3
    p = _bar("free expansion")
    p.bcs += [{"kind": "fixed_temperature", "on": "xmin", "value": T_ref + dT},
              {"kind": "fixed_temperature", "on": "xmax", "value": T_ref + dT}]
    p.add_bc("fixed", "xmin", dofs="x")
    p.add_bc("fixed", "ymin", dofs="y")
    p.add_bc("fixed", "zmin", dofs="z")
    mesh, m = _solved(p, E=200.0, nu=nu, alpha=alpha, conductivity=3.0, T_ref=T_ref)
    T = m.T.detach().numpy()
    assert np.max(np.abs(T - (T_ref + dT))) < 1e-8, "the temperature is not even"
    xyz = np.asarray(mesh.node_xyz, dtype=float)
    expected = alpha * dT * (xyz - xyz.min(axis=0))
    u = m.u.detach().numpy().reshape(-1, 3)
    error = np.max(np.abs(u - expected)) / np.max(np.abs(expected))
    assert error < 1e-8, f"largest difference {error:.2e} of the largest displacement"

def test_no_heating_gives_the_structural_compliance():
    """The bar held at T_ref at both ends, so no thermal strain anywhere: clamped on xmin and pulled down on
    xmax, its thermoelastic compliance (f + f_th) . u equals the structural solver's f . u for the same mesh,
    supports, load and material, and the thermal force is zero."""
    from lsr_mmto.physics.structural import structural_module as STRUCTURAL
    from lsr_mmto.pyto_ad_imports import linear_solvers, mat_lib
    T_ref, E, nu = 293.15, 200.0, 0.3
    p = _bar("no heating")
    p.bcs += [{"kind": "fixed_temperature", "on": "xmin", "value": T_ref},
              {"kind": "fixed_temperature", "on": "xmax", "value": T_ref}]
    p.add_bc("fixed", "xmin", dofs="xyz")
    p.add_bc("force", "xmax", total=[0.0, -1.0, 0.0])
    mesh, m = _solved(p, E=E, nu=nu, alpha=1e-5, conductivity=3.0, T_ref=T_ref)
    bcs, _ = p.supports_and_loads(mesh)                     # its own copy, for the structural solver
    mat_prop = mat_lib.create_material_with_defaults(
        "CustomMaterial", youngs_modulus=1.0, poissons_ratio=0.3, mass_density=1.0, yield_strength=1.0)
    n = mesh.num_elems
    s = SimpleNamespace(fe=STRUCTURAL.make_solver(mesh, mat_prop, {"u": bcs["u"]}, linear_solvers.Solvers.PARDISO,
                                                  None, None),
                        x_phys=m.x_phys, props={"Youngs_Modulus": torch.full((n,), E, dtype=torch.float64)},
                        settings=SimpleNamespace(use_constant_poissons_ratio=True, constant_poissons_ratio=nu,
                                                 pnorm_exponent=6.0, stress_relaxation_exponent=0.5))
    STRUCTURAL.analyse(s)
    f_th = m.f_th.detach().numpy()
    f = s.fe.total_force.detach().numpy()
    assert np.max(np.abs(f_th)) < 1e-10 * np.max(np.abs(f)), "a thermal force with no heating"
    thermoelastic = float(torch.dot(m.fe.total_force, m.u))
    structural = float(torch.dot(s.fe.total_force, s.u))
    assert abs(thermoelastic - structural) < 1e-10 * abs(structural), (thermoelastic, structural)

def test_each_simp_exponent_reaches_its_solver():
    """Thermoelastic's two SIMP exponents (set_simp_penalties): the stiffness's where PyTO-AD's structural solve
    reads it (its global), the conductivity's on the temperature's own solver."""
    from lsr_mmto.pyto_ad_imports import topopt_material_model
    p = _bar("simp")
    p.bcs += [{"kind": "fixed_temperature", "on": "xmin", "value": 293.15}]
    p.add_bc("fixed", "xmin", dofs="xyz")
    mesh, m = _solved(p, E=200.0, nu=0.3, alpha=1e-5, conductivity=3.0, T_ref=293.15)
    saved = topopt_material_model._SIMP_STRUCTURAL_PENALTY
    try:
        THERMOELASTIC.set_simp_penalties(m.fe, {"structural": 4.25, "thermal": 2.5})
        assert topopt_material_model._SIMP_STRUCTURAL_PENALTY == 4.25 and m.fe.thermal.conductivity_penalty == 2.5
    finally:
        topopt_material_model.set_SIMP_STRUCTURAL_PENALTY(saved)


def test_the_shared_dof_table_is_the_displacements_whatever_the_order():
    """Each solver keeps its own dof table (.edof); the mesh's shared one, which PyTO-AD's constructors point at
    their own, is the displacement's once thermoelastic's solver is made."""
    p = _bar("tables")
    p.bcs += [{"kind": "fixed_temperature", "on": "xmin", "value": 293.15}]
    p.add_bc("fixed", "xmin", dofs="xyz")
    mesh, m = _solved(p, E=200.0, nu=0.3, alpha=1e-5, conductivity=3.0, T_ref=293.15)
    assert mesh.edofMat is mesh.edofMatStructural
    assert np.array_equal(m.fe.edof.numpy(), mesh.edofMatStructural)
    assert np.array_equal(m.fe.thermal.edof.numpy(), mesh.edofMatThermal)


def test_gradients_agree_with_finite_differences():
    """J = (f + f_th) . u on the bar clamped and held at T_ref on xmin and heated on xmax (no applied force, so
    all of J is the heating's, and finite differences can see each element's part), at an
    uneven design (densities, expansion coefficients and conductivities all differ by element). Its autograd
    gradient for the density, the expansion coefficient and the conductivity of a few elements, each where the
    gradient is large, agrees with central differences: the whole chain T -> f_th -> u, the thermal solve
    included. The expansion coefficient and the conductivity reach J only through the heating."""
    T_ref, E, nu = 0, 200.0, 0.3
    p = _bar("gradients")
    p.bcs += [{"kind": "fixed_temperature", "on": "xmin", "value": T_ref},
              {"kind": "heat_source", "on": "xmax", "total": 1.0}]
    p.add_bc("fixed", "xmin", dofs="xyz")
    _, m = _solved(p, E=E, nu=nu, alpha=1e-5, conductivity=3.0, T_ref=T_ref)
    n = len(m.x_phys)
    rng = np.random.default_rng(0)
    design = {"x_phys": rng.uniform(0.3, 1.0, n), "Thermal_Expansion": 1e-5 * rng.uniform(0.5, 1.5, n),
              "Conductivity": 3.0 * rng.uniform(0.5, 1.5, n)}

    def J(values):
        """J at the design values, with the solver of m: a fresh model through analyse."""
        t = {k: torch.tensor(v, dtype=torch.float64, requires_grad=True) for k, v in values.items()}
        model = SimpleNamespace(fe=m.fe, x_phys=t["x_phys"], settings=m.settings,
                                props={"Youngs_Modulus": torch.full((n,), E, dtype=torch.float64),
                                       "Thermal_Expansion": t["Thermal_Expansion"],
                                       "Conductivity": t["Conductivity"]})
        THERMOELASTIC.analyse(model)
        return torch.dot(model.fe.total_force, model.u), t

    value, leaves = J(design)
    value.backward()
    for name, leaf in leaves.items():
        g = leaf.grad.numpy()
        assert np.any(g != 0), f"no gradient for {name}"
        for i in rng.choice(np.argsort(-np.abs(g))[:50], 3, replace=False):
            best = np.inf
            for step in (1e-4, 1e-5, 1e-6):                     # relative to the value
                h = step * design[name][i]
                plus, minus = {k: v.copy() for k, v in design.items()}, {k: v.copy() for k, v in design.items()}
                plus[name][i] += h
                minus[name][i] -= h
                with torch.no_grad():
                    fd = (float(J(plus)[0]) - float(J(minus)[0])) / (2 * h)
                best = min(best, abs(fd - g[i]) / abs(g[i]))
            assert best < 1e-5, f"{name}[{i}]: relative difference {best:.1e}"

def test_a_large_problem_uses_dpcg_and_gives_the_direct_answer():
    """Above the direct solver's limit the core uses DPCG for a physics with a deflation space; thermoelastic
    takes structural's for its displacement and thermal's for its temperature (one unknown per node), both on
    the same checked solver. The bar, heated and pulled down, solved this way (the limit passed by asking for
    more unknowns than it) gives the direct solver's temperature and displacement, and both solves are kept as
    converged."""
    from lsr_mmto.optimize.iterative_solve_check import CheckedDeflationSolver
    from lsr_mmto.optimize.run_setup import problem_solver
    from lsr_mmto.pyto_ad_imports import DIRECT_SOLVER_DOF_CUTOFF, linear_solvers, mat_lib
    T_ref, E, nu = 293.15, 200.0, 0.3
    p = _bar("large")
    p.bcs += [{"kind": "fixed_temperature", "on": "xmin", "value": T_ref},
              {"kind": "heat_source", "on": "xmax", "total": 1.0}]
    p.add_bc("fixed", "xmin", dofs="xyz")
    p.add_bc("force", "xmax", total=[0.0, -1.0, 0.0])
    p.complete_settings()
    mesh = p.mesh()
    mat_prop = mat_lib.create_material_with_defaults(          # placeholder, as Project.build gives it
        "CustomMaterial", youngs_modulus=1.0, poissons_ratio=0.3, mass_density=1.0, yield_strength=1.0)
    n = mesh.num_elems
    each = lambda v: torch.full((n,), v, dtype=torch.float64)  # noqa: E731
    answers, checked = {}, CheckedDeflationSolver()
    for name, n_dof in (("PARDISO", 1), ("DPCG", DIRECT_SOLVER_DOF_CUTOFF + 1)):
        bcs, _ = p.supports_and_loads(mesh)
        fe = problem_solver(THERMOELASTIC, mesh, mat_prop, bcs, None, SimpleNamespace(nDOFDesired=n_dof),
                            dsolver=checked if name == "DPCG" else None)
        assert fe.solver == getattr(linear_solvers.Solvers, name), (name, fe.solver)
        m = SimpleNamespace(fe=fe, x_phys=each(1.0),
                            props={"Youngs_Modulus": each(E), "Conductivity": each(3.0),
                                   "Thermal_Expansion": each(1e-5)},
                            settings=SimpleNamespace(reference_temperature=T_ref, te_use_constant_poissons_ratio=True,
                                                     te_constant_poissons_ratio=nu))
        THERMOELASTIC.analyse(m)
        answers[name] = (m.T.detach().numpy(), m.u.detach().numpy())
    for what, direct, iterative in zip(("temperature", "displacement"), answers["PARDISO"], answers["DPCG"]):
        error = np.max(np.abs(iterative - direct)) / np.max(np.abs(direct))
        assert error < 1e-6, f"{what}: largest difference {error:.1e} of its largest value"
    assert len(checked.solves) == 2 and all(ok for _, _, ok in checked.solves), checked.solves


def test_a_part_left_free_in_a_rigid_motion_is_refused():
    """The thermal force pushes in every direction, so the part must be held against all six rigid motions,
    whatever the applied loads: one left free makes the displacement's matrix singular, and the direct solver
    then gives a displacement with an arbitrary drift in it. Held in x on xmin and y on ymin only, the bar can
    still move in z; held in all directions along one edge only, it can still rotate about that edge. Each
    refusal names the motion left free. Held as in the free-expansion test (x, y and z on three faces), it is
    held against all six, though it can still grow."""
    from lsr_mmto.problem import selections as S
    T_ref = 293.15
    def error(p):
        p.bcs += [{"kind": "fixed_temperature", "on": "xmin", "value": T_ref},
                  {"kind": "heat_source", "on": "xmax", "total": 1.0}]
        p.complete_settings()
        mesh = p.mesh()
        bcs, _ = p.supports_and_loads(mesh)
        return THERMOELASTIC.supports_and_loads_error(mesh, bcs)
    p = _bar("z left free")
    p.add_bc("fixed", "xmin", dofs="x")
    p.add_bc("fixed", "ymin", dofs="y")
    assert "free to move in z" in (error(p) or ""), error(p)
    p = _bar("one edge")
    p.add_selection("edge", S.box([-1, -1, -1], [1001, 1, 1]))   # the nodes along x at y = 0, z = 0
    p.add_bc("fixed", "edge", dofs="xyz")
    assert "free to rotate about x" in (error(p) or ""), error(p)
    p = _bar("three faces")
    p.add_bc("fixed", "xmin", dofs="x")
    p.add_bc("fixed", "ymin", dofs="y")
    p.add_bc("fixed", "zmin", dofs="z")
    assert error(p) is None
