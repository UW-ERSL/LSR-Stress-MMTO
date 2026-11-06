"""
structural_quantities.py
========================
What the structural module measures: its objectives and constraints (QUANTITIES: compliance,
p-norm stress, p-norm failure factor), each iteration's stress diagnostics (max failure factor,
stress-violation counts) recorded into the history (iteration_diagnostics), and the final
designs' compliance, stresses and mass, printed by the final check (final_summary).
"""
import numpy as np
import torch

import lsr_mmto.pyto_ad_imports  # noqa: F401  puts PyTO-AD on sys.path
from lsr_mmto.pyto_ad_imports import topopt_material_model as ad_mm
from lsr_mmto.optimize.history import _property_to_numpy, compute_mass_from_x_and_z
from lsr_mmto.physics.structural.structural_physics import pnorm


def _compliance(m, opt):
    return torch.dot(m.fe.total_force, m.u)               # F . u, as main's total_force . sol


def _stress_pnorm(m, opt):
    return pnorm(m.svm, ad_mm.get_pNorm_exponent())     # no 1/N, as PyTO's pNormStress


def _failure_factor_pnorm(m, opt):
    p = ad_mm.get_pNorm_exponent()                      # mean form (1/N), as the manual code on main
    Y = torch.clamp(m.props["Yield_Strength"], min=1e-12)
    return (torch.sum((m.svm / Y) ** p) / m.N) ** (1.0 / p)


# the structural objectives and constraints, as the arguments of a Quantity (formulation/quantities.py)
QUANTITIES = [
    dict(name="Compliance", objective=True, constraint=True, needs=[], fn=_compliance, doc="F . u",
         shown="Compliance", help="How much the part gives under the loads. Lower is stiffer.",
         unit="{force}*{length}", scales_with={"Youngs_Modulus": -1}),
    dict(name="StressPNorm", objective=True, constraint=False, needs=[], fn=_stress_pnorm,
         doc="p-norm of the relaxed von Mises stress, no 1/N", shown="Stress (p-norm)",
         help="One value close to the highest von Mises stress in the part.", unit="{stress}"),
    dict(name="FailureFactorPNorm", objective=True, constraint=True, needs=["Yield_Strength"],
         fn=_failure_factor_pnorm, doc="mean p-norm of von Mises / yield strength, per element",
         shown="Failure factor (p-norm)",
         help="One value close to the highest stress / yield strength in the part. Above 1 means some element "
              "yields.", unit="no units", scales_with={"Yield_Strength": -1}, start_bound=1.0),
]


def compute_max_failure_factor(vm, material_properties):
    """
    Compute max failure factor = max(vonMises / Yield_Strength).

    Returns None if Yield_Strength is not present.
    """
    if "Yield_Strength" not in material_properties:
        return None

    vm = np.asarray(vm, dtype=float).reshape(-1)
    yield_strength = material_properties["Yield_Strength"].detach().cpu().numpy().astype(float).reshape(-1)

    return float(np.max(vm / np.maximum(yield_strength, 1e-12)))


def compute_stress_violation_diagnostics(
    vm,
    material_properties,
    x_vec=None,
    active_thresh=0.5,
    mesh=None,
    bc=None,
    violation_tol=0.0,
):
    """
    Compute elementwise stress-constraint diagnostics.

    Local constraint:
        FF_e = vonMises_e / YieldStrength_e
        Q_e  = FF_e - 1

    An element violates the local stress constraint when:
        Q_e > violation_tol

    The active-topology count uses x_vec > active_thresh.
    """
    yield_strength = _property_to_numpy(material_properties, "Yield_Strength")
    if yield_strength is None:
        return None

    vm = np.asarray(vm, dtype=float).reshape(-1)
    ff = vm / np.maximum(yield_strength, 1e-12)
    q_local = ff - 1.0

    n_elems = vm.size
    all_mask = np.ones(n_elems, dtype=bool)

    if x_vec is None:
        x_arr = None
        active_mask = all_mask.copy()
    else:
        x_arr = np.asarray(x_vec, dtype=float).reshape(-1)
        active_mask = x_arr > active_thresh

    violation_mask_all = q_local > violation_tol
    violation_mask_active = violation_mask_all & active_mask
    satisfied_mask_active = (~violation_mask_all) & active_mask

    active_count = int(np.sum(active_mask))
    n_viol_all = int(np.sum(violation_mask_all))
    n_viol_active = int(np.sum(violation_mask_active))
    n_sat_active = int(np.sum(satisfied_mask_active))

    active_ff = ff[active_mask] if active_count > 0 else np.array([], dtype=float)
    active_q = q_local[active_mask] if active_count > 0 else np.array([], dtype=float)

    viol_indices_active = np.where(violation_mask_active)[0]
    viol_indices_all = np.where(violation_mask_all)[0]

    out = {
        "ff": ff,
        "q_local": q_local,
        "active_mask": active_mask,
        "violation_mask_all": violation_mask_all,
        "violation_mask_active": violation_mask_active,
        "satisfied_mask_active": satisfied_mask_active,
        "violating_indices_active": viol_indices_active,
        "violating_indices_all": viol_indices_all,
        "num_elements": n_elems,
        "num_active": active_count,
        "num_violating_all": n_viol_all,
        "num_violating_active": n_viol_active,
        "num_satisfied_active": n_sat_active,
        "frac_violating_all": n_viol_all / max(n_elems, 1),
        "frac_violating_active": n_viol_active / max(active_count, 1),
        "max_ff_all": float(np.max(ff)) if ff.size else float("nan"),
        "max_q_all": float(np.max(q_local)) if q_local.size else float("nan"),
        "max_ff_active": float(np.max(active_ff)) if active_ff.size else float("nan"),
        "max_q_active": float(np.max(active_q)) if active_q.size else float("nan"),
    }

    if mesh is not None and hasattr(mesh, "elem_centers") and viol_indices_active.size > 0:
        centers = np.asarray(mesh.elem_centers, dtype=float)
        viol_centers = centers[viol_indices_active]

        out["violating_centroid"] = np.mean(viol_centers, axis=0)
        out["violating_bbox_min"] = np.min(viol_centers, axis=0)
        out["violating_bbox_max"] = np.max(viol_centers, axis=0)

        if bc is not None and hasattr(bc, "force") and hasattr(mesh, "node_xyz"):
            force = np.asarray(bc.force, dtype=float).reshape(-1)
            loaded_dofs = np.where(np.abs(force) > 1e-14)[0]
            loaded_nodes = np.unique(loaded_dofs // 3)

            if loaded_nodes.size > 0:
                load_xyz = np.asarray(mesh.node_xyz, dtype=float)[loaded_nodes]
                d = np.linalg.norm(
                    viol_centers[:, None, :] - load_xyz[None, :, :],
                    axis=2,
                )
                nearest = np.min(d, axis=1)

                out["num_loaded_nodes"] = int(loaded_nodes.size)
                out["nearest_load_distance_min"] = float(np.min(nearest))
                out["nearest_load_distance_mean"] = float(np.mean(nearest))
                out["nearest_load_distance_max"] = float(np.max(nearest))

    return out


def print_stress_violation_diagnostics(diag, iteration=None, prefix="[STRESS DIAG]"):
    """
    Print compact stress violation diagnostics.
    """
    if diag is None:
        if iteration is None or int(iteration) <= 1:   # once per run, not every iteration
            print(f"{prefix} No Yield_Strength in the material database, so stress violations "
                  "are not counted. This is expected for such databases.")
        return

    iter_txt = "" if iteration is None else f" Iter {int(iteration):4d} |"

    print(
        f"{prefix}{iter_txt} "
        f"active violations = {diag['num_violating_active']}/{diag['num_active']} "
        f"({100.0 * diag['frac_violating_active']:.3f}%) | "
        f"all violations = {diag['num_violating_all']}/{diag['num_elements']} "
        f"({100.0 * diag['frac_violating_all']:.3f}%) | "
        f"max FF active = {diag['max_ff_active']:.6g} | "
        f"max Q active = {diag['max_q_active']:.6g}"
    )

    if diag["num_violating_active"] > 0:
        sample_ids = diag["violating_indices_active"][:20]
        print(f"{prefix} violating active element IDs, first 20: {sample_ids}")

        if "violating_centroid" in diag:
            c = diag["violating_centroid"]
            bmin = diag["violating_bbox_min"]
            bmax = diag["violating_bbox_max"]
            print(
                f"{prefix} violating centroid xyz = "
                f"[{c[0]:.6g}, {c[1]:.6g}, {c[2]:.6g}]"
            )
            print(
                f"{prefix} violating bbox min xyz = "
                f"[{bmin[0]:.6g}, {bmin[1]:.6g}, {bmin[2]:.6g}]"
            )
            print(
                f"{prefix} violating bbox max xyz = "
                f"[{bmax[0]:.6g}, {bmax[1]:.6g}, {bmax[2]:.6g}]"
            )

        if "nearest_load_distance_min" in diag:
            print(
                f"{prefix} distance from violating active elements to nearest loaded node: "
                f"min={diag['nearest_load_distance_min']:.6g}, "
                f"mean={diag['nearest_load_distance_mean']:.6g}, "
                f"max={diag['nearest_load_distance_max']:.6g} "
                f"over {diag['num_loaded_nodes']} loaded nodes"
            )


def update_paper_history(
    history,
    x_phys,
    z_phys_elemmajor,
    material_properties,
    von_mises,
    elem_volume,
    design_domain_volume,
):
    """
    Update paper-style histories:
      - mass ratio: final mass / volume of the design domain (a density, kept under its old name)
      - maximum failure factor

    If Density or Yield_Strength are unavailable, store np.nan.
    """
    density_vals = None
    yield_vals = None

    if "Density" in material_properties:
        density_vals = material_properties["Density"].detach().cpu().numpy().astype(float)

    if "Yield_Strength" in material_properties:
        yield_vals = material_properties["Yield_Strength"].detach().cpu().numpy().astype(float)

    # Mass ratio
    if density_vals is not None:
        final_mass = float(np.sum(np.asarray(x_phys, dtype=float).reshape(-1) * density_vals) * elem_volume)
        mass_ratio = final_mass / max(float(design_domain_volume), 1e-12)
    else:
        mass_ratio = np.nan

    max_failure_factor = compute_max_failure_factor(von_mises, material_properties)
    if max_failure_factor is None:
        max_failure_factor = np.nan

    history["mass_ratio"].append(mass_ratio)
    history["max_failure_factor"].append(max_failure_factor)


def record_fea_diagnostics(history, iteration, x_phys, z, material_properties,
                           fe_solver, elem_volume, design_domain_volume,
                           plot_progress=False, plotter=None, verbose=True):
    """After the FEA solve: update paper history (mass ratio, max FF), record
    stress-violation diagnostics into history, optionally print them (verbose),
    and optionally draw the realtime topology (plot_progress)."""
    update_paper_history(
        history=history,
        x_phys=x_phys,
        z_phys_elemmajor=z,
        material_properties=material_properties,
        von_mises=fe_solver.vonMisesStress,
        elem_volume=elem_volume,
        design_domain_volume=design_domain_volume,
    )
    stress_diag = compute_stress_violation_diagnostics(
        vm=fe_solver.vonMisesStress,
        material_properties=material_properties,
        x_vec=x_phys,
        active_thresh=0.5,
        mesh=fe_solver.mesh,
        bc=fe_solver.bc,
        violation_tol=0.0,
    )

    if stress_diag is not None:
        history["num_stress_violating_active"].append(stress_diag["num_violating_active"])
        history["frac_stress_violating_active"].append(stress_diag["frac_violating_active"])
        history["num_stress_violating_all"].append(stress_diag["num_violating_all"])
        history["frac_stress_violating_all"].append(stress_diag["frac_violating_all"])
        history["max_q_active"].append(stress_diag["max_q_active"])
        history["max_q_all"].append(stress_diag["max_q_all"])

    if verbose:
        print_stress_violation_diagnostics(
            stress_diag, iteration=iteration + 1, prefix="[STRESS DIAG]"
        )
    if plot_progress:
        fe_solver.plot_pseudo_density_realtime(
            title=f"Iter {iteration + 1}", external_plotter=plotter
        )


def eval_stress_compliance_metrics(
    x_vec,
    z_vec,
    label,
    active_thresh,
    latentDim,
    matEncoder,
    fe_solver_structural,
    MaterialModel,
    elem_volume,
    design_domain_volume,
    decode,
):
    """The design's compliance, stresses and mass, printed under label. decode(z) gives the
    per-element (E, nu, properties) the optimization used for latent points z."""
    x_vec = np.asarray(x_vec, dtype=float).flatten()
    z_vec = np.asarray(z_vec, dtype=float).flatten()

    zPts = torch.tensor(np.asarray(z_vec, dtype=float).reshape(latentDim, -1).T, dtype=torch.get_default_dtype())

    # E and nu exactly as the optimization had them: the same decode, the same constant-nu setting
    with torch.no_grad():
        E, nu, material_properties = decode(zPts)
    fe_solver_structural.set_material_fields(E.detach(), nu.detach())

    sol_local = fe_solver_structural.solve(x_vec, MaterialModel.SIMP)
    fe_solver_structural.mesh.setPseudoDensity(torch.as_tensor(x_vec))   # PyTO-AD's mesh takes torch
    fe_solver_structural.postprocess()

    vm = np.asarray(fe_solver_structural.vonMisesStress, dtype=float)
    max_failure_factor = compute_max_failure_factor(vm, material_properties)
    pnorm_vm = float(fe_solver_structural.pNormStress)
    compliance = float(np.einsum("i,i->", fe_solver_structural.total_force, sol_local))

    active = x_vec > active_thresh
    vm_min_active = float(np.min(vm[active])) if np.any(active) else float("nan")
    vm_max_active = float(np.max(vm[active])) if np.any(active) else float("nan")

    final_mass = compute_mass_from_x_and_z(
        x_density=x_vec,
        z_flat=z_vec,
        matEncoder=matEncoder,
        latentDim=latentDim,
        elem_volume=elem_volume,
    )

    if final_mass is not None:
        mass_ratio = final_mass / max(design_domain_volume, 1e-12)
    else:
        mass_ratio = None

    print("\n" + "=" * 70)
    print(f"[FINAL METRICS] {label}")

    if final_mass is not None:
        print(f" Final mass       : {final_mass:.6g}")
        print(
            f" Mass / design volume : {mass_ratio:.6g} "
            f"(final mass divided by the volume of the design domain: a density, not a ratio)"
        )
    else:
        print(" Final mass       : N/A (Density not present in material database)")
        print(" Mass / design volume : N/A (Density not present in material database)")
    if max_failure_factor is not None:
        print(f" Max failure factor : {max_failure_factor:.6g}")
    else:
        print(" Max failure factor : N/A (Yield_Strength not present in material database)")

    print(f" Compliance       : {compliance:.6g}")
    print(f" p-norm(vonMises) : {pnorm_vm:.6g}")
    print(f" max(vonMises)    : {vm_max_active:.6g} (x > {active_thresh})")
    print(f" min(vonMises)    : {vm_min_active:.6g} (x > {active_thresh})")
    print("=" * 70 + "\n")

    return {
        "final_mass": final_mass,
        "mass_ratio": mass_ratio,
        "compliance": compliance,
        "pnorm_vm": pnorm_vm,
        "vm_max_active": vm_max_active,
        "vm_min_active": vm_min_active,
        "max_failure_factor": max_failure_factor,
    }


def iteration_diagnostics(history, iteration, solver, x_phys, properties, elem_volume, design_domain_volume,
                          plot_progress=False, plotter=None, verbose=True):
    """After the iteration's solve: the solver's stresses at x_phys (postprocess), then the mass ratio,
    the max failure factor and the stress-violation counts recorded into history, printed when verbose."""
    solver.mesh.setPseudoDensity(torch.as_tensor(x_phys))   # PyTO-AD's mesh takes torch
    solver.postprocess()
    record_fea_diagnostics(history, iteration, x_phys, None, properties, solver, elem_volume,
                           design_domain_volume, plot_progress=plot_progress, plotter=plotter, verbose=verbose)


def final_summary(designs, solver, properties_of, settings, encoder, latent_dim, elem_volume,
                  design_domain_volume):
    """The final check's numbers: each design (label, x, z element-major) analysed and printed, then the
    stress violations of the last one. properties_of(z) gives the decoded properties of latent points z.
    Returns the results a run saves (RESULT_FIELDS), of the last design: von Mises at each element, and the
    displacement (x, y, z) at each node, the solver's last solution."""
    from lsr_mmto.physics.structural.structural_physics import elastic_constants

    def decode(z):
        props = properties_of(z)
        E, nu = elastic_constants(props, settings)
        return E, nu, props

    for label, x, z in designs:
        eval_stress_compliance_metrics(
            x_vec=x, z_vec=np.asarray(z).T.reshape(-1), label=label, active_thresh=0.5, latentDim=latent_dim,
            matEncoder=encoder, fe_solver_structural=solver, MaterialModel=ad_mm.MaterialModel,
            elem_volume=elem_volume, design_domain_volume=design_domain_volume, decode=decode)
    _, x, z = designs[-1]
    with torch.no_grad():
        props = properties_of(z if torch.is_tensor(z) else torch.as_tensor(np.asarray(z), dtype=torch.get_default_dtype()))
    final_stress_diag = compute_stress_violation_diagnostics(
        vm=solver.vonMisesStress, material_properties=props, x_vec=x, active_thresh=0.5,
        mesh=solver.mesh, bc=solver.bc, violation_tol=0.0)
    print_stress_violation_diagnostics(final_stress_diag, iteration=None, prefix="[FINAL STRESS DIAG]")
    return {"von_mises": np.asarray(solver.vonMisesStress),
            "displacement": solver.sol.detach().cpu().numpy().reshape(-1, 3)}         # node by node: x, y, z
