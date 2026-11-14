"""
structural_results.py
=====================
What the structural module records and reports: the results a run saves (RESULT_FIELDS) and its
Summary rows (SUMMARY_ROWS), each iteration's stress diagnostics (max failure factor, stress-violation
counts) recorded into the history (iteration_diagnostics), and the final designs' compliance, stresses
and mass, printed by the final check (final_summary).
"""
from typing import TYPE_CHECKING, Any, Callable

import numpy as np
import torch

import lsr_mmto.pyto_ad_imports  # noqa: F401  puts PyTO-AD on sys.path
from lsr_mmto.pyto_ad_imports import topopt_material_model as ad_mm
from lsr_mmto.optimize.history import _property_to_numpy, compute_mass_from_x_and_z

if TYPE_CHECKING:
    import pyvista

    from pyto.autodiff.material_model import MaterialModel
    from pyto.core.bc import BC
    from pyto.core.hex_mesher import HexMesher
    from lsr_mmto.materials.encoder import MaterialEncoder
    from lsr_mmto.optimize.solid_part_analysis import _Materials
    from lsr_mmto.physics.structural.structural_physics import MultiMaterialHexFEA
    from lsr_mmto.problem.settings import RunSettings
    from lsr_mmto.optimize.design_model import MMTOModelAD
# the results a run saves, by name: the view that shows it in the Results panel, its title on the colour scale,
# what it measures (for its unit), where its values live and their shape (a number at each element unless it
# says otherwise), and its value from the solver after the final check
RESULT_FIELDS = {
    "von_mises": dict(view="Stress", title="von Mises stress", measures="stress"),
    "displacement": dict(view="Displacement", title="Displacement", measures="length", where="node", shape="vector"),
}

# the numbers of its own a run's Summary shows: the last value of each history key, when the run recorded it
SUMMARY_ROWS = {
    "max_failure_factor": dict(label="Max failure factor", unit="", format=".3f"),
}


# HELPER: of record_fea_diagnostics and eval_stress_compliance_metrics
def compute_max_failure_factor(vm: np.ndarray, material_properties: dict[str, torch.Tensor]) -> float | None:
    """
    Compute max failure factor = max(vonMises / Yield_Strength).

    Returns None if Yield_Strength is not present.
    """
    if "Yield_Strength" not in material_properties:
        return None

    vm = np.asarray(vm, dtype=float).reshape(-1)
    yield_strength = material_properties["Yield_Strength"].detach().cpu().numpy().astype(float).reshape(-1)

    return float(np.max(vm / np.maximum(yield_strength, 1e-12)))


# HELPER: of record_fea_diagnostics and final_summary
def compute_stress_violation_diagnostics(
    vm: np.ndarray,
    material_properties: dict[str, torch.Tensor],
    x_vec: np.ndarray | None = None,
    active_thresh: float = 0.5,
    mesh: "HexMesher | None" = None,
    bc: "BC | None" = None,
    violation_tol: float = 0.0,
) -> dict[str, Any] | None:
    """
    Compute elementwise stress-constraint diagnostics.

    Local constraint:
        FF_e = vonMises_e / YieldStrength_e
        Q_e  = FF_e - 1

    An element violates the local stress constraint when:
        Q_e > violation_tol

    The active-topology count uses x_vec > active_thresh (every element when x_vec is None).
    Returns a dict of the failure factors, masks, indices, counts, fractions and maxima, or None when the
    properties have no Yield_Strength. When some active element violates and mesh has elem_centers, also where
    the violations are; when bc also has a force and mesh node_xyz, how far they are from the loaded nodes.
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


# HELPER: of record_fea_diagnostics and final_summary
def print_stress_violation_diagnostics(diag: dict[str, Any] | None, iteration: int | None = None,
                                       prefix: str = "[STRESS DIAG]") -> None:
    """
    Print compact stress violation diagnostics (compute_stress_violation_diagnostics'). For None, a note that
    there is no yield strength, printed only for iteration None or <= 1.
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


# HELPER: of record_fea_diagnostics
def update_paper_history(
    history: dict[str, list[float | np.ndarray]],
    x_phys: np.ndarray,
    z_phys_elemmajor: np.ndarray | None,
    material_properties: dict[str, torch.Tensor],
    von_mises: np.ndarray,
    elem_volume: float,
    design_domain_volume: float,
) -> None:
    """
    Update paper-style histories:
      - mass ratio: final mass / volume of the design domain (a density, kept under its old name)
      - maximum failure factor

    If Density or Yield_Strength are unavailable, store np.nan. Appends to history["mass_ratio"] and
    history["max_failure_factor"]; z_phys_elemmajor is not used.
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


# HELPER: of iteration_diagnostics
def record_fea_diagnostics(history: dict[str, list[float | np.ndarray]], iteration: int, x_phys: np.ndarray,
                           z: np.ndarray | None, material_properties: dict[str, torch.Tensor],
                           fe_solver: "MultiMaterialHexFEA",
                           elem_volume: float, design_domain_volume: float, plot_progress: bool = False,
                           plotter: "pyvista.Plotter | None" = None, verbose: bool = True) -> None:
    """After the FEA solve: update paper history (mass ratio, max FF), record
    stress-violation diagnostics into history, optionally print them (verbose),
    and optionally draw the realtime topology (plot_progress). Reads the solver's vonMisesStress, so its
    postprocess() must have run at this design. z is passed on to update_paper_history, which does not use it."""
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


# HELPER: of final_summary
def eval_stress_compliance_metrics(
    x_vec: np.ndarray,
    z_vec: np.ndarray,
    label: str,
    active_thresh: float,
    latentDim: int,
    matEncoder: "MaterialEncoder | _Materials",
    fe_solver_structural: "MultiMaterialHexFEA",
    MaterialModel: "type[MaterialModel]",
    elem_volume: float,
    design_domain_volume: float,
    decode: "Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]]",
) -> dict[str, float | None]:
    """The design's compliance, stresses and mass, printed under label. decode(z) gives the
    per-element (E, nu, properties) the optimization used for latent points z. x_vec (N,); z_vec latent-major
    (latentDim * N). Solves on the solver (its state is left at this design) and returns the printed numbers;
    final_mass, mass_ratio and max_failure_factor are None when the database has no column for them."""
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


# the history keys structural records each iteration (iteration_diagnostics adds them)
HISTORY_KEYS = ("mass_ratio", "max_failure_factor", "num_stress_violating_active", "frac_stress_violating_active",
                "num_stress_violating_all", "frac_stress_violating_all", "max_q_active", "max_q_all")


# OPTIONAL: iteration_diagnostics, what each iteration records
def iteration_diagnostics(history: dict[str, list[float | np.ndarray]], iteration: int, m: "MMTOModelAD",
                          x_phys: np.ndarray, properties: dict[str, torch.Tensor], elem_volume: float,
                          design_domain_volume: float,
                          plot_progress: bool = False, plotter: "pyvista.Plotter | None" = None,
                          verbose: bool = True) -> None:
    """After the iteration's solve: the solver's stresses at x_phys (postprocess), then the mass ratio,
    the max failure factor and the stress-violation counts recorded into history, printed when verbose. Adds
    those keys (HISTORY_KEYS) to the history, each run's first time, so every structural record has them.
    m is the model after its solve; its solver is m.fe."""
    solver = m.fe
    for key in HISTORY_KEYS:
        history.setdefault(key, [])
    solver.mesh.setPseudoDensity(torch.as_tensor(x_phys))   # PyTO-AD's mesh takes torch
    solver.postprocess()
    record_fea_diagnostics(history, iteration, x_phys, None, properties, solver, elem_volume,
                           design_domain_volume, plot_progress=plot_progress, plotter=plotter, verbose=verbose)


# ESSENTIAL: final_summary, the final check and the results a run saves
def final_summary(designs: "list[tuple[str, np.ndarray, np.ndarray | torch.Tensor]]", solver: "MultiMaterialHexFEA",
                  properties_of: Callable[[torch.Tensor], dict[str, torch.Tensor]], settings: "RunSettings",
                  encoder: "MaterialEncoder | _Materials", latent_dim: int, elem_volume: float,
                  design_domain_volume: float) -> dict[str, np.ndarray]:
    """The final check's numbers: each design (label, x, z element-major) analysed and printed, then the
    stress violations of the last one. properties_of(z) gives the decoded properties of latent points z.
    Returns the results a run saves (RESULT_FIELDS), of the last design: von Mises at each element, and the
    displacement (x, y, z) at each node, the solver's last solution."""
    from lsr_mmto.physics.structural.structural_physics import elastic_constants

    def decode(z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
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
