"""
final_check.py
==============
The final design analysed three ways (continuous, binarized, binarized with snapped materials),
its stress diagnostics printed, and the run saved for later postprocessing.
"""
import numpy as np
import torch

from .history import (compute_mass_from_x_and_z, compute_max_failure_factor,
                      compute_stress_violation_diagnostics, print_stress_violation_diagnostics)


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



def run_postprocessing(history, zetaOptimal, fe_solver, matEncoder, mat_lib,
                       MaterialModel, physical_density,
                       zRealPoints, material_colors, num_elems, latentDim,
                       elem_volume, design_domain_volume,
                       binarize_topology, snap_to_real_material, save_dir=None,
                       run_config=None, example_name=None, material_labels=None, decode=None):
    """Reconstruct the optimal fields, evaluate stress/compliance metrics at the
    continuous / binarized / snapped designs, and draw the final plots.

    decode(z) -> (E, nu, properties) is the optimization's own decode, so the final check analyses
    the design with the same materials (a decoded or constant Poisson's ratio) as the optimization;
    physical_density(x) -> x_phys is its own density filter and projection, at the last beta."""
    if decode is None:
        raise ValueError("run_postprocessing needs decode: the optimization's E, nu for latent points")
    import os
    if save_dir is not None:
        pass  # plot autosave disabled; figures are drawn by results/figures.py from the saved record

    def _p(name):
        return os.path.join(save_dir, name) if save_dir else None

    # Reconstruct physical fields
    xOptimal_raw = zetaOptimal[0:num_elems].copy()
    zOptimal = zetaOptimal[num_elems:].reshape(latentDim, -1).T   # numpy, element-major

    xOptimal_cont = physical_density(xOptimal_raw)   # the optimization's own filter and projection

    eval_stress_compliance_metrics(
        x_vec=xOptimal_cont,
        z_vec=zOptimal.T.reshape(-1),
        label="Continuous (x_phys projected, raw z)",
        active_thresh=0.5,
        latentDim=latentDim,
        matEncoder=matEncoder,
        fe_solver_structural=fe_solver,
        MaterialModel=MaterialModel,
        elem_volume=elem_volume,
        design_domain_volume=design_domain_volume,
        decode=decode,
    )

    xOptimal = xOptimal_cont.copy()
    if binarize_topology:
        x_sorted = np.sort(xOptimal)
        threshold = x_sorted[int((1 - np.mean(xOptimal)) * len(xOptimal))]
        xOptimal = np.where(xOptimal < threshold, 0.0, 1.0)

    eval_stress_compliance_metrics(
        x_vec=xOptimal,
        z_vec=zOptimal.T.reshape(-1),
        label="Binarized topology (x projected -> binarized, raw z)",
        active_thresh=0.5,
        latentDim=latentDim,
        matEncoder=matEncoder,
        fe_solver_structural=fe_solver,
        MaterialModel=MaterialModel,
        elem_volume=elem_volume,
        design_domain_volume=design_domain_volume,
        decode=decode,
    )

    zOptimalPts = torch.tensor(zOptimal, dtype=torch.get_default_dtype())
    if snap_to_real_material:
        zSnappedPts = torch.tensor(matEncoder.getClosestRealMaterialZValues(zOptimalPts))
        zOptimalPts = zSnappedPts
    eval_stress_compliance_metrics(
        x_vec=xOptimal,
        z_vec=zOptimalPts.T.flatten().numpy(),
        label="Binarized + snapped materials (x binarized, snapped z)",
        active_thresh=0.5,
        latentDim=latentDim,
        matEncoder=matEncoder,
        fe_solver_structural=fe_solver,
        MaterialModel=MaterialModel,
        elem_volume=elem_volume,
        design_domain_volume=design_domain_volume,
        decode=decode,
    )

    decoded = matEncoder.vaeNet.decoder(zOptimalPts)
    material_properties = matEncoder.getMaterialProperties(decoded)
    Youngs_Modulus = material_properties["Youngs_Modulus"].detach().cpu().numpy()

    final_stress_diag = compute_stress_violation_diagnostics(
        vm=fe_solver.vonMisesStress,
        material_properties=material_properties,
        x_vec=xOptimal,
        active_thresh=0.5,
        mesh=fe_solver.mesh,
        bc=fe_solver.bc,
        violation_tol=0.0,
    )

    print_stress_violation_diagnostics(
        final_stress_diag,
        iteration=None,
        prefix="[FINAL STRESS DIAG]",
    )

    # --- persist everything for offline postprocessing (results/records.py, results/figures.py) ---
    material_indices = matEncoder.getClosestRealMaterialIndex(zOptimalPts)
    if hasattr(material_indices, "cpu"):
        material_indices = material_indices.cpu().numpy()
    material_names = [matEncoder.materialNames[i]
                      for i in range(len(matEncoder.materialNames))]

    if save_dir is not None:
        from lsr_mmto.optimize import save as run_save
        _zreal = (zRealPoints.detach().cpu().numpy()
                  if hasattr(zRealPoints, "detach") else np.asarray(zRealPoints))
        run_save.save_run_data(
            save_dir=save_dir,
            config=run_config if run_config is not None else {},
            fields={
                "zeta": np.asarray(zetaOptimal),
                "x_cont": np.asarray(xOptimal_cont),
                "x_bin": np.asarray(xOptimal),
                "z_elem": np.asarray(zOptimal),
                "youngs_modulus": np.asarray(Youngs_Modulus),
                "material_indices": np.asarray(material_indices),
                "von_mises": np.asarray(fe_solver.vonMisesStress),
            },
            vae_state_dict=matEncoder.vaeNet.state_dict(),
            history=history,
            example_name=example_name,
            z_real_points=_zreal,
            material_colors=material_colors,
            material_names=material_names,
            material_labels=material_labels,
            latent_dim=latentDim,
            num_elems=num_elems,
            num_nodes=int(fe_solver.mesh.num_nodes),
        )
        print(f"[SAVE] run_data.pkl + run_config.yaml -> {save_dir}")
