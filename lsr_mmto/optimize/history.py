"""
history.py
==========
What each iteration of a run records and prints: the history (objective, constraints, grey
elements, material counts, mass, max failure factor), the log lines, and the stress-violation
diagnostics. Nothing here changes the design.
"""
import numpy as np
import torch
import matplotlib.pyplot as plt


def compute_domain_volume(fe_solver_structural) -> float:
    """
    Original design-domain volume before optimization.
    """
    elem_volume = float(np.prod(fe_solver_structural.mesh.elem_size))
    return float(fe_solver_structural.mesh.num_elems * elem_volume)

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


def _property_to_numpy(material_properties, key):
    if key not in material_properties:
        return None
    val = material_properties[key]
    if hasattr(val, "detach"):
        return val.detach().cpu().numpy().astype(float).reshape(-1)
    return np.asarray(val, dtype=float).reshape(-1)


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




def compute_mass_from_x_and_z(
    x_density: np.ndarray,
    z_flat: np.ndarray,
    matEncoder,
    latentDim: int,
    elem_volume: float,
):
    """
    Compute total mass = sum_e x_e * rho_e * V_e using decoded material density.

    Returns
    -------
    mass : float | None
        None if 'Density' is not present in the decoded material properties.
    """
    x_density = np.asarray(x_density, dtype=float).reshape(-1)

    zPts = torch.tensor(
        np.asarray(z_flat, dtype=float).reshape(latentDim, -1).T,
        dtype=torch.get_default_dtype(),   # the decoder's dtype (float64 under PyTO-AD)
    )

    with torch.no_grad():
        decoded = matEncoder.vaeNet.decoder(zPts)
        material_properties = matEncoder.getMaterialProperties(decoded)

    if "Density" not in material_properties:
        return None

    rho_e = material_properties["Density"].detach().cpu().numpy().astype(float)
    mass = float(np.sum(x_density * rho_e) * elem_volume)
    return mass


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
      - mass ratio
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



# ============================================================================
# RUN I/O + PER-ITERATION DIAGNOSTICS + POST-PROCESSING
# Extracted from the driver (optimize/run.py) so it stays optimization-only.
# History is recorded in all modes; `verbose` gates console output only.
# ============================================================================

def init_run_history():
    """Return a fresh history dict for a topopt run."""
    return {
        "objective": [],
        "constraints": [],
        "J_phys": [],
        "P": [],
        "Pn": [],
        "Pz_n": [],
        "grey": [],
        "mass_ratio": [],
        "max_failure_factor": [],
        "num_stress_violating_active": [],
        "frac_stress_violating_active": [],
        "num_stress_violating_all": [],
        "frac_stress_violating_all": [],
        "max_q_active": [],
        "max_q_all": [],
    }


def log_design_state(history, iteration, x_phys, z, material_properties, zTorch,
                     matEncoder, num_elems, beta_proj, eta_proj, verbose=True):
    """Record the grey fraction; in verbose mode also print the iteration header,
    grey %, and the [MAT] active-material statistics.

    Yield_Strength / Density are only present when the active material database
    decodes those attributes. When absent (e.g. a 2-attribute E/nu database such
    as BM1_3Mats_2Attr) the corresponding stats are skipped instead of raising
    KeyError; the material-count breakdown is unaffected and still printed."""
    grey_elements = np.sum((x_phys > 0.1) & (x_phys < 0.9))
    fraction_grey = grey_elements / num_elems
    history["grey"].append(float(fraction_grey))

    if not verbose:
        return

    print("-------------- Iteration", iteration, "-----------------")
    print(
        f"Percentage grey elements (x_phys): {fraction_grey*100:.2f}% | "
        f"beta={beta_proj:.3g}, eta={eta_proj}"
    )

    act = x_phys > 0.5
    if not act.any():
        return

    # getClosestRealMaterialIndex depends only on the latent vector, not on which
    # attributes are decoded, so material counts print regardless of DB attributes.
    midx = matEncoder.getClosestRealMaterialIndex(zTorch).cpu().numpy()
    counts = np.bincount(midx[act], minlength=len(matEncoder.materialNames))

    # _property_to_numpy returns None when the key is absent (no .detach() on a
    # missing key) -> matches the existing guard style used everywhere else.
    Y = _property_to_numpy(material_properties, "Yield_Strength")
    rho = _property_to_numpy(material_properties, "Density")

    parts = []
    if Y is not None:
        parts.append(
            f"Y_active min/mean/max="
            f"{Y[act].min():.3g}/{Y[act].mean():.3g}/{Y[act].max():.3g}"
        )
    if rho is not None:
        parts.append(
            f"rho_active min/mean/max="
            f"{rho[act].min():.3g}/{rho[act].mean():.3g}/{rho[act].max():.3g}"
        )
    parts.append(f"counts={counts.tolist()}")
    print("[MAT] " + " | ".join(parts))


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


def log_objective_and_constraints(history, obj, J_phys, Jn, P, Pn, Pzn, cons,
                                  to_params, verbose=True):
    """Record objective/constraint history. In verbose mode print the full
    Obj line + per-constraint lines (exactly as before); in simple mode print
    one compact per-iteration line."""
    cons_flat = np.asarray(cons).flatten()
    constraint_names = [getattr(c[0], "name", str(c[0])) for c in to_params.Constraints]

    history["objective"].append(float(obj))
    history["constraints"].append(cons_flat.copy())
    history["J_phys"].append(J_phys)
    history["P"].append(P)
    history["Pn"].append(Pn)
    history["Pz_n"].append(Pzn)

    if verbose:
        print(
            f"Obj: J={J_phys:.4g} | Jn={Jn:.4g} | P={P:.3g} | "
            f"Pn={Pn:.3g} | Pz_n={Pzn:.3g} | f={float(obj):.4g}"
        )
        for idx, val in enumerate(cons_flat):
            inequality = "<="
            if constraint_names[idx] in ("STRESS_SAFETY_FACTOR", "TEMPERATURE_SAFETY_FACTOR"):
                inequality = ">="
            print(
                f"Constraint {idx+1} ({constraint_names[idx]}): "
                f"{(val + 1) * to_params.Constraints[idx][2]:.3g} "
                f"{inequality} {to_params.Constraints[idx][2]:.3g}?"
            )
    else:
        mass = history["mass_ratio"][-1] if history["mass_ratio"] else float("nan")
        maxff = history["max_failure_factor"][-1] if history["max_failure_factor"] else float("nan")
        cons_str = " ".join(
            f"{constraint_names[idx]}={(val + 1) * to_params.Constraints[idx][2]:.3g}"
            for idx, val in enumerate(cons_flat)
        )
        print(f"[iter] f={float(obj):.4g} J={J_phys:.4g} mass={mass:.3g} "
              f"maxFF={maxff:.3g} | {cons_str}")

