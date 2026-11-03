"""
history.py
==========
What each iteration of a run records and prints: the history (objective, constraints, grey
elements, material counts, mass) and the log lines. The stress diagnostics (max failure factor,
stress-violation counts) are recorded by physics/structural/structural_quantities.py. Nothing
here changes the design.
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

def _property_to_numpy(material_properties, key):
    if key not in material_properties:
        return None
    val = material_properties[key]
    if hasattr(val, "detach"):
        return val.detach().cpu().numpy().astype(float).reshape(-1)
    return np.asarray(val, dtype=float).reshape(-1)


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


def log_objective_and_constraints(history, obj, J_phys, Jn, P, Pn, Pzn, cons,
                                  to_params, verbose=True, material_labels=None):
    """Record objective/constraint history. In verbose mode print the full
    Obj line + per-constraint lines; in simple mode print one compact per-iteration
    line. Each constraint is named as the app names it."""
    from lsr_mmto.formulation.quantities import quantity_label
    cons_flat = np.asarray(cons).flatten()
    constraint_names = [quantity_label(c[0], c[1], material_labels) for c in to_params.Constraints]

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
            print(
                f"Constraint {idx+1} ({constraint_names[idx]}): "
                f"{(val + 1) * to_params.Constraints[idx][2]:.3g} "
                f"<= {to_params.Constraints[idx][2]:.3g}?"
            )
    else:
        mass = history["mass_ratio"][-1] if history["mass_ratio"] else float("nan")
        maxff = history["max_failure_factor"][-1] if history["max_failure_factor"] else float("nan")
        cons_str = " ".join(
            f"{constraint_names[idx]}={(val + 1) * to_params.Constraints[idx][2]:.3g}"
            for idx, val in enumerate(cons_flat)
        )
        print(f"[iter] f={float(obj):.4g} J={J_phys:.4g} mass/V={mass:.3g} "
              f"maxFF={maxff:.3g} | {cons_str}")

