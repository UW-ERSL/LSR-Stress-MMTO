"""
history.py
==========
What each iteration of a run records and prints: the history (objective, constraints, grey
elements, material counts, mass) and the log lines. The stress diagnostics (max failure factor,
stress-violation counts) are recorded by physics/structural/structural_quantities.py. Nothing
here changes the design.

The history is a dict of lists: init_run_history's keys, each penalty's keys (added by the loop's setup) and
those a physics' iteration_diagnostics appends to. Each iteration appends to the keys its steps record; a key a
physics does not record (the stress keys for a database without yield strength, or another physics) stays
short or empty.
"""
from typing import TYPE_CHECKING, Any, Mapping

import numpy as np
import torch
import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from lsr_mmto.materials.encoder import MaterialEncoder
    from lsr_mmto.optimize.mmto_to_params import MMTOParams
    from lsr_mmto.optimize.solid_part_analysis import _Materials


def compute_domain_volume(fe_solver_structural: Any) -> float:
    """
    Original design-domain volume before optimization: the number of elements times one element's volume
    (every element the same size). Any physics' solver (what its make_solver returns), despite the name.
    """
    elem_volume = float(np.prod(fe_solver_structural.mesh.elem_size))
    return float(fe_solver_structural.mesh.num_elems * elem_volume)

def _property_to_numpy(material_properties: Mapping[str, torch.Tensor | np.ndarray], key: str) -> np.ndarray | None:
    """One property's values as a flat float numpy array, from a torch tensor or anything numpy reads;
    None when the properties have no such key."""
    if key not in material_properties:
        return None
    val = material_properties[key]
    if hasattr(val, "detach"):
        return val.detach().cpu().numpy().astype(float).reshape(-1)
    return np.asarray(val, dtype=float).reshape(-1)


def compute_mass_from_x_and_z(
    x_density: np.ndarray,
    z_flat: np.ndarray,
    matEncoder: "MaterialEncoder | _Materials",
    latentDim: int,
    elem_volume: float,
) -> float | None:
    """
    Compute total mass = sum_e x_e * rho_e * V_e using decoded material density.
    x_density is (N,); z_flat is latent-major (latentDim * N), as in the design vector. Decoded outside the tree.

    Returns
    -------
    mass : float | None
        None if no decoded column measures density.
    """
    x_density = np.asarray(x_density, dtype=float).reshape(-1)

    zPts = torch.tensor(
        np.asarray(z_flat, dtype=float).reshape(latentDim, -1).T,
        dtype=torch.get_default_dtype(),   # the decoder's dtype (float64 under PyTO-AD)
    )

    with torch.no_grad():
        decoded = matEncoder.vaeNet.decoder(zPts)
        material_properties = matEncoder.getMaterialProperties(decoded)

    from lsr_mmto.materials.databases import density_column
    column = density_column(material_properties)
    if column is None:
        return None

    rho_e = material_properties[column].detach().cpu().numpy().astype(float)
    mass = float(np.sum(x_density * rho_e) * elem_volume)
    return mass


# ============================================================================
# RUN I/O + PER-ITERATION DIAGNOSTICS + POST-PROCESSING
# Extracted from the driver (optimize/optimization_loop.py) so it stays optimization-only.
# History is recorded in all modes; `verbose` gates console output only.
# ============================================================================

def init_run_history() -> dict[str, list[float | np.ndarray]]:
    """Return a fresh history dict for a topopt run; the loop adds each penalty's keys (Penalty.keys)."""
    return {
        "objective": [],
        "constraints": [],
        "J_phys": [],
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


def log_design_state(history: dict[str, list[float | np.ndarray]], iteration: int, x_phys: np.ndarray, z: np.ndarray,
                     material_properties: dict[str, torch.Tensor], zTorch: torch.Tensor,
                     matEncoder: "MaterialEncoder", num_elems: int, beta_proj: float, eta_proj: float,
                     verbose: bool = True) -> None:
    """Record the grey fraction (appends to history["grey"]); in verbose mode also print the iteration header,
    grey %, and the [MAT] active-material statistics. x_phys is (N,); z and zTorch are element-major
    (N, latentDim), z unused.

    The [MAT] line gives min/mean/max over the solid elements of each decoded column whose
    attribute has an iteration_label (materials/databases.py: Yield strength, Density), in the order
    of their rank; a database without them prints only the material counts."""
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

    from lsr_mmto.materials.databases import iteration_columns
    parts = []
    for column, label in iteration_columns(material_properties):
        v = _property_to_numpy(material_properties, column)
        parts.append(f"{label}_active min/mean/max={v[act].min():.3g}/{v[act].mean():.3g}/{v[act].max():.3g}")
    parts.append(f"counts={counts.tolist()}")
    print("[MAT] " + " | ".join(parts))


def constraints_over_limit(history: dict[str, list[float | np.ndarray]], to_params: "MMTOParams",
                           material_labels: list[str] | None = None) -> list[str]:
    """The constraints of the last evaluated design that are over their limit, as lines of text
    ("Mass: 3.2, 6.4 times its limit of 0.5"); [] when none is, or nothing was evaluated."""
    from lsr_mmto.formulation.quantities import quantity_label
    if not history.get("constraints"):
        return []
    lines = []
    for (name, options, bound), g in zip(to_params.Constraints, np.asarray(history["constraints"][-1]).flatten()):
        if not np.isfinite(g) or g > 0:
            label = quantity_label(name, options, material_labels)
            value = (g + 1) * bound
            times = f"{value / bound:.3g} times its limit" if np.isfinite(g) and bound else "not a number"
            lines.append(f"{label}: {value:.4g}, {times} of {bound:.4g}")
    return lines


def log_objective_and_constraints(history: dict[str, list[float | np.ndarray]], obj: float, J_phys: float, Jn: float,
                                  penalties: dict[str, float], cons: np.ndarray, to_params: "MMTOParams",
                                  verbose: bool = True, material_labels: list[str] | None = None) -> None:
    """Record objective/constraint history (appends to objective, constraints, J_phys and each penalty key);
    penalties is {history key: value} of the penalties (add_penalties in the loop). In verbose mode print the
    full Obj line + per-constraint lines; in simple mode print one compact per-iteration line. Each constraint
    is named as the app names it."""
    from lsr_mmto.formulation.quantities import quantity_label
    cons_flat = np.asarray(cons).flatten()
    constraint_names = [quantity_label(c[0], c[1], material_labels) for c in to_params.Constraints]

    history["objective"].append(float(obj))
    history["constraints"].append(cons_flat.copy())
    history["J_phys"].append(J_phys)
    for key, value in penalties.items():
        history[key].append(value)

    if verbose:
        print(
            f"Obj: J={J_phys:.4g} | Jn={Jn:.4g} | "
            + "".join(f"{key}={value:.3g} | " for key, value in penalties.items())
            + f"f={float(obj):.4g}"
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

