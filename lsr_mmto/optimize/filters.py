"""
filters.py
==========
The density filter and the material (latent) filter of a run.
"""
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    import numpy as np
    from scipy.sparse import spmatrix

    from lsr_mmto.optimize.mmto_to_params import MMTOParams


def create_density_and_material_filters(
        fe_solver_structural: Any, to_params: "MMTOParams",
        createFilters: "Callable[[Any, MMTOParams], tuple[spmatrix, np.ndarray]]",
) -> "tuple[spmatrix, np.ndarray, spmatrix, np.ndarray]":
    """
    Build two filters:
      - density filter: uses to_params.RelativeFilterRadius
      - material filter: uses to_params.MaterialFilterRadius if provided,
                         otherwise falls back to density filter
    Returns (H_density, Hs_density, H_material, Hs_material): each sparse matrix with its row sums. With equal
    radii the material filter is the same objects as the density filter. createFilters is PyTO-AD's
    (pyto.topopt.common), given the physics' solver (any physics, despite the name) and to_params.
    """
    H_density, Hs_density = createFilters(fe_solver_structural, to_params)

    mat_radius = (
        to_params.RelativeFilterRadius
        if to_params.MaterialFilterRadius is None
        else to_params.MaterialFilterRadius
    )

    if abs(mat_radius - to_params.RelativeFilterRadius) < 1e-15:
        return H_density, Hs_density, H_density, Hs_density

    to_params_mat = replace(to_params, RelativeFilterRadius=mat_radius)
    H_material, Hs_material = createFilters(fe_solver_structural, to_params_mat)

    return H_density, Hs_density, H_material, Hs_material
