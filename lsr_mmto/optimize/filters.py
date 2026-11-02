"""
filters.py
==========
The density filter and the material (latent) filter of a run.
"""
from dataclasses import replace


def create_density_and_material_filters(fe_solver_structural, to_params, createFilters):
    """
    Build two filters:
      - density filter: uses to_params.RelativeFilterRadius
      - material filter: uses to_params.MaterialFilterRadius if provided,
                         otherwise falls back to density filter
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
