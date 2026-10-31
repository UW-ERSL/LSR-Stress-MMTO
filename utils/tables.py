"""
tables.py
=========
Tables for one run and for the material database behind it.

summary(run)              setup, result and constraint utilisation
materials(encoder)        the candidate materials, in the colours they are drawn in
pca(encoder)              principal components, d_eff and the latent dimension
encoding_errors(encoder)  how faithfully the decoder reproduces each material
"""
from __future__ import annotations


import numpy as np



def label_of(run):
    from .records import label
    return label(run)


def summary(run, show=True):
    """One run at a glance: setup, result, constraint utilisation."""
    names = {"mass_cap": "mass budget", "cost_cap": "cost budget",
             "r_rho": "r_rho", "r_z": "r_z", "nx": "grid", "n_dof": "target DOF",
             "material": "material", "P": "P", "fixed_radius": "fixed radius"}
    rows = [("example", run.example), ("run", run.descriptor)]
    units = (run.config or {}).get("units") or {}
    if units.get("system") not in (None, "none"):
        rows.append(("units", f"{units['system']}: {units['length']}, {units['force']}, "
                              f"{units['stress']}, {units['mass']}"))
    for k, v in run.params.items():
        if k == "params_from":
            continue
        lbl = names.get(k, k.replace("_", " "))
        if k == "nx":
            rows.append((lbl, f"{int(v)} x {int(v)}"))
        elif k in ("mass_cap", "cost_cap"):
            rows.append((lbl, f"{v:.3g}"))
        else:
            rows.append((lbl, f"{v:g}" if isinstance(v, float) else str(v)))
    rows.append(("iterations", str(run.meta.get("iterations", "?"))))
    rows.append(("elements", f"{run.n_elems}"))
    rows.append(("solid elements",
                 f"{int(run.solid.sum())}  ({100 * run.solid.mean():.1f}%)"))
    rows.append(("latent dimension", str(run.n_latent)))
    rows.append(("materials available", str(len(run.material_names))))
    rows.append(("materials used", str(int(run.materials_used().size))))
    rows.append(("objective, Phi_p", f"{run.phi_p:.4f}"))
    if np.isfinite(run.max_ff):
        rows.append(("max failure factor", f"{run.max_ff:.4f}"))
    rows.append(("grayness", f"{100 * run.grey:.2f}%"))
    pretty = {"MASS": "mass", "COST": "cost",
              "VOLUME_FRACTION": "volume fraction", "COMPLIANCE": "compliance"}
    for i, spec in enumerate(run.constraint_spec):
        name = pretty.get(str(spec[0]), str(spec[0]).lower().replace("_", " "))
        aux = spec[1] if len(spec) > 1 else None
        if isinstance(aux, dict) and "material_id" in aux:
            name += f", material {int(aux['material_id']) + 1}"
        rows.append((f"{name}, fraction of budget",
                     f"{run.utilisation(i):.4f}"))

    from .display import Table
    return Table([[k, v] for k, v in rows],
                 headers=["quantity", "value"],
                 title=f"{label_of(run)}")


# --------------------------------------------------------------------------- #
# the database, its PCA, and how well the network encodes it                   #
# --------------------------------------------------------------------------- #

def materials(mat_encoder, example=None):
    """The candidate materials, with the colour each is drawn in.

    materialAttributes holds metadata per attribute (column index, unit,
    scaling); the values themselves are the columns of rawData.
    """
    from .display import Table, swatch
    from .render import material_colours, material_labels

    names = material_labels(list(mat_encoder.materialNames), example)
    attrs = mat_encoder.materialAttributes
    raw = np.asarray(mat_encoder.rawData, float)
    colours = material_colours(len(names), example)

    headers = ["", "material"]
    for name, meta in attrs.items():
        unit = meta.get("unit")
        unit = "" if unit is None else str(unit).strip()
        if unit.lower() in ("", "nan", "none", "-"):
            headers.append(name)
        else:
            headers.append(f"{name} [{unit}]")

    rows = []
    for i, name in enumerate(names):
        row = [swatch(colours[i]), str(name)]
        for meta in attrs.values():
            row.append(f"{raw[i, int(meta['idx'])]:.4g}")
        rows.append(row)
    return Table(rows, headers=headers,
                 title=f"Candidate materials ({len(names)})",
                 symbol_cols=(1,))


def pca(mat_encoder):
    """Explained variance per principal component, and the latent dimension.

    d_eff is the participation ratio of the PCA eigenvalues; the latent
    dimension is the smallest integer at or above it.
    """
    from .display import Note, Table, Tables

    ratios = np.asarray(mat_encoder.pca_explained_variance_ratio, float)
    eigs = np.asarray(mat_encoder.pca_explained_variance, float)
    d_eff = float(mat_encoder.intrinsic_dim)
    n_z = int(mat_encoder.vae_params.latentDim)

    cum = np.cumsum(ratios)
    t1 = Table([[f"{i + 1}", f"{eigs[i]:.4g}", f"{100 * ratios[i]:.2f}",
                 f"{100 * cum[i]:.2f}"] for i in range(ratios.size)],
               headers=["component", "eigenvalue", "explained variance %",
                        "cumulative %"],
               title="Principal components of the material database")

    t2 = Table([["effective dimension, d_eff", f"{d_eff:.4f}"],
                ["latent dimension, n_z", f"{n_z}"]],
               headers=["quantity", "value"])

    n_attr = ratios.size
    lead = Note(
        f"The database has {n_attr} attributes, but its effective dimension is "
        f"d_eff = {d_eff:.2f}: the first principal component alone carries "
        f"{100 * ratios[0]:.1f}% of the variance. A latent space of "
        f"n_z = {n_z} is enough to represent it.")
    return Tables(lead, t1, t2)


def encoding_errors(mat_encoder):
    """How faithfully the network reproduces each real material.

    Each material is encoded to its latent point and decoded back; the error is
    reported per attribute, over the normalized attribute values.
    """
    import torch
    from .display import Table

    with torch.no_grad():
        z_real = mat_encoder.vaeNet.encoder(mat_encoder.scaledMaterialData)
        decoded = mat_encoder.vaeNet.decoder(z_real)
        diff = (mat_encoder.scaledMaterialData - decoded).abs().cpu().numpy()

    names = list(mat_encoder.materialAttributes.keys())
    rows = []
    for col, name in enumerate(names):
        err = 100.0 * diff[:, col]
        rows.append([name, f"{np.median(err):.4f}", f"{err.max():.4f}"])
    return Table(rows, headers=["attribute", "median error %", "max error %"],
                 title="Reconstruction error of the real materials")
