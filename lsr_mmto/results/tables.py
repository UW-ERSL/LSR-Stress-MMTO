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
    from .load_results import label
    return label(run)


def summary(run, show=True):
    """One run at a glance: setup, result, constraint utilisation."""
    rows = [("example", run.example), ("run", run.descriptor)]
    units = (run.config or {}).get("units") or {}
    if units.get("system") not in (None, "none"):
        rows.append(("units", f"{units['system']}: {units['length']}, {units['force']}, "
                              f"{units['stress']}, {units['mass']}"))
    rows.append(("iterations", str(run.iterations)))     # the starting design is not counted
    rows.append(("elements", f"{run.n_elems}"))
    rows.append(("solid elements",
                 f"{int(run.solid.sum())}  ({100 * run.solid.mean():.1f}%)"))
    rows.append(("latent dimension", str(run.n_latent)))
    rows.append(("materials available", str(len(run.material_names))))
    rows.append(("materials used", str(int(run.materials_used().size))))
    rows.append((f"objective: {run.objective_label}", f"{run.final_objective:.4g}"))
    if np.isfinite(run.max_ff):
        rows.append(("max failure factor", f"{run.max_ff:.4f}"))
    rows.append(("grayness", f"{100 * run.grey:.2f}%"))
    for i in range(len(run.constraint_spec)):
        rows.append((f"{run.constraint_label(i)}, fraction of budget",
                     f"{run.utilisation(i):.4f}"))

    from .display import Table
    return Table([[k, v] for k, v in rows],
                 headers=["quantity", "value"],
                 title=f"{label_of(run)}")


# --------------------------------------------------------------------------- #
# the database, its PCA, and how well the network encodes it                   #
# --------------------------------------------------------------------------- #

def materials(mat_encoder):
    """The candidate materials, with the colour and label each is shown with (the
    database's Display sheet).

    materialAttributes holds metadata per attribute (column index, unit,
    scaling); the values themselves are the columns of rawData.
    """
    from .display import Table, swatch
    from lsr_mmto.materials.databases import material_display

    colours, names = material_display(mat_encoder.excel_file, list(mat_encoder.materialNames))
    attrs = mat_encoder.materialAttributes
    raw = np.asarray(mat_encoder.rawData, float)

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


def encoding_errors(mat_encoder, per_material=False):
    """How faithfully the network reproduces each real material.

    Each material is encoded to its latent point and decoded back. Per property: the median and
    the largest error over the materials, in % of the property's range in the database, and the
    largest difference in the property's own unit. per_material=True: one row per material and
    property, with the database's value and the network's.
    """
    from .display import Table
    from lsr_mmto.materials.databases import SHOWN
    from lsr_mmto.materials.encoder import largest_errors

    rows = mat_encoder.reconstruction()
    if per_material:
        return Table([[r["material"], SHOWN.get(r["property"], r["property"]), r["unit"] or "",
                       f"{r['database']:.6g}", f"{r['network']:.6g}", f"{r['percent_of_range']:.4f}"]
                      for r in rows],
                     headers=["material", "property", "unit", "database", "network", "error [% of range]"],
                     title="Each real material as the network gives it back")
    table = []
    for name, (pct, diff, unit) in largest_errors(rows).items():
        errs = [r["percent_of_range"] for r in rows if r["property"] == name]
        table.append([SHOWN.get(name, name), f"{np.median(errs):.4f}", f"{pct:.4f}",
                      f"{diff:.6g} {unit or ''}".strip()])
    return Table(table, headers=["property", "median error [% of range]", "largest error [% of range]",
                                 "largest difference"],
                 title="Reconstruction error of the real materials")
