"""
figures.py
==========
Figures for one optimization run.

design                the optimized design: a 2D panel or a 3D voxel render
latent_scatter        the latent field against the real-material anchors
convergence           the objective and constraints over the iterations
boundary_conditions   the loads and supports on the meshed design domain

Panels are raw: one quad per element in 2D, one voxel face per element in 3D.
"""
from __future__ import annotations

import os

import numpy as np

from .records import Run
from .render import (MIRROR_AXES, RenderError, draw_panel, project_name,
                     material_colours, material_labels, mesh_for, screenshot,
                     voxel_surface)

FS_LEGEND = 11
DPI = 400


def _rc():
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.linewidth": 0.6,
                         "mathtext.fontset": "cm"})


def _save(fig, out, *, dpi=DPI):
    """Write PNG and PDF. out is a path without extension; None returns fig.

    A failure on one format is reported and does not lose the other, which
    matters on Windows where an open viewer locks the file.
    """
    import matplotlib.pyplot as plt
    if out is None:
        return fig
    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    written, failed = [], []
    for ext in ("png", "pdf"):
        p = f"{out}.{ext}"
        try:
            fig.savefig(p, dpi=dpi, bbox_inches="tight", facecolor="white")
            written.append(p)
        except OSError as e:
            failed.append(f"{os.path.basename(p)}: {e}")
    plt.close(fig)
    if failed:
        print("    could not write " + "; ".join(failed)
              + " (a file open in a viewer will do this)")
    if not written:
        raise RenderError(f"no output written for {out}: " + "; ".join(failed))
    return written


def _legend_beside(ax, run: Run):
    """Material legend to the right of a single panel, clear of the design."""
    from matplotlib.patches import Patch
    names = material_labels(run.material_names, run.example)
    colours = material_colours(len(names), run.example)
    used = sorted(run.materials_used().tolist())

    available = set((run.config.get("materials") or {})
                    .get("real_material_properties", {}))
    for key, sym in (("Yield_Strength", r"\sigma_Y"), ("Youngs_Modulus", "E")):
        if key in available:
            vals, symbol = run.material_property(key), sym
            break
    else:
        vals, symbol = None, None

    handles = []
    for i in used:
        base = names[i] if i < len(names) else f"material {i}"
        if vals is not None and i < vals.size:
            base += f"  (${symbol}$={vals[i]:g})"
        handles.append(Patch(fc=colours[i], ec="0.3", lw=0.5, label=base))
    if handles:
        ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.01, 0.5),
                  frameon=False, fontsize=FS_LEGEND - 1)


def _panel_image(run, mesh, camera, size, png_path, view="iso", angles=(0, 0, 0, 1)):
    """One 3D panel as an RGB array, optionally also written to disk."""
    import pyvista as pv
    from .render import _set_view, _rotate
    az, el, roll, zoom = angles
    if png_path is not None:
        screenshot(run, png_path, mesh=mesh, camera=camera, view=view,
                   azimuth=az, elevation=el, roll=roll, zoom=zoom, size=size)
        import matplotlib.image as mpimg
        return mpimg.imread(png_path)

    surf, _, _ = voxel_surface(run, mesh=mesh)
    p = pv.Plotter(off_screen=True, window_size=size)
    p.set_background("white")
    p.add_mesh(surf, scalars="rgb", rgb=True, show_scalar_bar=False,
               smooth_shading=False, specular=0.3)
    p.enable_anti_aliasing()
    if camera is not None:
        p.camera_position = camera
    else:
        _set_view(p, view)
        _rotate(p, az, el, roll, zoom)
    img = p.screenshot(return_img=True)
    p.close()
    return img


_OBJECTIVE_LABEL = {
    "PNORM_STRESS": r"$p$-norm failure factor $\Phi_p$",
    "COMPLIANCE":   "Compliance",
    "MASS":         "Mass",
}


def _constraint_label(spec):
    """Display name for one constraint, e.g. VOLUME_FRACTION with a material_id
    becomes 'Volume fraction, material 1'."""
    name = str(spec[0])
    pretty = {"MASS": "Mass constraint", "COST": "Cost constraint",
              "VOLUME_FRACTION": "Volume fraction",
              "COMPLIANCE": "Compliance constraint"}.get(name, name + " constraint")
    aux = spec[1] if len(spec) > 1 else None
    if isinstance(aux, dict) and "material_id" in aux:
        pretty += f", material {int(aux['material_id']) + 1}"
    return pretty


def convergence(run: Run, out=None):
    """Objective and every constraint as a fraction of its budget.

    Ported from postprocess.py's plot_convergence, which produced the published
    Fig. 7 and Fig. 21. Objective on the left axis, constraints on the right,
    with a dashed line at 1.0 for the budget.
    """
    import matplotlib.pyplot as plt

    J = np.asarray(run.history.get("J_phys", []), dtype=float).ravel()
    if J.size == 0:
        J = np.asarray(run.history.get("objective", []), dtype=float).ravel()
    if J.size == 0:
        raise RenderError(f"{run.descriptor}: no objective history to plot")
    it = np.arange(1, J.size + 1)

    obj = (run.config.get("topopt_params") or {}).get("Objective")
    obj_label = _OBJECTIVE_LABEL.get(str(obj[0]) if obj else "",
                                     r"$\Phi_p$ (objective)")

    cons = np.asarray(run.history.get("constraints", []), dtype=float)
    cons = cons.reshape(cons.shape[0], -1) if cons.size else cons
    n_con = cons.shape[1] if cons.size else 0
    labels = [_constraint_label(c) for c in run.constraint_spec][:n_con]
    while len(labels) < n_con:
        labels.append(f"Constraint {len(labels) + 1}")

    plt.rcParams.update({
        "font.size": 11, "font.family": "serif",
        "axes.linewidth": 0.8, "axes.edgecolor": "#333333",
        "xtick.direction": "out", "ytick.direction": "out",
    })
    C_OBJ = "#2b3a67"
    CON_COLORS = ["#b2452e", "#2a7f62", "#7a5c99", "#b58b00", "#3a6ea5", "#a0522d"]

    fig, axL = plt.subplots(figsize=(6.4, 4.0))
    handles = [axL.plot(it, J, color=C_OBJ, lw=1.9, label=obj_label)[0]]
    axL.set_xlabel("Iteration")
    axL.set_ylabel(obj_label, color=C_OBJ)
    axL.tick_params(axis="y", colors=C_OBJ)
    axL.set_xlim(1, J.size)
    axL.grid(True, color="#cccccc", lw=0.5, alpha=0.6)
    axL.set_axisbelow(True)

    if n_con:
        axR = axL.twinx()
        for i in range(n_con):
            frac = cons[:, i] + 1.0        # stored as value/budget - 1
            handles.append(axR.plot(np.arange(1, frac.size + 1), frac,
                                    color=CON_COLORS[i % len(CON_COLORS)], lw=1.6,
                                    label=labels[i])[0])
        handles.append(axR.axhline(1.0, color="#999999", lw=1.0, ls="--",
                                   label="Budget"))
        axR.set_ylabel("Constraint value (fraction of budget)")

    axL.legend(handles=handles, loc="upper right", frameon=True, framealpha=0.95,
               edgecolor="#cccccc", fontsize=9.5)
    fig.tight_layout()
    return _save(fig, out)


def latent_scatter(run: Run, out=None, *, max_points=20000, labels="letters"):
    """The converged latent field against the real-material anchors.

    Follows materialEncoder.plotLSR: red circles for the optimized elements,
    black stars for the anchors. A 3D axes when the latent space is 3D, a plane
    when it is 2D, and a PCA projection above that.

    labels: "letters" relabels the materials A, B, C..., as plot_latent did;
    "names" keeps the database names.
    """
    import matplotlib.pyplot as plt
    _rc()

    z = run.z_solid()
    if z.shape[0] > max_points:
        sel = np.random.default_rng(0).choice(z.shape[0], max_points, replace=False)
        z = z[sel]
    zr = np.asarray(run.z_real, float)
    n_z = zr.shape[1]

    if labels == "letters":
        names = material_labels(run.material_names, run.example)
    else:
        names = [run.material_names[i] if i < len(run.material_names) else str(i)
                 for i in range(zr.shape[0])]
    while len(names) < zr.shape[0]:
        names.append(str(len(names)))

    if n_z > 3:
        from sklearn.decomposition import PCA
        p = PCA(n_components=2).fit(zr)
        zr, z, n_z = p.transform(zr), p.transform(z), 2
        title = f"latent space, PCA projection from {run.n_latent}D"
    else:
        title = f"latent space ({n_z}D)"

    if n_z == 3:
        fig = plt.figure(figsize=(6.4, 5.4))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(z[:, 0], z[:, 1], z[:, 2], c="red", marker="o", s=20,
                   alpha=0.6, label="Optimized materials")
        ax.scatter(zr[:, 0], zr[:, 1], zr[:, 2], c="black", marker="*", s=200,
                   alpha=0.6, label="Real materials")
        for i, nm in enumerate(names):
            ax.text(zr[i, 0] + 0.1, zr[i, 1], zr[i, 2], nm, fontsize=10)
        ax.set_xlabel("$z_1$"); ax.set_ylabel("$z_2$"); ax.set_zlabel("$z_3$")
    else:
        fig, ax = plt.subplots(figsize=(6.0, 5.0))
        ax.scatter(z[:, 0], z[:, 1], c="red", marker="o", s=20, alpha=0.2,
                   label="Optimized materials")
        ax.scatter(zr[:, 0], zr[:, 1], c="black", marker="*", s=200, alpha=0.4,
                   label="Real materials")
        for i, nm in enumerate(names):
            ax.text(zr[i, 0] + 0.1, zr[i, 1], nm, fontsize=12, color="black",
                    ha="center", va="bottom")
        ax.set_xlabel("$z_1$"); ax.set_ylabel("$z_2$")
        ax.grid(True)

    ax.set_title(title, fontsize=11)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    return _save(fig, out)


# --------------------------------------------------------------------------- #
# single-run views                                                            #
# --------------------------------------------------------------------------- #


def is_3d(run: Run) -> bool:
    """True for the examples rendered as voxels rather than 2D panels."""
    return project_name(run.example) in MIRROR_AXES


def design(run: Run, out=None, *, field="material", size=(1200, 950),
           view="iso", **kw):
    """The optimized design, drawn the way this example wants.

    2D examples get a matplotlib panel; 3D examples get a mirrored voxel render
    placed on an axes, so both return a figure and display the same way.
    field applies to the 2D path only: material | topology | stress | density.
    """
    import matplotlib.pyplot as plt
    _rc()

    if is_3d(run):
        if field != "material":
            raise RenderError(
                f"{run.descriptor} is a 3D example; only the material view is "
                f"rendered, not {field!r}")
        img = _panel_image(run, mesh_for(run), None, size, None,
                           view=view, **kw)
        fig, ax = plt.subplots(figsize=(size[0] / 190.0, size[1] / 190.0))
        ax.imshow(img)
        ax.axis("off")
        _legend_beside(ax, run)
        fig.tight_layout()
        return _save(fig, out)

    fig, ax = plt.subplots(figsize=(6.6, 5.0))
    pc = draw_panel(ax, run, field, mesh=mesh_for(run))
    if field in ("stress", "density"):
        cb = fig.colorbar(pc, ax=ax, shrink=0.8, pad=0.02)
        cb.set_label("von Mises stress" if field == "stress" else "density")
    elif field == "material":
        _legend_beside(ax, run)
    fig.tight_layout()
    return _save(fig, out)


def boundary_conditions(problem, out=None, *, size=(1400, 1050), zoom=0.82):
    """Loads and supports of a Project on its meshed design domain, drawn by PyTO.

    A plotter is supplied so the render is full size rather than PyTO's default
    400x300, and so the camera can be widened afterwards: its "iso" branch fits
    the view and then applies zoom(1.25), which crops the load arrow and the
    clamped face. zoom < 1 here undoes that.
    """
    from PyTOImports import hex_structural_fea, linear_solvers, deflation  # type: ignore
    import pyvista as pv

    mesh, mat_prop, bc, body_force, to_params, _ = problem.build()

    # plot_mesh never solves, so the solver choice does not matter here
    fe = hex_structural_fea.HexStructuralFEA(
        mesh=mesh, mat_prop=mat_prop, bc=bc,
        solver=linear_solvers.Solvers.PARDISO,
        dsolver=deflation.DeflationSolver(),
        rtol=1e-8, elem_body_force=body_force)

    p = pv.Plotter(off_screen=out is not None, window_size=list(size))
    p.enable_anti_aliasing()
    fe.plot_mesh(title="Loads and boundary conditions", plot_bc=True,
                 rel_arrow_scale=0.1, offsetArrow=True,
                 camera_position="iso", plotter=p)
    p.camera.zoom(zoom)

    if out is None:
        p.show()
        return None
    png = out if out.endswith(".png") else out + ".png"
    os.makedirs(os.path.dirname(os.path.abspath(png)) or ".", exist_ok=True)
    p.screenshot(png)
    p.close()
    return png
