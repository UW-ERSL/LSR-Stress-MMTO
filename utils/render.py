"""
render.py
=========
Raw element-level rendering.

draw_panel(ax, run, field)   2D: one matplotlib quad per element
voxel_surface(run)           3D: whole cells thresholded, then extract_surface
screenshot(run, path)        3D: off-screen render to PNG
mesh_for(run)                the run's voxel mesh, rebuilt and cached

Colour comes from the run's saved material_indices. Mesh rebuilding needs PyTO;
nothing else here does.
"""
from __future__ import annotations

import os

import numpy as np

from .records import Run


class RenderError(RuntimeError):
    pass


# Cached on (example, nDOFDesired). Both parts matter: the U-bracket appears at
# five DOF targets in the mesh ladder.
_MESH_CACHE: dict = {}


REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Runs made before the examples became project files carry these names.
OLD_NAMES = {
    "LBracketTopLoad_Stress_BM1": "L-bracket",
    "UBracket_Stress_FF":         "U-bracket",
    "Beam_Liao_Pareto_FF":        "beam",
    "AirplaneBracket_Stress":     "Airplane bracket",
}


def project_name(example):
    """The project name of an example, for runs made under its old name too."""
    return OLD_NAMES.get(example, example)


def _problem(run: Run):
    """The Project a run was made from."""
    try:
        from project import Project  # type: ignore
    except ImportError as e:
        raise RenderError(
            "could not import project, which needs PyTO on sys.path. Call "
            "project_paths.bootstrap() first, and check that PyTO is cloned beside "
            f"this repository. Original error: {e}") from e
    spec = ((run.config or {}).get("arguments_resolved") or {}).get("to_problem")
    if isinstance(spec, dict) and "project" in spec:
        return Project.from_record(spec)
    if isinstance(spec, dict) and "pyto" in spec:          # made in the GUI, before projects
        p = Project.load(spec["pyto"])
        p.n_dof, p.single_layer = int(spec.get("n_dof", 30000)), bool(spec.get("single_layer", False))
        return p
    path = os.path.join(REPO, "projects", f"{project_name(run.example)}.mmto")
    if not os.path.isfile(path):
        raise RenderError(f"{run.descriptor}: no project file for example {run.example!r} ({path})")
    return Project.load(path)


def mesh_for(run: Run):
    """The voxel mesh this run used, rebuilt and cached.

    nDOFDesired comes from the run's own config, so the rebuild matches the grid
    the run solved on rather than the example's current default.
    """
    ndof = (run.config.get("topopt_params") or {}).get("nDOFDesired")
    key = (run.example, ndof)

    if key not in _MESH_CACHE:
        from pyto_project import make_mesh  # type: ignore
        p = _problem(run)
        _MESH_CACHE[key] = make_mesh(p.stl_path(), p.n_dof if ndof is None else int(ndof), p.single_layer)

    mesh = _MESH_CACHE[key]
    if int(mesh.num_elems) != run.n_elems:
        raise RenderError(
            f"{run.descriptor}: rebuilt mesh has {int(mesh.num_elems)} elements, "
            f"record has {run.n_elems}. The project or the STL has "
            f"changed since the run"
            + (f" (nDOFDesired={ndof})." if ndof is not None
               else ", and the config carries no nDOFDesired to rebuild with."))
    return mesh


def material_colours(n_needed: int, example: str = None) -> list:
    """Palette as hex strings, at least n_needed long.

    An example with an entry in EXAMPLE_PALETTE uses that; everything else uses
    materialColors.material_colors.
    """
    import matplotlib.colors as mcolors
    palette = EXAMPLE_PALETTE.get(project_name(example))
    if palette is None:
        try:
            from materialColors import material_colors  # type: ignore
        except ImportError as e:
            raise RenderError(
                "could not import materialColors from 0-Common. Call "
                f"project_paths.bootstrap() first. Original error: {e}") from e
        palette = material_colors
    return [mcolors.to_hex(palette[i % len(palette)])
            for i in range(max(n_needed, 1))]


# --------------------------------------------------------------------------- #
# 2D
# --------------------------------------------------------------------------- #

def build_quads(mesh):
    """(n_elems, 4, 2) corners, one axis-aligned quad per element."""
    c = np.asarray(mesh.elem_centers, float)
    dx, dy = float(mesh.elem_size[0]), float(mesh.elem_size[1])
    hx, hy = dx / 2.0, dy / 2.0
    x, y = c[:, 0], c[:, 1]
    return np.stack([
        np.stack([x - hx, y - hy], axis=1),
        np.stack([x + hx, y - hy], axis=1),
        np.stack([x + hx, y + hy], axis=1),
        np.stack([x - hx, y + hy], axis=1),
    ], axis=1)


def draw_panel(ax, run: Run, field: str = "material", *,
               mesh=None, quads=None, edges: bool = False,
               cmap: str = "viridis", vmin=None, vmax=None, colours=None):
    """Draw one run as raw element quads.

    field: material | topology | stress   solid elements only
           density                        every element, shaded by x_cont

    Returns the PolyCollection so a caller can attach a colourbar.
    """
    from matplotlib.collections import PolyCollection

    if mesh is None:
        mesh = mesh_for(run)
    if quads is None:
        quads = build_quads(mesh)
    if quads.shape[0] != run.n_elems:
        raise RenderError(
            f"{run.descriptor}: {quads.shape[0]} quads for {run.n_elems} elements")

    solid = run.solid
    edge_kw = dict(edgecolors=("#404040" if edges else "none"),
                   linewidths=(0.15 if edges else 0.0))

    if field == "material":
        if colours is None:
            colours = material_colours(int(run.material_indices.max()) + 1,
                                       run.example)
        pc = PolyCollection(quads[solid],
                            facecolors=[colours[i] for i in run.material_indices[solid]],
                            **edge_kw)
    elif field == "topology":
        pc = PolyCollection(quads[solid], facecolors="#4a4a4a", **edge_kw)
    elif field == "stress":
        vals = run.von_mises[solid]
        pc = PolyCollection(quads[solid], array=vals, cmap=cmap, **edge_kw)
        pc.set_clim(vmin if vmin is not None else float(vals.min()),
                    vmax if vmax is not None else float(vals.max()))
    elif field == "density":
        pc = PolyCollection(quads, array=run.x_cont, cmap="Greys", **edge_kw)
        pc.set_clim(0.0, 1.0)
    else:
        raise RenderError(f"unknown field {field!r}; expected material, "
                          "topology, stress or density")

    ax.add_collection(pc)
    c = np.asarray(mesh.elem_centers, float)
    dx, dy = float(mesh.elem_size[0]), float(mesh.elem_size[1])
    ax.set_xlim(c[:, 0].min() - 2 * dx, c[:, 0].max() + 2 * dx)
    ax.set_ylim(c[:, 1].min() - 2 * dy, c[:, 1].max() + 2 * dy)
    ax.set_aspect("equal")
    ax.axis("off")
    return pc


# --------------------------------------------------------------------------- #
# 3D
# --------------------------------------------------------------------------- #

# Axes to reflect about, giving 2**k copies of the solved domain.
MIRROR_AXES = {
    "Airplane bracket": (2,),         # half model, symmetry plane z = 0
    "beam":             (0, 1, 2),    # octant model
}

# Per-example palettes. materialColors.py carries one active list plus
# commented alternatives; the L-bracket benchmark uses the three-colour set its
# material names refer to.
EXAMPLE_PALETTE = {
    "L-bracket": ["#FF0000", "#4b00f1", "#23FF24"],
}

# Databases whose spreadsheet names are internal codes. The paper labels these
# A, B, C..., and so does everything here.
LETTER_LABELS = {"Airplane bracket"}


def material_labels(names, example=None) -> list:
    """Display names for the materials: letters where the paper uses letters."""
    if project_name(example) in LETTER_LABELS:
        return [chr(ord("A") + i) for i in range(len(names))]
    return [str(n) for n in names]


def _reflect_centroids(c, mat, d, axes):
    """Mirror centroids about the minimum plane of each axis.

    The symmetry plane sits half an element outside the first centroid layer,
    at min(centre) - d/2, so the reflection lands with no overlapping layer.
    Follows beam_pareto_parts.py and postprocess_mirror_bracket_rawvoxel.py.
    """
    C, M = c, mat
    for ax in axes:
        plane = C[:, ax].min() - d[ax] / 2.0
        C_ref = C.copy()
        C_ref[:, ax] = 2.0 * plane - C_ref[:, ax]
        C = np.vstack([C, C_ref])
        M = np.concatenate([M, M])
    return C, M


def voxel_surface(run: Run, *, mesh=None, mirror: bool = True):
    """Raw voxel surface of the design. Returns (surf, n_solid, materials)."""
    try:
        import pyvista as pv
    except ImportError as e:
        raise RenderError(
            "3D rendering needs pyvista (pip install pyvista). On a headless "
            "Linux machine VTK falls back to software rendering; if it cannot, "
            "run under xvfb-run.") from e
    import matplotlib.colors as mcolors

    if mesh is None:
        mesh = mesh_for(run)

    centers = np.asarray(mesh.elem_centers, float)
    dx, dy, dz = (float(s) for s in mesh.elem_size)

    solid = run.solid
    n_half = int(solid.sum())
    axes = MIRROR_AXES.get(project_name(run.example), ()) if mirror else ()
    C, M = _reflect_centroids(centers[solid], run.material_indices[solid],
                              (dx, dy, dz), axes)

    def lattice(coord, step):
        o = coord.min()
        return np.rint((coord - o) / step).astype(int), o

    ix, ox = lattice(C[:, 0], dx)
    iy, oy = lattice(C[:, 1], dy)
    iz, oz = lattice(C[:, 2], dz)
    nx, ny, nz = ix.max() + 1, iy.max() + 1, iz.max() + 1

    grid = np.full((nx, ny, nz), -1, int)
    grid[ix, iy, iz] = M

    ImageData = getattr(pv, "ImageData", None) or pv.UniformGrid
    img = ImageData(dimensions=(nx + 1, ny + 1, nz + 1), spacing=(dx, dy, dz),
                    origin=(ox - dx / 2, oy - dy / 2, oz - dz / 2))

    flat = grid.flatten(order="F")
    img.cell_data["material"] = flat

    colours = material_colours(int(run.material_indices.max()) + 1, run.example)
    rgb = np.full((flat.size, 3), 255, np.uint8)
    for v in np.unique(flat[flat >= 0]):
        rgb[flat == v] = (np.array(mcolors.to_rgb(colours[int(v)])) * 255).astype(np.uint8)
    img.cell_data["rgb"] = rgb

    solid_blocks = img.threshold(-0.5, scalars="material")
    try:
        # newer pyvista warns that the default changes in a future release
        surf = solid_blocks.extract_surface(algorithm="dataset_surface")
    except TypeError:
        # older pyvista has no such keyword; its default is dataset_surface
        surf = solid_blocks.extract_surface()
    return surf, n_half, M


def screenshot(run: Run, png_path: str, *, mesh=None, mirror: bool = True,
               camera=None, view: str = "iso", azimuth=0.0, elevation=0.0,
               roll=0.0, zoom=1.0, size=(2000, 1600),
               edges: bool = False, transparent: bool = False):
    """Render one 3D run off-screen to a PNG.

    view: iso | xy | xz | yz | -xy | -xz | -yz, used when camera is None.
    azimuth, elevation, roll: degrees applied after the view, for tuning an
    angle by eye. zoom > 1 fills more of the frame.
    camera: a pyvista camera_position triple, which overrides view and the
    angles. Pass one to share a camera across a row of panels.
    """
    import pyvista as pv

    surf, n_half, M = voxel_surface(run, mesh=mesh, mirror=mirror)

    p = pv.Plotter(off_screen=True, window_size=size)
    p.set_background("white")
    p.add_mesh(surf, scalars="rgb", rgb=True, show_scalar_bar=False,
               smooth_shading=False, specular=0.3,
               show_edges=edges, edge_color="#404040", line_width=0.3)
    p.enable_anti_aliasing()
    if camera is not None:
        p.camera_position = camera      # after add_mesh, so no reset_camera
    else:
        _set_view(p, view)
        _rotate(p, azimuth, elevation, roll, zoom)
    os.makedirs(os.path.dirname(os.path.abspath(png_path)), exist_ok=True)
    p.screenshot(png_path, transparent_background=transparent)
    p.close()
    return png_path, n_half, M


def _rotate(plotter, azimuth=0.0, elevation=0.0, roll=0.0, zoom=1.0):
    """Nudge the camera off a named view, in degrees."""
    if azimuth:
        plotter.camera.azimuth += float(azimuth)
    if elevation:
        plotter.camera.elevation += float(elevation)
    if roll:
        plotter.camera.roll += float(roll)
    if zoom and zoom != 1.0:
        plotter.camera.zoom(float(zoom))


def _set_view(plotter, view: str):
    """Apply one of pyvista's built-in cameras by name."""
    key = view.lstrip("-")
    if key not in ("iso", "xy", "xz", "yz"):
        raise RenderError(
            f"unknown view {view!r}; expected iso, xy, xz, yz, or one of those "
            "prefixed with '-' for the opposite side")
    if key == "iso":
        plotter.view_isometric()
    else:
        getattr(plotter, f"view_{key}")(negative=view.startswith("-"))
