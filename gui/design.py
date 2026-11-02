"""
design.py
=========
An optimized design on the 3D view: one block per element, drawn in place of the part.

    live(window, mesh, x, material, colours)    while a run goes: density and material
    record(window, run, mesh, field, ...)       a finished run: Material, Density, Stress or Solid
    run_mesh(window, folder, run)               the mesh a saved run was made on

Every actor drawn here is named "result_...", so clearing a preview (actors named "mmto_...")
never removes a design. The window's Setup view and Design view (main_window.py) decide
whether a design or the part is shown.
"""
import os

import numpy as np

FIELDS = ("Material", "Density", "Stress", "Solid")
FAINT = 0.1                                   # elements below this density are empty: not drawn
WEAKEST = 0.25                                # colour strength of an element just above FAINT
SOLID_COLOUR = "#9aa5b1"
PREFIX = "result_"
_MESHES = {}                                  # (STL, dofs, single layer) -> mesh, for saved runs


def rgb(colour):
    colour = colour.lstrip("#")
    return np.array([int(colour[i:i + 2], 16) for i in (0, 2, 4)], dtype=float)


def strength(x):
    """Colour strength of each element from its density: WEAKEST just above FAINT, 1 when solid."""
    return WEAKEST + (1.0 - WEAKEST) * np.clip((np.asarray(x, dtype=float) - FAINT) / (1.0 - FAINT), 0.0, 1.0)


def faded(colours, s):
    """Each colour mixed with white: s = 1 is the colour itself, smaller is paler, never white."""
    s = np.asarray(s, dtype=float)[:, None]
    return np.round(255.0 - s * (255.0 - colours)).astype(np.uint8)


def _blocks(centres, h, data):
    """A cube at each centre; data (name -> one value per centre) carried onto every cube."""
    import pyvista as pv
    cloud = pv.PolyData(np.asarray(centres, dtype=float))
    for key, value in data.items():
        cloud.point_data[key] = value
    return cloud.glyph(geom=pv.Cube(x_length=h[0], y_length=h[1], z_length=h[2]), scale=False, orient=False)


def draw(window, mesh, mask, colours=None, values=None, title=None, clim=None, shift=0.0, name="design"):
    """The masked elements of mesh as blocks: colours (an rgb row per element of the mesh), or
    values (one per element) on a colour scale with title, or else SOLID_COLOUR. shift moves
    them along x."""
    mask = np.asarray(mask, dtype=bool)
    centres = np.asarray(mesh.elem_centers, dtype=float)[mask] + np.array([shift, 0.0, 0.0])
    if not len(centres):
        return None
    h = np.asarray(mesh.elem_size, dtype=float)
    kwargs = dict(pickable=False, name=PREFIX + name, reset_camera=False)
    if colours is not None:
        blocks = _blocks(centres, h, {"rgb": np.asarray(colours, dtype=np.uint8)[mask]})
        return window.plotter.add_mesh(blocks, scalars="rgb", rgb=True, **kwargs)
    if values is not None:
        blocks = _blocks(centres, h, {"value": np.asarray(values, dtype=float)[mask]})
        window.design_bars.add(title)
        return window.plotter.add_mesh(blocks, scalars="value", cmap="viridis", clim=clim,
                                       scalar_bar_args={"title": title, "vertical": False, "position_x": 0.3,
                                                        "position_y": 0.04, "width": 0.45, "height": 0.07,
                                                        "color": "#333333", "fmt": "%.3g"},
                                       **kwargs)
    return window.plotter.add_mesh(_blocks(centres, h, {}), color=SOLID_COLOUR, **kwargs)


def live(window, mesh, x, material, colours):
    """A design while it runs: elements at FAINT density or more, in their material's colour at a
    strength that follows their density (pale when half solid, full colour when solid)."""
    x = np.asarray(x, dtype=float).ravel()
    table = np.array([rgb(c) for c in colours])
    per_element = faded(table[np.asarray(material) % len(table)], strength(x))
    return draw(window, mesh, x >= FAINT, colours=per_element)


def stress_title(run):
    units = (run.config or {}).get("units") or {}
    return f"von Mises stress [{units['stress']}]" if units.get("stress") else "von Mises stress"


def field_range(runs, field):
    """One colour scale for several runs drawn together, so their colours compare."""
    if field == "Density":
        return (0.0, 1.0)
    if field == "Stress":
        top = max((float(np.max(r.von_mises[r.solid])) for r in runs if r.solid.any()), default=1.0)
        return (0.0, top if top > 0 else 1.0)
    return None


def record(window, run, mesh, field="Material", shift=0.0, clim=None, name="design"):
    """A finished run, drawn as field: Material (its solid elements in their material's colour),
    Density (every element at FAINT or more, on a scale), Stress (von Mises on its solid
    elements, on a scale) or Solid (its solid elements, one colour)."""
    if field == "Material":
        table = np.array([rgb(c) for c in run.material_colors]).astype(np.uint8)
        return draw(window, mesh, run.solid, colours=table[np.asarray(run.material_indices) % len(table)],
                    shift=shift, name=name)
    if field == "Density":
        return draw(window, mesh, np.asarray(run.x_cont) >= FAINT, values=run.x_cont, title="Density",
                    clim=clim or (0.0, 1.0), shift=shift, name=name)
    if field == "Stress":
        return draw(window, mesh, run.solid, values=run.von_mises, title=stress_title(run),
                    clim=clim or field_range([run], "Stress"), shift=shift, name=name)
    return draw(window, mesh, run.solid, shift=shift, name=name)


def smooth(window, pieces, apart=False):
    """STL files on the part, each in its colour: [(path, colour)]. apart moves each piece away from the
    middle by a quarter of its distance from it, so each can be seen alone."""
    import pyvista as pv
    meshes = [(pv.read(path), colour) for path, colour in pieces]
    middle = np.mean([m.center for m, _ in meshes], axis=0) if meshes else np.zeros(3)
    for k, (m, colour) in enumerate(meshes):
        if apart:
            m = m.translate(0.25 * (np.asarray(m.center) - middle), inplace=False)
        window.plotter.add_mesh(m, color=colour, smooth_shading=True, pickable=False, name=f"{PREFIX}smooth_{k}",
                                reset_camera=False)


class MeshError(RuntimeError):
    pass


def mesh_of(window, project):
    """The mesh a run of project uses: the window's own when it is the same one, else built once."""
    from lsr_mmto.problem.mesh import make_mesh
    key = (os.path.abspath(project.stl_path()), project.n_dof, project.single_layer)
    if key == getattr(window, "_mesh_key", None) and window._mesh is not None:
        return window._mesh
    if key not in _MESHES:
        _MESHES[key] = make_mesh(*key)
    return _MESHES[key]


def run_mesh(window, folder, run):
    """The mesh a saved run was made on, rebuilt from the copy of the project saved with it (the
    window's own mesh when it is the same one). MeshError when it cannot match the run."""
    from lsr_mmto.problem.project import Project
    from lsr_mmto.problem.project_folder import run_project_file
    path = run_project_file(folder)
    if path is None:
        raise MeshError("this run has no copy of its project beside it, so its part is not known")
    p = Project.load(path)
    if not os.path.isfile(p.stl_path()):
        raise MeshError(f"its part is missing: {p.stl_path()}")
    mesh = mesh_of(window, p)
    if int(mesh.num_elems) != int(run.n_elems):
        raise MeshError(f"its part has changed since the run: the mesh has {int(mesh.num_elems)} elements, "
                        f"the run {int(run.n_elems)}")
    return mesh


def width(mesh):
    """The extent of the mesh along x, for drawing a second design beside it."""
    c = np.asarray(mesh.elem_centers, dtype=float)
    return float(np.ptp(c[:, 0]) + np.asarray(mesh.elem_size, dtype=float)[0]) if len(c) else 0.0
