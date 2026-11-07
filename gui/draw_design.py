"""
draw_design.py
==============
An optimized design on the 3D view: one block per element, drawn in place of the part.

    live(window, mesh, x, material, colours, labels)   while a run goes: density and material
    record(window, run, mesh, field, ...)       a finished run: Material, Density, a result (Stress,
                                                Displacement) or Solid
    run_mesh(window, run)                       the mesh a saved run was made on, at the size it was run at

A drawing of materials (a run going, the Material view, the smooth pieces) also adds a column to
window.design_legend: each material of the database with its colour, label and share (none when the
design does not use it), for the window's legend (one column per design drawn, so two runs compared get two).

Every actor drawn here is named "result_...", so clearing a preview (actors named "mmto_...")
never removes a design. The window's Setup view and Design view (main_window.py) decide
whether a design or the part is shown.
"""
import os

import numpy as np

def views(*runs):
    """The ways finished runs can be shown: Material, Density, one per result the runs saved (Stress for a
    structural run, from the run's own physics), Solid."""
    shown = [info["view"] for run in runs for info in run.result_info.values()]
    return ("Material", "Density", *dict.fromkeys(shown), "Solid")


COMPONENTS = ("magnitude", "x", "y", "z")     # how a vector result is shown: its size, or one of its parts
DEFORMED_SHARE = 0.05                         # the deformed shape's first scale: the largest movement drawn as this
                                              # share of the part's largest extent
FAINT = 0.1                                   # elements below this density are empty: not drawn
WEAKEST = 0.25                                # colour strength of an element just above FAINT
SOLID_COLOUR = "#9aa5b1"
PREFIX = "result_"
# The colour scale of a value view (Density, a result): the lowest and the highest value at the two
# ends, under the bar, and the title centred over it, at fixed sizes, so the numbers never run into
# each other or into the title, however large or small the values, however long the title and whatever
# the size of the view.
VALUE_BAR = {"vertical": False, "position_x": 0.3, "position_y": 0.05, "width": 0.4, "height": 0.05,
             "n_labels": 2, "fmt": "%.3g", "unconstrained_font_size": True, "title_font_size": 16,
             "label_font_size": 14, "color": "#333333"}
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
        return _on_scale(window, blocks, title, clim, kwargs)
    return window.plotter.add_mesh(_blocks(centres, h, {}), color=SOLID_COLOUR, **kwargs)


def _on_scale(window, shape, title, clim, kwargs):
    """shape drawn coloured by its "value" on the colour scale with title, the scale under the view."""
    window.design_bars.add(title)
    actor = window.plotter.add_mesh(shape, scalars="value", cmap="viridis", clim=clim,
                                    scalar_bar_args=VALUE_BAR | {"title": title}, **kwargs)
    bar = window.plotter.scalar_bars[title]
    bar.SetTextPositionToPrecedeScalarBar()                # the numbers under the bar
    bar.SetTextPad(3)
    bar.SetTitle("")                                       # the title over it, apart from the numbers
    window.plotter.add_actor(scale_title(title), name=PREFIX + "scale title " + title, reset_camera=False,
                             pickable=False)
    return actor


def scale_title(title):
    """The colour scale's title as a text of its own, centred over the bar, its foot just above the bar's top: on
    the other side of the bar from the numbers, so the two never overlap."""
    import vtk
    text = vtk.vtkTextActor()
    text.SetInput(title)
    text.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
    text.SetPosition(VALUE_BAR["position_x"] + VALUE_BAR["width"] / 2,
                     VALUE_BAR["position_y"] + VALUE_BAR["height"] + 0.008)
    p = text.GetTextProperty()
    p.SetFontSize(VALUE_BAR["title_font_size"])
    p.SetJustificationToCentered()
    p.SetVerticalJustificationToBottom()
    p.SetColor(*(rgb(VALUE_BAR["color"]) / 255.0))
    return text


def draw_nodes(window, mesh, mask, values, title, clim, shift=0.0, name="design", moved=None):
    """The masked elements of mesh as one surface coloured by values, one per node of the mesh: the colour
    blends smoothly over each face, from corner to corner, instead of one colour per element. moved: each
    node's movement (nodes, 3), added to where it is (the deformed shape), or None. shift moves it along x."""
    import pyvista as pv
    corners = np.asarray(mesh.elemArray)[:, :8][np.asarray(mask, dtype=bool)]
    if not len(corners):
        return None
    points = np.asarray(mesh.node_xyz, dtype=float) + np.array([shift, 0.0, 0.0])
    if moved is not None:
        points = points + np.asarray(moved, dtype=float)
    cells = np.column_stack((np.full(len(corners), 8), corners)).ravel()
    grid = pv.UnstructuredGrid(cells, np.full(len(corners), pv.CellType.HEXAHEDRON, dtype=np.uint8), points)
    grid.point_data["value"] = np.asarray(values, dtype=float)
    return _on_scale(window, grid.extract_surface(), title, clim,
                     dict(pickable=False, name=PREFIX + name, reset_camera=False))


def legend_rows(colours, labels, shares):
    """[(colour, label, share)] of every material of the database, largest first; shares as fractions of the
    whole (0 for a material not in the design: the legend shows only those with a share, but a database of
    several materials keeps its legend when the design uses one). labels: one per material (a material
    without one is named by its number)."""
    shares = np.asarray(shares, dtype=float)
    total = shares.sum()
    rows = [(colours[i % len(colours)], labels[i] if i < len(labels) else f"material {i}",
             shares[i] / total if total > 0 else 0.0) for i in range(len(shares))]
    return sorted(rows, key=lambda r: -r[2])


def add_legend(window, rows):
    """A column of the window's legend (window.design_legend), when the window keeps one."""
    if hasattr(window, "design_legend"):
        window.design_legend.append(rows)


def live(window, mesh, x, material, colours, labels=()):
    """A design while it runs: elements at FAINT density or more, in their material's colour at a
    strength that follows their density (pale when half solid, full colour when solid). Its legend:
    each material's share of the density of those elements."""
    x = np.asarray(x, dtype=float).ravel()
    table = np.array([rgb(c) for c in colours])
    m = np.asarray(material).ravel() % len(table)
    per_element = faded(table[m], strength(x))
    on = x >= FAINT
    add_legend(window, legend_rows(colours, list(labels), np.bincount(m[on], weights=x[on], minlength=len(table))))
    return draw(window, mesh, on, colours=per_element)


def result_of(run, field):
    """The name of the run's result shown by the view field (von_mises for Stress), or None."""
    return next((name for name, info in run.result_info.items() if info.get("view") == field), None)


def result_title(run, name, component="magnitude"):
    """The colour scale's title of a result, with its unit: von Mises stress [MPa]; a vector's with the part
    shown: Displacement x [mm]."""
    info = run.result_info[name]
    units = (run.config or {}).get("units") or {}
    unit = units.get(info.get("measures"))
    title = f"{info['title']} {component}" if info.get("shape") == "vector" else info["title"]
    return f"{title} [{unit}]" if unit else title


def result_values(run, name, component="magnitude"):
    """A result's numbers to colour by: the values themselves, or for a vector its magnitude or its x, y or z."""
    values = np.asarray(run.results[name], dtype=float)
    if run.result_info[name].get("shape") != "vector":
        return values
    return np.linalg.norm(values, axis=1) if component == "magnitude" else values[:, "xyz".index(component)]


def deforms(info):
    """True for a result that can draw the deformed shape: a vector at each node that measures length."""
    return info.get("where") == "node" and info.get("shape") == "vector" and info.get("measures") == "length"


def _on(run, mesh, name):
    """Where a result is drawn: the solid elements, or for a result at the nodes, the nodes of those."""
    solid = np.asarray(run.solid, dtype=bool)
    if run.result_info[name].get("where") != "node":
        return solid
    return np.unique(np.asarray(mesh.elemArray)[:, :8][solid])


def field_range(shown, field, component="magnitude"):
    """One colour scale for several runs drawn together, so their colours compare: shown is [(run, mesh)]. It
    starts at 0, or lower for values below 0 (the x, y or z of a vector)."""
    if field == "Density":
        return (0.0, 1.0)
    values = [result_values(run, result_of(run, field), component)[_on(run, mesh, result_of(run, field))]
              for run, mesh in shown if result_of(run, field) is not None and np.asarray(run.solid).any()]
    if not any(result_of(run, field) is not None for run, _ in shown):
        return None
    lo = min([0.0] + [float(v.min()) for v in values if v.size])
    hi = max([lo] + [float(v.max()) for v in values if v.size])
    return (lo, hi if hi > lo else lo + 1.0)


def deformed_scale(shown, field):
    """The Scale of the deformed shape that makes the largest movement DEFORMED_SHARE of the part's largest
    extent: shown is [(run, mesh)], field the view of the movement."""
    largest, extent = 0.0, 0.0
    for run, mesh in shown:
        name = result_of(run, field)
        if name is None or not np.asarray(run.solid).any():
            continue
        on = _on(run, mesh, name)
        largest = max(largest, float(np.linalg.norm(np.asarray(run.results[name], dtype=float)[on], axis=1).max()))
        extent = max(extent, float(np.ptp(np.asarray(mesh.node_xyz, dtype=float)[on], axis=0).max()))
    return DEFORMED_SHARE * extent / largest if largest > 0 else 1.0


def record(window, run, mesh, field="Material", shift=0.0, clim=None, name="design", component="magnitude",
           scale=None):
    """A finished run, drawn as field: Material (its solid elements in their material's colour),
    Density (every element at FAINT or more, on a scale), a result's view (Stress: von Mises on its
    solid elements; Displacement: at their nodes, coloured smoothly; on a scale) or Solid (its solid
    elements, one colour). component: magnitude, x, y or z of a vector result; scale: for a result that
    deforms (deforms), the part drawn moved by it times scale, or None for not moved."""
    if field == "Material":
        table = np.array([rgb(c) for c in run.material_colors]).astype(np.uint8)
        m = np.asarray(run.material_indices).ravel() % len(table)
        solid = np.asarray(run.solid, dtype=bool).ravel()
        add_legend(window, legend_rows(list(run.material_colors), list(run.material_labels),
                                       np.bincount(m[solid], minlength=len(table))))
        return draw(window, mesh, run.solid, colours=table[m], shift=shift, name=name)
    if field == "Density":
        return draw(window, mesh, np.asarray(run.x_cont) >= FAINT, values=run.x_cont, title="Density",
                    clim=clim or (0.0, 1.0), shift=shift, name=name)
    result = result_of(run, field)
    if result is None:
        return draw(window, mesh, run.solid, shift=shift, name=name)
    values, title = result_values(run, result, component), result_title(run, result, component)
    clim = clim or field_range([(run, mesh)], field, component)
    info = run.result_info[result]
    if info.get("where") == "node":
        moved = np.asarray(run.results[result], dtype=float) * scale if scale and deforms(info) else None
        return draw_nodes(window, mesh, run.solid, values, title, clim, shift=shift, name=name, moved=moved)
    return draw(window, mesh, run.solid, values=values, title=title, clim=clim, shift=shift, name=name)


def smooth(window, pieces, apart=False, absent=()):
    """STL files on the part, each in its colour: [(path, colour, label)]. apart sets them side by side in a row
    along x, largest first: the k-th moved by k times the part's width along x and a gap of a tenth of it,
    so each can be seen alone. The legend gets each piece's label and its share of the volume, and the
    database's materials with no piece, absent [(colour, label)], at none. Returns the pieces as drawn,
    largest first."""
    import pyvista as pv
    meshes = sorted([(pv.read(path), colour, label) for path, colour, label in pieces],
                    key=lambda p: -abs(p[0].volume))
    if not meshes:
        return []
    lo = min(m.bounds[0] for m, _, _ in meshes)
    step = 1.1 * (max(m.bounds[1] for m, _, _ in meshes) - lo)
    drawn = []
    for k, (m, colour, _) in enumerate(meshes):
        if apart and k:
            m = m.translate((k * step, 0.0, 0.0), inplace=False)
        window.plotter.add_mesh(m, color=colour, smooth_shading=True, pickable=False, name=f"{PREFIX}smooth_{k}",
                                reset_camera=False)
        drawn.append(m)
    add_legend(window, legend_rows([c for _, c, _ in meshes] + [c for c, _ in absent],
                                   [lab for _, _, lab in meshes] + [lab for _, lab in absent],
                                   [abs(m.volume) for m, _, _ in meshes] + [0.0] * len(absent)))
    return drawn


class MeshError(RuntimeError):
    pass


def mesh_of(window, project):
    """The mesh a run of project uses: the window's own when it is the same one, else built once."""
    from lsr_mmto.problem.mesh import make_mesh
    key = (os.path.abspath(project.stl_path()), project.n_dof, project.single_layer, project.unknowns_per_node())
    if key == getattr(window, "_mesh_key", None) and window._mesh is not None:
        return window._mesh
    if key not in _MESHES:
        _MESHES[key] = make_mesh(*key)
    return _MESHES[key]


def forget_meshes(stl):
    """The meshes kept for the STL at stl dropped, after its file changed (File > Scale part)."""
    for key in [k for k in _MESHES if os.path.normcase(k[0]) == os.path.normcase(os.path.abspath(stl))]:
        del _MESHES[key]


def _scaled(mesh, factor):
    """A copy of mesh at factor times its size: its nodes, element centres, element size and the part's surface
    (which Export STL cuts the design from); everything else is the same, and mesh itself is not changed."""
    import copy
    out = copy.copy(mesh)
    out.node_xyz = np.asarray(mesh.node_xyz, dtype=float) * factor
    out.elem_centers = np.asarray(mesh.elem_centers, dtype=float) * factor
    out.elem_size = np.asarray(mesh.elem_size, dtype=float) * factor
    out.stlMesh = mesh.stlMesh.scale(factor, inplace=False)
    return out


def run_mesh(window, run):
    """The mesh a saved run was made on, rebuilt from the project its record keeps (the window's own
    mesh when it is the same one). A part scaled since the run (File > Scale part: the same elements, each
    larger or smaller by one factor in x, y and z than the run recorded) gives a copy of the mesh scaled back
    to the run's own size. MeshError when it cannot match the run."""
    from lsr_mmto.results.load_results import RecordError
    from lsr_mmto.problem.project import Project
    spec = ((run.config or {}).get("arguments_resolved") or {}).get("to_problem")
    if not isinstance(spec, dict) or "project" not in spec:
        raise MeshError("this run's record does not hold its project, so its part is not known")
    p = Project.from_record(spec)
    if not os.path.isfile(p.stl_path()):
        raise MeshError(f"its part is missing: {p.stl_path()}")
    mesh = mesh_of(window, p)
    if int(mesh.num_elems) != int(run.n_elems):
        raise MeshError(f"its part has changed since the run: the mesh has {int(mesh.num_elems)} elements, "
                        f"the run {int(run.n_elems)}")
    try:
        recorded = run.elem_size
    except RecordError:                                         # an older record: as it is
        return mesh
    ratio = np.asarray(mesh.elem_size, dtype=float) / recorded
    if np.allclose(ratio, 1.0, rtol=1e-9, atol=0.0):
        return mesh
    if not np.allclose(ratio, ratio[0], rtol=1e-9, atol=0.0):
        raise MeshError(f"its part has changed since the run: the mesh's elements are "
                        f"{' x '.join(f'{v:.4g}' for v in mesh.elem_size)}, the run's "
                        f"{' x '.join(f'{v:.4g}' for v in recorded)}")
    return _scaled(mesh, 1.0 / float(ratio[0]))


def width(mesh):
    """The extent of the mesh along x, for drawing a second design beside it."""
    c = np.asarray(mesh.elem_centers, dtype=float)
    return float(np.ptp(c[:, 0]) + np.asarray(mesh.elem_size, dtype=float)[0]) if len(c) else 0.0
