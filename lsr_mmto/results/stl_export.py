"""
stl_export.py
=============
A run's design as smooth STL files, for printing or for another program.

    export(run, mesh, folder)    writes design.stl and design_<material>.stl into folder,
                                 and returns a Report of the checks
    exported(run, folder)        the files an earlier export wrote there, or None

design.stl, the whole part: PyTO-AD's STL recovery (pyto/io/topopt_stl_recovery.py), as its
GUI's Postprocess does. The density is smoothed by a Gaussian of SIGMA elements, the void
surface is taken at density 0.5 and subtracted from the STL the mesh was made from, so faces
the optimizer did not touch keep their exact CAD shape.

design_<material>.stl, one per material the design uses: each material's field (the density
where the element is that material) is smoothed by the same
Gaussian on a grid FINE times finer. A fine cell is solid where the smoothed density is 0.5 or more,
as for the whole part, and a solid cell goes to the material whose smoothed field is strongest. VTK's
multi-label surface extraction (vtkSurfaceNets3D) then gives every surface between two
materials once, shared by both, so the pieces cannot overlap or leave gaps. Each face is
turned to point away from its material, decided where every face lies on a cell face (VTK's own
orientation is wrong inside cavities). The shared surface is then smoothed once with Taubin's
non-shrinking smoothing (VTK's windowed sinc): points move, faces stay, so the pieces stay matched.
(VTK's own smoothing in the extraction lost 6% of a plate one element thick.)

Where two cells of one material meet only along an edge or at a corner (common in speckled
designs), a cell between them is given that material first (contacts_joined): otherwise the
surface would not be closed, or two bodies would touch at a point. It changes a few of the fine
cells, a third of an element each.

Limits, as the tests on shapes with known answers showed: features one element across are
removed (as by the whole-part recovery at the same width); square corners of thin members
round off.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

import numpy as np

SIGMA = 0.5        # Gaussian width, in elements: PyTO-AD's STL recovery default
FINE = 3           # each element split FINE x FINE x FINE for the per-material surfaces
TAUBIN = (30, 0.05) # Taubin smoothing of the shared surface: iterations, pass band (removes the
                   # third-of-an-element staircase without shrinking, unlike plain smoothing)


@dataclass
class Piece:
    label: str
    path: str
    closed: bool
    volume: float
    design_volume: float          # the material's solid elements, as blocks


@dataclass
class Report:
    whole_path: str = ""
    whole_closed: bool = False
    whole_volume: float = float("nan")
    design_volume: float = float("nan")
    whole_error: str = ""
    pieces: list = field(default_factory=list)
    pieces_whole_volume: float = float("nan")   # the per-material set's own outer surface

    @property
    def sum_off(self):
        """How far the pieces' volumes sum from the per-material set's whole: a gap or an overlap shows here."""
        return sum(p.volume for p in self.pieces) / self.pieces_whole_volume - 1.0

    def lines(self, unit=""):
        u = f" {unit}³" if unit else ""
        out = []
        if self.whole_error:
            out.append(f"Whole part not written: {self.whole_error}")
        else:
            out.append(f"Whole part: {os.path.basename(self.whole_path)}, {self.whole_volume:,.4g}{u} "
                       f"(design {self.design_volume:,.4g}){'' if self.whole_closed else ', NOT closed'}")
        for p in self.pieces:
            out.append(f"{p.label}: {os.path.basename(p.path)}, {p.volume:,.4g}{u} (design {p.design_volume:,.4g})"
                       f"{'' if p.closed else ', NOT closed'}")
        if self.pieces:
            out.append(f"The material pieces add up to their whole within {abs(100 * self.sum_off):.2g}%; "
                       + ("all closed." if all(p.closed for p in self.pieces) else
                          "some are not closed: where two materials touch only along an edge"))
        return out


def _grid(mesh):
    """Element (i, j, k) indices on the voxel grid, its shape, its lowest corner and the element size."""
    c = np.asarray(mesh.elem_centers, dtype=float)
    h = np.asarray(mesh.elem_size, dtype=float)
    lo = c.min(axis=0)
    ijk = np.rint((c - lo) / h).astype(int)
    return ijk, ijk.max(axis=0) + 1, lo - h / 2, h


def _vtu(mesh, density):
    """The hex mesh with the element density, for PyTO-AD's STL recovery."""
    import pyvista as pv
    e = np.asarray(mesh.elemArray)[:, :8]
    n = e.shape[0]
    cells = np.column_stack((np.full(n, 8), e)).ravel()
    grid = pv.UnstructuredGrid(cells, np.full(n, pv.CellType.HEXAHEDRON, dtype=np.uint8),
                               np.asarray(mesh.node_xyz, dtype=float))
    grid.cell_data["density"] = np.asarray(density, dtype=float)
    return grid


def whole_part(run, mesh):
    """PyTO-AD's STL recovery of the run's design: (part, design volume)."""
    from lsr_mmto.pyto_ad_imports import stl_recovery
    extract_isosurface, subtract_voids_from_stl = stl_recovery().extract_isosurface, stl_recovery().subtract_voids_from_stl
    stl = mesh.stlMesh.triangulate().compute_normals()
    voids = extract_isosurface(_vtu(mesh, run.x_cont), isovalue=0.5, method="gaussian", sigma=SIGMA)
    part = subtract_voids_from_stl(stl, voids)
    return part, float(np.count_nonzero(run.solid)) * float(np.prod(mesh.elem_size))


def material_labels(run, mesh, fine=FINE, sigma=SIGMA):
    """The fine label grid: 0 void, m + 1 material m; its lowest corner and spacing."""
    from scipy.ndimage import gaussian_filter
    ijk, shape, origin, h = _grid(mesh)
    x = np.clip(np.asarray(run.x_cont, dtype=float), 0.0, 1.0)
    mat = np.asarray(run.material_indices, dtype=int)
    used = np.unique(mat[np.asarray(run.solid, dtype=bool)])
    def smoothed(f):                                               # as the whole-part recovery smooths: on the
        return gaussian_filter(f.repeat(fine, 0).repeat(fine, 1).repeat(fine, 2),   # grid itself, its edges
                               sigma * fine, mode="nearest")                         # extended (padding first
    density = np.zeros(shape)                                                        # would blur a plate one
    density[ijk[:, 0], ijk[:, 1], ijk[:, 2]] = x                                     # element thick away)
    solid = smoothed(density) >= 0.5                               # 1. solid or not: the density, as the whole part
    label = np.zeros(solid.shape, dtype=int)
    best = np.full(solid.shape, -np.inf)
    for m in used:                                                 # 2. inside the solid, the strongest material
        f = np.zeros(shape)
        sel = mat == m
        f[ijk[sel, 0], ijk[sel, 1], ijk[sel, 2]] = x[sel]
        g = smoothed(f)
        win = solid & (g > best)
        best = np.where(win, g, best)
        label[win] = int(m) + 1
    pad = 1                                                        # then void around, so every surface closes
    label, _ = contacts_joined(np.pad(label, pad, constant_values=0))
    return label, origin - pad * h / fine, h / fine


def contacts_joined(label, max_rounds=200):
    """The label grid with every edge-only or corner-only contact made a face contact. Where two cells of
    one label meet only along an edge (a 2 x 2 checkerboard) or only at a corner (a 2 x 2 x 2 window with
    none of its other cells theirs), one cell of the window changes: an edge contact makes a surface that
    is not closed (an edge shared by four faces), a corner contact joins two bodies at a point, which some
    slicers refuse.

    A cell only ever changes to a higher label (the void, 0, lowest), so the fixes cannot undo one another
    and the loop ends: the label of the contact grows into a cell between when that cell's label is
    lower, and otherwise the higher label of that cell grows over the contact. Returns (grid, cells changed)."""
    label = label.copy()
    before = label.copy()
    shape = label.shape
    flat = label.reshape(-1)

    def raise_to(cells, new):                                       # cells (flat indices) to at least new
        np.maximum.at(flat, cells, new)

    def index(i, j, k):
        return np.ravel_multi_index((i, j, k), shape)

    for _ in range(max_rounds):
        start = flat.copy()
        for a, b in ((0, 1), (0, 2), (1, 2)):                       # edges, in each plane
            c = [s_ for s_ in range(3) if s_ not in (a, b)][0]
            L = np.moveaxis(label, (a, b, c), (0, 1, 2))
            n0, n1, n2 = L.shape
            i, j, k = np.meshgrid(np.arange(n0 - 1), np.arange(n1 - 1), np.arange(n2), indexing="ij")
            corners = [(i, j), (i + 1, j + 1), (i + 1, j), (i, j + 1)]          # p, q (one diagonal), r, t
            lab = [L[x, y, k] for x, y in corners]

            def cell(x, y):
                axes = [None] * 3
                axes[a], axes[b], axes[c] = x, y, k
                return index(*axes)
            for (d1, d2, o1, o2) in ((0, 1, 2, 3), (2, 3, 0, 1)):
                m = lab[d1]
                bad = (lab[d1] == lab[d2]) & (lab[o1] != m) & (lab[o2] != m)
                if not bad.any():
                    continue
                grow1 = bad & (lab[o1] < m)                         # the contact's label into a lower cell
                grow2 = bad & ~grow1 & (lab[o2] < m)
                over = bad & ~grow1 & ~grow2                        # both higher: the higher one grows over
                raise_to(cell(*corners[o1])[grow1], m[grow1])
                raise_to(cell(*corners[o2])[grow2], m[grow2])
                raise_to(cell(*corners[d1])[over], lab[o1][over])
        nx_, ny_, nz_ = shape                                       # corners, in each 2 x 2 x 2 window
        I, J, K = np.meshgrid(np.arange(nx_ - 1), np.arange(ny_ - 1), np.arange(nz_ - 1), indexing="ij")
        win = {w: label[I + w[0], J + w[1], K + w[2]] for w in np.ndindex(2, 2, 2)}
        for u, v in (((0, 0, 0), (1, 1, 1)), ((1, 0, 0), (0, 1, 1)), ((0, 1, 0), (1, 0, 1)), ((0, 0, 1), (1, 1, 0))):
            m = win[u]
            others = np.zeros(m.shape, dtype=bool)
            for w in win:
                if w not in (u, v):
                    others |= win[w] == m
            bad = (win[u] == win[v]) & ~others
            if not bad.any():
                continue
            step = (v[0], u[1], u[2])                               # the cell one step from u towards v
            between = win[step]
            grow = bad & (between < m)
            over = bad & ~grow
            raise_to(index(I + step[0], J + step[1], K + step[2])[grow], m[grow])
            raise_to(index(I + u[0], J + u[1], K + u[2])[over], between[over])
        if np.array_equal(flat, start):
            break
    return label, int(np.count_nonzero(label != before))


def material_surfaces(label, origin, spacing):
    """(points, quads, boundary labels, label behind each quad) of the multi-label surface: VTK's
    extraction without its own smoothing, each quad's side decided where quads lie on cell faces, then
    Taubin's smoothing (non-shrinking) of the shared surface, so the pieces stay matched."""
    import pyvista as pv
    nx, ny, nz = label.shape
    img = pv.ImageData(dimensions=(nx + 1, ny + 1, nz + 1), spacing=tuple(spacing), origin=tuple(origin))
    img.cell_data["m"] = label.ravel(order="F")
    raw = img.contour_labels("all", smoothing=False, output_mesh_type="quads", scalars="m", simplify_output=False,
                             orient_faces=False)
    quads = raw.faces.reshape(-1, 5)[:, 1:]
    bl = np.asarray(raw.cell_data["boundary_labels"])
    normals = raw.compute_normals(cell_normals=True, point_normals=False, auto_orient_normals=False,
                                  consistent_normals=False, split_vertices=False)
    n, c = np.asarray(normals.cell_data["Normals"]), np.asarray(raw.cell_centers().points)
    sp = np.asarray(spacing, dtype=float)
    i = np.floor((c - 0.25 * sp * n - np.asarray(origin)) / sp).astype(int)   # a quarter cell behind each face
    inside = np.all((i >= 0) & (i < [nx, ny, nz]), axis=1)
    i = np.clip(i, 0, [nx - 1, ny - 1, nz - 1])
    behind = np.where(inside, label[i[:, 0], i[:, 1], i[:, 2]], 0)
    if not np.all((behind == bl[:, 0]) | (behind == bl[:, 1])):
        raise RuntimeError("a face lies between other materials than its labels say")
    surface = pv.PolyData(np.asarray(raw.points), np.hstack([np.full((len(quads), 1), 4), quads]).ravel())
    smooth = surface.smooth_taubin(n_iter=TAUBIN[0], pass_band=TAUBIN[1], non_manifold_smoothing=True,
                                   boundary_smoothing=False, feature_smoothing=False, normalize_coordinates=True)
    return np.asarray(smooth.points), quads, bl, behind


def piece(points, quads, bl, behind, m):
    """The closed surface of label m, its faces pointing away from it (m = 0: the whole, from outside)."""
    import trimesh
    keep = (bl == m).any(axis=1) if m else (bl == 0).any(axis=1)
    q, b = quads[keep].copy(), behind[keep]
    flip = (b != m) if m else (b == 0)
    q[flip] = q[flip][:, ::-1]
    return trimesh.Trimesh(points, np.vstack([q[:, [0, 1, 2]], q[:, [0, 2, 3]]]), process=True)


def safe_name(label):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(label)).strip("_") or "material"


def piece_stem(name, i, used):
    """The file name (no .stl) of the piece of the material named name, label i of the grid: design_<name>,
    or design_<name>_<i> when another piece took that name (two names that clean to the same file; Windows
    ignores case). used: the stems taken so far, in lower case; this one is added."""
    base = f"design_{safe_name(name)}"
    stem = base if base.lower() not in used else f"{base}_{i}"
    used.add(stem.lower())
    return stem


def exported(run, folder):
    """The STL files an export of run wrote into folder: (design.stl, [(material index, path)]) for the
    files there, or None when folder holds no design.stl. Each material is matched to its file by the
    names export gives (piece_stem), in the order of the materials."""
    whole = os.path.join(folder, "design.stl")
    if not os.path.isfile(whole):
        return None
    files = {os.path.splitext(f)[0].lower(): os.path.join(folder, f) for f in os.listdir(folder)
             if f.lower().endswith(".stl") and f.lower().startswith("design_")}
    pieces = []
    for m, name in enumerate(run.material_labels):
        base = f"design_{safe_name(name)}".lower()
        stem = next((s for s in (base, f"{base}_{m + 1}") if s in files), None)
        if stem is not None:
            pieces.append((m, files.pop(stem)))
    return whole, pieces


def export(run, mesh, folder, materials=True):
    """design.stl (and design_<material>.stl) into folder, as Export STL in the GUI writes them; returns
    the Report, whose lines() say what was written and whether each file is closed.

    run: a Run, from load
    mesh: the mesh it was solved on, from mesh_for(run)
    folder: the folder the files go into, usually the run's own
    materials: True also writes one file for each material, False only the whole part
    Example: report = export(r, mesh, folder)
    """
    report = Report()
    h3 = float(np.prod(mesh.elem_size))
    try:
        part, report.design_volume = whole_part(run, mesh)
        report.whole_path = os.path.join(folder, "design.stl")
        part.save(report.whole_path)
        import trimesh
        t = trimesh.load(report.whole_path, force="mesh")
        report.whole_closed, report.whole_volume = bool(t.is_watertight), float(abs(t.volume))
    except Exception as e:                                          # noqa: BLE001
        report.whole_error = str(e)
    if not materials:
        return report
    label, origin, spacing = material_labels(run, mesh)
    pts, quads, bl, behind = material_surfaces(label, origin, spacing)
    report.pieces_whole_volume = float(piece(pts, quads, bl, behind, 0).volume)
    solid = np.asarray(run.solid, dtype=bool)
    mat = np.asarray(run.material_indices, dtype=int)
    used = set()
    for i in sorted(set(np.unique(label)) - {0}):
        m = i - 1
        name = run.material_labels[m] if m < len(run.material_labels) else f"material {m + 1}"
        t = piece(pts, quads, bl, behind, i)
        path = os.path.join(folder, piece_stem(name, i, used) + ".stl")
        t.export(path)
        report.pieces.append(Piece(name, path, bool(t.is_watertight), float(t.volume),
                                   float(np.count_nonzero(solid & (mat == m))) * h3))
    return report
