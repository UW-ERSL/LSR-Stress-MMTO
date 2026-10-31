"""
pyto_project.py
===============
A structural problem from a PyTO-AD project file (.pyto).

The project file is made in the PyTO-AD GUI: load an STL, pick faces or nodes, apply
supports and loads, and save the project. This module reads the geometry, the supports
and the loads from it, the same way the GUI maps them onto the mesh. The material in the
project file is not used: the material properties come from the VAE, as in the other
examples. Thermal loads and body forces in the project file are not used either.

problem_from_pyto(path, nDOFDesired, single_layer)   mesh, mat_prop, bc, elem_body_force
make_mesh(stl_path, nDOFDesired, single_layer)        the voxel mesh
gui_bcs_on_mesh(project, mesh)                        fixed dofs, force of the GUI's supports and loads

Supports: Fixed XYZ, Fixed X, Fixed Y, Fixed Z, on picked faces, picked nodes or box
selections. Loads: a force (x, y, z) on picked faces, picked nodes or box selections,
split equally over the nodes it lands on.

Forces are used as entered in the GUI, with no unit conversion. They must be in the same
units as the material database.
"""
import json
import os

import numpy as np

from PyTOImports import stl_reader, hex_mesher, mat_lib, bound_cond  # type: ignore

# degrees of freedom each support type fixes, and its marker for the boundary plot
_SUPPORT_DOFS = {"Fixed XYZ": (0, 1, 2), "Fixed X": (0,), "Fixed Y": (1,), "Fixed Z": (2,)}
_MARK_FIXED, _MARK_LOAD, _MARK_ROLLER = 1, 2, 3


def read_project(path):
    """The project file as a dict, with the STL path made absolute."""
    with open(path, "r") as fh:
        project = json.load(fh)
    stl = project.get("stl_file_path")
    if not stl:
        raise ValueError(f"{path}: no stl_file_path in the project file")
    if not os.path.isabs(stl):
        stl = os.path.join(os.path.dirname(os.path.abspath(path)), stl)
    if not os.path.isfile(stl):
        raise FileNotFoundError(f"{path}: the STL file {stl} does not exist")
    project["stl_file_path"] = stl
    return project


def _supports_and_loads(project):
    """Supports as (type, triangles, points, regions) and loads as
    (triangles, points, regions, force), from either project file layout."""
    supports, loads = [], []
    sbc = project.get("structuralBC")
    if sbc is not None:                       # layout written by the current GUI
        for key, kind in (("fixed_faces_indices", "Fixed XYZ"), ("fixed_faces_x_indices", "Fixed X"),
                          ("fixed_faces_y_indices", "Fixed Y"), ("fixed_faces_z_indices", "Fixed Z")):
            if sbc.get(key):
                supports.append((kind, sbc[key], [], []))
        for p in sbc.get("point_supports", []):
            supports.append((p["type"], [], p.get("points") or [], p.get("regions") or []))
        for tris, f in zip(sbc.get("load_faces_indices", []), sbc.get("load_forces", [])):
            loads.append((tris, [], [], f))
        for p in sbc.get("point_loads", []):
            loads.append(([], p.get("points") or [], p.get("regions") or [], p["force"]))
    else:                                     # older layout
        for c in project.get("constraint_data", []):
            supports.append((c["type"], c.get("triangles") or [], c.get("points") or [],
                             c.get("regions") or []))
        for f in project.get("force_data", []):
            if f.get("type", "force_xyz") != "force_xyz":
                raise NotImplementedError(f"load type {f.get('type')!r} is not supported")
            loads.append((f.get("triangles") or [], f.get("points") or [], f.get("regions") or [],
                          [f.get("force_x", 0.0), f.get("force_y", 0.0), f.get("force_z", 0.0)]))
    for s in supports:
        if s[0] not in _SUPPORT_DOFS:
            raise NotImplementedError(f"support type {s[0]!r} is not supported; "
                                      f"use one of {', '.join(_SUPPORT_DOFS)}")
    return supports, loads


def _points_in_region(points, region):
    """Points inside a box selection: on the inner side of each of its planes,
    n . (x - o) <= 0, as in the GUI."""
    n = np.asarray(region["normals"], dtype=float)
    o = np.asarray(region["origins"], dtype=float)
    return np.all(points @ n.T - np.sum(n * o, axis=1)[None, :] <= 1e-12, axis=1)


def _nodes_of(selection, mesh, stl, boundary_nodes, boundary_points, tol, what):
    """Surface nodes of one selection: its faces, its picked nodes and its boxes."""
    triangles, points, regions = selection
    nodes = set()
    for t in triangles:
        if int(t) >= stl.stl_n_triangles:
            raise ValueError(f"{what}: face {t} is not in the STL, which has {stl.stl_n_triangles} "
                             f"faces; the project file was saved for a different STL")
        d = stl.find_points_triangle_distances_vectorized(boundary_points, int(t))
        nodes.update(boundary_nodes[d < tol].tolist())
    if len(points):
        from scipy.spatial import cKDTree
        dist, idx = cKDTree(boundary_points).query(np.asarray(points, dtype=float).reshape(-1, 3))
        if np.any(dist > 2.0 * max(mesh.elem_size)):
            print(f"  warning: {what}: {int(np.sum(dist > 2.0 * max(mesh.elem_size)))} picked "
                  f"node(s) are more than two elements from the mesh; the nearest node is used")
        nodes.update(boundary_nodes[idx].tolist())
    for r in regions:
        nodes.update(boundary_nodes[_points_in_region(boundary_points, r)].tolist())
    if not nodes:
        raise ValueError(f"{what} does not land on any node of the mesh")
    return np.array(sorted(nodes), dtype=int)


def make_mesh(stl_path, nDOFDesired, single_layer=False):
    """The voxel mesh of an STL, with nDOFDesired / 3 elements asked for."""
    mesh = hex_mesher.HexMesher()
    if single_layer:
        mesh.minVoxelsPerAxis = 1
    mesh.createMeshFromSTLFile(stl_path, nElemsDesired=nDOFDesired / 3)
    mesh.createEdofMatStructural()
    return mesh


def gui_bcs_on_mesh(project, mesh):
    """Fixed dofs and force vector of the supports and loads picked in the GUI, on this mesh.
    Empty when the project has none."""
    supports, loads = _supports_and_loads(project)
    force = np.zeros(3 * mesh.num_nodes)
    if not supports and not loads:
        return np.zeros(0, dtype=int), force

    stl = stl_reader.STLGeom(project["stl_file_path"])
    boundary_nodes = np.asarray(mesh.get_boundary_nodes())
    boundary_points = np.asarray(mesh.node_xyz, dtype=float)[boundary_nodes]
    tol = 0.8 * min(mesh.elem_size)                    # as the GUI

    fixed_dofs = []
    for i, (kind, *selection) in enumerate(supports):
        nodes = _nodes_of(selection, mesh, stl, boundary_nodes, boundary_points, tol,
                          f"support {i + 1} ({kind})")
        fixed_dofs.extend((3 * nodes[:, None] + np.array(_SUPPORT_DOFS[kind])[None, :]).ravel())
        mesh.node_indices[nodes, 3] = _MARK_FIXED if kind == "Fixed XYZ" else _MARK_ROLLER
    fixed_dofs = np.unique(np.asarray(fixed_dofs, dtype=int))

    for i, (*selection, f) in enumerate(loads):
        nodes = _nodes_of(selection, mesh, stl, boundary_nodes, boundary_points, tol, f"load {i + 1}")
        force.reshape(-1, 3)[nodes] += np.asarray(f, dtype=float) / len(nodes)
        mesh.node_indices[nodes, 3] = _MARK_LOAD
    return fixed_dofs, force


def problem_from_pyto(path, nDOFDesired=30000, single_layer=False):
    """Mesh, placeholder material, boundary conditions and body force from a .pyto file.

    single_layer: one element through the thickness, for a flat part (as the L-bracket
    and the U-bracket)
    """
    project = read_project(path)
    supports, loads = _supports_and_loads(project)
    if not supports:
        raise ValueError(f"{path}: the project has no supports")
    if not loads:
        raise ValueError(f"{path}: the project has no loads")

    mesh = make_mesh(project["stl_file_path"], nDOFDesired, single_layer)
    fixed_dofs, force = gui_bcs_on_mesh(project, mesh)
    dirichlet_values = np.zeros_like(fixed_dofs, dtype=float)
    bc = bound_cond.BC(force=force, fixed_dofs=fixed_dofs, dirichlet_values=dirichlet_values)

    # placeholder single material; the real properties come from the VAE
    mat_prop = mat_lib.create_material_with_defaults(
        "CustomMaterial", youngs_modulus=1.0, poissons_ratio=0.3, mass_density=1.0, yield_strength=1.0)

    name = os.path.splitext(os.path.basename(path))[0]
    units = (project.get("settings") or {}).get("unit_system", "not stated")
    volume = mesh.num_elems * float(np.prod(mesh.elem_size))
    print(f"[{name}] grid={mesh.grid}  elems={mesh.num_elems}  h={mesh.elem_size[0]:.4g}  "
          f"domain volume={volume:.4g}")
    print(f"[{name}] {len(fixed_dofs)} fixed dofs, total force "
          f"{np.round(force.reshape(-1, 3).sum(axis=0), 6).tolist()} as entered in the GUI "
          f"(GUI units: {units}). The material database must use the same units.")
    if any((project.get("thermalBC") or {}).get(k) for k in ("fixed_temps", "heat_sources",
                                                            "total_heat_sources")):
        print(f"[{name}] the thermal loads in the project file are not used")

    return mesh, mat_prop, bc, None
