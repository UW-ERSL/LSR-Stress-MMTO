import numpy as np
import os
import enum
script_dir = os.path.dirname(os.path.abspath(__file__))
from PyTOImports import *
import scipy.sparse as spy_sprs
from scipy.sparse import lil_matrix
class StructuralFEAExamples(enum.Enum):
	LBracket_BM1 = enum.auto()
	Beam_Liao = enum.auto()
	AirplaneBracket = enum.auto()
	UBracket = enum.auto()
 
def getStructuralFEAProblem(problem: StructuralFEAExamples,nDOFDesired: int = 20000, **kwargs):
  """Returns a structural problem based on the given problem name.

  Parameters:
  ----------
  problem : StructuralExamples
    The name of the problem to return.
  **kwargs : dict
    Additional keyword arguments to pass to the problem creation function.

  Returns:
  -------
  tuple
    A tuple containing the mesh, material properties, and boundary conditions for the problem.
  """ 
  if problem == StructuralFEAExamples.LBracket_BM1:
    return createLBracket_BM1_Problem(nDOFDesired=nDOFDesired,**kwargs)
  elif problem == StructuralFEAExamples.Beam_Liao:
    return createBeam_Liao_Problem(nDOFDesired=nDOFDesired, **kwargs)
  elif problem == StructuralFEAExamples.AirplaneBracket:
    return createAirplaneBracket_Problem(nDOFDesired=nDOFDesired, **kwargs)
  elif problem == StructuralFEAExamples.UBracket:
    return createUBracket_Problem(nDOFDesired=nDOFDesired, **kwargs)
  else:
    raise ValueError("Invalid structural example name.")

def createLBracket_BM1_Problem(nDOFDesired: int = 10000, topload = 0,midload = 1000):
  stl_file = os.path.join(script_dir, './Models/LBracket_BM1/LBracket_BM1.STL')

  nElemsDesired = nDOFDesired/3

  mesh = hex_mesher.HexMesher()
  mesh.minVoxelsPerAxis = 1   # single 1 mm through-thickness layer (default floor is 2)
  mesh.createMeshFromSTLFile(stl_file, nElemsDesired=nElemsDesired)
  _nz = mesh.grid[2]
  _inplane = mesh.num_elems // max(_nz, 1)
  print(f"[LBracket_BM1] grid={mesh.grid}  total_elems={mesh.num_elems}  "
        f"in-plane={_inplane}  nz={_nz}  h={mesh.elem_size[0]:.3f} mm   "
        f"(Banh target: 12800 in-plane, h=1.414 mm)")
  mesh.createEdofMatStructural()

  triList0 = [12,13]
  fixed_nodes = mesh.get_nodes_on_triangles(triList0)

  fixed_dofs = np.array([3 * fixed_nodes,
              3 * fixed_nodes + 1,
              3 * fixed_nodes + 2]).flatten().astype(int)
  
  dirichlet_values = 0*np.ones_like(fixed_dofs, dtype = float)
  mesh.node_indices[fixed_nodes, 3] = 1 # for plotting

  force = np.zeros(3*mesh.num_nodes)

  if(abs(topload) > 0):
    triList1 = [12,13]
    topload_nodes = mesh.get_nodes_on_triangles(triList1)
    topload_dofs = 3 * topload_nodes + 1  
    mesh.node_indices[topload_nodes, 3] = 2 # for plotting
    force[topload_dofs] = -topload/len(topload_nodes)

  if(abs(midload) > 0):
    # triList2 = [6,7]
    # midload_nodes = mesh.get_nodes_on_triangles(triList2)
    # midload_dofs = 3 * midload_nodes + 1
    # force[midload_dofs] = -midload/len(midload_nodes)
    # Banh-faithful load: top 8 node-rows of the right edge (x = xmax).
    # nz=1 gives 2 z-layers, so 8 rows = 16 mesh nodes; total force is still
    # `midload`, i.e. -midload/8 per row, exactly as in Banh's 2-D mesh.
    hx, hy = mesh.elem_size[0], mesh.elem_size[1]
    xmax = mesh.node_xyz[:, 0].max()
    on_right = np.abs(mesh.node_xyz[:, 0] - xmax) < 0.25 * hx
    yrow = np.round(mesh.node_xyz[:, 1] / hy).astype(int)
    top_rows = np.unique(yrow[on_right])[-8:]          # 8 highest rows = the loaded corner
    midload_nodes = np.where(on_right & np.isin(yrow, top_rows))[0]
    midload_dofs = 3 * midload_nodes + 1
    force[midload_dofs] = -midload / len(midload_nodes)
    mesh.node_indices[midload_nodes, 3] = 2  # for plotting

  bc = bound_cond.BC(force = force,fixed_dofs = fixed_dofs,dirichlet_values = dirichlet_values)

  # Define material properties
  mat_prop = mat_lib.create_material_with_defaults("CustomMaterial", youngs_modulus=1.0, poissons_ratio=0.3, mass_density=1.0, yield_strength=1.0)

  elem_body_force = None

  return mesh, mat_prop, bc, elem_body_force


def createUBracket_Problem(nDOFDesired: int = 30000, load = 1.0):
  """U-bracket: unit square [0,1]^2 with the top slot x in [0.5,0.75], y in [0.5,1.0]
  removed (baked into the STL -> passive void). Single through-thickness layer (nz=1).

  BCs (coordinate-based, mirroring the Banh BM1 philosophy but for a -x load):
    Clamp : x = xmin (left face), all DOFs              -> Gamma_u
    Load  : total -x force `load` spread over the top 8 node-rows of the right face
            (x = xmax), to avoid the corner stress singularity in the figure.
  The -x load is normal to the right face; distributing it down the top rows
  mirrors createLBracket_BM1_Problem's "top-8-rows of the right edge" load.
  """
  stl_file = os.path.join(script_dir, './Models/UBracket/UBracket.STL')

  nElemsDesired = nDOFDesired/3

  mesh = hex_mesher.HexMesher()
  mesh.minVoxelsPerAxis = 1   # single through-thickness layer (default floor is 2)
  mesh.createMeshFromSTLFile(stl_file, nElemsDesired=nElemsDesired)
  _nz = mesh.grid[2]
  _inplane = mesh.num_elems // max(_nz, 1)
  print(f"[UBracket] grid={mesh.grid}  total_elems={mesh.num_elems}  "
        f"in-plane={_inplane}  nz={_nz}  h={mesh.elem_size[0]:.4f}  "
        f"(want nz=1; raise/lower nDOFDesired if nz flips)")
  mesh.createEdofMatStructural()

  hx = mesh.elem_size[0]
  hy = mesh.elem_size[1]
  xmin = mesh.node_xyz[:, 0].min()
  xmax = mesh.node_xyz[:, 0].max()

  # --- clamp: left face x = xmin, all DOFs ---
  fixed_nodes = np.where(np.abs(mesh.node_xyz[:, 0] - xmin) < 0.25 * hx)[0]
  fixed_dofs = np.array([3 * fixed_nodes,
                         3 * fixed_nodes + 1,
                         3 * fixed_nodes + 2]).flatten().astype(int)
  dirichlet_values = 0 * np.ones_like(fixed_dofs, dtype=float)
  mesh.node_indices[fixed_nodes, 3] = 1  # for plotting

  # --- load: -x over the top 8 node-rows of the right face x = xmax ---
  force = np.zeros(3 * mesh.num_nodes)
  on_right = np.abs(mesh.node_xyz[:, 0] - xmax) < 0.25 * hx
  yrow = np.round(mesh.node_xyz[:, 1] / hy).astype(int)
  top_rows = np.unique(yrow[on_right])[-8:]          # 8 highest rows = the loaded corner
  load_nodes = np.where(on_right & np.isin(yrow, top_rows))[0]
  load_dofs = 3 * load_nodes + 0                      # x-direction
  force[load_dofs] = -load / len(load_nodes)
  mesh.node_indices[load_nodes, 3] = 2               # for plotting

  bc = bound_cond.BC(force=force, fixed_dofs=fixed_dofs, dirichlet_values=dirichlet_values)

  # Placeholder single material; real A-F properties come from the VAE / Excel in the MMTO pipeline.
  mat_prop = mat_lib.create_material_with_defaults(
      "CustomMaterial", youngs_modulus=1.0, poissons_ratio=0.3, mass_density=1.0, yield_strength=1.0)

  elem_body_force = None

  return mesh, mat_prop, bc, elem_body_force

def createBeam_Liao_Problem(nDOFDesired: int = 60000, traction: float = 1.0):
    """3D supported beam, Liao et al. (2024) Fig.7. QUARTER model, units mm.

    Geometry from Liao's own mesh (Ex1_Beam_Fine.msh) -> Ex1_Beam_design.stl.
    Design domain ~ 300 (x, half-length) x 50 (y, half-width) x 100 (z, height).
    Regions taken from Liao's physical groups (verified against the .msh):
      Traction  z~zmax, x in [0,10],  y in [0,8.7]            -> -z load (t in ksi-z)
      RollerU   x = xmin plane                                -> u_x = 0 (mid-span symmetry)
      RollerV   y = ymin plane                                -> u_y = 0 (longitudinal symmetry)
      RollerW   z = zmin, x in [xmax-10, xmax], y in [ymax-10, ymax] -> u_z = 0 (support, lower corner)
    nDOFDesired/3 ~ #hex elements; 60000 -> ~20k hex (~4 mm). Lower to ~30000 for Liao's 5 mm.
    """
    stl_file = os.path.join(script_dir, './Models/Beam_Liao/Ex1_Beam_design.stl')
    nElemsDesired = nDOFDesired / 3

    mesh = hex_mesher.HexMesher()
    mesh.createMeshFromSTLFile(stl_file, nElemsDesired=nElemsDesired)
    mesh.createEdofMatStructural()

    p = mesh.node_xyz
    x, y, z = p[:, 0], p[:, 1], p[:, 2]
    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()
    zmin, zmax = z.min(), z.max()
    tol = 0.5 * np.min(mesh.elem_size)
    zlayer = 1.5 * np.max(mesh.elem_size)   # a couple of voxels to grab a surface layer

    # --- RollerU: x = xmin -> u_x = 0  (mid-span symmetry) ---
    nU = np.where(np.abs(x - xmin) < tol)[0]
    dofU = (3 * nU + 0).astype(int)

    # --- RollerV: y = ymin -> u_y = 0  (longitudinal symmetry) ---
    nV = np.where(np.abs(y - ymin) < tol)[0]
    dofV = (3 * nV + 1).astype(int)

    # --- RollerW: support, bottom outer-end corner -> u_z = 0 ---
    nW = np.where((z < zmin + zlayer) &
                  (x > xmax - 10.0) & (y > ymax - 10.0))[0]
    dofW = (3 * nW + 2).astype(int)

    fixed_dofs = np.unique(np.concatenate([dofU, dofV, dofW])).astype(int)
    dirichlet_values = np.zeros_like(fixed_dofs, dtype=float)
    mesh.node_indices[nU, 3] = 3
    mesh.node_indices[nV, 3] = 3
    mesh.node_indices[nW, 3] = 1   # support shown as clamp colour


# --- Traction: -z load on top-center patch, total force = t * Liao's actual Gamma_t area ---
    in_xy = (x <= xmin + 11.0) & (y <= ymin + 9.5)
    z_top = z[in_xy].max() if in_xy.any() else zmax
    load_nodes = np.where(in_xy & (z > z_top - zlayer))[0]
    force = np.zeros(3 * mesh.num_nodes)
    if len(load_nodes) > 0:
        LIAO_TRACTION_AREA = 88.515                      # mm^2, exact from Ex1_Beam_Fine.msh (14 tris)
        total_force = traction * LIAO_TRACTION_AREA
        force[3 * load_nodes + 2] = -total_force / len(load_nodes)   # -z, split over captured nodes
        mesh.node_indices[load_nodes, 3] = 2
    else:
        print("[Beam_Liao] WARNING: no load nodes selected")

    bc = bound_cond.BC(force=force, fixed_dofs=fixed_dofs, dirichlet_values=dirichlet_values)

    mat_prop = mat_lib.create_material_with_defaults(
        "CustomMaterial", youngs_modulus=1.0, poissons_ratio=0.37,   # Liao Table 1: nu=0.37
        mass_density=1.0, yield_strength=1.0)

    elem_body_force = None
    return mesh, mat_prop, bc, elem_body_force
import numpy as np
import os
# (script_dir, hex_mesher, bound_cond, mat_lib all already available in StructuralFEAExamples.py)

def createAirplaneBracket_Problem(nDOFDesired: int = 75000, traction: float = 75.0):
    """Airplane bearing bracket (Liao et al. 2024 geometry), z-HALF model.

    Voxelised from AirplaneBracket.stl (Design + NonDesign union; the source
    'External' void box is discarded). BCs are taken from the source .msh
    physical groups by COORDINATE region (the organic surface has ~14.8k tris,
    so L-beam-style triangle-index lists are not usable):

        Fixed    : flat patch at y~0,  x in [-4.6, 41.4],  z in [3.7, 14.1]  -> clamp (all DOF)
        RollerW  : z = 0 plane                                              -> u_z = 0 (symmetry roller)
        Traction : bearing bore, x in [94.5,108.7], y in [-9.2,-2.2], z in [0,3.6] -> -y load

    Non-design: NonDesign collar tagged via AirplaneBracket_NonDesign.stl and stored
    on mesh.nonDesignElems (consumed by the MMTO layer below).
    """
    import pyvista as pv  # mesher already depends on pyvista

    stl_file    = os.path.join(script_dir, './Models/AirplaneBracket/AirplaneBracket.stl')
    nd_stl_file = os.path.join(script_dir, './Models/AirplaneBracket/AirplaneBracket_NonDesign.stl')

    nElemsDesired = nDOFDesired / 3
    mesh = hex_mesher.HexMesher()
    mesh.createMeshFromSTLFile(stl_file, nElemsDesired=nElemsDesired)
    mesh.createEdofMatStructural()

    node_pts = mesh.node_xyz
    tol = 0.5 * np.min(mesh.elem_size)

    # ---- Fixed clamp: y~0 patch, localised in x,z (all DOF) ----
# ---- Clamp: inner faces of the two mounting bores (Liao's Gamma_u), axis ~ y ----
    BOLT_HOLES = [(0.1, 9.5), (36.8, 8.2)]   # accurate (x,z) bore centers
    R_BORE     = 4.5    # bore radius (both ~Ø9mm)
    WALL_BAND  = 1.5    # grab the lining: void interior is empty, so this picks r in [R_BORE, R_BORE+WALL_BAND]
    Y_BORE     = (-3.5, 3.0)   # clamp the inner face over the lower bore depth (raise top for a longer grip)
    clamp_mask = np.zeros(len(node_pts), dtype=bool)
    for (hx, hz) in BOLT_HOLES:
        rr = np.sqrt((node_pts[:, 0] - hx) ** 2 + (node_pts[:, 2] - hz) ** 2)
        clamp_mask |= (rr <= R_BORE + WALL_BAND) & \
                      (node_pts[:, 1] >= Y_BORE[0]) & (node_pts[:, 1] <= Y_BORE[1])
    fixed_nodes = np.where(clamp_mask)[0]
    clamp_dofs = np.array([3*fixed_nodes, 3*fixed_nodes+1, 3*fixed_nodes+2]).flatten().astype(int)
    mesh.node_indices[fixed_nodes, 3] = 1  # plotting

    # ---- RollerW symmetry plane z = 0: u_z = 0 only ----
    sym_nodes = np.where(np.abs(node_pts[:, 2] - 0.0) < tol)[0]
    sym_dofs = (3 * sym_nodes + 2).astype(int)
    mesh.node_indices[sym_nodes, 3] = 3  # plotting

    fixed_dofs = np.unique(np.concatenate([clamp_dofs, sym_dofs])).astype(int)
    dirichlet_values = np.zeros_like(fixed_dofs, dtype=float)

    # ---- Traction: straight-down (-y) net force over the bearing bore ----
    ld = ((node_pts[:, 0] >= 94.5 - tol) & (node_pts[:, 0] <= 108.7 + tol) &
          (node_pts[:, 1] >= -9.2 - tol) & (node_pts[:, 1] <=  -2.2 + tol) &
          (node_pts[:, 2] >=  0.0 - tol) & (node_pts[:, 2] <=   3.6 + tol))
    load_nodes = np.where(ld)[0]
    load_pts = node_pts[load_nodes]
    loaded_area = max(np.ptp(load_pts[:, 0]), tol) * max(np.ptp(load_pts[:, 2]), tol)  # x-z projection for -y load
    total_force = traction * loaded_area
    force = np.zeros(3 * mesh.num_nodes)
    force[3 * load_nodes + 1] = -total_force / max(len(load_nodes), 1)  # -y, downward
    mesh.node_indices[load_nodes, 3] = 2  # plotting

    bc = bound_cond.BC(force=force, fixed_dofs=fixed_dofs, dirichlet_values=dirichlet_values)

    # ---- tag NonDesign voxels (same select_enclosed_points the mesher uses internally) ----
    nd_surf = pv.read(nd_stl_file)
    centers = pv.PolyData(np.asarray(mesh.elem_centers))
    sel = centers.select_enclosed_points(nd_surf, tolerance=0.0, check_surface=False)
    nd_mask = np.asarray(sel.point_data['SelectedPoints']).astype(bool)
    mesh.nonDesignElems = np.where(nd_mask)[0]   # <- consumed by the MMTO layer (EDIT 3)
# Keep the bolt-bore (clamp) elements solid as NON-DESIGN, WITHOUT freezing the symmetry plane.
    ec = np.asarray(mesh.elem_centers)
    clamp_elem_mask = np.zeros(mesh.num_elems, dtype=bool)
    for (hx, hz) in BOLT_HOLES:
        rr = np.sqrt((ec[:, 0] - hx) ** 2 + (ec[:, 2] - hz) ** 2)
        clamp_elem_mask |= (rr <= R_BORE + WALL_BAND) & (ec[:, 1] >= Y_BORE[0]) & (ec[:, 1] <= Y_BORE[1])
    mesh.nonDesignElems = np.unique(np.concatenate([mesh.nonDesignElems, np.where(clamp_elem_mask)[0]])).astype(int)
    print(f"[AirplaneBracket] elems={mesh.num_elems}  elem_size~{np.round(mesh.elem_size,3)}  "
          f"clamp_nodes={len(fixed_nodes)}  sym_nodes={len(sym_nodes)}  "
          f"load_nodes={len(load_nodes)}  nonDesignElems={len(mesh.nonDesignElems)}")

    # placeholder material; real M1/M2/M3 come from the VAE / Excel in the MMTO pipeline
    mat_prop = mat_lib.create_material_with_defaults(
        "CustomMaterial", youngs_modulus=1.0, poissons_ratio=0.3, mass_density=1.0, yield_strength=1.0)
    elem_body_force = None
    return mesh, mat_prop, bc, elem_body_force
