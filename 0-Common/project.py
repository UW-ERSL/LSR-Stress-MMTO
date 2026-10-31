"""
project.py
==========
One MMTO problem as a project: geometry, named selections, boundary conditions,
keep-solid regions, material database, objective and constraints, and settings.
A project is saved in a PyTO-AD project file (.pyto), in its own "mmto" section;
the rest of what the PyTO-AD GUI saved in the same file is kept, except its supports
and loads, which become part of the mmto section (see below).

    from project import Project
    from selections import face, box
    p = Project("bracket", stl="bracket.stl", n_dof=30000)
    p.add_selection("left", face("xmin"))
    p.add_support("left", dofs="xyz")
    p.add_force(box([9, 0, 0], [10, 10, 1]), total=[0, -1000, 0])
    p.database = "6Materials_AF_3Attr"
    p.objective("FailureFactorPNorm()")
    p.constraint("Mass()", "<=", 3.5e-3)
    p.save("bracket.pyto")
    q = Project.load("bracket.pyto")
    mesh, mat_prop, bc, body_force, to_params, vae_params = q.build()
    run(q)                                  # optimizes; results and a copy of q in results/bracket
    q.to_script("bracket_setup.py")         # the same project as Python calls, to read or edit

build() returns the mesh, the material, the boundary conditions, the body force and the
optimization and VAE parameters that run_topopt starts from.

Settings: every setting of run_topopt is stored, under run_topopt's own name, in
settings.common, settings.advanced and settings.vae (DEFAULT_SETTINGS below). A new
project starts from the settings of the paper's examples; a file missing some is
completed from the same defaults, and the run says which.

Boundary conditions (each acts on a named selection, evaluated on the mesh nodes):
    fixed          dofs "x", "y", "z" or any combination, value 0
    displacement   dofs as above, a nonzero imposed value
    force          a total force [Fx, Fy, Fz], split equally over the nodes
    traction       pressure * area along a direction, split equally over the nodes;
                   area is a number, or "bbox": the bounding box of the loaded nodes
                   seen along an axis-aligned direction (each side at least half an
                   element). "bbox" is approximate; it is there for the airplane bracket.
Supports and loads picked in the PyTO-AD GUI (its "structuralBC" section) are turned into
selections and boundary conditions here when the project is loaded, with the GUI's own rules
(stl_faces with tol 0.8, nearest boundary node, halfspaces), and the GUI's lists are emptied:
the mmto section is the whole problem.

Forces are used as given, with no unit conversion: they must be in the units of the
material database.

Objective and constraints are written as PyTO-AD stores them: an expression, a sense
(minimize) or an operator (<=) and a bound. An expression is exactly one quantity of the
registry in AutodiffPureStructural.py, e.g. "Mass()", "FailureFactorPNorm()",
"MaterialVolumeFraction(2, tau=0.5)".
"""
import json
import os
import re

import numpy as np

import selections as S
from databases import (DATABASE_DIR, DATABASE_VAE, VAEParams, check_database, database_attributes,
                       recommended_latent_dim)

SCHEMA_VERSION = 1
# _UPGRADES[n] turns an mmto section of version n into version n + 1. When the format
# changes: raise SCHEMA_VERSION and add the function here; old files then still load.
_UPGRADES = {}
_AXES = {"x": 0, "y": 1, "z": 2}
_MARK_FIXED, _MARK_LOAD, _MARK_ROLLER = 1, 2, 3

# Every setting of run_topopt a project stores, under run_topopt's own names, with the value
# a new project starts from: the settings of the paper's examples (comparison/run_both.py).
# A setting missing from a file is filled from here and reported; a setting not listed here
# is an error. material_filter_radius None means "the same as relative_filter_radius";
# saveNet None means the .nt file next to the database's .xlsx.
DEFAULT_SETTINGS = {
    "common": {"maxIterations": 300, "relative_filter_radius": 2.5, "material_filter_radius": None,
               "pnorm_exponent": 6.0, "beta_schedule_scheme": "PAPER"},
    "advanced": {"timeLimit": 36000, "rel_conv_tol": 1e-7,
                 "use_penalization": True, "gamma_init": 1e-3, "gamma_max": 10.0, "gamma_factor": 1.1,
                 "penalty_start_iter": 0, "distance_penalty_disable_after_iter": 250,
                 "use_dynamic_gamma_scheduling": True,
                 "use_z_smoothing": True, "z_smoothing_weight": 1e-4, "z_smoothing_disable_after_iter": 250,
                 "use_heaviside_projection": True, "eta_proj": 0.5, "use_continuation": False,
                 "use_constant_poissons_ratio": False, "constant_poissons_ratio": 0.3,
                 "snap_to_real_material": False, "z0_init_method": "ORIGIN", "binarize_topology": True},
    "vae": {"use_pretrained_vae": True, "saveNet": None},
}
# one line per setting, for docs/project_reference.md (and later the GUI)
SETTING_DOCS = {
    "maxIterations": "largest number of MMA iterations",
    "relative_filter_radius": "density filter radius, in element sizes",
    "material_filter_radius": "radius of the latent smoothing neighbourhood, in element sizes; None = relative_filter_radius",
    "pnorm_exponent": "p of the p-norm that aggregates stress or failure factor",
    "beta_schedule_scheme": "Heaviside sharpness schedule: beta = 1, then x1.5 every 5 iterations after 50, "
                            "up to 32 (PAPER) or 16 (PAPER_BM1)",
    "timeLimit": "wall-clock limit of the optimization, seconds",
    "rel_conv_tol": "stop when the objective and constraints change by less than this, relatively",
    "use_penalization": "penalty on the distance from each solid element's latent point to the nearest real material",
    "gamma_init": "starting weight of that penalty",
    "gamma_max": "largest weight of that penalty",
    "gamma_factor": "the weight is multiplied by this each iteration while it ramps up",
    "penalty_start_iter": "first iteration of the penalty, without dynamic scheduling",
    "distance_penalty_disable_after_iter": "iteration at which the penalty is switched off, without dynamic scheduling",
    "use_dynamic_gamma_scheduling": "start the weight's ramp when the objective plateaus, instead of at fixed iterations",
    "use_z_smoothing": "penalty on latent points that differ from their neighbours' (material flicker)",
    "z_smoothing_weight": "weight of that penalty",
    "z_smoothing_disable_after_iter": "iteration at which that penalty is switched off",
    "use_heaviside_projection": "push the filtered density toward 0 or 1",
    "eta_proj": "threshold of that projection",
    "use_continuation": "raise the SIMP penalty by 0.25 every 10 iterations",
    "use_constant_poissons_ratio": "use constant_poissons_ratio instead of the decoded Poisson's ratio",
    "constant_poissons_ratio": "the Poisson's ratio used when it is held constant",
    "snap_to_real_material": "in the reported design, replace each latent point by the nearest real material's",
    "z0_init_method": "starting latent point: ORIGIN (z = 0), LIGHTEST, HEAVIEST material, or UNIFORM random",
    "binarize_topology": "in the reported design, round the density to 0 or 1, keeping the volume",
    "use_pretrained_vae": "load the trained VAE from saveNet if it exists; otherwise train one",
    "saveNet": "the VAE file; None = the .nt file next to the database's .xlsx",
    "latentDim": "latent dimension of the VAE",
    "learningRate": "VAE training learning rate",
    "vae_hiddenDim": "width of the VAE's hidden layers",
    "maxAttributeErrorPercent": "VAE training stops once every attribute is reproduced within this percent",
    "klFactor": "weight of the KL term in VAE training",
    "numEpochs": "largest number of VAE training epochs",
}
# the VAE's own settings, also in settings.vae; they start from the database's known values
VAE_PARAMS = ("latentDim", "learningRate", "vae_hiddenDim", "maxAttributeErrorPercent", "klFactor", "numEpochs")
# run_topopt arguments about one run's output, not the problem; n_dof_desired is mesh.n_dof
NOT_STORED = ("to_problem", "save_dir", "plot_progress", "plot_bcs", "verbose", "plotter")


def _dofs(dofs):
    if not dofs or any(c not in _AXES for c in dofs) or len(set(dofs)) != len(dofs):
        raise ValueError(f"dofs must be letters from 'xyz', e.g. 'xyz' or 'y', not {dofs!r}")
    return [_AXES[c] for c in sorted(dofs, key="xyz".index)]


def upgrade(m, name="project"):
    """An mmto section brought to the current SCHEMA_VERSION, through _UPGRADES."""
    m = dict(m)
    version = m.get("schema_version", SCHEMA_VERSION)
    if version > SCHEMA_VERSION:
        raise ValueError(f"{name}: written by a newer version (schema {version})")
    while version < SCHEMA_VERSION:
        if version not in _UPGRADES:
            raise ValueError(f"{name}: schema {version} has no upgrade to {version + 1}")
        m = _UPGRADES[version](m)
        version += 1
        m["schema_version"] = version
    return m


# ---------------------------------------------------------------------- boundary conditions
class BCKind:
    """One kind of boundary condition. role is "support" or "load"; check(bc) gives a list
    of errors; apply(bc, nodes, mesh, state, where) puts it on the mesh, in
    state["values"] (dof -> imposed value) or state["force"]; script(bc, on) is the
    Project call that makes it. Supports are applied before loads, each in the order given."""

    def __init__(self, doc, role, check, apply, script):
        self.doc, self.role, self.check, self.apply, self.script = doc, role, check, apply, script


def _check_dofs(bc):
    try:
        _dofs(bc.get("dofs"))
        return []
    except ValueError as e:
        return [str(e)]


def _check_direction(bc):
    d = np.asarray(bc.get("direction", []), dtype=float)
    if d.shape != (3,) or not np.any(d):
        return ["direction must be three numbers, not all zero"]
    if bc.get("area") == "bbox" and np.count_nonzero(d) != 1:
        return ["area 'bbox' needs a direction along x, y or z; give a number instead"]
    return []


def _apply_support(bc, n, mesh, state, where):
    values, v = state["values"], float(bc.get("value", 0.0))
    for d in (3 * n[:, None] + np.array(_dofs(bc["dofs"]))[None, :]).ravel():
        if values.get(int(d), v) != v:
            raise ValueError(f"{where}: dof {int(d)} is already given the value {values[int(d)]}, here {v}")
        values[int(d)] = v
    mesh.node_indices[n, 3] = _MARK_FIXED if len(bc["dofs"]) == 3 else _MARK_ROLLER


def _add_load(total, n, mesh, state):
    state["force"].reshape(-1, 3)[n] += total / len(n)
    mesh.node_indices[n, 3] = _MARK_LOAD


def _apply_force(bc, n, mesh, state, where):
    _add_load(np.asarray(bc["total"], dtype=float), n, mesh, state)


def _apply_traction(bc, n, mesh, state, where):
    d = np.asarray(bc["direction"], dtype=float)
    d = d / np.linalg.norm(d)
    area = bc["area"]
    if area == "bbox":
        tol = 0.5 * float(np.min(mesh.elem_size))
        p = np.asarray(mesh.node_xyz, dtype=float)[n]
        a, b = [k for k in range(3) if d[k] == 0]
        area = max(np.ptp(p[:, a]), tol) * max(np.ptp(p[:, b]), tol)
    _add_load(bc["pressure"] * area * d, n, mesh, state)


BC_KINDS = {
    "fixed": BCKind(
        "the given dofs (x, y, z) held at 0", "support", _check_dofs, _apply_support,
        lambda bc, on: f"p.add_support({on}, dofs={bc['dofs']!r})"),
    "displacement": BCKind(
        "the given dofs held at a nonzero value", "support", _check_dofs, _apply_support,
        lambda bc, on: f"p.add_support({on}, dofs={bc['dofs']!r}, value={bc['value']!r})"),
    "force": BCKind(
        "a total force [Fx, Fy, Fz], split equally over the nodes", "load", lambda bc: [], _apply_force,
        lambda bc, on: f"p.add_force({on}, total={bc['total']!r})"),
    "traction": BCKind(
        "pressure * area along a direction, split equally over the nodes; area a number or 'bbox'",
        "load", _check_direction, _apply_traction,
        lambda bc, on: f"p.add_traction({on}, pressure={bc['pressure']!r}, "
                       f"direction={bc['direction']!r}, area={bc['area']!r})"),
}


def _from_repo(path):
    """path relative to the repository folder, as one types it there."""
    from utils.project_paths import find_root                               # type: ignore
    root = find_root(os.path.dirname(path))
    return os.path.relpath(path, root).replace("\\", "/") if root else os.path.basename(path)


def parse_expression(text):
    """'Name(a, b=1)' -> ('Name', [a], {'b': 1}). Only a single quantity is accepted."""
    m = re.fullmatch(r"\s*([A-Za-z_]\w*)\s*\(([^()]*)\)\s*", text or "")
    if not m:
        raise ValueError(f"expression {text!r}: write one quantity, e.g. 'Mass()'; "
                         "sums and products of quantities are not supported yet")
    name, inside = m.group(1), m.group(2).strip()
    args, kwargs = [], {}
    for part in [p.strip() for p in inside.split(",")] if inside else []:
        key, eq, value = part.partition("=")
        try:
            v = float(value if eq else key)
        except ValueError:
            raise ValueError(f"expression {text!r}: argument {part!r} is not a number") from None
        if eq:
            kwargs[key.strip()] = v
        else:
            args.append(v)
    return name, args, kwargs


def to_params_form(text, role, bound=None):
    """The old to_params tuple for an expression: (TO_QOI, option) or (TO_QOI, option, bound)."""
    from AutodiffPureStructural import QUANTITIES          # type: ignore
    from PyTOImports import TO_QOI                         # type: ignore
    name, args, kwargs = parse_expression(text)
    by_name = {q.name: q for q in QUANTITIES}
    if name not in by_name:
        raise ValueError(f"expression {text!r}: unknown quantity {name!r}; "
                         f"known: {', '.join(by_name)}")
    q = by_name[name]
    form = q.objective if role == "objective" else q.constraint
    if form is None:
        raise ValueError(f"{name} cannot be the {role}")
    qoi, kind = form
    if kind == "material":
        if len(args) != 1 or set(kwargs) - {"tau"}:
            raise ValueError(f"expression {text!r}: write MaterialVolumeFraction(material, tau=...)")
        opt = {"material_id": int(args[0]), "tau": float(kwargs.get("tau", 0.5))}
    else:
        if args or kwargs:
            raise ValueError(f"expression {text!r}: {name} takes no arguments")
        opt = kind
    return (TO_QOI[qoi], opt) if role == "objective" else (TO_QOI[qoi], opt, float(bound))


class Project:
    def __init__(self, name, stl, n_dof=30000, single_layer=False, base_dir=None):
        self.name = name
        self.stl = stl                          # as given; relative paths are relative to base_dir
        self.base_dir = os.path.abspath(base_dir or os.getcwd())
        self.n_dof, self.single_layer = int(n_dof), bool(single_layer)
        self.comment = ""
        self.physics = "structural"
        self.selections = {}                    # name -> Selection
        self.bcs = []                           # dicts, see BC_KINDS
        self.keep_solid_names = []
        self.keep_support_elements = False
        self.database = None
        self.formulation = {"objective": None, "constraints": []}
        self.settings = {k: dict(v) for k, v in DEFAULT_SETTINGS.items()}
        self._gui = {}                          # everything else in the .pyto, kept for the GUI

    # ------------------------------------------------------------------ building it up
    def add_selection(self, name, selection):
        if not isinstance(selection, S.Selection):
            raise TypeError(f"{name}: a Selection is expected, e.g. face('xmin')")
        self.selections[name] = selection
        return name

    def _free_name(self, prefix):
        i = 1
        while f"{prefix}{i}" in self.selections:
            i += 1
        return f"{prefix}{i}"

    def _on(self, on, kind):
        """A selection name; a Selection given directly is stored under a generated name."""
        if isinstance(on, S.Selection):
            return self.add_selection(self._free_name(kind), on)
        return on

    def _adopt_gui_picks(self):
        """Supports and loads picked in the PyTO-AD GUI become selections and boundary
        conditions of this project, with the GUI's own rules; the GUI's lists are emptied,
        so the problem is defined in one place."""
        from pyto_project import _SUPPORT_DOFS, _supports_and_loads                  # type: ignore
        supports, loads = _supports_and_loads(self._gui) if self._gui else ([], [])
        if not supports and not loads:
            return []

        def selection(triangles, points, regions, what):
            parts = [S.stl_faces(triangles, tol=0.8)] if len(triangles) else []
            parts += [S.nearest(point, boundary=True) for point in points]
            parts += [S.halfspaces(r["normals"], r["origins"]) for r in regions]
            if not parts:
                raise ValueError(f"[{self.name}] {what} picked in the PyTO-AD GUI has no faces, "
                                 f"nodes or boxes")
            return parts[0] if len(parts) == 1 else S.union(*parts)

        names = []
        for i, (kind, triangles, points, regions) in enumerate(supports):
            name = self.add_selection(self._free_name("gui_support"),
                                      selection(triangles, points, regions, f"support {i + 1}"))
            self.add_support(name, dofs="".join("xyz"[d] for d in _SUPPORT_DOFS[kind]))
            names.append(name)
        for i, (triangles, points, regions, f) in enumerate(loads):
            name = self.add_selection(self._free_name("gui_load"),
                                      selection(triangles, points, regions, f"load {i + 1}"))
            self.add_force(name, total=f)
            names.append(name)
        sbc = self._gui.get("structuralBC")
        if isinstance(sbc, dict):
            self._gui["structuralBC"] = {k: [] if isinstance(v, list) else
                                         {kk: 0 for kk in v} if isinstance(v, dict) else v
                                         for k, v in sbc.items()}
        for k in ("constraint_data", "force_data"):
            if k in self._gui:
                self._gui[k] = []
        print(f"[{self.name}] the supports and loads picked in the PyTO-AD GUI are now selections "
              f"{', '.join(names)} of this project")
        return names

    def add_support(self, on, dofs="xyz", value=0.0):
        _dofs(dofs)
        kind = "fixed" if float(value) == 0.0 else "displacement"
        bc = {"kind": kind, "on": self._on(on, "support"), "dofs": dofs}
        if kind == "displacement":
            bc["value"] = float(value)
        self.bcs.append(bc)

    def add_force(self, on, total):
        self.bcs.append({"kind": "force", "on": self._on(on, "load"), "total": [float(v) for v in total]})

    def add_traction(self, on, pressure, direction, area="bbox"):
        self.bcs.append({"kind": "traction", "on": self._on(on, "load"), "pressure": float(pressure),
                         "direction": [float(v) for v in direction],
                         "area": area if area == "bbox" else float(area)})

    def keep_solid(self, *names):
        for n in names:
            n = self._on(n, "keep")
            if n not in self.keep_solid_names:
                self.keep_solid_names.append(n)

    def objective(self, expression, sense="minimize"):
        self.formulation["objective"] = {"expression": expression, "sense": sense}

    def constraint(self, expression, op, bound):
        self.formulation["constraints"].append({"expression": expression, "op": op, "bound": float(bound)})

    # ------------------------------------------------------------------ checks without a mesh
    def stl_path(self):
        return self.stl if os.path.isabs(self.stl) else os.path.join(self.base_dir, self.stl)

    def database_paths(self):
        """(xlsx, VAE defaults or None) for the database name or path."""
        if self.database in DATABASE_VAE:
            return os.path.join(DATABASE_DIR, f"{self.database}.xlsx"), DATABASE_VAE[self.database]
        path = self.database if os.path.isabs(self.database) else os.path.join(self.base_dir, self.database)
        return path, None

    def validate(self):
        """Problems that can be found without building the mesh; an empty list means none."""
        errors = []
        if not os.path.isfile(self.stl_path()):
            errors.append(f"STL file {self.stl_path()} does not exist")
        if self.physics != "structural":
            errors.append(f"physics {self.physics!r}: only 'structural' for now")
        for i, bc in enumerate(self.bcs, 1):
            where = f"boundary condition {i} ({bc.get('kind')})"
            if bc.get("kind") not in BC_KINDS:
                errors.append(f"{where}: kind must be one of {', '.join(BC_KINDS)}")
                continue
            if bc.get("on") not in self.selections:
                errors.append(f"{where}: no selection named {bc.get('on')!r}")
            errors += [f"{where}: {e}" for e in BC_KINDS[bc["kind"]].check(bc)]
        roles = {BC_KINDS[bc["kind"]].role for bc in self.bcs if bc.get("kind") in BC_KINDS}
        if "support" not in roles:
            errors.append("no supports")
        if "load" not in roles:
            errors.append("no loads")
        for n in self.keep_solid_names:
            if n not in self.selections:
                errors.append(f"keep_solid: no selection named {n!r}")
        if self.database is None:
            errors.append("no material database")
        else:
            xlsx, _ = self.database_paths()
            db_errors = check_database(xlsx)[0]
            errors += [f"material database {os.path.basename(xlsx)}: {e}" for e in db_errors]
            if not db_errors:
                errors += self._missing_columns(database_attributes(xlsx))
        for section, values in self.settings.items():
            known = set(DEFAULT_SETTINGS.get(section, {})) | (set(VAE_PARAMS) if section == "vae" else set())
            if section not in DEFAULT_SETTINGS:
                errors.append(f"settings: unknown section {section!r}")
                continue
            for k in values:
                if k not in known:
                    errors.append(f"settings.{section}: unknown setting {k!r}")
        from MainPureStructural import BetaScheduleScheme, Z0InitMethod                # type: ignore
        for section, key, enum in (("common", "beta_schedule_scheme", BetaScheduleScheme),
                                   ("advanced", "z0_init_method", Z0InitMethod)):
            v = self.settings.get(section, {}).get(key)
            if v is not None and v not in enum.__members__:
                errors.append(f"settings.{section}.{key}: {v!r} is not one of {', '.join(enum.__members__)}")
        obj = self.formulation.get("objective")
        if not obj:
            errors.append("no objective")
        else:
            if obj.get("sense", "minimize") != "minimize":
                errors.append("objective: only 'minimize' for now")
            try:
                to_params_form(obj.get("expression"), "objective")
            except ValueError as e:
                errors.append(f"objective: {e}")
        for i, c in enumerate(self.formulation.get("constraints", []), 1):
            if c.get("op") != "<=":
                errors.append(f"constraint {i}: only '<=' for now")
            if not c.get("bound"):
                errors.append(f"constraint {i}: the bound must be a nonzero number")
            try:
                to_params_form(c.get("expression"), "constraint", c.get("bound") or 1.0)
            except ValueError as e:
                errors.append(f"constraint {i}: {e}")
        return errors

    def _missing_columns(self, columns):
        """Columns the physics and the objective and constraints read that the database lacks."""
        from AutodiffPureStructural import PHYSICS_NEEDS, QUANTITIES                  # type: ignore
        out = [f"material database: the {self.physics} model needs a {c} column"
               for c in PHYSICS_NEEDS.get(self.physics, ()) if c not in columns]
        by_name = {q.name: q for q in QUANTITIES}
        f = self.formulation
        for expr in [(f.get("objective") or {}).get("expression")] + \
                    [c.get("expression") for c in f.get("constraints", [])]:
            try:
                q = by_name.get(parse_expression(expr)[0])
            except Exception:                                                          # noqa: BLE001
                continue                                    # a bad expression is reported on its own
            for c in (q.needs if q else ()):
                if c not in columns:
                    out.append(f"material database: {q.name} needs a {c} column")
        return out

    def warnings(self):
        """What is allowed but deserves a look before a run: the database's warnings, and a
        latent dimension other than the one recommended for the database."""
        out = []
        if self.database is None:
            return out
        xlsx, _ = self.database_paths()
        errors, db_warnings = check_database(xlsx)
        if errors:
            return out
        out += [f"material database {os.path.basename(xlsx)}: {w}" for w in db_warnings]
        d_eff, n_z = recommended_latent_dim(xlsx)
        latent = self.settings.get("vae", {}).get("latentDim")
        if latent is not None and int(latent) != n_z:
            out.append(f"latentDim is {latent}; the database's effective dimension is d_eff = {d_eff:.2f}, "
                       f"so {n_z} is recommended. A latent space larger than the database needs has "
                       f"stopped runs from converging before.")
        return out

    def complete_settings(self):
        """Fill every missing setting from the defaults; returns what was filled."""
        filled = []
        for section, defaults in DEFAULT_SETTINGS.items():
            values = self.settings.setdefault(section, {})
            for k, v in defaults.items():
                if k not in values:
                    values[k] = v
                    filled.append(f"{section}.{k}")
        vae_defaults = {k: getattr(VAEParams(), k) for k in VAE_PARAMS}
        if self.database is not None:
            xlsx, known = self.database_paths()
            if known:                                       # a shipped database: what its network was trained with
                vae_defaults.update(known)
            elif "latentDim" not in self.settings["vae"] and not check_database(xlsx)[0]:
                vae_defaults["latentDim"] = recommended_latent_dim(xlsx)[1]
        for k in VAE_PARAMS:
            if k not in self.settings["vae"]:
                self.settings["vae"][k] = vae_defaults[k]
                filled.append(f"vae.{k}")
        return filled

    # ------------------------------------------------------------------ the problem
    def build(self, verbose=True):
        """mesh, mat_prop, bc, elem_body_force, to_params, vae_params."""
        filled = self.complete_settings()
        if filled and verbose:
            print(f"[{self.name}] settings not in the project, taken from the defaults: {', '.join(filled)}")
        errors = self.validate()
        if errors:
            raise ValueError(f"project {self.name!r}:\n  " + "\n  ".join(errors))
        from PyTOImports import (bound_cond, mat_lib, topopt_material_model,          # type: ignore
                                 find_elements_with_fixedDOF)
        from mmto_common import MMTOParams                                            # type: ignore
        from pyto_project import make_mesh                                            # type: ignore

        mesh = make_mesh(self.stl_path(), self.n_dof, self.single_layer)
        nodes_of = {}

        def nodes(name, where):
            if name not in nodes_of:
                nodes_of[name] = self.selections[name].evaluate(mesh, "nodes", self.base_dir)
            if len(nodes_of[name]) == 0:
                raise ValueError(f"{where}: selection {name!r} has no nodes on this mesh")
            return nodes_of[name]

        # ---- supports, then loads (BC_KINDS)
        state = {"values": {}, "force": np.zeros(3 * mesh.num_nodes)}   # values: dof -> imposed value
        for role in ("support", "load"):
            for i, bc in enumerate(self.bcs, 1):
                kind = BC_KINDS[bc["kind"]]
                if kind.role == role:
                    where = f"boundary condition {i}"
                    kind.apply(bc, nodes(bc["on"], where), mesh, state, where)
        values, force = state["values"], state["force"]

        fixed_dofs = np.array(sorted(values), dtype=int)
        dirichlet_values = np.array([values[d] for d in fixed_dofs], dtype=float)
        bc = bound_cond.BC(force=force, fixed_dofs=fixed_dofs, dirichlet_values=dirichlet_values)

        # ---- optimization parameters
        common = self.settings.get("common", {})
        to_params = MMTOParams()
        to_params.Comment = self.comment or self.name
        to_params.ExtrudeZ = self.single_layer
        to_params.KeepFixedElems = self.keep_support_elements
        to_params.nDOFDesired = self.n_dof
        to_params.RelativeFilterRadius = float(common["relative_filter_radius"])
        if common["material_filter_radius"] is not None:
            to_params.MaterialFilterRadius = float(common["material_filter_radius"])
        topopt_material_model._PNORM_EXPONENT = float(common["pnorm_exponent"])
        xlsx, _ = self.database_paths()
        to_params.MaterialsExcelFile = xlsx
        obj = self.formulation["objective"]
        to_params.Objective = to_params_form(obj["expression"], "objective")
        to_params.Constraints = [to_params_form(c["expression"], "constraint", c["bound"])
                                 for c in self.formulation.get("constraints", [])]

        vae_params = VAEParams()
        for k in VAE_PARAMS:
            setattr(vae_params, k, self.settings["vae"][k])

        # ---- keep-solid elements
        to_params.ElemsToKeep = None
        if self.keep_support_elements:
            to_params.ElemsToKeep = find_elements_with_fixedDOF(mesh, bc, nDOFPerNode=3)
        if self.keep_solid_names:
            keep = np.unique(np.concatenate([self.selections[n].evaluate(mesh, "elements", self.base_dir)
                                             for n in self.keep_solid_names])).astype(int)
            mesh.nonDesignElems = keep
            to_params.ElemsToKeep = keep if to_params.ElemsToKeep is None else np.union1d(to_params.ElemsToKeep, keep)

        mat_prop = mat_lib.create_material_with_defaults(          # placeholder; the VAE gives the real ones
            "CustomMaterial", youngs_modulus=1.0, poissons_ratio=0.3, mass_density=1.0, yield_strength=1.0)
        if verbose:
            print(f"[{self.name}] elems={mesh.num_elems}  h={np.round(mesh.elem_size, 4).tolist()}  "
                  f"fixed dofs={len(fixed_dofs)}  total force={np.round(force.reshape(-1, 3).sum(0), 6).tolist()}  "
                  f"keep solid={0 if to_params.ElemsToKeep is None else len(to_params.ElemsToKeep)}")
        return mesh, mat_prop, bc, None, to_params, vae_params

    # ------------------------------------------------------------------ files
    def to_dict(self):
        return {"schema_version": SCHEMA_VERSION, "name": self.name, "physics": self.physics,
                "comment": self.comment,
                "mesh": {"n_dof": self.n_dof, "single_layer": self.single_layer},
                "selections": {k: s.to_dict() for k, s in self.selections.items()},
                "bcs": [dict(bc) for bc in self.bcs],
                "keep_solid": list(self.keep_solid_names),
                "keep_support_elements": self.keep_support_elements,
                "database": self.database,
                "formulation": {"objective": self.formulation.get("objective"),
                                "constraints": [dict(c) for c in self.formulation.get("constraints", [])]},
                "settings": {k: dict(v) for k, v in self.settings.items()}}

    def _file_data(self, path):
        """The JSON a project file at path holds, with the STL path relative to it."""
        data = dict(self._gui)
        stl = self.stl_path()
        try:
            stl = os.path.relpath(stl, os.path.dirname(path))
        except ValueError:                                      # another drive on Windows
            pass
        data["stl_file_path"] = stl.replace("\\", "/")
        data["mmto"] = self.to_dict()
        return data

    def save(self, path):
        """Write the project, every setting included, with the other sections it was loaded with."""
        path = os.path.abspath(path)
        self.complete_settings()
        data = self._file_data(path)
        with open(path, "w") as fh:
            json.dump(data, fh, indent=2)
        self.base_dir, self.stl = os.path.dirname(path), data["stl_file_path"]

    def to_script(self, path):
        """Write a Python script that rebuilds this project with the same calls, next to it.

        Settings that differ from a new project's are written; the rest are the defaults.        """
        path = os.path.abspath(path)
        if os.path.normcase(os.path.dirname(path)) != os.path.normcase(self.base_dir):
            raise ValueError(f"save the script in {self.base_dir}, next to the project, so that the "
                             "relative paths in it stay valid")
        stl = self.stl if not os.path.isabs(self.stl) else os.path.relpath(self.stl, self.base_dir)
        stl = stl.replace("\\", "/")
        reference = Project(self.name, stl, base_dir=self.base_dir)
        reference.database = self.database
        reference.complete_settings()
        mine = self.copy()
        mine.complete_settings()
        types = sorted({type(s).__name__.lower() for s in self.selections.values()} |
                       {type(t).__name__.lower() for s in self.selections.values()
                        for t in getattr(s, "parts", ())})
        lines = [f'"""{self.name}: rebuilds the project {self.name!r}. Written by Project.to_script.',
                 "",
                 f"    python {_from_repo(path)}       (from the repository folder; saves {self.name}.pyto next to it)",
                 '"""',
                 "import os",
                 "import sys",
                 "",
                 "sys.path.insert(0, os.getcwd())",
                 "from utils import project_paths",
                 "",
                 "project_paths.bootstrap()",
                 "",
                 "from project import Project",
                 f"from selections import {', '.join(types)}" if types else "",
                 "",
                 "HERE = os.path.dirname(os.path.abspath(__file__))",
                 f"p = Project({self.name!r}, stl={stl!r}, n_dof={self.n_dof}, single_layer={self.single_layer},",
                 "            base_dir=HERE)"]
        if self.comment:
            lines.append(f"p.comment = {self.comment!r}")
        if self.physics != "structural":
            lines.append(f"p.physics = {self.physics!r}")
        lines.append("")
        for name, sel in self.selections.items():
            lines.append(f"p.add_selection({name!r}, {sel!r})")
        for bc in self.bcs:
            lines.append(BC_KINDS[bc["kind"]].script(bc, repr(bc["on"])))
        if self.keep_solid_names:
            lines.append(f"p.keep_solid({', '.join(map(repr, self.keep_solid_names))})")
        if self.keep_support_elements:
            lines.append("p.keep_support_elements = True")
        lines += ["", f"p.database = {self.database!r}"]
        obj = self.formulation.get("objective")
        if obj:
            lines.append(f"p.objective({obj['expression']!r})")
        for c in self.formulation.get("constraints", []):
            lines.append(f"p.constraint({c['expression']!r}, {c['op']!r}, {c['bound']!r})")
        changed = [(sec, k, v) for sec, values in mine.settings.items() for k, v in values.items()
                   if reference.settings.get(sec, {}).get(k, object()) != v]
        if changed:
            lines += ["", "# settings that differ from a new project's; the rest are the defaults"]
            lines += [f"p.settings[{sec!r}][{k!r}] = {v!r}" for sec, k, v in changed]
        lines += ["", f"p.save(os.path.join(HERE, {self.name + '.pyto'!r}))", ""]
        with open(path, "w") as fh:
            fh.write("\n".join(lines))
        return path

    def copy(self, **changes):
        import copy
        p = copy.deepcopy(self)
        for k, v in changes.items():
            setattr(p, k, v)
        return p

    def to_record(self):
        """The project as a run record stores it."""
        return {"project": self.to_dict(), "stl_file_path": self.stl_path(), "base_dir": self.base_dir,
                "gui": self._gui}

    @classmethod
    def from_record(cls, record):
        data = dict(record.get("gui") or {}, mmto=record["project"], stl_file_path=record["stl_file_path"])
        return cls._from_data(data, record["base_dir"], record["project"].get("name", "project"))

    @classmethod
    def load(cls, path):
        path = os.path.abspath(path)
        with open(path) as fh:
            data = json.load(fh)
        return cls._from_data(data, os.path.dirname(path), os.path.splitext(os.path.basename(path))[0])

    @classmethod
    def _from_data(cls, data, base_dir, default_name):
        m = upgrade(data.get("mmto") or {}, default_name)
        mesh = m.get("mesh", {})
        p = cls(m.get("name") or default_name, data.get("stl_file_path", ""),
                n_dof=mesh.get("n_dof", 30000), single_layer=mesh.get("single_layer", False),
                base_dir=base_dir)
        p.comment, p.physics = m.get("comment", ""), m.get("physics", "structural")
        p.selections = {k: S.from_dict(v) for k, v in m.get("selections", {}).items()}
        p.bcs = [dict(bc) for bc in m.get("bcs", [])]
        p.keep_solid_names = list(m.get("keep_solid", []))
        p.keep_support_elements = bool(m.get("keep_support_elements", False))
        p.database = m.get("database")
        f = m.get("formulation") or {}
        p.formulation = {"objective": f.get("objective"), "constraints": list(f.get("constraints", []))}
        stored = m.get("settings") or {}
        p.settings = {k: dict(stored.get(k, {})) for k in DEFAULT_SETTINGS}    # missing ones: filled at build
        p.settings.update({k: dict(v) for k, v in stored.items() if k not in DEFAULT_SETTINGS})
        p._gui = {k: v for k, v in data.items() if k not in ("mmto", "stl_file_path")}
        p._adopt_gui_picks()
        return p


# ---------------------------------------------------------------------- running
def run_arguments(project, save_dir=None, verbose=True, plot_progress=False, plot_bcs=False, plotter=None):
    """The keyword arguments run_topopt gets for this project."""
    from MainPureStructural import BetaScheduleScheme, Z0InitMethod                    # type: ignore
    project.complete_settings()
    args = dict(project.settings["common"], **project.settings["advanced"],
                use_pretrained_vae=project.settings["vae"]["use_pretrained_vae"],
                saveNet=project.settings["vae"]["saveNet"])
    args["beta_schedule_scheme"] = BetaScheduleScheme[args["beta_schedule_scheme"]]
    args["z0_init_method"] = Z0InitMethod[args["z0_init_method"]]
    args.update(to_problem=project, n_dof_desired=project.n_dof,
                save_dir=save_dir if save_dir is not None else os.path.join("results", project.name),
                verbose=verbose, plot_progress=plot_progress, plot_bcs=plot_bcs, plotter=plotter)
    return args


def run(project, save_dir=None, **run_options):
    """Optimize the project with run_topopt; a copy of the project goes into save_dir with the results."""
    import run_io                                                                      # type: ignore
    from MainPureStructural import run_topopt                                          # type: ignore
    filled = project.complete_settings()
    if filled:
        print(f"[{project.name}] settings not in the project, taken from the defaults: {', '.join(filled)}")
    errors = project.validate()
    if errors:
        raise ValueError(f"project {project.name!r}:\n  " + "\n  ".join(errors))
    for w in project.warnings():
        print(f"[{project.name}] warning: {w}")
    snapshot = project.copy()                    # the run gets, and records, the project as it is now
    args = run_arguments(snapshot, save_dir, **run_options)
    if run_io.WRITE_FILES:
        os.makedirs(args["save_dir"], exist_ok=True)
        path = os.path.abspath(os.path.join(args["save_dir"], f"{project.name}.pyto"))
        with open(path, "w") as fh:
            json.dump(snapshot._file_data(path), fh, indent=2)
    return run_topopt(**args)
