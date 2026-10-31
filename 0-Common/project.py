"""
project.py
==========
One MMTO problem as a project: geometry, named selections, boundary conditions,
keep-solid regions, material database, objective and constraints, and settings.
A project is saved in a PyTO-AD project file (.pyto), in its own "mmto" section;
whatever the PyTO-AD GUI saved in the same file is kept as it is.

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

build() returns what getMMTOProblemPureStructural returns for a benchmark.

Boundary conditions (each acts on a named selection, evaluated on the mesh nodes):
    fixed          dofs "x", "y", "z" or any combination, value 0
    displacement   dofs as above, a nonzero imposed value
    force          a total force [Fx, Fy, Fz], split equally over the nodes
    traction       pressure * area along a direction, split equally over the nodes;
                   area is a number, or "bbox": the bounding box of the loaded nodes
                   seen along an axis-aligned direction (each side at least half an
                   element). "bbox" is approximate; it is there for the airplane bracket.
Supports and loads picked in the PyTO-AD GUI (its "structuralBC" section) are added to these.

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

SCHEMA_VERSION = 1
DATABASE_DIR = os.path.join("1-PureStructural", "MaterialDataPureStructural")
BC_KINDS = ("fixed", "displacement", "force", "traction")
_AXES = {"x": 0, "y": 1, "z": 2}
_MARK_FIXED, _MARK_LOAD, _MARK_ROLLER = 1, 2, 3

# settings.vae key -> VAEParams attribute
_VAE_KEYS = {"latent_dim": "latentDim", "learning_rate": "learningRate", "hidden_dim": "vae_hiddenDim",
             "max_attribute_error_percent": "maxAttributeErrorPercent", "kl_factor": "klFactor",
             "num_epochs": "numEpochs"}


def _dofs(dofs):
    if not dofs or any(c not in _AXES for c in dofs) or len(set(dofs)) != len(dofs):
        raise ValueError(f"dofs must be letters from 'xyz', e.g. 'xyz' or 'y', not {dofs!r}")
    return [_AXES[c] for c in sorted(dofs, key="xyz".index)]


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
        self.settings = {"common": {}, "advanced": {}, "vae": {}}
        self._gui = {}                          # everything else in the .pyto, kept for the GUI

    # ------------------------------------------------------------------ building it up
    def add_selection(self, name, selection):
        if not isinstance(selection, S.Selection):
            raise TypeError(f"{name}: a Selection is expected, e.g. face('xmin')")
        self.selections[name] = selection
        return name

    def _on(self, on, kind):
        """A selection name; a Selection given directly is stored under a generated name."""
        if isinstance(on, S.Selection):
            i = 1
            while f"{kind}{i}" in self.selections:
                i += 1
            return self.add_selection(f"{kind}{i}", on)
        return on

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
        from ExamplesPureStructural import _DATABASE_VAE   # type: ignore
        if self.database in _DATABASE_VAE:
            return os.path.join(DATABASE_DIR, f"{self.database}.xlsx"), _DATABASE_VAE[self.database]
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
            if bc["kind"] in ("fixed", "displacement"):
                try:
                    _dofs(bc.get("dofs"))
                except ValueError as e:
                    errors.append(f"{where}: {e}")
            if bc["kind"] == "traction":
                d = np.asarray(bc.get("direction", []), dtype=float)
                if d.shape != (3,) or not np.any(d):
                    errors.append(f"{where}: direction must be three numbers, not all zero")
                elif bc.get("area") == "bbox" and np.count_nonzero(d) != 1:
                    errors.append(f"{where}: area 'bbox' needs a direction along x, y or z; give a number instead")
        from pyto_project import _supports_and_loads                      # type: ignore
        gui_supports, gui_loads = _supports_and_loads(self._gui) if self._gui else ([], [])
        if not gui_supports and not any(bc.get("kind") in ("fixed", "displacement") for bc in self.bcs):
            errors.append("no supports")
        if not gui_loads and not any(bc.get("kind") in ("force", "traction") for bc in self.bcs):
            errors.append("no loads")
        for n in self.keep_solid_names:
            if n not in self.selections:
                errors.append(f"keep_solid: no selection named {n!r}")
        if self.database is None:
            errors.append("no material database")
        else:
            xlsx, _ = self.database_paths()
            if not os.path.isfile(xlsx):
                errors.append(f"material database {xlsx} does not exist")
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

    # ------------------------------------------------------------------ the problem
    def build(self, verbose=True):
        """mesh, mat_prop, bc, elem_body_force, to_params, vae_params, as for a benchmark."""
        errors = self.validate()
        if errors:
            raise ValueError(f"project {self.name!r}:\n  " + "\n  ".join(errors))
        from PyTOImports import (bound_cond, mat_lib, topopt_material_model,          # type: ignore
                                 find_elements_with_fixedDOF)
        from mmto_common import MMTOParams                                            # type: ignore
        from ExamplesPureStructural import VAEParams                                  # type: ignore
        from pyto_project import make_mesh, gui_bcs_on_mesh                           # type: ignore

        mesh = make_mesh(self.stl_path(), self.n_dof, self.single_layer)
        nodes_of = {}

        def nodes(name, where):
            if name not in nodes_of:
                nodes_of[name] = self.selections[name].evaluate(mesh, "nodes", self.base_dir)
            if len(nodes_of[name]) == 0:
                raise ValueError(f"{where}: selection {name!r} has no nodes on this mesh")
            return nodes_of[name]

        # ---- supports
        values = {}                                                    # dof -> imposed value
        for i, bc in enumerate(self.bcs, 1):
            if bc["kind"] not in ("fixed", "displacement"):
                continue
            n = nodes(bc["on"], f"boundary condition {i}")
            v = float(bc.get("value", 0.0))
            for d in (3 * n[:, None] + np.array(_dofs(bc["dofs"]))[None, :]).ravel():
                if values.get(int(d), v) != v:
                    raise ValueError(f"boundary condition {i}: dof {int(d)} is already given the value "
                                     f"{values[int(d)]}, here {v}")
                values[int(d)] = v
            mesh.node_indices[n, 3] = _MARK_FIXED if len(bc["dofs"]) == 3 else _MARK_ROLLER

        # ---- loads
        force = np.zeros(3 * mesh.num_nodes)
        loaded = set()
        for i, bc in enumerate(self.bcs, 1):
            if bc["kind"] not in ("force", "traction"):
                continue
            n = nodes(bc["on"], f"boundary condition {i}")
            if bc["kind"] == "force":
                total = np.asarray(bc["total"], dtype=float)
            else:
                d = np.asarray(bc["direction"], dtype=float)
                d = d / np.linalg.norm(d)
                area = bc["area"]
                if area == "bbox":
                    tol = 0.5 * float(np.min(mesh.elem_size))
                    p = np.asarray(mesh.node_xyz, dtype=float)[n]
                    a, b = [k for k in range(3) if d[k] == 0]
                    area = max(np.ptp(p[:, a]), tol) * max(np.ptp(p[:, b]), tol)
                total = bc["pressure"] * area * d
            force.reshape(-1, 3)[n] += total / len(n)
            loaded.update(n.tolist())
            mesh.node_indices[n, 3] = _MARK_LOAD

        # ---- supports and loads picked in the GUI, if any
        if self._gui:
            gui = dict(self._gui, stl_file_path=self.stl_path())
            gui_fixed, gui_force = gui_bcs_on_mesh(gui, mesh)
            for d in gui_fixed:
                if values.get(int(d), 0.0) != 0.0:
                    raise ValueError(f"dof {int(d)} is fixed in the GUI and given {values[int(d)]} here")
                values[int(d)] = 0.0
            both = loaded & set(np.flatnonzero(np.any(gui_force.reshape(-1, 3) != 0, axis=1)).tolist())
            if both and verbose:
                print(f"[{self.name}] warning: {len(both)} nodes are loaded both in the GUI and here; "
                      f"the forces add up")
            force += gui_force

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
        if common.get("filter_radius") is not None:
            to_params.RelativeFilterRadius = float(common["filter_radius"])
        if common.get("material_filter_radius") is not None:
            to_params.MaterialFilterRadius = float(common["material_filter_radius"])
        topopt_material_model._PNORM_EXPONENT = float(common.get("pnorm_exponent", 6))
        xlsx, vae_defaults = self.database_paths()
        to_params.MaterialsExcelFile = xlsx
        obj = self.formulation["objective"]
        to_params.Objective = to_params_form(obj["expression"], "objective")
        to_params.Constraints = [to_params_form(c["expression"], "constraint", c["bound"])
                                 for c in self.formulation.get("constraints", [])]

        vae_params = VAEParams()
        for k, v in (vae_defaults or {}).items():
            setattr(vae_params, k, v)
        for k, v in self.settings.get("vae", {}).items():
            if k in _VAE_KEYS:
                setattr(vae_params, _VAE_KEYS[k], v)

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
                "settings": {k: dict(self.settings.get(k, {})) for k in ("common", "advanced", "vae")}}

    def save(self, path):
        """Write the project; other sections of an existing .pyto are kept."""
        path = os.path.abspath(path)
        data = dict(self._gui)
        if os.path.isfile(path):
            with open(path) as fh:
                data.update({k: v for k, v in json.load(fh).items() if k not in ("mmto", "stl_file_path")})
        stl = self.stl_path()
        try:
            stl = os.path.relpath(stl, os.path.dirname(path))
        except ValueError:                                      # another drive on Windows
            pass
        data["stl_file_path"] = stl.replace("\\", "/")
        data["mmto"] = self.to_dict()
        with open(path, "w") as fh:
            json.dump(data, fh, indent=2)
        self.base_dir, self.stl = os.path.dirname(path), data["stl_file_path"]

    @classmethod
    def load(cls, path):
        path = os.path.abspath(path)
        with open(path) as fh:
            data = json.load(fh)
        m = data.get("mmto") or {}
        if m.get("schema_version", SCHEMA_VERSION) > SCHEMA_VERSION:
            raise ValueError(f"{path}: written by a newer version (schema {m['schema_version']})")
        mesh = m.get("mesh", {})
        p = cls(m.get("name") or os.path.splitext(os.path.basename(path))[0], data.get("stl_file_path", ""),
                n_dof=mesh.get("n_dof", 30000), single_layer=mesh.get("single_layer", False),
                base_dir=os.path.dirname(path))
        p.comment, p.physics = m.get("comment", ""), m.get("physics", "structural")
        p.selections = {k: S.from_dict(v) for k, v in m.get("selections", {}).items()}
        p.bcs = [dict(bc) for bc in m.get("bcs", [])]
        p.keep_solid_names = list(m.get("keep_solid", []))
        p.keep_support_elements = bool(m.get("keep_support_elements", False))
        p.database = m.get("database")
        f = m.get("formulation") or {}
        p.formulation = {"objective": f.get("objective"), "constraints": list(f.get("constraints", []))}
        s = m.get("settings") or {}
        p.settings = {k: dict(s.get(k, {})) for k in ("common", "advanced", "vae")}
        p._gui = {k: v for k, v in data.items() if k not in ("mmto", "stl_file_path")}
        return p
