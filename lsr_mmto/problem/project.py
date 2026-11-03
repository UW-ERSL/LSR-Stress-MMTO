"""
project.py
==========
One MMTO problem as a project: geometry, named selections, boundary conditions,
keep-solid regions, material database, objective and constraints, and settings.
A project is saved as an LSR MMTO project file (.mmto): the layout of a PyTO-AD project
file, with the problem in its own "mmto" section. Project.load reads .mmto files only.

    from lsr_mmto.problem.project import Project
    from lsr_mmto.problem.selections import face, box
    p = Project("bracket", stl="bracket.stl", n_dof=30000)
    p.add_selection("left", face("xmin"))
    p.add_support("left", dofs="xyz")
    p.add_force(box([9, 0, 0], [10, 10, 1]), total=[0, -1000, 0])
    p.database = "6Materials_AF_3Attr"
    p.objective("FailureFactorPNorm()")
    p.constraint("Mass()", "<=", 3.5e-3)
    p.save("bracket.mmto")
    q = Project.load("bracket.mmto")
    mesh, mat_prop, bc, body_force, to_params, vae_params = q.build()
    run(q)                                  # optimizes; results and a copy of q in results/bracket
    q.to_script("bracket_setup.py")         # the same project as Python calls, to read or edit

build() returns the mesh, the material, the boundary conditions, the body force and the
optimization and VAE parameters that _run_topopt starts from.

Settings: every setting is one entry of SETTINGS in problem/settings.py, stored in
settings.common, settings.advanced and settings.vae. A new project starts from the
settings of the paper's examples; a file missing some is completed from the same
defaults, and the run says which.

Boundary conditions (each acts on a named selection, evaluated on the mesh nodes):
    fixed          dofs "x", "y", "z" or any combination, value 0
    displacement   dofs as above, a nonzero imposed value
    force          a total force [Fx, Fy, Fz], split equally over the nodes
    traction       pressure * area along a direction, split equally over the nodes;
                   area is a number, or "bbox": the bounding box of the loaded nodes
                   seen along an axis-aligned direction (each side at least half an
                   element). "bbox" is approximate; it is there for the airplane bracket.
Supports and loads picked on the part in the window (faces, nodes, boxes) become selections
and boundary conditions of the project as soon as they are applied (adopt_picks), with the
picking tools' own rules (stl_faces with tol 0.8, nearest boundary node, halfspaces): the mmto
section is the whole problem.

Forces are used as given, with no unit conversion: they must be in the units of the
material database.

Objective and constraints are written as PyTO-AD stores them: an expression, a sense
(minimize) or an operator (<=) and a bound. An expression is exactly one quantity of the
registry in formulation/quantities.py, e.g. "Mass()", "FailureFactorPNorm()",
"MaterialVolumeFraction(2, tau=0.5)".
"""
import json
import os

import numpy as np

from lsr_mmto.problem import selections as S
from lsr_mmto.materials.databases import (ATTRIBUTES, DATABASE_DIR, DATABASE_VAE, UNIT_SYSTEMS, VAEParams,
                                         check_database, database_attributes, database_units, in_shipped_folder,
                                         recommended_latent_dim, unit_ratio)
from lsr_mmto.formulation.formulation_text import parse_expression, to_params_form
from lsr_mmto.formulation.quantities import default_of, takes
from lsr_mmto.physics.available_physics import PHYSICS
from lsr_mmto.problem.boundary_conditions import BC_KINDS
from lsr_mmto.problem.settings import (BY_KEY, DEFAULT_SETTINGS, MESH_DEFAULTS, SETTINGS, VAE_PARAMS, pyto_ad_value,
                                       run_settings)

DEFAULT_PHYSICS = next(iter(PHYSICS))          # a project that names no physics: the first one listed

SCHEMA_VERSION = 1
# _UPGRADES[n] turns an mmto section of version n into version n + 1. When the format
# changes: raise SCHEMA_VERSION and add the function here; old files then still load.
_UPGRADES = {}


def network_shape(path):
    """(inputs, hidden width, latent dimension) of the VAE saved in path, read from its weights;
    None when the file is not such a network."""
    import torch
    try:
        sd = torch.load(path, map_location="cpu", weights_only=True)
        hidden, n_in = sd["encoder.linear1.weight"].shape
        return int(n_in), int(hidden), int(sd["encoder.linear2.weight"].shape[0])
    except Exception:                                                                  # noqa: BLE001
        return None


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


REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))     # the repository folder


def _supports_and_loads(picks):
    """Supports and loads as (kind, values, triangles, points, regions), from the window's
    constraint_data (supports) and force_data (loads) lists, each made a kind of BC_KINDS by the
    kind's picks."""
    found = {}
    for role, key in (("load", "force_data"), ("support", "constraint_data")):
        found[role] = []
        for entry in picks.get(key, []):
            t = entry.get("type")
            kinds = [(name, k.picks[t]) for name, k in BC_KINDS.items() if k.role == role and t in k.picks]
            if not kinds:
                known = [p for k in BC_KINDS.values() if k.role == role for p in k.picks if p is not None]
                raise NotImplementedError(f"{role} type {t!r} is not supported; use one of {', '.join(known)}")
            name, values = kinds[0]
            found[role].append((name, values(entry), entry.get("triangles") or [], entry.get("points") or [],
                                entry.get("regions") or []))
    return found["support"], found["load"]


class Project:
    def __init__(self, name, stl, n_dof=None, single_layer=None, base_dir=None):
        self.name = name
        self.stl = stl                          # as given; relative paths are relative to base_dir
        self.base_dir = os.path.abspath(base_dir or os.getcwd())
        self.n_dof = int(MESH_DEFAULTS["n_dof"] if n_dof is None else n_dof)
        self.single_layer = bool(MESH_DEFAULTS["single_layer"] if single_layer is None else single_layer)
        self.comment = ""
        self.physics = DEFAULT_PHYSICS
        self.selections = {}                    # name -> Selection
        self.bcs = []                           # dicts, see BC_KINDS
        self.keep_solid_names = []
        self.keep_support_elements = False
        self.database = None
        self.formulation = {"objective": None, "constraints": []}
        self.settings = {k: dict(v) for k, v in DEFAULT_SETTINGS.items()}
        self._gui = {}                          # everything else in the file, kept for the GUI

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

    def adopt_picks(self, gui):
        """Supports and loads picked on the part in the window (its constraint_data and force_data
        lists) as selections and boundary conditions of this project, with the picking tools' own
        rules: faces as stl_faces (tol 0.8), a clicked node as its nearest boundary node, a dragged
        box as halfspaces. Returns the names of the new selections."""
        supports, loads = _supports_and_loads(gui)
        if not supports and not loads:
            return []

        def selection(triangles, points, regions, what):
            parts = [S.stl_faces(triangles, tol=0.8)] if len(triangles) else []
            parts += [S.nearest(point, boundary=True) for point in points]
            parts += [S.halfspaces(r["normals"], r["origins"]) for r in regions]
            if not parts:
                raise ValueError(f"[{self.name}] {what} picked on the part has no faces, "
                                 f"nodes or boxes")
            return parts[0] if len(parts) == 1 else S.union(*parts)

        names = []
        for role, found in (("support", supports), ("load", loads)):
            for i, (kind, values, triangles, points, regions) in enumerate(found):
                name = self.add_selection(self._free_name(f"gui_{role}"),
                                          selection(triangles, points, regions, f"{role} {i + 1}"))
                self.bcs.append({"kind": kind, "on": name, **values})
                names.append(name)
        return names

    def add_support(self, on, dofs="xyz", value=0.0):
        kind = "fixed" if float(value) == 0.0 else "displacement"
        errors = BC_KINDS[kind].check({"dofs": dofs})
        if errors:
            raise ValueError(errors[0])
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

    def evaluate(self, selection, mesh, on="nodes"):
        """Indices of a selection (or of the selection with that name) on mesh, with every selection(name) in
        it looked up now, so a combination follows later edits of its parts. Elements of a selection that
        picks only nodes (stl_faces, rows, ...) are the elements touching those nodes."""
        seen = ()
        if isinstance(selection, str):
            seen, selection = (selection,), self.selections[selection]
        s = S.resolved(selection, self.selections, seen)
        try:
            return s.evaluate(mesh, on, self.base_dir)
        except ValueError as e:
            if on != "elements" or "nodes only" not in str(e):
                raise
        nodes = s.evaluate(mesh, "nodes", self.base_dir)
        touching = np.isin(np.asarray(mesh.elemArray), nodes).any(axis=1)
        return np.flatnonzero(touching).astype(int)

    def rename_selection(self, old, new):
        """The selection and everything that names it: supports and loads, keep solid, other selections."""
        self.selections = {new if k == old else k: S.renamed(v, old, new) for k, v in self.selections.items()}
        for bc in self.bcs:
            if bc.get("on") == old:
                bc["on"] = new
        self.keep_solid_names = [new if n == old else n for n in self.keep_solid_names]

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

    @property
    def units(self):
        """The unit system: "MKS", "mmKS", "IPS" (those of the PyTO-AD GUI, kept where the GUI
        keeps it, settings.unit_system) or "none". Numbers are never converted; the unit
        system says what they are meant to be in, so that a database in other units is reported."""
        return (self._gui.get("settings") or {}).get("unit_system", "none")

    @units.setter
    def units(self, system):
        if system == "none":
            (self._gui.get("settings") or {}).pop("unit_system", None)
            return
        settings = self._gui.setdefault("settings", {"temperature_unit": "Kelvin", "angle_unit": "Degree"})
        settings["unit_system"] = system

    def unit_record(self):
        """The units of the run's numbers, for the run record."""
        if self.units not in UNIT_SYSTEMS:
            return {"system": self.units}
        return dict(system=self.units, **UNIT_SYSTEMS[self.units])

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
        if self.physics not in PHYSICS:
            errors.append(f"physics {self.physics!r}: one of {', '.join(PHYSICS)}")
        if self.units != "none" and self.units not in UNIT_SYSTEMS:
            errors.append(f"units {self.units!r}: one of {', '.join(UNIT_SYSTEMS)} or 'none'")
        for i, bc in enumerate(self.bcs, 1):
            where = f"boundary condition {i} ({bc.get('kind')})"
            if bc.get("kind") not in BC_KINDS:
                errors.append(f"{where}: kind must be one of {', '.join(BC_KINDS)}")
                continue
            if bc.get("on") not in self.selections:
                errors.append(f"{where}: no selection named {bc.get('on')!r}")
            errors += [f"{where}: {e}" for e in BC_KINDS[bc["kind"]].check(bc)]
            if self.physics in PHYSICS and BC_KINDS[bc["kind"]].field not in PHYSICS[self.physics].FIELDS:
                errors.append(f"{where}: not a kind of {self.physics} support or load")
        roles = {BC_KINDS[bc["kind"]].role for bc in self.bcs if bc.get("kind") in BC_KINDS}
        if "support" not in roles:
            errors.append("no supports")
        if "load" not in roles:
            errors.append("no loads")
        for n in self.keep_solid_names:
            if n not in self.selections:
                errors.append(f"keep_solid: no selection named {n!r}")
        for n, sel in self.selections.items():                   # selections inside selections: known, no loops
            try:
                S.resolved(sel, self.selections, (n,))
            except ValueError as e:
                errors.append(f"selection {n}: {e}")
        if self.database is None:
            errors.append("no material database")
        else:
            xlsx, _ = self.database_paths()
            db_errors = check_database(xlsx)[0]
            errors += [f"material database {os.path.basename(xlsx)}: {e}" for e in db_errors]
            if not db_errors:
                errors += self._missing_columns(database_attributes(xlsx))
                errors += self.vae_check()[0]
        for section, values in self.settings.items():
            if section not in DEFAULT_SETTINGS:
                errors.append(f"settings: unknown section {section!r}")
                continue
            for k, v in values.items():
                entry = BY_KEY.get(k)
                if entry is None or entry.section != section:
                    errors.append(f"settings.{section}: unknown setting {k!r}")
                elif entry.choices is not None and v is not None and v not in entry.choices.__members__:
                    errors.append(f"settings.{section}.{k}: {v!r} is not one of "
                                  f"{', '.join(entry.choices.__members__)}")
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
        taus = set()
        for c in self.formulation.get("constraints", []):
            try:
                name, _, kwargs = parse_expression(c.get("expression"))
            except ValueError:
                continue
            if takes(name, "tau"):                              # one tau for every material volume fraction
                taus.add(float(kwargs.get("tau", default_of(name, "tau"))))
        if len(taus) > 1:
            listed = ", ".join(f"{t:g}" for t in sorted(taus))
            errors.append(f"the material volume fraction constraints use different tau ({listed}); one tau "
                          f"applies to all of them: use the same value")
        if any(t <= 0 for t in taus):
            errors.append("tau must be greater than 0")
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
        from lsr_mmto.formulation.quantities import QUANTITIES
        out = [f"material database: the {self.physics} model needs a {c} column"
               for c in (PHYSICS[self.physics].PROPERTIES if self.physics in PHYSICS else ()) if c not in columns]
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

    def _unit_warnings(self, xlsx):
        """A database column in units other than the project's, with what it does to the results."""
        from lsr_mmto.formulation.quantities import QUANTITIES
        if self.units not in UNIT_SYSTEMS:
            return []
        system = UNIT_SYSTEMS[self.units]
        f = self.formulation
        objective = parse_expression((f.get("objective") or {}).get("expression") or "x()")[0]
        bounds = {}
        for c in f.get("constraints", []):
            try:
                bounds.setdefault(parse_expression(c["expression"])[0], []).append(c["bound"])
            except Exception:                                                          # noqa: BLE001
                pass
        units = database_units(xlsx)
        ratio = {n: unit_ratio(ATTRIBUTES[n].measures, u, self.units) for n, u in units.items()
                 if u is not None and ATTRIBUTES[n].measures != "ratio"}

        def times(r):
            e = np.log10(r)
            return f"10^{int(round(e))}" if abs(e - round(e)) < 1e-9 else f"{r:.4g}"

        out = []
        for name, r in ratio.items():
            if abs(r - 1.0) < 1e-12:
                continue
            a = ATTRIBUTES[name]
            say = (f"the database gives {name.replace('_', ' ')} in {units[name]}, but the project's units "
                   f"({self.units}) expect {system[a.measures]}: its numbers are {times(r)} times what they "
                   f"should be.")
            if name == "Youngs_Modulus":
                say += (f" Displacement and compliance come out {times(r)} times too small; stress and "
                        f"failure factor are unaffected.")
            elif name == "Yield_Strength":
                if "FailureFactorPNorm" in bounds:
                    say += (f" The failure factor comes out {times(1 / r)} times its value, so the constraint "
                            f"FailureFactorPNorm() <= {bounds['FailureFactorPNorm'][0]:g} is a different "
                            f"problem: it allows a true failure factor of "
                            f"{bounds['FailureFactorPNorm'][0] * r:g}.")
                elif objective == "FailureFactorPNorm":
                    say += (f" The failure factor is reported {times(1 / r)} times its value; the design is "
                            f"unaffected, since the objective is only scaled.")
            elif name == "Density" and "Mass" in bounds:
                say += (f" Mass comes out {times(r)} times its value in {system['mass']}: the bound "
                        f"{bounds['Mass'][0]:g} means {bounds['Mass'][0] / r:g} {system['mass']}.")
            out.append(say)
        if "Cost" in bounds and "Cost" in ratio and "Density" in ratio:
            r = ratio["Cost"] * ratio["Density"]
            if abs(r - 1.0) > 1e-12:            # said once, under Cost if it is off, else under Density
                k = [i for i, w in enumerate(out) if w.startswith("the database gives Cost ")] or \
                    [i for i, w in enumerate(out) if w.startswith("the database gives Density ")]
                out[k[0]] += (f" Cost() is density * cost per mass * volume: with these units it comes out "
                              f"{times(r)} times its value in USD, so the bound {bounds['Cost'][0]:g} means "
                              f"{bounds['Cost'][0] / r:g} USD.")
        return out

    def warnings(self):
        """What is allowed but deserves a look before a run: the database's warnings, a
        database in units other than the project's, and a latent dimension other than the
        one recommended for the database."""
        out = []
        if self.database is None:
            return out
        xlsx, _ = self.database_paths()
        errors, db_warnings = check_database(xlsx)
        if errors:
            return out
        out += [f"material database {os.path.basename(xlsx)}: {w}" for w in db_warnings]
        out += self._unit_warnings(xlsx)
        out += self.vae_check()[1]
        return out

    def network_path(self):
        """The VAE file a run loads or trains: settings.vae.saveNet, or else the .nt next to the
        database's .xlsx with the same name (my_db.xlsx, my_db.nt)."""
        xlsx, _ = self.database_paths()
        return self.settings.get("vae", {}).get("saveNet") or os.path.splitext(xlsx)[0] + ".nt"

    def vae_check(self):
        """(errors, warnings, status) about the VAE: a run that would train into the shipped folder,
        a trained network that does not fit the settings or the database, a network a run would
        replace, a latent dimension other than the recommended one; status says which network the
        database has. For a database that passes its check."""
        vae = self.settings.get("vae", {})
        xlsx, _ = self.database_paths()
        nt = self.network_path()
        name = os.path.basename(nt)
        shipped, exists = in_shipped_folder(nt), os.path.isfile(nt) and os.path.getsize(nt) > 0
        train = not vae.get("use_pretrained_vae", True) or not exists
        errors, warnings_ = [], []
        if train and shipped:
            errors.append(f"a run would train a new VAE and save it over the shipped {name}; to train a different "
                          f"network, copy {os.path.basename(xlsx)} to a folder of your own and use the copy")
        elif train and exists:
            warnings_.append(f"{name} exists: the next run trains a new network and replaces it (the old one is "
                             f"kept as {os.path.splitext(name)[0]}.previous.nt). To keep both, copy "
                             f"{os.path.basename(xlsx)} first.")
        if not train:
            shape = network_shape(nt)
            if shape is None:
                errors.append(f"{name} could not be read as a VAE network")
            else:
                n_in, hidden, latent = shape
                n_columns = len(database_attributes(xlsx))
                if n_in != n_columns:
                    errors.append(f"{name} takes {n_in} attributes, but {os.path.basename(xlsx)} has {n_columns}: "
                                  f"it was trained on another database")
                for key, have in (("latentDim", latent), ("vae_hiddenDim", hidden)):
                    if vae.get(key) is not None and int(vae[key]) != have:
                        errors.append(f"{key} is {vae[key]}, but {name} was trained with {have}: set it to {have}, "
                                      f"or switch off use_pretrained_vae to train a new network")
        d_eff, n_z = recommended_latent_dim(xlsx)
        latent = vae.get("latentDim")
        if latent is not None and int(latent) > n_z:
            warnings_.append(f"latentDim is {latent}; the database's effective dimension is d_eff = {d_eff:.2f}, "
                             f"so {n_z} is recommended. A latent space larger than the database needs has "
                             f"stopped runs from converging before.")
        elif latent is not None and int(latent) < n_z:
            warnings_.append(f"latentDim is {latent}; the database's effective dimension is d_eff = {d_eff:.2f}, "
                             f"so {n_z} is recommended. A latent space smaller than the database needs cannot "
                             f"reproduce every material: expect large reconstruction errors, and decoded "
                             f"properties that do not match the real materials.")
        status = f"Network: {name}, " + ("shipped." if shipped and exists else "found." if exists else
                                         f"none yet. The first run trains {name}.")
        return errors, warnings_, status

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
        if verbose:
            shown = self.warnings()
            if shown != getattr(self, "_shown_warnings", None):          # once per change, not every build
                for w in shown:
                    print(f"[{self.name}] warning: {w}")
                self._shown_warnings = shown
        from lsr_mmto.pyto_ad_imports import mat_lib, find_elements_with_fixedDOF
        from lsr_mmto.optimize.mmto_to_params import MMTOParams
        from lsr_mmto.problem.mesh import make_mesh

        mesh = make_mesh(self.stl_path(), self.n_dof, self.single_layer)
        bcs, _ = self.supports_and_loads(mesh)
        error = self.supports_and_loads_error(mesh, bcs)            # as PyTO-AD's FEA does, but before the
        if error:                                                   # VAE and the run
            raise ValueError(f"project {self.name!r}: {error}")

        # ---- optimization parameters
        to_params = MMTOParams()
        to_params.Comment = self.comment or self.name
        to_params.KeepFixedElems = self.keep_support_elements
        for e in SETTINGS:                                          # every setting TOParams has a field for
            if e.pyto_ad_field is not None:
                value = getattr(self, e.key) if e.section == "mesh" else self.settings[e.section][e.key]
                setattr(to_params, e.pyto_ad_field, pyto_ad_value(e, value))
        for e in SETTINGS:                                          # every setting PyTO-AD keeps elsewhere
            if e.at_build is not None:
                e.at_build(getattr(self, e.key) if e.section == "mesh" else self.settings[e.section][e.key])
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
            for field, n in PHYSICS[self.physics].FIELDS.items():
                held = find_elements_with_fixedDOF(mesh, bcs[field], nDOFPerNode=n)
                to_params.ElemsToKeep = held if to_params.ElemsToKeep is None else np.union1d(to_params.ElemsToKeep,
                                                                                               held)
        if self.keep_solid_names:
            keep = np.unique(np.concatenate([self.evaluate(n, mesh, "elements")
                                             for n in self.keep_solid_names])).astype(int)
            mesh.nonDesignElems = keep
            to_params.ElemsToKeep = keep if to_params.ElemsToKeep is None else np.union1d(to_params.ElemsToKeep, keep)

        mat_prop = mat_lib.create_material_with_defaults(          # placeholder; the VAE gives the real ones
            "CustomMaterial", youngs_modulus=1.0, poissons_ratio=0.3, mass_density=1.0, yield_strength=1.0)
        if verbose:
            held = "  ".join(f"{field}: fixed dofs={len(bc.fixed_dofs)}  total load="
                             f"{np.round(bc.force.reshape(-1, PHYSICS[self.physics].FIELDS[field]).sum(0), 6).tolist()}"
                             for field, bc in bcs.items())
            print(f"[{self.name}] elems={mesh.num_elems}  h={np.round(mesh.elem_size, 4).tolist()}  {held}  "
                  f"keep solid={0 if to_params.ElemsToKeep is None else len(to_params.ElemsToKeep)}")
        return mesh, mat_prop, bcs, None, to_params, vae_params

    def supports_and_loads(self, mesh):
        """({field: PyTO-AD BC}, {selection name: its nodes}): the supports, then the loads (BC_KINDS), on
        mesh, one boundary condition for each field of the project's physics. ValueError for a selection
        with no nodes or a dof given two values."""
        from lsr_mmto.pyto_ad_imports import bound_cond
        nodes_of = {}

        def nodes(name, where):
            if name not in nodes_of:
                nodes_of[name] = self.evaluate(name, mesh, "nodes")
            if len(nodes_of[name]) == 0:
                raise ValueError(f"{where}: selection {name!r} has no nodes on this mesh")
            return nodes_of[name]

        # values: dof -> imposed value; force: the load on each dof of the field
        states = {field: {"values": {}, "force": np.zeros(n * mesh.num_nodes)}
                  for field, n in PHYSICS[self.physics].FIELDS.items()}
        for role in ("support", "load"):
            for i, bc in enumerate(self.bcs, 1):
                kind = BC_KINDS[bc["kind"]]
                if kind.role == role:
                    where = f"boundary condition {i}"
                    kind.apply(bc, nodes(bc["on"], where), mesh, states[kind.field], where)
        bcs = {}
        for field, state in states.items():
            values, force = state["values"], state["force"]
            fixed_dofs = np.array(sorted(values), dtype=int)
            dirichlet_values = np.array([values[d] for d in fixed_dofs], dtype=float)
            bcs[field] = bound_cond.BC(force=force, fixed_dofs=fixed_dofs, dirichlet_values=dirichlet_values)
        return bcs, nodes_of

    def supports_and_loads_error(self, mesh, bcs):
        """Why the supports and loads leave the problem without a solution (the part free to move), or None:
        the physics' own check."""
        return PHYSICS[self.physics].supports_and_loads_error(mesh, bcs)

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
        """Write the project, every setting included, with the other sections it was loaded with.
        Always as .mmto. Its folder becomes the project's folder, as the window keeps it: the part and
        the database with its network are copied into part/ and materials/ there (project_folder.pack),
        unless they are in it already."""
        from lsr_mmto.problem.project_folder import pack
        if not path.lower().endswith(".mmto"):
            raise ValueError(f"{path}: an LSR MMTO project is saved as a .mmto file")
        path = os.path.abspath(path)
        pack(self, os.path.dirname(path))
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
                 "Run it with File > Run script in LSR MMTO, which opens the project it saves, or from any folder",
                 f"with: python {os.path.basename(path)}       (saves {self.name}.mmto next to it)",
                 "",
                 "LSR MMTO writes this file again each time the project is saved in its window, so edits",
                 "made here are replaced then. Saving from this script does not rewrite it.",
                 '"""',
                 "import os",
                 "",
                 f"from lsr_mmto import {', '.join(['Project'] + types)}",
                 "",
                 "HERE = os.path.dirname(os.path.abspath(__file__))",
                 f"p = Project({self.name!r}, stl={stl!r}, n_dof={self.n_dof}, single_layer={self.single_layer},",
                 "            base_dir=HERE)"]
        if self.comment:
            lines.append(f"p.comment = {self.comment!r}")
        if self.physics != DEFAULT_PHYSICS:
            lines.append(f"p.physics = {self.physics!r}")
        if self.units != "none":
            lines.append(f"p.units = {self.units!r}")
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
        lines += ["", f"p.save(os.path.join(HERE, {self.name + '.mmto'!r}))", ""]
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
    def from_record(cls, record, folder=None):
        """The project a run record stores. folder: the project's folder now, when it may have been
        moved since the run (zipped and sent): the part is then looked for at the same place
        relative to it."""
        data = dict(record.get("gui") or {}, mmto=record["project"], stl_file_path=record["stl_file_path"])
        p = cls._from_data(data, record["base_dir"], record["project"].get("name", "project"))
        if folder is not None and not os.path.isfile(p.stl_path()):
            try:
                rel = os.path.relpath(record["stl_file_path"], record["base_dir"])
            except ValueError:                                  # another drive on Windows
                return p
            if os.path.isfile(os.path.join(folder, rel)):
                p.base_dir, p.stl = os.path.abspath(folder), rel.replace("\\", "/")
        return p

    @classmethod
    def load(cls, path):
        path = os.path.abspath(path)
        if not path.lower().endswith(".mmto"):
            raise ValueError(f"{path}: not an LSR MMTO project; a project is a .mmto file")
        with open(path) as fh:
            data = json.load(fh)
        if "mmto" not in data:
            raise ValueError(f"{path}: no mmto section; it is not an LSR MMTO project")
        return cls._from_data(data, os.path.dirname(path), os.path.splitext(os.path.basename(path))[0])

    @classmethod
    def _from_data(cls, data, base_dir, default_name):
        m = upgrade(data.get("mmto") or {}, default_name)
        mesh = m.get("mesh", {})
        p = cls(m.get("name") or default_name, data.get("stl_file_path", ""),
                n_dof=mesh.get("n_dof"), single_layer=mesh.get("single_layer"),
                base_dir=base_dir)
        p.comment, p.physics = m.get("comment", ""), m.get("physics", DEFAULT_PHYSICS)
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
        return p


# ---------------------------------------------------------------------- running
def _run_arguments(project, save_dir=None, verbose=True, plot_progress=False, plot_bcs=False, plotter=None,
                  progress=None, should_stop=None, on_network_error=None):
    """The keyword arguments _run_topopt gets for this project: its settings as one object s
    (settings.run_settings), the project, where the results go, and how the run is watched.
    Internal: a project runs through run(p). With no save_dir, the results go where the window puts
    them: results/<date and time> in the project's folder (project_folder.new_run_folder)."""
    from lsr_mmto.problem.project_folder import new_run_folder
    project.complete_settings()
    return dict(s=run_settings(project), to_problem=project,
                save_dir=save_dir if save_dir is not None else new_run_folder(project),
                verbose=verbose, plot_progress=plot_progress, plot_bcs=plot_bcs, plotter=plotter,
                progress=progress, should_stop=should_stop, on_network_error=on_network_error)


def _prepared(project):
    """The project as a run gets it: its settings completed (saying which came from the defaults),
    checked, and copied, so that the run records the project as it is now."""
    filled = project.complete_settings()
    if filled:
        print(f"[{project.name}] settings not in the project, taken from the defaults: {', '.join(filled)}")
    errors = project.validate()
    if errors:
        raise ValueError(f"project {project.name!r}:\n  " + "\n  ".join(errors))
    return project.copy()


def run(project, save_dir=None, **run_options):
    """Optimize the project with _run_topopt. The run's record keeps the project as it was run
    (Project.from_record rebuilds it)."""
    from lsr_mmto.optimize.optimization_loop import _run_topopt
    return _run_topopt(**_run_arguments(_prepared(project), save_dir, **run_options))
