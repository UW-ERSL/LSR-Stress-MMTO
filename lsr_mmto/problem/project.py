"""
project.py
==========
One MMTO problem as a project: geometry, named selections, boundary conditions,
keep-solid regions, material database, objective and constraints, and settings.
A project is saved as an LSR MMTO project file (.mmto): the layout of a PyTO-AD project
file, with the problem in its own "mmto" section. Project.load reads .mmto files only.

    from lsr_mmto.problem.project import Project, run
    from lsr_mmto.problem.selections import face, box
    p = Project("bracket", stl="bracket.stl", n_dof=30000)
    p.add_selection("left", face("xmin"))
    p.add_bc("fixed", "left", dofs="xyz")
    p.add_bc("force", box([9, 0, 0], [10, 10, 1]), total=[0, -1000, 0])
    p.database = "6Materials_AF_3Attr"
    p.objective("FailureFactorPNorm()")
    p.constraint("Mass()", "<=", 3.5e-3)
    p.save("bracket.mmto")
    q = Project.load("bracket.mmto")
    folder = run(q)          # optimizes; the results and a copy of q go in a new folder, named after the
                             # date and time, in results in the project's folder; folder is that folder

run builds the project itself (Project.build): the mesh, the material, the supports and loads of each
field, the body force and the optimization and VAE parameters the run starts from.

Settings: every setting is one entry of SETTINGS in problem/settings.py, stored in
settings.common, settings.advanced and settings.vae. A new project starts from their
defaults; a file missing some is completed from the same defaults, and the run says which.

Supports and loads each act on a named selection, evaluated on the mesh nodes. Their kinds are
those of the project's physics (the <name>_supports_and_loads.py file of each physics); every
kind with its values is listed in docs/reference.md, made from the code.
Supports and loads picked on the part in the window (faces, nodes, boxes) become selections
and boundary conditions of the project as soon as they are applied (adopt_picks), with the
picking tools' own rules (stl_faces with tol 0.8, nearest boundary node, halfspaces): the mmto
section is the whole problem.

Forces are used as given, with no unit conversion: they must be in the units of the
material database.

Objective and constraints are written as PyTO-AD stores them: an expression, a sense
(minimize) or an operator (<= at most, >= at least) and a bound.
An expression is exactly one quantity, e.g. "Mass()"
or "MaterialVolumeFraction(2, tau=0.5)"; every quantity is listed in docs/reference.md.
"""
import copy
import json
import os
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

from lsr_mmto.problem import selections as S
from lsr_mmto.problem.project_folder import PART
from lsr_mmto.problem.shapes import ShapeError, check, plain, to_script
from lsr_mmto.materials.databases import (ATTRIBUTES, DATABASE_DIR, DATABASE_VAE, UNIT_SYSTEMS, UNITS, VAEParams,
                                         check_database, database_attributes, database_units, in_shipped_folder,
                                         recommended_latent_dim, unit_ratio)
from lsr_mmto.formulation.formulation_text import parse_expression, to_params_form
from lsr_mmto.formulation.quantities import OPS, default_of, find_quantity, takes
from lsr_mmto.physics.available_physics import PHYSICS
from lsr_mmto.problem.boundary_conditions import BC_KINDS
from lsr_mmto.problem.settings import (BY_KEY, DEFAULT_SETTINGS, MESH_DEFAULTS, SETTINGS, VAE_PARAMS, pyto_ad_value,
                                       run_settings)

if TYPE_CHECKING:
    import pyvista

    from pyto.core.bc import BC
    from pyto.core.hex_mesher import HexMesher
    from pyto.core.mat_lib import Material
    from lsr_mmto.formulation.quantities import Quantity
    from lsr_mmto.materials.database_versions import Version
    from lsr_mmto.materials.network_stamp import Comparison
    from lsr_mmto.optimize.mmto_to_params import MMTOParams

DEFAULT_PHYSICS = next(iter(PHYSICS))          # a project that names no physics: the first one listed

# where a run's material properties come from (Project.material_source; run_setup.material_source)
MATERIAL_SOURCES = ("network", "one_database_material")

SCHEMA_VERSION = 2
# _UPGRADES[n] turns an mmto section of version n into version n + 1. When the format
# changes: raise SCHEMA_VERSION and add the function here; old files then still load.


def _named_sharpness_schedule_to_numbers(m: dict[str, Any]) -> dict[str, Any]:
    """Schema 1 to 2: the Heaviside sharpness schedule was a name, common.beta_schedule_scheme, PAPER or
    PAPER_BM1; it is now its five numbers, advanced.beta_init ... beta_start_iter, the same for both but the
    largest sharpness (32 or 16). Any other name stays, and the project's check calls it an unknown setting."""
    settings = {k: dict(v) for k, v in (m.get("settings") or {}).items()}
    common = settings.get("common", {})
    if common.get("beta_schedule_scheme") in ("PAPER", "PAPER_BM1"):
        scheme = common.pop("beta_schedule_scheme")
        settings.setdefault("advanced", {}).update(
            beta_init=1.0, beta_max=16.0 if scheme == "PAPER_BM1" else 32.0, beta_factor=1.5, beta_update_every=5,
            beta_start_iter=50)
    return dict(m, settings=settings) if "settings" in m else m


_UPGRADES = {1: _named_sharpness_schedule_to_numbers}

# The steps of a project's setup in the GUI, one per sidebar button, each with what makes it begun: an error
# Project.validate finds names its step (Problem.gui_step), so the window can mark the step it is about; a step
# not begun is never marked wrong.
GUI_STEPS = {"part": lambda p: p.has_part(),
         "units": lambda p: p.units != "none",
         "materials": lambda p: p.database is not None,
         "supports and loads": lambda p: bool(p.bcs),
         "formulation": lambda p: bool(p.formulation.get("objective")),
         "settings": lambda p: True}


class Problem(str):
    """An error Project.validate finds: its text, with .gui_step, the step of GUI_STEPS it is about."""

    gui_step: str

    def __new__(cls, step: str, text: str) -> "Problem":
        if step not in GUI_STEPS:
            raise ValueError(f"{step!r} is not a step of the setup ({', '.join(GUI_STEPS)})")
        problem = super().__new__(cls, text)
        problem.gui_step = step
        return problem


def network_shape(path: str) -> tuple[int, int, int] | None:
    """(inputs, hidden width, latent dimension) of the VAE saved in path, read from its weights;
    None when the file is not such a network."""
    from lsr_mmto.materials.network_stamp import read_network
    try:
        sd = read_network(path)[0]
        hidden, n_in = sd["encoder.linear1.weight"].shape
        return int(n_in), int(hidden), int(sd["encoder.linear2.weight"].shape[0])
    except Exception:                                                                  # noqa: BLE001
        return None


def upgrade(m: dict[str, Any], name: str = "project") -> dict[str, Any]:
    """An mmto section brought to the current SCHEMA_VERSION, through _UPGRADES, as a shallow copy of m.
    ValueError for a newer schema, or an older one with no upgrade."""
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


def _supports_and_loads(picks: dict[str, list[dict[str, Any]]],
                        fields: "dict[str, int] | None" = None) -> tuple[list[tuple], list[tuple]]:
    """Supports and loads as (kind, values, triangles, points, regions), from the picking windows' lists:
    constraint_data (supports) and force_data (loads) of the structural one, and picked (either, the kind
    saying which) of the others, each entry made a kind of BC_KINDS by the kind's picks (its "type"), among
    the kinds acting on fields (the project's physics' FIELDS; None: every kind)."""
    found = {"support": [], "load": []}
    for role, key in (("load", "force_data"), ("support", "constraint_data"), (None, "picked")):
        for entry in picks.get(key, []):
            t = entry.get("type")
            kinds = [(name, k) for name, k in BC_KINDS.items() if (role is None or k.role == role) and t in k.picks
                     and (fields is None or k.field in fields)]
            if not kinds:
                known = [p for k in BC_KINDS.values() if role is None or k.role == role for p in k.picks
                         if p is not None]
                raise NotImplementedError(f"{role or 'picked'} type {t!r} is not supported; use one of "
                                          f"{', '.join(known)}")
            name, kind = kinds[0]
            found[kind.role].append((name, kind.picks[t](entry), entry.get("triangles") or [],
                                     entry.get("points") or [], entry.get("regions") or []))
    return found["support"], found["load"]


class Project:
    """One problem: the part, its selections, supports and loads, keep-solid regions, material database,
    objective and constraints, and settings.

    name: the project's name, text; its runs are named after it
    stl: the part, an STL file: its path, absolute or relative to base_dir; None when part_shapes is given
    part_shapes: the part as a list of shape steps (lsr_mmto.shapes, see Making the part from shapes), built
        into an STL in part/ of base_dir when it is needed; None when stl is given. A project has one of the
        two, or neither while its part is still to come (has_part; validate says "no part")
    n_dof: how fine the mesh is, a whole number of unknowns (see Settings, mesh); None is the default there
    single_layer: True for one element through the thickness (a flat part such as a plate), False for a
        solid part; None is the default under Settings, mesh
    base_dir: the folder relative paths start from; None is the folder Python runs in
    Example: p = Project("my bracket", stl="data/models/UBracket/UBracket.STL", n_dof=8000)

    Set on a project once it is made (its project script sets them the same way):

    comment: a description, text
        Example: p.comment = "U-bracket, fixed on the left"
    physics: the physics the optimization uses, one of those under Physics
        Example: p.physics = "structural"
    material_source: where each element's material properties come from: "network" (the material
        network, from the database; Multi material in the Materials window) or "one_database_material" (one
        material of the database, pinned_material, at its database values, with no network; Single material)
        Example: p.material_source = "network"
    pinned_material: one database material's name, every element that material. With "network": through
        the network (Fix every element to, in the Materials window), and None lets the optimization choose
        each element's material. With "one_database_material": at its database values (Material:), and None
        only for a database of one material, which is then that one
        Example: p.pinned_material = None
    settings[section][name]: one setting: section is "common", "advanced" or "vae", name a setting of
        that section; every setting, its default and what it does are under Settings
        Default: the values under Settings
        Example: p.settings["common"]["maxIterations"] = 50
    """

    name: str
    stl: str | None                             # as given; relative paths are relative to base_dir
    part_shapes: "list[dict[str, Any]] | None"  # the shape steps, or None for an imported STL
    base_dir: str                               # absolute
    n_dof: int
    single_layer: bool
    comment: str
    physics: str                                # should be a name of PHYSICS; validate reports any other
    selections: "dict[str, S.Selection]"
    bcs: list[dict[str, Any]]                   # {"kind": should be a name of BC_KINDS, "on": a selection name, values}
    keep_solid_names: list[str]
    _database: str | None                       # a shipped database's name or a path, behind the database property
    material_source: str                        # one of MATERIAL_SOURCES
    pinned_material: str | None                 # a material's name, or None: see the docstring
    formulation: dict[str, Any]                 # {"objective": {"expression", "sense"} or None, "constraints": [...]}
    settings: dict[str, dict[str, Any]]         # {section: {key: value}}
    _gui: dict[str, Any]                        # everything else in the file, kept for the GUI
    _from_record: bool                          # a run's project (from_record): its built part removes no other

    def __init__(self, name: str, stl: str | None = None, n_dof: int | None = None, single_layer: bool | None = None,
                 base_dir: str | None = None, part_shapes: "list[dict[str, Any]] | None" = None) -> None:
        if stl is not None and part_shapes is not None:
            raise ValueError("a project's part is an STL file (stl=...) or a list of shapes (part_shapes=...): "
                             "give one of the two, or neither for a project whose part comes later")
        self.name = name
        if part_shapes is not None:
            check(part_shapes)
            part_shapes = plain(part_shapes)
        self.part_shapes = part_shapes
        self.stl = stl                          # relative paths: to base_dir
        self.base_dir = os.path.abspath(base_dir or os.getcwd())
        self.n_dof = int(MESH_DEFAULTS["n_dof"] if n_dof is None else n_dof)
        self.single_layer = bool(MESH_DEFAULTS["single_layer"] if single_layer is None else single_layer)
        self.comment = ""
        self.physics = DEFAULT_PHYSICS
        self.selections = {}                    # name -> Selection
        self.bcs = []                           # dicts, see BC_KINDS
        self.keep_solid_names = []
        self._keeps_supports = False        # an older file's "keep_support_elements": true (validate says so)
        self._database = None
        self.material_source = "network"        # or "one_database_material": see the docstring
        self.pinned_material = None             # None, or the material every element is: see the docstring
        self.formulation = {"objective": None, "constraints": []}
        self.settings = {k: dict(v) for k, v in DEFAULT_SETTINGS.items()}
        self._gui = {}                          # everything else in the file, kept for the GUI
        self._from_record = False

    # ------------------------------------------------------------------ building it up
    def add_selection(self, name: str, selection: "S.Selection") -> str:
        """Adds a selection under a name, which supports, loads and keep solid can then use; returns the name.

        name: the selection's name, text
        selection: one of the selection types above, e.g. face("xmin")
        Example: p.add_selection("left", left)
        """
        if not isinstance(selection, S.Selection):
            raise TypeError(f"{name}: a Selection is expected, e.g. face('xmin')")
        self.selections[name] = selection
        return name

    def _free_name(self, prefix: str) -> str:
        """prefix1, prefix2, ...: the first that names no selection yet."""
        i = 1
        while f"{prefix}{i}" in self.selections:
            i += 1
        return f"{prefix}{i}"

    def _on(self, on: "S.Selection | str", kind: str) -> str:
        """A selection name; a Selection given directly is stored under a generated name."""
        if isinstance(on, S.Selection):
            return self.add_selection(self._free_name(kind), on)
        return on

    def adopt_picks(self, gui: dict[str, Any]) -> list[str]:
        """Supports and loads picked on the part in the window (its constraint_data and force_data
        lists) as selections and boundary conditions of this project, with the picking tools' own
        rules: faces as stl_faces (tol 0.8), a clicked node as its nearest boundary node, a dragged
        box as halfspaces. Returns the names of the new selections."""
        fields = PHYSICS[self.physics].FIELDS if self.physics in PHYSICS else None
        supports, loads = _supports_and_loads(gui, fields)
        if not supports and not loads:
            return []

        def selection(triangles: list[int], points: list[list[float]], regions: list[dict[str, Any]],
                      what: str) -> "S.Selection":
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

    def add_bc(self, kind: str, on: "S.Selection | str", **values: Any) -> None:
        """A support or load of one of the kinds of the project's physics (Boundary conditions lists them, with
        each kind's physics) on a selection, its name or a Selection (stored under a new name), with the kind's values,
        e.g. p.add_bc("fixed", "clamp", dofs="xyz"), p.add_bc("force", "corner", total=[0, -4, 0]).
        The values are checked as the project's check checks them; a wrong one is an error here.

        kind: one of the kinds under Boundary conditions, such as "fixed" or "force"; a kind of another physics
            is taken here and reported by the project's check (validate)
        on: the selection it acts on: its name, or a selection type above (added under a new name)
        values: the kind's values, as keywords, e.g. dofs="xyz" or total=[0, -1000, 0]; each kind's values
            are under Boundary conditions
        Example: p.add_bc("fixed", "left", dofs="xyz")
        """
        if kind not in BC_KINDS:
            raise ValueError(f"kind must be one of {', '.join(BC_KINDS)}, not {kind!r}")
        k = BC_KINDS[kind]
        errors = k.errors({"kind": kind, **values})
        if errors:
            raise ValueError(f"{kind}: {errors[0]}")
        self.bcs.append({"kind": kind, "on": self._on(on, k.role),
                         **{v.name: v.stored(values[v.name]) for v in k.values}})

    def evaluate(self, selection: "S.Selection | str", mesh: "HexMesher", on: str = "nodes") -> np.ndarray:
        """Indices of a selection (or of the selection with that name) on mesh, with every selection(name) in
        it looked up now, so a combination follows later edits of its parts. Elements of a selection that
        picks only nodes (stl_faces, rows, ...) are the elements touching those nodes.

        selection: a selection's name, or a selection type above
        mesh: the mesh, from p.mesh()
        on: "nodes" for node numbers, "elements" for element numbers
        Example: print(len(p.evaluate("left", mesh)))
        """
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

    def rename_selection(self, old: str, new: str) -> None:
        """The selection and everything that names it: supports and loads, keep solid, other selections."""
        self.selections = {new if k == old else k: S.renamed(v, old, new) for k, v in self.selections.items()}
        for bc in self.bcs:
            if bc.get("on") == old:
                bc["on"] = new
        self.keep_solid_names = [new if n == old else n for n in self.keep_solid_names]

    @property
    def keep_support_elements(self) -> bool:
        """An older project's switch (keep every element at a support solid), no longer offered: True only for
        an older file that had it on, which validate refuses until the project is saved."""
        return self._keeps_supports

    @keep_support_elements.setter
    def keep_support_elements(self, on: bool) -> None:
        """An older project script's line: off changes nothing; on is an error saying what to use instead."""
        if on:
            raise ValueError("keep_support_elements is no longer offered: use p.keep_solid with each support's "
                             "selection (nearly the same elements, not always exactly)")

    def keep_solid(self, *names: "S.Selection | str") -> None:
        """Keeps these selections solid: their elements are always material.

        names: one or more selections: their names, or selection types above (each added under a new name)
        Example: p.keep_solid(boss)
        """
        for n in names:
            n = self._on(n, "keep")
            if n not in self.keep_solid_names:
                self.keep_solid_names.append(n)

    def objective(self, expression: str, sense: str = "minimize") -> None:
        """What the optimization makes as small as it can.

        expression: one quantity, text, such as "Mass()" or "FailureFactorPNorm()" (see Quantities)
        sense: "minimize", the only one for now
        Example: p.objective("FailureFactorPNorm()")
        """
        self.formulation["objective"] = {"expression": expression, "sense": sense}

    def constraint(self, expression: str, op: str, bound: float) -> None:
        """A limit on a quantity, expression op bound; each call adds one more constraint.

        expression: one quantity, text, such as "Mass()" or "MaterialVolumeFraction(2, tau=0.5)" (see
            Quantities)
        op: "<=" (at most bound) or ">=" (at least bound), for any quantity
        bound: the limit, a number in the project's units
        Example: p.constraint("Mass()", "<=", 0.0035)
        """
        self.formulation["constraints"].append({"expression": expression, "op": op, "bound": float(bound)})

    # ------------------------------------------------------------------ checks without a mesh
    def has_part(self) -> bool:
        """Whether the project has its part: an STL file or at least one shape step (built or not)."""
        return self.stl is not None or bool(self.part_shapes)

    def stl_path(self) -> str:
        """The part's STL. A part made of shapes is built into part/ first when no STL of these shapes is there
        (part_from_shapes.part_stl); for one that cannot be made or written, a path where no file is (validate
        says why)."""
        if self.part_shapes is not None:
            from lsr_mmto.problem.part_from_shapes import part_stl
            try:
                return part_stl(self)
            except (ShapeError, OSError):
                return os.path.join(self.base_dir, PART, "shapes_not_built.stl")
        if self.stl is None:                                # neither: validate says so
            return os.path.join(self.base_dir, PART, "no_part.stl")
        return self.stl if os.path.isabs(self.stl) else os.path.join(self.base_dir, self.stl)

    def part_warnings(self) -> list[str]:
        """What may not come out as meant in a part made of shapes: a cut that misses the part, a part in pieces,
        a feature under 2 elements across at the mesh settings. An empty list for an imported STL. ShapeError for
        a part that cannot be made. Part of warnings, which a run prints.

        Example: print(p.part_warnings())
        """
        if self.part_shapes is None:
            return []
        from lsr_mmto.problem.part_from_shapes import build_warnings, small_feature_warnings
        return build_warnings(self) + small_feature_warnings(self)

    def unknowns_per_node(self) -> int:
        """How many unknowns the project's physics solves for at each node (3 for structural: the
        displacement in x, y and z); the mesh is built for n_dof unknowns in all."""
        return sum(PHYSICS[self.physics if self.physics in PHYSICS else DEFAULT_PHYSICS].FIELDS.values())

    @property
    def units(self) -> str:
        """The unit system: "MKS", "mmKS", "IPS" (those of the PyTO-AD GUI, kept where the GUI
        keeps it, settings.unit_system) or "none". Numbers are never converted; the unit
        system says what they are meant to be in, so that a database in other units is reported.

        units: "MKS", "mmKS", "IPS" or "none" (see Unit systems)
        Example: p.units = "MKS"
        """
        return (self._gui.get("settings") or {}).get("unit_system", "none")

    @units.setter
    def units(self, system: str) -> None:
        if system == "none":
            (self._gui.get("settings") or {}).pop("unit_system", None)
            return
        settings = self._gui.setdefault("settings", {"temperature_unit": "Kelvin", "angle_unit": "Degree"})
        settings["unit_system"] = system

    def unit_record(self) -> dict[str, str]:
        """The units of the run's numbers, for the run record."""
        if self.units not in UNIT_SYSTEMS:
            return {"system": self.units}
        return dict(system=self.units, **UNIT_SYSTEMS[self.units])

    @property
    def database(self) -> str | None:
        """The material database. When the database chosen has a trained network, and the trained network is
        used (the vae setting use_pretrained_vae), the settings that give the network its shape (latentDim,
        vae_hiddenDim) are set from the network itself, and a printed line says which changed.

        database: the name of one in data/databases, such as "6Materials_AF_3Attr", or the path of an
            .xlsx file, absolute or relative to base_dir
        Example: p.database = "6Materials_AF_3Attr"
        """
        return self._database

    @database.setter
    def database(self, value: str | None) -> None:
        cleared = value != self._database and self.clear_hand_set_network()
        self._database = value
        for said in (cleared, self.match_network()):
            if said:
                print(f"[{self.name}] {said}")

    def clear_hand_set_network(self) -> str | None:
        """A VAE file set by hand (settings.vae.saveNet) cleared, as it belongs to the database chosen before;
        the network next to the database is then used. Returns the line saying so, or None when none was set."""
        vae = self.settings.setdefault("vae", {})
        if not vae.get("saveNet"):
            return None
        old = vae.pop("saveNet")
        return (f"The VAE file set by hand ({os.path.basename(old)}) was for the database chosen before, so it no "
                f"longer applies: the network next to the database is used.")

    def match_network(self) -> str | None:
        """The settings that give the network its shape (latentDim, vae_hiddenDim) set from the database's
        trained network, when it has one and the trained network is used. Returns the line saying what
        changed from what a run would have used, or None when nothing did."""
        vae = self.settings.setdefault("vae", {})
        if self._database is None or not vae.get("use_pretrained_vae", True):
            return None
        nt = self.network_path()
        shape = network_shape(nt) if os.path.isfile(nt) and os.path.getsize(nt) > 0 else None
        if shape is None:
            return None
        before = self.copy()
        before.complete_settings()                          # what a run would use now
        changed = []
        for key, value in (("vae_hiddenDim", shape[1]), ("latentDim", shape[2])):
            now = before.latent_dim() if key == "latentDim" else before.settings["vae"].get(key)
            if now != value:
                vae[key] = value
                changed.append(f"{BY_KEY[key].label[:1].lower()}{BY_KEY[key].label[1:]} {value}")
        return f"Network settings set to match the trained network: {', '.join(changed)}" if changed else None

    def database_paths(self) -> tuple[str, dict[str, float] | None]:
        """(xlsx, VAE defaults or None) for the database name or path."""
        if self.database in DATABASE_VAE:
            return os.path.join(DATABASE_DIR, f"{self.database}.xlsx"), DATABASE_VAE[self.database]
        path = self.database if os.path.isabs(self.database) else os.path.join(self.base_dir, self.database)
        return path, None

    def validate(self, network: bool = True) -> list[Problem]:
        """Problems that can be found without building the mesh; an empty list means none. Each is a
        Problem: its text, and the step of the setup it is about (Problem.gui_step). network False leaves out
        the checks of the trained network (vae_check), for an analysis, which uses the database's own values."""
        errors = []
        if self.part_shapes is not None:
            from lsr_mmto.problem.part_from_shapes import part_stl
            try:
                part_stl(self)
            except ShapeError as e:
                errors.append(Problem("part", f"the part's shapes: {e}"))
            except OSError as e:
                errors.append(Problem("part", f"the part made of shapes could not be written in "
                                              f"{os.path.join(self.base_dir, PART)}: {e}"))
        elif self.stl is None:
            errors.append(Problem("part", "no part yet: give the project its part in Geometry"))
        elif not os.path.isfile(self.stl_path()):
            errors.append(Problem("part", f"STL file {self.stl_path()} does not exist"))
        if self.physics not in PHYSICS:
            errors.append(Problem("part", f"physics {self.physics!r}: one of {', '.join(PHYSICS)}"))
        if self.units != "none" and self.units not in UNIT_SYSTEMS:
            errors.append(Problem("units", f"units {self.units!r}: one of {', '.join(UNIT_SYSTEMS)} or 'none'"))
        for i, bc in enumerate(self.bcs, 1):
            where = f"boundary condition {i} ({bc.get('kind')})"
            if bc.get("kind") not in BC_KINDS:
                errors.append(Problem("supports and loads", f"{where}: kind must be one of {', '.join(BC_KINDS)}"))
                continue
            if bc.get("on") not in self.selections:
                errors.append(Problem("supports and loads", f"{where}: no selection named {bc.get('on')!r}"))
            errors += [Problem("supports and loads", f"{where}: {e}") for e in BC_KINDS[bc["kind"]].errors(bc)]
            if self.physics in PHYSICS and BC_KINDS[bc["kind"]].field not in PHYSICS[self.physics].FIELDS:
                errors.append(Problem("supports and loads", f"{where}: not a kind of {self.physics} support or load"))
        if self._keeps_supports:                                  # an older file's switch, no longer offered
            errors.append(Problem("supports and loads", "this project keeps every element at a support solid, a "
                                  "switch no longer offered. Tick Keep solid on each support's selection instead "
                                  "(nearly the same elements, not always exactly); saving the project then drops "
                                  "the switch"))
        roles = {BC_KINDS[bc["kind"]].role for bc in self.bcs if bc.get("kind") in BC_KINDS}
        needs = PHYSICS[self.physics].NEEDS if self.physics in PHYSICS else ("support", "load")
        for role in needs:                                        # the physics says what a problem must have
            if role not in roles:
                errors.append(Problem("supports and loads", f"no {role}s"))
        for n in self.keep_solid_names:
            if n not in self.selections:
                errors.append(Problem("supports and loads", f"keep_solid: no selection named {n!r}"))
        for n, sel in self.selections.items():                   # selections inside selections: known, no loops
            try:
                S.resolved(sel, self.selections, (n,))
            except ValueError as e:
                errors.append(Problem("supports and loads", f"selection {n}: {e}"))
        if self.database is None:
            errors.append(Problem("materials", "no material database"))
        else:
            xlsx, _ = self.database_paths()
            db_errors = check_database(xlsx)[0]
            errors += [Problem("materials", f"material database {os.path.basename(xlsx)}: {e}") for e in db_errors]
            if not db_errors:
                errors += self._missing_columns(database_attributes(xlsx))
                errors += self.material_source_errors(xlsx)
                if self.material_source == "network" and network:
                    errors += self.vae_check()[0]
        for section, values in self.settings.items():
            if section not in DEFAULT_SETTINGS:
                errors.append(Problem("settings", f"settings: unknown section {section!r}"))
                continue
            for k, v in values.items():
                entry = BY_KEY.get(k)
                if entry is None or entry.section != section:
                    errors.append(Problem("settings", f"settings.{section}: unknown setting {k!r}"))
                elif entry.choices is not None and v is not None and v not in entry.choices.__members__:
                    errors.append(Problem("settings", f"settings.{section}.{k}: {v!r} is not one of "
                                                      f"{', '.join(entry.choices.__members__)}"))
        obj = self.formulation.get("objective")
        if not obj:
            errors.append(Problem("formulation", "no objective"))
        else:
            if obj.get("sense", "minimize") != "minimize":
                errors.append(Problem("formulation", "objective: only 'minimize' for now"))
            try:
                to_params_form(obj.get("expression"), "objective")
            except ValueError as e:
                errors.append(Problem("formulation", f"objective: {e}"))
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
            errors.append(Problem("formulation", f"the material volume fraction constraints use different tau "
                                                 f"({listed}); one tau applies to all of them: use the same value"))
        if any(t <= 0 for t in taus):
            errors.append(Problem("formulation", "tau must be greater than 0"))
        for i, c in enumerate(self.formulation.get("constraints", []), 1):
            if c.get("op") not in OPS:
                errors.append(Problem("formulation", f"constraint {i}: the operator must be '<=' or '>=', not "
                                                     f"{c.get('op')!r}"))
            if not c.get("bound"):
                errors.append(Problem("formulation", f"constraint {i}: the bound must be a nonzero number"))
            try:
                to_params_form(c.get("expression"), "constraint", c.get("bound") or 1.0)
            except ValueError as e:
                errors.append(Problem("formulation", f"constraint {i}: {e}"))
        named = [("objective", obj or {})] + [(f"constraint {i}", c)
                                               for i, c in enumerate(self.formulation.get("constraints", []), 1)]
        for where, c in named:                                     # each quantity of this physics or of every one
            try:
                q = find_quantity(parse_expression(c.get("expression"))[0])
            except (ValueError, KeyError):                         # an unknown or unreadable one: said above
                continue
            if self.physics in PHYSICS and not q.applies_to(self.physics):
                errors.append(Problem("formulation", f"{where}: {q.shown} is not a {self.physics} quantity"))
            if q.reads_latent_points and self.single_material():
                errors.append(Problem("formulation", f"{where}: {q.shown} needs a mix of materials, and this design "
                                                     f"has one material"))
        return errors

    def single_material(self) -> bool:
        """True when every element is one material: one database material, or the network pinned to one."""
        return self.material_source != "network" or self.pinned_material is not None

    def material_source_errors(self, xlsx: str) -> list[Problem]:
        """What is wrong with the material source for the database xlsx (which passes its check): a source not in
        MATERIAL_SOURCES; with one database material, no material chosen from a database of several; with the
        network, a database of one material; and a pinned material the database does not have."""
        from lsr_mmto.materials.databases import database_table
        names = [str(n) for n in database_table(xlsx)[0]]
        db = os.path.basename(xlsx)
        out = []
        if self.material_source not in MATERIAL_SOURCES:
            out.append(Problem("materials", f"Materials in the design: {self.material_source!r} is not a choice "
                                            f"LSR MMTO knows; choose Single material or Multi material (in a "
                                            f"script, material_source is {' or '.join(map(repr, MATERIAL_SOURCES))})"))
        elif self.material_source == "one_database_material":
            if self.pinned_material is None and len(names) > 1:
                out.append(Problem("materials", f"Single material: choose the material, one of the "
                                                f"{len(names)} in {db}"))
            elif self.pinned_material is not None and str(self.pinned_material) not in names:
                out.append(Problem("materials", f"Material: {self.pinned_material!r} is not in {db}; one of "
                                                f"{', '.join(names)}"))
        elif len(names) < 2:
            out.append(Problem("materials", f"Multi material: {db} has one material, and the network needs at "
                                            f"least two to learn from; choose Single material"))
        elif self.pinned_material is not None and str(self.pinned_material) not in names:
            out.append(Problem("materials", f"Fix every element to: {self.pinned_material!r} is not in {db}; one "
                                            f"of {', '.join(names)}"))
        return out

    def gui_step_started(self, step: str) -> bool:
        """True once the step of GUI_STEPS is begun (units chosen, a database, a support or load, an
        objective)."""
        return GUI_STEPS[step](self)

    def needed_properties(self) -> list[str]:
        """The material properties (database columns) this project reads: its physics' PROPERTIES, and those
        its objective and constraints read (each quantity's needs), in that order, each once."""
        from lsr_mmto.formulation.quantities import QUANTITIES
        by_name, f = {q.name: q for q in QUANTITIES}, self.formulation
        out = list(PHYSICS[self.physics].PROPERTIES if self.physics in PHYSICS else ())
        for expr in [(f.get("objective") or {}).get("expression")] + \
                    [c.get("expression") for c in f.get("constraints", [])]:
            try:
                q = by_name.get(parse_expression(expr)[0])
            except Exception:                                                          # noqa: BLE001
                continue
            out += list(q.needs if q else ())
        return list(dict.fromkeys(out))

    def starting_properties(self) -> list[str]:
        """The columns a new database for this project starts with: Density, then the properties it needs
        (needed_properties), then those its physics reads when present (OPTIONAL_PROPERTIES), each once and
        each group in the order of the property list (databases.ATTRIBUTES)."""
        from lsr_mmto.materials.databases import ATTRIBUTES
        optional = PHYSICS[self.physics].OPTIONAL_PROPERTIES if self.physics in PHYSICS else ()
        needed = set(self.needed_properties())
        return list(dict.fromkeys(["Density"] + [k for k in ATTRIBUTES if k in needed]
                                  + [k for k in ATTRIBUTES if k in optional]))

    def _missing_columns(self, columns: list[str]) -> list[Problem]:
        """Columns the physics and the objective and constraints read that the database lacks."""
        from lsr_mmto.formulation.quantities import QUANTITIES
        out = [Problem("materials", f"material database: the {self.physics} model needs a {c} column")
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
                    out.append(Problem("materials", f"material database: {q.name} needs a {c} column"))
        return out

    def _unit_ratios(self, xlsx: str) -> tuple[dict[str, str | None], dict[str, float]]:
        """(units, ratio): each database column's unit as written, and how many times larger the numbers of
        each column with a unit (not a ratio) are than the same values in the project's units."""
        units = database_units(xlsx)
        return units, {n: unit_ratio(ATTRIBUTES[n].measures, u, self.units) for n, u in units.items()
                       if u is not None and ATTRIBUTES[n].measures != "ratio"}

    @staticmethod
    def _quantity_ratio(q: "Quantity", ratio: dict[str, float]) -> float:
        """How many times its real value a quantity comes out with columns that many times too large."""
        return float(np.prod([ratio[c] ** k for c, k in q.scales_with.items() if c in ratio]))

    def quantity_unit(self, q: "Quantity") -> str:
        """A quantity's unit in the project's units, with a space before it; "" for one without units."""
        system = UNIT_SYSTEMS[self.units]
        unit = q.unit.format(**system) if "{" in q.unit else q.unit
        return "" if unit == "no units" or unit.startswith("fraction") else f" {unit}"

    def unit_conversion(self) -> "tuple[list[tuple[str, str, str, float]], list[tuple[int, Quantity, float, float]]]":
        """What converting the database into the project's units changes: (columns, limits). columns: (name,
        unit, the project's unit, factor) for each column in other units, its numbers to be multiplied by
        factor; limits: (index of the constraint, its quantity, limit, converted limit) for each limit the
        conversion changes the meaning of: the converted limit means for the converted database what the
        limit means now. Both empty when there is nothing to convert (no units, no database, or one with
        errors)."""
        from lsr_mmto.formulation.quantities import QUANTITIES
        if self.units not in UNIT_SYSTEMS or self.database is None:
            return [], []
        xlsx, _ = self.database_paths()
        if check_database(xlsx)[0]:
            return [], []
        system = UNIT_SYSTEMS[self.units]
        units, ratio = self._unit_ratios(xlsx)
        ratio = {n: r for n, r in ratio.items() if abs(r - 1.0) >= 1e-12}
        columns = []
        for n in ratio:
            kind = UNITS[ATTRIBUTES[n].measures]
            target = system[ATTRIBUTES[n].measures]
            columns.append((n, units[n], target, kind[units[n]] / kind[target]))
        by_name, limits = {q.name: q for q in QUANTITIES}, []
        for i, c in enumerate(self.formulation.get("constraints", [])):
            try:
                q = by_name.get(parse_expression(c["expression"])[0])
            except Exception:                                                          # noqa: BLE001
                continue
            factor = self._quantity_ratio(q, ratio) if q is not None else 1.0
            if abs(factor - 1.0) >= 1e-12:
                limits.append((i, q, c["bound"], c["bound"] / factor))
        return columns, limits

    def _unit_warnings(self, xlsx: str) -> list[str]:
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
        units, ratio = self._unit_ratios(xlsx)

        def times(r: float) -> str:
            e = np.log10(r)
            return f"10^{int(round(e))}" if abs(e - round(e)) < 1e-9 else f"{r:.4g}"

        out, at = [], {}                     # at: the warning of each column, by its name
        for name, r in ratio.items():
            if abs(r - 1.0) < 1e-12:
                continue
            a = ATTRIBUTES[name]
            say = (f"the database gives {a.shown} in {units[name]}, but the project's units "
                   f"({self.units}) expect {system[a.measures]}: its numbers are {times(r)} times what they "
                   f"should be.")
            if a.wrong_unit_effect:
                say += " " + a.wrong_unit_effect.format(times=times(r))
            at[name] = len(out)
            out.append(say)
        # what the wrong units do to each quantity the project uses, from the quantity's own scales_with,
        # said once, under the first of its columns that is off
        by_name = {q.name: q for q in QUANTITIES}
        used = [(n, bs[0]) for n, bs in bounds.items()] + ([] if objective in bounds else [(objective, None)])
        for name, bound in used:
            q = by_name.get(name)
            if q is None:
                continue
            factor = self._quantity_ratio(q, ratio)
            off = [c for c in q.scales_with if c in at]
            if abs(factor - 1.0) < 1e-12 or not off:
                continue
            unit = self.quantity_unit(q)
            if bound is None:
                out[at[off[0]]] += (f" {q.shown} comes out {times(factor)} times its real value. As the "
                                    f"objective that changes only the numbers shown, not the design.")
            else:
                out[at[off[0]]] += (f" {q.shown} comes out {times(factor)} times its real value, so its limit "
                                    f"of {bound:g} is really a limit of {bound / factor:g}{unit}.")
        return out

    def warnings(self) -> list[str]:
        """What is allowed but deserves a look before a run: what may not come out as meant in a part made of
        shapes (part_warnings), the database's warnings, a database in units other than the project's, a latent
        dimension other than the one recommended for the database, and what a constraint's quantity says about
        its limit. A run prints them before it starts; none of them stops it.

        Example: print(p.warnings())
        """
        try:
            out = [f"part: {w}" for w in self.part_warnings()]
        except (ShapeError, OSError):
            out = []                                        # a part that cannot be made: validate says why
        if self.database is None:
            return out
        xlsx, _ = self.database_paths()
        errors, db_warnings = check_database(xlsx)
        if errors:
            return out
        out += [f"material database {os.path.basename(xlsx)}: {w}" for w in db_warnings]
        out += self._unit_warnings(xlsx)
        if self.material_source == "network":
            out += self.vae_check()[1]
        out += self._bound_warnings()
        return out

    def _bound_warnings(self) -> list[str]:
        """What each constraint's quantity says about its limit (Quantity.bound_warning)."""
        from lsr_mmto.formulation.quantities import QUANTITIES
        by_name = {q.name: q for q in QUANTITIES}
        out = []
        for i, c in enumerate(self.formulation.get("constraints", []), 1):
            try:
                q = by_name.get(parse_expression(c["expression"])[0])
                said = q.bound_warning(self, float(c["bound"])) if q and q.bound_warning and \
                    c.get("op", "<=") == "<=" else None                     # its warnings are about "at most"
            except Exception:                                                          # noqa: BLE001
                continue                                    # a bad constraint is an error of its own
            if said:
                out.append(f"constraint {i} ({q.shown} <= {float(c['bound']):g}): {said}")
        return out

    def network_path(self) -> str:
        """The VAE file a run loads or trains: settings.vae.saveNet, or else the .nt next to the
        database's .xlsx with the same name (my_db.xlsx, my_db.nt)."""
        xlsx, _ = self.database_paths()
        return self.settings.get("vae", {}).get("saveNet") or os.path.splitext(xlsx)[0] + ".nt"

    def vae_check(self) -> tuple[list[Problem], list[str], str]:
        """(errors, warnings, status) about the VAE: a run that would train into the shipped folder,
        a trained network that does not fit the settings or the database, a network a run would
        replace, a latent dimension other than the recommended one; status says which network the
        database has. For a database that passes its check. Each error says how to put it right.

        Example: errors, warnings, status = p.vae_check()
        """
        vae = self.settings.get("vae", {})
        xlsx, _ = self.database_paths()
        nt = self.network_path()
        name = os.path.basename(nt)
        shipped, exists = in_shipped_folder(nt), os.path.isfile(nt) and os.path.getsize(nt) > 0
        train = not vae.get("use_pretrained_vae", True) or not exists
        errors, warnings_ = [], []
        if train and shipped:
            errors.append(Problem("materials", f"a run would train a new VAE and save it over the shipped {name}; to "
                                               f"train a different network, copy {os.path.basename(xlsx)} to a "
                                               f"folder of your own and use the copy"))
        elif train and exists:
            warnings_.append(f"{name} exists: the next run trains a new network and replaces it (the old one is "
                             f"kept as {os.path.splitext(name)[0]}.previous.nt). To keep both, copy "
                             f"{os.path.basename(xlsx)} first.")
        if not train:
            shape = network_shape(nt)
            if shape is None:
                errors.append(Problem("materials", f"{name} could not be read as a VAE network"))
            else:
                n_in, hidden, latent = shape
                n_columns = len(database_attributes(xlsx))
                if n_in != n_columns:
                    errors.append(Problem("materials", f"{name} takes {n_in} attributes, but "
                                                       f"{os.path.basename(xlsx)} has {n_columns}: it was trained on "
                                                       f"another database"))
                for key, have in (("latentDim", latent), ("vae_hiddenDim", hidden)):
                    value = self.latent_dim() if key == "latentDim" else vae.get(key)
                    shown = f"{value} (recommended for the database)" if vae.get(key) is None else f"{value}"
                    if value is not None and int(value) != have:
                        errors.append(Problem("settings", f"{key} is {shown}, but {name} was trained with {have}: "
                                                          f"set it to {have}, or switch off use_pretrained_vae to "
                                                          f"train a new network"))
                if n_in == n_columns:
                    errors += self._stamp_errors(xlsx, nt, shipped)
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

    def _stamp_errors(self, xlsx: str, nt: str, shipped: bool) -> list[Problem]:
        """The database against the record its trained network keeps of it (network_check), as errors: a
        database changed since its network was trained, and any change to one that comes with LSR MMTO. A change
        a yes settles (CONFIRMABLE) is no error here: the window asks before a run, and a script's run refuses
        until confirm_network (_ready_to_run)."""
        from lsr_mmto.materials.network_stamp import CHANGED, SAME
        check = self.network_check()
        if check is None or check.state == SAME:
            return []
        db = os.path.basename(xlsx)
        if shipped:
            return [Problem("materials", f"{db}, which comes with LSR MMTO, does not match its network: "
                                         f"{check.text}. Put the file back as it came, or make your own copy with "
                                         f"Edit... in Materials")]
        if check.state == CHANGED:
            return [Problem("materials", f"{db}: {check.text}. Train a new network (untick Use the trained network "
                                         f"in Settings), or bring back the version it was trained on (Versions... "
                                         f"in Materials)")]
        return []

    def network_check(self) -> "Comparison | None":
        """The database as it is now against the record its trained network keeps of the database it was trained
        on (materials/network_stamp.py): a Comparison, its state SAME, CHANGED (no run until a new network is
        trained), DISPLAY, TO_CONFIRM or NO_STAMP (a run asks once, confirm_network) and a line saying what
        changed. None when a run uses no trained network: single material, a network to be trained, none to read.
        A network whose record can't be read is CHANGED: a run never goes ahead unchecked.

        Example: check = p.network_check()
        """
        from lsr_mmto.materials.network_stamp import CHANGED, Comparison, compare
        vae = self.settings.get("vae", {})
        if self.material_source != "network" or self._database is None or not vae.get("use_pretrained_vae", True):
            return None
        nt = self.network_path()
        if not (os.path.isfile(nt) and os.path.getsize(nt) > 0):
            return None
        try:
            return compare(self.database_paths()[0], nt)
        except Exception as e:                                                         # noqa: BLE001
            return Comparison(CHANGED, f"its network's record of the database could not be read ({e})")

    def confirm_network(self) -> None:
        """A yes to using the trained network on the database as it is now, when network_check asks for one
        (only colours or labels changed, units converted, or a network from before networks kept a record): the
        network is stamped with the database as it is now, so it is not asked about again. Nothing to do when
        the network matches its database or a run uses none. ValueError when the database has changed since the
        network was trained, or the network is one that comes with the app.

        Example: p.confirm_network()
        """
        from lsr_mmto.materials.network_stamp import CONFIRMABLE, SAME, stamp
        check, nt = self.network_check(), self.network_path()
        if check is None or check.state == SAME:
            return
        if check.state not in CONFIRMABLE:
            raise ValueError(f"{os.path.basename(nt)}: {check.text}; a new network has to be trained")
        if in_shipped_folder(nt):
            raise ValueError(f"{os.path.basename(nt)} comes with LSR MMTO and is never changed")
        stamp(nt, self.database_paths()[0])

    def keep_database_version(self) -> "Version | None":
        """The project's database kept as a version when it is in a state no version has, a change made in Excel
        say, and its trained network kept with the version in use when it was trained on it
        (database_versions.record). Called as a project opens in the window and as a run starts and ends, never
        by validate. Only a database in the project's folder: one elsewhere (a shared folder, another project) is
        never written beside. The version in use, or None (no database, one that comes with LSR MMTO, one outside
        the project's folder, one that cannot be read just now).

        Example: p.keep_database_version()
        """
        from lsr_mmto.materials.database_versions import record
        from lsr_mmto.problem.project_folder import inside
        if self._database is None:
            return None
        try:
            xlsx = self.database_paths()[0]
        except Exception:                                                              # noqa: BLE001
            return None
        return record(xlsx) if inside(xlsx, self.base_dir) else None

    def network_error(self) -> tuple[list[dict[str, Any]], str] | tuple[None, None]:
        """How closely the database's trained network gives back its materials: (rows, line). rows: one for
        each material and property, the database's value and the network's; line: the same in one sentence,
        as the Materials panel shows it. (None, None) when the database has no trained network yet, or the
        network cannot be read for it (vae_check says why).

        Example: rows, line = p.network_error()
        """
        from lsr_mmto.materials.encoder import network_reconstruction
        nt = self.network_path()
        if not (os.path.isfile(nt) and os.path.getsize(nt) > 0):
            return None, None
        try:
            rows = network_reconstruction(self.database_paths()[0], nt)
        except Exception:                                                              # noqa: BLE001
            return None, None
        if not rows:
            return None, None
        worst = max(r["percent_of_range"] for r in rows)
        return rows, f"The trained network gives back every material within {worst:.2g}% of each property's range."

    def complete_settings(self) -> list[str]:
        """Fill every missing setting from the defaults; returns what was filled."""
        filled = []
        for section, defaults in DEFAULT_SETTINGS.items():
            values = self.settings.setdefault(section, {})
            for k, v in defaults.items():
                if k not in values:
                    values[k] = v
                    filled.append(f"{section}.{k}")
        vae_defaults = {k: getattr(VAEParams(), k) for k in VAE_PARAMS}
        vae_defaults["latentDim"] = None                    # empty: the one recommended for the database (latent_dim)
        if self.database is not None:
            _, known = self.database_paths()
            if known:                                       # a shipped database: what its network was trained with
                vae_defaults.update(known)
        for k in VAE_PARAMS:
            if k not in self.settings["vae"]:
                self.settings["vae"][k] = vae_defaults[k]
                filled.append(f"vae.{k}")
        return filled

    def latent_dim(self) -> int:
        """The latent dimension a run uses: the setting when a number is set, else the one recommended for the
        database (recommended_latent_dim), else VAEParams' when there is no database or it fails its check.

        Example: n_z = p.latent_dim()
        """
        value = self.settings.get("vae", {}).get("latentDim")
        if value is not None:
            return int(value)
        if self.database is not None:
            xlsx, _ = self.database_paths()
            if os.path.isfile(xlsx) and not check_database(xlsx)[0]:
                return recommended_latent_dim(xlsx)[1]
        return int(VAEParams().latentDim)

    # ------------------------------------------------------------------ the problem
    def mesh(self) -> "HexMesher":
        """The voxel mesh of the part, made from the mesh settings (n_dof, single_layer) as a run makes it.

        Example: mesh = p.mesh()
        """
        from lsr_mmto.problem.mesh import make_mesh
        return make_mesh(self.stl_path(), self.n_dof, self.single_layer, self.unknowns_per_node())

    def formulation_params(self) -> "tuple[tuple, list[tuple]]":
        """The objective and the constraints as a run gets them (to_params.Objective, to_params.Constraints):
        (name, options) and [(name, options, limit, op)], op "<=" (at most) or ">=" (at least)."""
        obj = self.formulation["objective"]
        return (to_params_form(obj["expression"], "objective"),
                [to_params_form(c["expression"], "constraint", c["bound"]) + (c.get("op", "<="),)
                 for c in self.formulation.get("constraints", [])])

    def build(self, verbose: bool = True, network: bool = True) \
            -> "tuple[HexMesher, Material, dict[str, BC], None, MMTOParams, VAEParams]":
        """mesh, mat_prop, bc, elem_body_force, to_params, vae_params, after the project's check (validate; with
        network False, for an analysis or a picture of the mesh, without the trained network's checks)."""
        filled = self.complete_settings()
        if filled and verbose:
            print(f"[{self.name}] settings not in the project, taken from the defaults: {', '.join(filled)}")
        errors = self.validate(network)
        if errors:
            raise ValueError(f"project {self.name!r}:\n  " + "\n  ".join(errors))
        if verbose:
            shown = self.warnings()
            if shown != getattr(self, "_shown_warnings", None):          # once per change, not every build
                for w in shown:
                    print(f"[{self.name}] warning: {w}")
                self._shown_warnings = shown
        from lsr_mmto.pyto_ad_imports import mat_lib
        from lsr_mmto.optimize.mmto_to_params import MMTOParams

        mesh = self.mesh()
        bcs, _ = self.supports_and_loads(mesh)
        error = self.supports_and_loads_error(mesh, bcs)            # as PyTO-AD's FEA does, but before the
        if error:                                                   # VAE and the run
            raise ValueError(f"project {self.name!r}: {error}")

        # ---- optimization parameters
        to_params = MMTOParams()
        to_params.Comment = self.comment or self.name
        for e in SETTINGS:                                          # every setting TOParams has a field for
            if e.pyto_ad_field is not None:
                value = getattr(self, e.key) if e.section == "mesh" else self.settings[e.section][e.key]
                setattr(to_params, e.pyto_ad_field, pyto_ad_value(e, value))
        for e in SETTINGS:                                          # every setting PyTO-AD keeps elsewhere
            if e.at_build is not None and e.applies_to(self.physics):   # this project's physics' only
                e.at_build(getattr(self, e.key) if e.section == "mesh" else self.settings[e.section][e.key])
        xlsx, _ = self.database_paths()
        to_params.MaterialsExcelFile = xlsx
        to_params.Objective, to_params.Constraints = self.formulation_params()

        vae_params = VAEParams()
        for k in VAE_PARAMS:
            setattr(vae_params, k, self.settings["vae"][k])
        vae_params.latentDim = self.latent_dim()            # an empty setting: the database's recommendation

        # ---- keep-solid elements
        to_params.ElemsToKeep = None
        if self.keep_solid_names:
            keep = np.unique(np.concatenate([self.evaluate(n, mesh, "elements")
                                             for n in self.keep_solid_names])).astype(int)
            mesh.nonDesignElems = keep
            to_params.ElemsToKeep = keep

        mat_prop = mat_lib.create_material_with_defaults(          # placeholder; the VAE gives the real ones
            "CustomMaterial", youngs_modulus=1.0, poissons_ratio=0.3, mass_density=1.0, yield_strength=1.0)
        if verbose:
            held = "  ".join(f"{field}: fixed dofs={len(bc.fixed_dofs)}  total load="
                             f"{np.round(bc.force.reshape(-1, PHYSICS[self.physics].FIELDS[field]).sum(0), 6).tolist()}"
                             for field, bc in bcs.items())
            print(f"[{self.name}] elems={mesh.num_elems}  h={np.round(mesh.elem_size, 4).tolist()}  {held}  "
                  f"keep solid={0 if to_params.ElemsToKeep is None else len(to_params.ElemsToKeep)}")
        return mesh, mat_prop, bcs, None, to_params, vae_params

    def supports_and_loads(self, mesh: "HexMesher") -> "tuple[dict[str, BC], dict[str, np.ndarray]]":
        """({field: PyTO-AD BC}, {selection name: its nodes}): the supports, then the loads (BC_KINDS), on
        mesh, one boundary condition for each field of the project's physics. ValueError for a selection
        with no nodes or a dof given two values."""
        from lsr_mmto.pyto_ad_imports import bound_cond
        nodes_of = {}

        def nodes(name: str, where: str) -> np.ndarray:
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

    def supports_and_loads_error(self, mesh: "HexMesher", bcs: "dict[str, BC]") -> str | None:
        """Why the supports and loads leave the problem without a solution (the part free to move), or None:
        the physics' own check."""
        return PHYSICS[self.physics].supports_and_loads_error(mesh, bcs)

    # ------------------------------------------------------------------ files
    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": SCHEMA_VERSION, "name": self.name, "physics": self.physics,
                "comment": self.comment,
                "mesh": {"n_dof": self.n_dof, "single_layer": self.single_layer},
                "part_shapes": None if self.part_shapes is None else plain(self.part_shapes),
                "selections": {k: s.to_dict() for k, s in self.selections.items()},
                "bcs": [dict(bc) for bc in self.bcs],
                "keep_solid": list(self.keep_solid_names),
                "database": self.database,
                "material_source": self.material_source,
                "pinned_material": self.pinned_material,
                "formulation": {"objective": self.formulation.get("objective"),
                                "constraints": [dict(c) for c in self.formulation.get("constraints", [])]},
                "settings": {k: dict(v) for k, v in self.settings.items()}}

    def _file_data(self, path: str) -> dict[str, Any]:
        """The JSON a project file at path holds, with the STL path relative to it."""
        data = dict(self._gui)
        if not self.has_part() or self.part_shapes is not None:
            data["stl_file_path"] = None                        # no part yet, or its shapes, under mmto
        else:
            stl = self.stl_path()
            try:
                stl = os.path.relpath(stl, os.path.dirname(path))
            except ValueError:                                  # another drive on Windows
                pass
            data["stl_file_path"] = stl.replace("\\", "/")
        data["mmto"] = self.to_dict()
        return data

    def save(self, path: str) -> None:
        """Write the project, every setting included, with the other sections it was loaded with.
        Always as .mmto. Its folder becomes the project's folder, as the window keeps it: the part and
        the database with its network are copied into part/ and materials/ there (project_folder.pack),
        unless they are in it already.

        path: the .mmto file, absolute or relative to the folder Python runs in
        Example: p.save("my bracket/my bracket.mmto")
        """
        from lsr_mmto.problem.project_folder import pack
        if not path.lower().endswith(".mmto"):
            raise ValueError(f"{path}: an LSR MMTO project is saved as a .mmto file")
        path = os.path.abspath(path)
        self._keeps_supports = False                # the file it writes has no such switch
        pack(self, os.path.dirname(path))
        self.complete_settings()
        data = self._file_data(path)
        with open(path, "w") as fh:
            json.dump(data, fh, indent=2)
        self.base_dir, self.stl = os.path.dirname(path), data["stl_file_path"]      # None for shapes

    def save_as_script(self, path: str) -> str:
        """Write a Python script that rebuilds this project with the same calls, next to it.

        The physics and the material source are always written, since they say what problem this is; settings
        that differ from a new project's are written, and the rest are the defaults.

        path: the .py file, in the project's folder
        Example: p.save_as_script("my bracket/my bracket_setup.py")
        """
        path = os.path.abspath(path)
        if os.path.normcase(os.path.dirname(path)) != os.path.normcase(self.base_dir):
            raise ValueError(f"save the script in {self.base_dir}, next to the project, so that the "
                             "relative paths in it stay valid")
        shapes = self.part_shapes is not None
        if shapes:
            check(self.part_shapes)                             # ShapeError: a script of these could not run
            stl = None
        elif self.stl is None:                                  # no part yet
            stl = None
        else:
            stl = self.stl if not os.path.isabs(self.stl) else os.path.relpath(self.stl, self.base_dir)
            stl = stl.replace("\\", "/")
        reference = Project(self.name, stl, base_dir=self.base_dir,                  # its settings: a new project's
                            part_shapes=self.part_shapes if shapes else None)
        reference.database = self.database
        reference.complete_settings()
        mine = self.copy()
        mine.complete_settings()
        def explained(block: str, what: str) -> list[str]:      # the comment above each block of calls
            return [f"# {block}: {what} are in docs/reference.md",
                    "# (in the LSR MMTO folder), or in the GUI under Help > Researcher guide > Reference."]

        types = sorted({type(s).__name__.lower() for s in self.selections.values()} |
                       {type(t).__name__.lower() for s in self.selections.values()
                        for t in getattr(s, "parts", ())})
        lines = [f'"""{self.name}: rebuilds the project {self.name!r}. Written by Project.save_as_script.',
                 "",
                 "Run it with File > Run script in LSR MMTO, which opens the project it saves, or from any folder",
                 f"with: python {os.path.basename(path)}       (saves {self.name}.mmto next to it)",
                 "",
                 "LSR MMTO writes this file again each time the project is saved in its window, so edits",
                 "made here are replaced then. Saving from this script does not rewrite it.",
                 '"""',
                 "import os",
                 "",
                 f"from lsr_mmto import {', '.join(['Project'] + types + (['shapes as S'] if shapes else []))}",
                 "",
                 "HERE = os.path.dirname(os.path.abspath(__file__))",
                 ""] + explained("Part and mesh", "Project, its arguments and the units")
        if shapes:
            lines += ["# the part, made of shapes: every shape and step is in docs/reference.md, Making the part from shapes",
                      "part_shapes = ["] + to_script(self.part_shapes) + [
                      "]",
                      f"p = Project({self.name!r}, part_shapes=part_shapes, n_dof={self.n_dof}, "
                      f"single_layer={self.single_layer},",
                      "            base_dir=HERE)"]
        elif stl is None:
            lines += ["# no part yet: give it stl=... or part_shapes=... (docs/reference.md)",
                      f"p = Project({self.name!r}, n_dof={self.n_dof}, single_layer={self.single_layer}, base_dir=HERE)"]
        else:
            lines += [f"p = Project({self.name!r}, stl={stl!r}, n_dof={self.n_dof}, single_layer={self.single_layer},",
                      "            base_dir=HERE)"]
        if self.comment:
            lines.append(f"p.comment = {self.comment!r}")
        lines.append(f"p.physics = {self.physics!r}")                 # always: it says what problem this is
        if self.units != "none":
            lines.append(f"p.units = {self.units!r}")
        if self.selections:
            lines += [""] + explained("Selections", "every selection type, its arguments and an example")
            lines += [f"p.add_selection({name!r}, {sel!r})" for name, sel in self.selections.items()]
        if self.bcs:
            lines += [""] + explained("Supports and loads", "p.add_bc and the values of each kind")
            lines += [BC_KINDS[bc["kind"]].script(bc["kind"], bc, repr(bc["on"])) for bc in self.bcs]
        if self.keep_solid_names:
            lines += [""] + explained("Keep solid", "p.keep_solid")
            lines.append(f"p.keep_solid({', '.join(map(repr, self.keep_solid_names))})")
        lines += [""] + explained("Materials", "p.database and the columns of a database")
        lines.append(f"p.database = {self.database!r}")
        lines.append(f"p.material_source = {self.material_source!r}")     # always, as the physics
        if self.pinned_material is not None:
            lines.append(f"p.pinned_material = {self.pinned_material!r}")
        obj = self.formulation.get("objective")
        if obj or self.formulation.get("constraints"):
            lines += [""] + explained("Objective and constraints", "p.objective, p.constraint and every quantity")
        if obj:
            lines.append(f"p.objective({obj['expression']!r})")
        for c in self.formulation.get("constraints", []):
            lines.append(f"p.constraint({c['expression']!r}, {c['op']!r}, {c['bound']!r})")
        changed = [(sec, k, v) for sec, values in mine.settings.items() for k, v in values.items()
                   if reference.settings.get(sec, {}).get(k, object()) != v]
        if changed:
            lines += [""] + explained("Settings", "every setting, its default and what it does")
            lines += ["# settings that differ from a new project's; the rest are the defaults"]
            lines += [f"p.settings[{sec!r}][{k!r}] = {v!r}" for sec, k, v in changed]
        lines += ["", f"p.save(os.path.join(HERE, {self.name + '.mmto'!r}))", ""]
        with open(path, "w") as fh:
            fh.write("\n".join(lines))
        return path

    def copy(self, **changes: Any) -> "Project":
        import copy
        p = copy.deepcopy(self)
        for k, v in changes.items():
            setattr(p, k, v)
        return p

    def to_record(self) -> dict[str, Any]:
        """The project as a run record stores it."""
        return {"project": self.to_dict(), "stl_file_path": self.stl_path(), "base_dir": self.base_dir,
                "gui": self._gui}

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> "Project":
        """The project a run record stores, its paths as load gives them (absolute, in the project folder
        where it is now: load_results.absolute_paths)."""
        data = dict(record.get("gui") or {}, mmto=record["project"], stl_file_path=record["stl_file_path"])
        p = cls._from_data(data, record["base_dir"], record["project"].get("name", "project"))
        p._from_record = True                       # its built part removes no other (part_from_shapes.part_stl)
        return p

    @classmethod
    def load(cls, path: str) -> "Project":
        """A project read from its .mmto file, as the GUI and p.save write it.

        path: the .mmto file, absolute or relative to the folder Python runs in
        Example: p = Project.load("projects/U-bracket.mmto")
        """
        path = os.path.abspath(path)
        if not path.lower().endswith(".mmto"):
            raise ValueError(f"{path}: not an LSR MMTO project; a project is a .mmto file")
        with open(path) as fh:
            data = json.load(fh)
        if "mmto" not in data:
            raise ValueError(f"{path}: no mmto section; it is not an LSR MMTO project")
        return cls._from_data(data, os.path.dirname(path), os.path.splitext(os.path.basename(path))[0])

    @classmethod
    def _from_data(cls, data: dict[str, Any], base_dir: str, default_name: str) -> "Project":
        """A project from the JSON a project file holds (upgraded to SCHEMA_VERSION first); base_dir is the folder
        its relative paths start from, default_name its name when the data gives none."""
        m = upgrade(data.get("mmto") or {}, default_name)
        mesh = m.get("mesh", {})
        p = cls(m.get("name") or default_name, data.get("stl_file_path") or None,      # None: no part yet
                n_dof=mesh.get("n_dof"), single_layer=mesh.get("single_layer"), base_dir=base_dir)
        if m.get("part_shapes") is not None:                # a part made of shapes, checked by validate as loaded
            p.stl, p.part_shapes = None, plain(m["part_shapes"])
        p.comment, p.physics = m.get("comment", ""), m.get("physics", DEFAULT_PHYSICS)
        p.selections = {k: S.from_dict(v) for k, v in m.get("selections", {}).items()}
        p.bcs = [dict(bc) for bc in m.get("bcs", [])]
        p.keep_solid_names = list(m.get("keep_solid", []))
        p._keeps_supports = bool(m.get("keep_support_elements", False))    # no longer offered: validate says so
        p._database = m.get("database")                    # as saved: its settings stay as they are
        p.material_source = m.get("material_source", "network")    # older files: the network, not pinned
        p.pinned_material = m.get("pinned_material")
        f = m.get("formulation") or {}
        p.formulation = {"objective": f.get("objective"), "constraints": list(f.get("constraints", []))}
        stored = m.get("settings") or {}
        p.settings = {k: dict(stored.get(k, {})) for k in DEFAULT_SETTINGS}    # missing ones: filled at build
        p.settings.update({k: dict(v) for k, v in stored.items() if k not in DEFAULT_SETTINGS})
        p._gui = {k: v for k, v in data.items() if k not in ("mmto", "stl_file_path")}
        return p


# ---------------------------------------------------------------------- running
def _run_arguments(project: Project, save_dir: str | None = None, verbose: bool = True, plot_progress: bool = False,
                   plot_bcs: bool = False, plotter: "pyvista.Plotter | None" = None,
                   progress: Callable[[dict[str, Any]], None] | None = None,
                   should_stop: Callable[[], bool] | None = None,
                   on_network_error: Callable[[str], bool] | None = None) -> dict[str, Any]:
    """The keyword arguments _run_topopt gets for this project: its settings as one object s
    (settings.run_settings), the project, where the results go, and how the run is watched.
    Internal: a project runs through run(p). With no save_dir, the results go where the window puts
    them: results/<date and time> in the project's folder (project_folder.new_run_folder)."""
    from lsr_mmto.problem.project_folder import new_run_folder
    project.complete_settings()
    return dict(s=run_settings(project), project=project,
                save_dir=save_dir if save_dir is not None else new_run_folder(project),
                verbose=verbose, plot_progress=plot_progress, plot_bcs=plot_bcs, plotter=plotter,
                progress=progress, should_stop=should_stop, on_network_error=on_network_error)


def _ready_to_run(project: Project, network: bool = True) -> Project:
    """The project as a run gets it: its settings completed (saying which came from the defaults),
    checked (validate, with network: False for an analysis, which uses no network), and copied, so that the
    run records the project as it is now."""
    filled = project.complete_settings()
    if filled:
        print(f"[{project.name}] settings not in the project, taken from the defaults: {', '.join(filled)}")
    errors = project.validate(network)
    if errors:
        raise ValueError(f"project {project.name!r}:\n  " + "\n  ".join(errors))
    from lsr_mmto.materials.network_stamp import CONFIRMABLE
    check = project.network_check() if network else None
    if check is not None and check.state in CONFIRMABLE:
        raise ValueError(f"project {project.name!r}: {os.path.basename(project.network_path())}: {check.text}. If "
                         f"it is the network for the database as it is now, call confirm_network() on the project "
                         f"once, then run again")
    return project.copy()


def run(project: Project, save_dir: str | None = None, **run_options: Any) -> str | None:
    """Optimizes the project. The run's record keeps the project as it was run (Project.from_record
    rebuilds it). Returns the folder the results went to; None when nothing was saved.

    project: the project, a Project
    save_dir: the folder the results go into; None is a new folder in results in the project's folder,
        named after the date and time
    run_options: any of the options below, as keywords
        Default: none
    verbose: True prints each iteration's numbers
    plot_progress: True draws the design in a window as it changes
    plot_bcs: True opens a window with the supports and loads before the optimization starts
    plotter: a PyVista plotter the design is drawn into instead of a new window, or None
    progress: a function called after each iteration with a dict of its numbers (iteration, objective,
        constraints, x_phys, z, z_real, and material: each element's material, its database row), or None
    should_stop: a function the run asks as it goes; when it returns True the run stops and saves the
        design reached so far; or None
    on_network_error: a function called with the warning when the trained network is further from the
        database than the vae setting network_error_warning; when it returns False nothing is
        optimized; or None
    Example: folder = run(p)
    """
    from lsr_mmto.optimize.optimization_loop import _run_topopt
    from lsr_mmto.problem.project_folder import is_run
    project.keep_database_version()                 # the database as this run reads it, kept as a version
    arguments = _run_arguments(_ready_to_run(project), save_dir, **run_options)
    _run_topopt(**arguments)
    project.keep_database_version()                 # with the network the run trained on it, if it trained one
    return arguments["save_dir"] if is_run(arguments["save_dir"]) else None
