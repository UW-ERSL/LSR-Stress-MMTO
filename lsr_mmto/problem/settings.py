"""
settings.py
===========
Every setting of a project, each in one entry of SETTINGS: its section, its name, the value a new
project starts from, the label the app shows, its one help text (the app's (i) and the settings
page), and the group of the Settings panel it sits in. Everything else is made from this list:
DEFAULT_SETTINGS, the Settings panel, docs/reference.md, and the settings object s the
optimization reads (s.gamma_factor).

A new setting is one Setting here (or, for a setting of one physics, one entry of SETTINGS on its
module's front page), plus the line in optimize/optimization_loop.py that reads s.<name>. A setting
PyTO-AD's TOParams already has a field for names it (pyto_ad_field): build() copies it there, and
the run reads it from there (to_params.MaxIterations), so the two never disagree.

Sections: "mesh" is Project.n_dof and Project.single_layer; "common", "advanced" and "vae" are
project.settings[section]. A VAE setting whose default is FROM_DATABASE starts from the database's
known values (databases.DATABASE_VAE), or VAEParams for another database.
"""
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable

from lsr_mmto.physics.available_physics import MODULES

if TYPE_CHECKING:
    from lsr_mmto.problem.project import Project


class Z0InitMethod(Enum):
    LIGHTEST = "lightest"
    HEAVIEST = "heaviest"
    ORIGIN = "origin"
    UNIFORM = "uniform"


class BetaScheduleScheme(Enum):
    PAPER = "paper"
    PAPER_BM1 = "paper_bm1"


FROM_DATABASE = "from the database"
OPTIMIZATION, VAE = "Optimization", "VAE"                 # the Settings panel's two tabs


class Setting:
    """One setting. none_means: what an empty value (None) means, for a setting that may be empty;
    integer: a whole number; choices: the Enum whose member names are its only values; pyto_ad_field:
    the field of the optimization parameters (TOParams) that holds it during a run; at_build(value):
    for a setting PyTO-AD keeps elsewhere, puts it there when the project is built. physics: the
    physics it belongs to (set for the settings on a module's front page), None for a setting every
    project has; the Settings panel shows a project only its own physics' settings."""
    __slots__ = ("section", "key", "default", "label", "help", "group", "none_means", "integer", "choices",
                 "pyto_ad_field", "at_build", "physics")

    section: str                         # "common", "advanced", "vae" or "mesh"
    key: str
    default: float | int | bool | str | None       # or FROM_DATABASE
    label: str
    help: str
    group: str
    none_means: str | None
    integer: bool
    choices: type[Enum] | None
    pyto_ad_field: str | None
    at_build: Callable[[Any], None] | None
    physics: str | None

    def __init__(self, section: str, key: str, default: float | int | bool | str | None, label: str, help: str,
                 group: str, none_means: str | None = None, integer: bool = False,
                 choices: type[Enum] | None = None,
                 pyto_ad_field: str | None = None, at_build: Callable[[Any], None] | None = None,
                 physics: str | None = None) -> None:
        self.section, self.key, self.default, self.label, self.help = section, key, default, label, help
        self.group, self.none_means, self.integer, self.choices = group, none_means, integer, choices
        self.pyto_ad_field, self.at_build, self.physics = pyto_ad_field, at_build, physics

    def applies_to(self, physics: str) -> bool:
        """True for a setting of that physics, or one every project has."""
        return self.physics is None or self.physics == physics

    @property
    def tab(self) -> str:
        """The Settings panel's tab it is on: VAE for a setting of the vae section, else OPTIMIZATION."""
        return VAE if self.section == "vae" else OPTIMIZATION


_PLATEAU = "Start the ramp when the objective plateaus"

# in the order the Settings panel shows them, group by group
SETTINGS = [
    # ---- Main
    Setting("mesh", "n_dof", 30000, "Mesh dofs",
            "How fine the mesh is: the number of unknowns it is built for, counting each unknown at each "
            "node (three per node for a structural problem: the movement in x, y and z). More is finer, and "
            "slower.",
            "Main", integer=True, pyto_ad_field="nDOFDesired"),
    Setting("mesh", "single_layer", False, "Single element layer (2D)",
            "One element through the thickness, for a flat (2D) part.", "Main", pyto_ad_field="ExtrudeZ"),
    Setting("common", "maxIterations", 300, "Max iterations",
            "The optimization stops after this many iterations.", "Main", integer=True,
            pyto_ad_field="MaxIterations"),
    Setting("common", "relative_filter_radius", 2.5, "Density filter radius",
            "The smallest feature the design may have, in element sizes. Larger gives smoother, simpler designs.",
            "Main", pyto_ad_field="RelativeFilterRadius"),
    Setting("common", "material_filter_radius", None, "Latent smoothing radius",
            "How far material choices are smoothed between neighbouring elements, in element sizes. Ticked: "
            "the same as the density filter radius.", "Main", none_means="same as density filter",
            pyto_ad_field="MaterialFilterRadius"),
    Setting("common", "beta_schedule_scheme", "PAPER", "Projection sharpness schedule",
            "How fast the design is pushed to clear solid and empty: the sharpness starts at 1 and grows 1.5 "
            "times every 5 iterations after iteration 50, up to 32 (PAPER) or 16 (PAPER_BM1).", "Main",
            choices=BetaScheduleScheme),
    # ---- Run length
    Setting("advanced", "timeLimit", 36000, "Time limit [s]",
            "The optimization stops after this many seconds.", "Run length", integer=True),
    Setting("advanced", "rel_conv_tol", 1e-7, "Convergence tolerance",
            "The optimization stops early once the objective changes by less than this, relatively, between "
            "iterations, and every constraint is met to within this.", "Run length"),
    # ---- Projection
    Setting("advanced", "use_heaviside_projection", True, "Heaviside projection",
            "Pushes each element towards fully solid or fully empty.", "Projection",
            pyto_ad_field="HeavisideProjection"),
    Setting("advanced", "eta_proj", 0.5, "Projection threshold",
            "The density at which that push splits solid from empty.", "Projection"),
    # ---- Distance penalty
    Setting("advanced", "use_penalization", True, "Distance penalty",
            "Pushes each element's material towards one of the real materials in the database.",
            "Distance penalty"),
    Setting("advanced", "gamma_init", 1e-3, "Starting weight", "How strong that push is at the start.",
            "Distance penalty"),
    Setting("advanced", "gamma_max", 10.0, "Largest weight", "How strong it can get.", "Distance penalty"),
    Setting("advanced", "gamma_factor", 1.1, "Weight growth per iteration",
            "How much stronger it gets each iteration while it ramps up.", "Distance penalty"),
    Setting("advanced", "penalty_start_iter", 0, "First iteration",
            f"The iteration it starts at, when {_PLATEAU} is off.", "Distance penalty", integer=True),
    Setting("advanced", "distance_penalty_disable_after_iter", 250, "Off after iteration",
            f"The iteration it stops at, when {_PLATEAU} is off.", "Distance penalty", integer=True),
    Setting("advanced", "use_dynamic_gamma_scheduling", True, _PLATEAU,
            "Start the push when the design stops improving, instead of at a set iteration.", "Distance penalty"),
    Setting("advanced", "gamma_plateau_window", 10, "Plateau window",
            "The objective has plateaued when it changed by less than the plateau tolerance over this many "
            "iterations.", "Distance penalty", integer=True),
    Setting("advanced", "gamma_plateau_tolerance", 1e-3, "Plateau tolerance",
            "How little the objective may change over the plateau window, relatively, to count as plateaued.",
            "Distance penalty"),
    Setting("advanced", "gamma_plateau_earliest_iter", 20, "Earliest start",
            "The push never starts before this iteration.", "Distance penalty", integer=True),
    Setting("advanced", "gamma_ramp_end_margin", 30, "Iterations left at full weight",
            "If the objective never plateaus, the push starts in time to reach its largest weight this many "
            "iterations before Max iterations.", "Distance penalty", integer=True),
    # ---- Latent smoothing
    Setting("advanced", "use_z_smoothing", True, "Latent smoothing",
            "Discourages neighbouring elements from flickering between materials.", "Latent smoothing"),
    Setting("advanced", "z_smoothing_weight", 1e-4, "Weight", "How strong that smoothing is.", "Latent smoothing"),
    Setting("advanced", "z_smoothing_disable_after_iter", 250, "Off after iteration",
            "The iteration it stops at.", "Latent smoothing", integer=True),
    # ---- Material model
    Setting("advanced", "use_continuation", False, "SIMP continuation",
            "Raises the SIMP penalty by 0.25 every 10 iterations during the run.", "Material model"),
    # ---- Start and reported design
    Setting("advanced", "z0_init_method", "ORIGIN", "Starting latent point",
            "The material every element starts with: ORIGIN (the middle of the material map), the LIGHTEST or "
            "HEAVIEST material, or UNIFORM (random).", "Start and reported design", choices=Z0InitMethod),
    Setting("advanced", "snap_to_real_material", False, "Snap to real materials",
            "In the final design, give each element the nearest real material.", "Start and reported design"),
    Setting("advanced", "binarize_topology", True, "Binarize the topology",
            "In the final design, make each element fully solid or empty, keeping the volume.",
            "Start and reported design"),
    # ---- VAE
    Setting("vae", "use_pretrained_vae", True, "Use the trained network",
            "Use the network already trained for this database. Unticked, the next run trains a new one.", "VAE"),
    Setting("vae", "saveNet", None, "VAE file",
            "The trained network: always the .nt file next to the database, with the same name.", "VAE",
            none_means="next to the database"),
    Setting("vae", "network_error_warning", 5.0, "Warn above network error [%]",
            "Before a run, you are warned if the network reproduces any material property less closely than "
            "this, in % of the property's range in the database. You can still run.", "VAE"),
    Setting("vae", "latentDim", FROM_DATABASE, "Latent dimension",
            "How many numbers describe one material in the material map.", "VAE", integer=True),
    Setting("vae", "learningRate", FROM_DATABASE, "Learning rate", "The step size of the network's training.",
            "VAE"),
    Setting("vae", "vae_hiddenDim", FROM_DATABASE, "Hidden layer width", "The size of the network's inner layers.",
            "VAE", integer=True),
    Setting("vae", "maxAttributeErrorPercent", FROM_DATABASE, "Stop at attribute error [%]",
            "Training stops once the network reproduces every material property within this percent.", "VAE"),
    Setting("vae", "klFactor", FROM_DATABASE, "KL weight",
            "How strongly training keeps the material map compact and smooth.", "VAE"),
    Setting("vae", "numEpochs", FROM_DATABASE, "Max epochs", "Training stops after this many passes at the latest.",
            "VAE", integer=True),
]
SETTINGS += [Setting(physics=m.NAME, **args) for m in MODULES for args in m.SETTINGS]   # each physics' own, after the rest
# the views made from SETTINGS, kept in step by _make_views
BY_KEY, DEFAULT_SETTINGS, MESH_DEFAULTS = {}, {}, {}
VAE_PARAMS = RUN_KEYS = ()


def _make_views() -> None:
    """BY_KEY, DEFAULT_SETTINGS, MESH_DEFAULTS, VAE_PARAMS and RUN_KEYS from SETTINGS. The dicts are
    filled in place, so a module that imported them sees a setting added later (as the tests do)."""
    global VAE_PARAMS, RUN_KEYS
    BY_KEY.clear()
    BY_KEY.update((e.key, e) for e in SETTINGS)
    for section in ("common", "advanced", "vae"):
        DEFAULT_SETTINGS.setdefault(section, {}).clear()
        DEFAULT_SETTINGS[section].update((e.key, e.default) for e in SETTINGS
                                         if e.section == section and e.default is not FROM_DATABASE)
    MESH_DEFAULTS.clear()
    MESH_DEFAULTS.update((e.key, e.default) for e in SETTINGS if e.section == "mesh")
    VAE_PARAMS = tuple(e.key for e in SETTINGS if e.default is FROM_DATABASE)
    # what the optimization reads as s.<key>: the stored settings, without the mesh (Project.n_dof is the
    # mesh built) and the VAE's own settings (they make the network, through VAEParams)
    RUN_KEYS = tuple(e.key for e in SETTINGS if e.section != "mesh" and e.default is not FROM_DATABASE)


_make_views()


class RunSettings:
    """The settings one run gets, read as s.<key>; an unknown name is an error, not None. Its fields are the
    names in RUN_KEYS, which _make_views makes from SETTINGS, so they are not declared here and a type checker
    cannot see them. From run_settings, a setting with fixed choices holds the Enum member, not its name."""

    def __init__(self, values: dict[str, Any]) -> None:
        missing = [k for k in RUN_KEYS if k not in values]
        unknown = [k for k in values if k not in RUN_KEYS]
        if missing or unknown:
            raise ValueError(f"run settings: missing {missing}, unknown {unknown}")
        self.__dict__.update(values)

    def as_dict(self) -> dict[str, Any]:
        """{key: value} of every RUN_KEYS setting."""
        return {k: self.__dict__[k] for k in RUN_KEYS}

    def __repr__(self) -> str:
        return f"RunSettings({self.as_dict()!r})"


def pyto_ad_value(setting: Setting, value: float | int | bool | str | None) -> float | int | bool | str | None:
    """A setting's value as its TOParams field holds it: a whole number, a float, or as it is."""
    if value is None or isinstance(value, bool):
        return value
    if setting.integer:
        return int(value)
    return float(value) if isinstance(value, (int, float)) else value


def run_settings(project: "Project") -> RunSettings:
    """The RunSettings of a project whose settings are complete: each stored value, and for a setting
    with fixed choices, the Enum member it names."""
    values = {}
    for k in RUN_KEYS:
        s = BY_KEY[k]
        v = project.settings[s.section][k]
        values[k] = s.choices[v] if s.choices is not None else v
    return RunSettings(values)
