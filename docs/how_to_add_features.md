# How to add features

Most additions are one entry in a list. `docs/how_the_code_works.md` says where each part of the
code is, and `docs/code_map.md` leads from what you want to add to the section of this page to follow. The tests listed under each one fail when the entry is incomplete. After any addition, run

    python docs/make_project_file_reference.py
    python -m gui.user_guide_window --write
    python -m pytest tests -q

The first rewrites `docs/project_file_reference.md`, the second the lists in `docs/user_guide.md`
(the settings, quantities, kinds of support and load, selection types and material attributes, as
the user guide shows them); `tests/test_reference.py` and `tests/test_user_guide.py` fail until
they are rewritten. Each section below ends with what else to run.

## A quantity (an objective or a constraint)

`lsr_mmto/physics/template/template_quantities.py` has one quantity of a physics, commented: what it
is given, what it returns, and how its gradient comes from autograd.

A quantity every physics has (mass, cost, a volume fraction) is one entry in `QUANTITIES` in
`lsr_mmto/formulation/quantities.py`. A quantity of one physics is one entry in `QUANTITIES` on that
physics' front page, given as the arguments of a `Quantity` without `physics` (the structural ones
are in `lsr_mmto/physics/structural/structural_quantities.py`). Either way, its function computes it
as torch from the model's state; autograd gives the gradient.

    def _cost(m, opt):
        return torch.sum(m.props["Density"] * m.props["Cost"] * m.x_phys) * m.elem_volume

    Quantity("Cost", False, True, [], ["Density", "Cost"],
             _cost, "sum of density * cost * x_phys * element volume", "Cost",
             "The cost of the part, from each material's density and cost per mass.", "USD"),

- `m` holds `x_phys`, `z` and `props` (every column of the database, decoded), and whatever the
  physics' `analyse` left on it (the structural one leaves `u` and `svm`, the von Mises stress).
- The entry holds its name in the app, its help and the unit of a bound on it; the Formulation
  panel and the reference page read them from there.
- `needs`: the database columns it reads. A database without them gives a clear error.
- `arguments`: what its formula takes, e.g. `MaterialVolumeFraction(2, tau=0.5)` takes a material
  by position and `tau` by name. The formula text, the project's checks and the Formulation panel
  read them from there; a quantity with none is written `Name()`. A material argument may also
  take words in place of a material's row, given as a fourth item: `MassFraction` declares
  `("material_id", "material", None, ("heaviest", "lightest"))`, so `MassFraction(lightest)` is
  allowed and the panel offers *the heaviest* and *the lightest* after the materials. A word is an
  entry of `MATERIAL_WORDS` (its text in the app and the database column that picks the row), and
  the quantity's function gets the word: `material_row` turns it into the row.
- `bound_warning` (optional): `bound_warning(project, bound)` gives a warning about a constraint's
  limit, shown with the project's other warnings before a run, or None. Mass warns about a limit
  below 1 % of the part solid in the lightest material.
- `start_bound` (optional): the limit a new constraint row in the Formulation panel starts at, a
  number or a function of the project (the fractions give 0.4, the failure factor 1, Mass 0.4 times
  the part solid in the lightest material). Without it a new row starts at 0.
- A project names it by its name; the objective and the constraints are kept by that name. A
  quantity of one physics is refused by the project's check in a project of another physics.
- Add it to `QUANTITIES` in `tests/test_ad_quantities.py`, which checks the gradient against
  finite differences. What else to run: both generators above (the reference page and the user
  guide list every quantity).

## A penalty (a term added to the objective)

`lsr_mmto/physics/template/template_module.py` has one penalty of a physics, commented: what it is
given, its weight and its scale.

A penalty every physics has (latent smoothing, the distance penalty) is one entry in `PENALTIES` in
`lsr_mmto/design/latent_penalties.py`. A penalty of one physics is one entry in `PENALTIES` on that
physics' front page, given as the arguments of a `Penalty` without `physics`. Each iteration the loop
adds them in that order, those every physics has first. Its function computes it as torch from the
model's state; autograd gives the gradient.

    Penalty("LatentSmoothing", "Latent smoothing",
            "Discourages neighbouring elements from flickering between materials.",
            lambda m: latent_smoothing(m.x_phys, m.z, m.Hm, m.Hsm), _latent_smoothing_weight, "raw", "Pz_n"),

- `fn(m)`: its value, from `m` as for a quantity. `m.x_phys.detach()` keeps the density out of its
  gradient, as the distance penalty does.
- `weight(state, it)`: its weight this iteration; 0 leaves it out of that iteration. Its schedule
  lives here: latent smoothing gives `z_smoothing_weight` until `z_smoothing_disable_after_iter`,
  then 0; the distance penalty's weight grows each iteration it is on. `state.s` holds the settings,
  `state.iteration` the iteration, `it.Jn` the objective divided by its first value.
- `scale`: stated in every entry, `"raw"` (the weight multiplies the value) or `"first value"` (the
  value is divided by its value at the first iteration it is on). An entry without it is refused
  when the app starts.
- `key`, and `raw_key` if wanted: the history keys of what it adds to the objective and of its value
  before weight and scale. The run saves them; both are 0 at an iteration it is off.
- Its settings are entries of `SETTINGS` (see "A setting"), on the physics' front page for a penalty
  of one physics.

`tests/test_template_physics.py` registers a penalty on the template, checks the gradient against
finite differences and reads its history key; copy those tests for a penalty of your own. A new
penalty every physics has changes every run unless its weight is 0 by default:
`tests/test_regression.py` shows it.

## A selection type

`lsr_mmto/problem/selections.py`. Subclass `Selection`, give it a `TYPE`, a `fields` tuple and
`field_kinds`, how each field is given (a `Field`: a length, a point, a choice, a tick, a file, a
number or a list of numbers; optional or not), and write `_mask` (a True/False per point) or
`evaluate` (indices). Saving and loading find it by `TYPE`; the docstring becomes its line in the
reference page.

    class face(Selection):
        """The plane at the mesh's extreme on one side: xmin, xmax, ymin, ymax, zmin or zmax."""
        TYPE, fields = "face", ("side", "pad")
        field_kinds = {"side": SIDE, "pad": PAD}

The selection form makes each field's box from its `Field`: a length or a point gets the project's
length unit beside it, a choice a list, a tick a checkbox, a file a Browse button. Before adding a
type, look at the existing ones: a tube, for example, is a `cylinder` with an `inner_radius`.

The code line in the GUI picks it up by itself. For the selection form, add it to `FORM` in
`gui/supports_and_loads_panel.py`: the name the form shows, a plain description, and a label for each field
(the form never shows the code names).

Add it to `tests/test_selections.py`, including the round trip through `to_dict`. What else to run:
both generators above (the user guide lists every selection type).

## A kind of support or load

`lsr_mmto/physics/template/template_supports_and_loads.py` has one support and one load, with what
each piece of an entry is, who reads it, and a worked example of how it reaches the solve.

One entry in `SUPPORT_AND_LOAD_KINDS` on the front page of its physics (the structural ones are in
`lsr_mmto/physics/structural/structural_supports_and_loads.py`), given as the arguments of a
`BCKind` (`lsr_mmto/problem/boundary_conditions.py`):

- `role`: `"support"` or `"load"`; supports are applied before loads, each in the order given.
- `apply(bc, nodes, mesh, state, where)`: writes into `state["values"]` (dof to imposed value) or
  `state["force"]`, both of its field. The only physics in the entry.
- `shown`, `help`: its name and what it does, in one plain sentence, as the GUI, the user guide and
  the reference page show them.
- `values`: its values, each a dict of `name`, `kind` (`"dofs"`, `"number"`, `"vector"` or
  `"area"`), `label` (its name in the form), `measures` (what the number is, for its unit; None for
  none) and `doc`. The form, the checks that each value is there and is what its kind says, and the
  setup-script line are made from them.
- `check(bc)` (optional): errors the value kinds do not catch, as a list.
- `field` (optional): the field it acts on; left out by a physics of one field, which is then the
  one.
- `picks` (optional): how one picked on the part in the Pick window becomes this kind.

A project stores it as `{"kind": name, "on": selection, ...values}`, and a script adds it with
`p.add_bc(name, on, **values)`, e.g. `p.add_bc("force", "corner", total=[0, -4, 0])`; a wrong value
is an error there, as in the project's check.

`test_every_kind_round_trips_through_a_setup_script` in `tests/test_project.py` adds every kind,
writes it into a script and runs it; add a build test there for what the kind does. What else to
run: both generators above (the reference page and the user guide list every kind).

## A setting

One entry in `SETTINGS` in `lsr_mmto/problem/settings.py`, or, for a setting of one physics, one
entry in `SETTINGS` on its front page (as the arguments of a `Setting`), plus the line that reads it
as `s.<name>` (in `lsr_mmto/optimize/optimization_loop.py`, or in the physics' own code, which gets
the same `s`):

    Setting("advanced", "gamma_factor", 1.1, "Weight growth per iteration",
            "How much stronger it gets each iteration while it ramps up.", "Distance penalty"),

The entry holds its section (`common`, `advanced` or `vae`), the value a new project starts from
(files without it are completed from it), its label and help in the app, and the group of the
Settings panel it sits in; `none_means`, `integer` and `choices` when it needs them;
`pyto_ad_field` when PyTO-AD's parameters have a field for it, `at_build` when PyTO-AD keeps it
somewhere else (the p-norm exponent). The defaults, the Settings panel, the reference page and the
settings a run gets are made from it. `test_settings_cover_run_topopt` in
`tests/test_run_project.py` fails when the loop reads a name that is not a setting;
`tests/test_extending.py` shows one entry reaching every place. What else to run: both generators
above (the reference page and the user guide list every setting).

## A constraint method

What is done with the constraints during a run. Today there is one, MMA, which hands them to its
subproblem as constraints (g <= 0). A method is one entry of `CONSTRAINT_METHODS` in
`lsr_mmto/optimize/optimization_loop.py`, `{"run": run, "record": record}`:

- `run(state, evaluate, zeta0, lower, upper)` returns `(zeta_optimal, record)`. `evaluate(zeta)` gives,
  for a design, the objective (divided by its first value, with the penalties), the constraint values
  (met when <= 0) and the gradients of both with respect to the optimizer's variables, in the shapes
  (1, 1), (n, 1), (m, 1) and (m, n). `zeta0` is the start design, `lower` and `upper` the bounds of
  every variable. `run` calls `evaluate` as often as it needs and returns the best design it reached
  (shape (n,)) and what the run record should keep of how it ran.
- `record(state)` is that record on its own, used when the run is stopped from the window.
- `state` is the run's `RunState`: its settings as `state.s`, the iteration count, the problem's
  parameters (`state.to_params.Constraints`, `MaxIterations`), the number of variables.
- What `evaluate` does for each design is `ITERATION_STEPS` in the same file, in order; the module's
  header lists them. A method changes nothing there.

Its settings are entries of `SETTINGS` (see "A setting"), read as `state.s.<name>`. With a second
method comes the setting that chooses between them; `CONSTRAINT_METHOD` names the one used today.

What else to run: `tests/test_regression.py` compares the first iterations of every shipped project
with stored values; with MMA chosen they must not change.

## A material attribute

One entry in `ATTRIBUTES` in `lsr_mmto/materials/databases.py`: its name, what it measures (one of
`UNITS`, which gives its units), a one-line description, its name in the app, its allowed range,
and other names it may be written under in a database. An attribute that measures `"ratio"` is
dimensionless; its units cell may be left blank.

    Attribute("Yield_Strength", "stress", "yield strength, the stress at which the material yields",
              "Yield strength", low=0, aliases=("Yield Strength", "Yield")),

A new kind of quantity (a conductivity, say) belongs to the physics that uses it: an entry in
`UNIT_KINDS` on its front page, with each unit's value in SI and its unit in each unit system
(`structural_module.py` gives force and stress this way). Only a kind every problem has (length,
density, mass, cost per mass) is an entry in `UNITS` and `UNIT_SYSTEMS` in
`lsr_mmto/materials/databases.py`.

A database column not in `ATTRIBUTES` is an error, so a new attribute is added here before
it is used. The check, the template and the database editor follow. Add a case to
`tests/test_databases.py`. What else to run: both generators above (the reference page lists the
units, the user guide the attributes).

## A physics

Start from `lsr_mmto/physics/template/`: a small working steady diffusion, commented line by line.
Copy the folder, rename its four files, change the lines marked `FILL`, and add one line to
`MODULES`; the steps are at the top of `template/template_module.py`.

A physics is a folder in `lsr_mmto/physics/`, e.g. `thermoelastic/`, with four files:

| File | What goes in it |
|---|---|
| `thermoelastic_module.py` | the front page: every entry `MODULE_ENTRIES` lists, by name |
| `thermoelastic_physics.py` | the solver (`make_solver`) and one iteration's solve (`analyse`) |
| `thermoelastic_supports_and_loads.py` | its kinds of support and load, and its check that they hold the part |
| `thermoelastic_quantities.py` | its objectives and constraints, what each iteration records, the final check's numbers |

plus one line in `MODULES` in `lsr_mmto/physics/available_physics.py`. The entries are described in
`MODULE_ENTRIES` there; `structural/structural_module.py` is a complete example, and
`tests/test_template_physics.py` registers and runs the template. A project names its physics with
`"physics"` in its file (`p.physics` in code); an unknown name is an error before the run. What else to run: both generators
above, since its settings, quantities and kinds join their lists.

A module's files import the layers below them (`elements/`, `design/`, `pyto_ad_imports.py`), never
the core's lists: its quantities, kinds and settings are given as plain arguments, and the core
makes the entries.

## A material database

An `.xlsx` in `data/databases/`: row 1 the attribute names, row 2 their units, then one row
per material. A project names it by file name without `.xlsx`, or by a path. Make it in the
app (Materials, New), which checks it as it is filled in; the Materials panel also gives the
latent dimension recommended for the database.

An optional sheet `Display` sets the colour (#rrggbb) and the label each material is shown
with; the app's database editor writes it.

To ship a trained VAE with it, put the `.nt` file beside the `.xlsx` and add the settings it
was trained with to `DATABASE_VAE` in `lsr_mmto/materials/databases.py`. A database's network is always
the `.nt` beside it with the same name; a run never trains into the shipped folder, and a run
that replaces a network keeps the old one as `<name>.previous.nt`. A database not listed there
starts from `VAEParams`, with the recommended latent dimension; a project that sets another
one runs with a warning.

## A change to the project file format

Raise `SCHEMA_VERSION` in `lsr_mmto/problem/project.py` and add to `_UPGRADES` the function that
turns the previous version into the new one. Older files then load as before. See
`test_schema_upgrades` in `tests/test_project.py`.
