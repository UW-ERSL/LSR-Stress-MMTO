# Extending the code

Most additions are one entry in a registry. The tests listed under each one fail when the
entry is incomplete. After any addition, run

    python docs/make_project_file_reference.py
    python -m pytest tests -q

The first rewrites `docs/project_file_reference.md`; `tests/test_reference.py` fails until it is
rewritten.

## A quantity (an objective or a constraint)

One entry in `QUANTITIES` in `lsr_mmto/physics/structural.py`, with the function that computes it
as torch from the model's state; autograd gives the gradient.

    def _cost(m, opt):
        return torch.sum(m.props["Density"] * m.props["Cost"] * m.x_phys) * m.elem_volume

    Quantity("Cost", None, ("COST", None), [STRUCTURAL], ["Density", "Cost"],
             _cost, "sum of density * cost * x_phys * element volume", "Cost",
             "The cost of the part, from each material's density and cost per mass."),

- The entry holds its name in the app and its help; the Formulation panel and the reference
  page read them from there.
- `needs`: the database columns it reads. A database without them gives a clear error.
- The objective and constraint forms name a member of PyTO-AD's `TO_QOI`. A quantity with no
  `TO_QOI` counterpart also needs a change in `to_params_form` in `lsr_mmto/problem/project.py`.
- Add it to `QUANTITIES` in `tests/test_ad_quantities.py`, which checks the gradient against
  finite differences, and the unit of its bound to `bound_unit` in `gui/panels.py`.

## A selection type

`lsr_mmto/problem/selections.py`. Subclass `Selection`, give it a `TYPE` and a `fields` tuple, and
write `_mask` (a True/False per point) or `evaluate` (indices). Saving and loading find it by
`TYPE`; the docstring becomes its line in the reference page.

    class face(Selection):
        """The plane at the mesh's extreme on one side: xmin, xmax, ymin, ymax, zmin or zmax."""
        TYPE, fields = "face", ("side", "pad")

The code line in the GUI picks it up by itself. For the region form, add it to `FORM` in
`gui/supports.py`: the name the form shows, a plain description, and a label for each field
(the form never shows the code names).

Add it to `tests/test_selections.py`, including the round trip through `to_dict`.

## A kind of boundary condition

`BC_KINDS` in `lsr_mmto/problem/project.py`. A `BCKind` has a description, a role (`"support"` or
`"load"`), `check(bc)` returning a list of errors, `apply(bc, nodes, mesh, state, where)`
writing into `state["values"]` (dof to imposed value) or `state["force"]`, `script(bc, on)`
returning the `Project` call that makes it, and `fields`, its values as
`(name, kind, measures, doc)`, from which the GUI makes its form; give it a name, a plain
description and labels in `BC_FORM`, `gui/supports.py`. Add an `add_...` method to `Project`
for it.

Add a build test to `tests/test_project.py`; `test_script_round_trip` checks `script`.

## A setting

One entry in `SETTINGS` in `lsr_mmto/problem/settings.py`, plus the line in
`lsr_mmto/optimize/run.py` that reads it as `s.<name>`:

    Setting("advanced", "gamma_factor", 1.1, "Weight growth per iteration",
            "How much stronger it gets each iteration while it ramps up.", "Distance penalty"),

The entry holds its section (`common`, `advanced` or `vae`), the value a new project starts from
(files without it are completed from it), its label and help in the app, and the group of the
Settings panel it sits in; `none_means`, `integer` and `choices` when it needs them. The
defaults, the Settings panel, the reference page and the settings a run gets are made from it.
`test_settings_cover_run_topopt` in `tests/test_run_project.py` fails when `run.py` reads a name
that is not a setting; `tests/test_extending.py` shows one entry reaching every place.

## A material attribute

One entry in `ATTRIBUTES` in `lsr_mmto/materials/databases.py`: its name, what it measures (one of
`UNITS`, which gives its units), a one-line description, its name in the app, its allowed range,
and other names it may be written under in a database. An attribute that measures `"ratio"` is
dimensionless; its units cell may be left blank.

    Attribute("Yield_Strength", "stress", "yield strength, the stress at which the material yields",
              "Yield strength", low=0, aliases=("Yield Strength", "Yield")),

A new kind of quantity (a conductivity, say) is a new entry in `UNITS`, with each unit's value
in SI, and one in each of `UNIT_SYSTEMS`.

A database column not in `ATTRIBUTES` is an error, so a new attribute is added here before
it is used. The check, the template and the database editor follow. Add a case to
`tests/test_databases.py`.

## A physics

A file in `lsr_mmto/physics/` offering `Solver`, `Model`, `decode`, `distance_penalty` and
`latent_smoothing` (as `structural.py` does), plus one entry in `PHYSICS` in
`lsr_mmto/physics/__init__.py`: its name, its file, and the database columns it reads. A project
names it with `"physics"` in its file (`p.physics` in code); the optimization builds the solver and
the model of that physics, and an unknown name is an error before the run. Its quantities are
entries of `QUANTITIES` that name it in their `physics`.

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
