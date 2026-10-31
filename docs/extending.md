# Extending the code

Most additions are one entry in a registry. The tests listed under each one fail when the
entry is incomplete. After any addition, run

    python utils/reference.py
    python -m pytest tests -q

The first rewrites `docs/project_reference.md`; `tests/test_reference.py` fails until it is
rewritten.

## A quantity (an objective or a constraint)

`QUANTITIES` in `1-PureStructural/AutodiffPureStructural.py`. Write the value as a torch
function of the model's state and add an entry; autograd gives the gradient.

    def _cost(m, opt):
        return torch.sum(m.props["Density"] * m.props["Cost"] * m.x_phys) * m.elem_volume

    Quantity("Cost", None, ("COST", None), [STRUCTURAL], ["Density", "Cost"],
             _cost, "sum of density * cost * x_phys * element volume"),

- `needs`: the database columns it reads. A database without them gives a clear error.
- The objective and constraint forms name a member of PyTO-AD's `TO_QOI`. A quantity with no
  `TO_QOI` counterpart also needs a change in `to_params_form` in `0-Common/project.py`.
- Add it to `QUANTITIES` in `tests/test_ad_quantities.py`, which checks the gradient against
  finite differences.

## A selection type

`0-Common/selections.py`. Subclass `Selection`, give it a `TYPE` and a `fields` tuple, and
write `_mask` (a True/False per point) or `evaluate` (indices). Saving and loading find it by
`TYPE`; the docstring becomes its line in the reference page.

    class face(Selection):
        """The plane at the mesh's extreme on one side: xmin, xmax, ymin, ymax, zmin or zmax."""
        TYPE, fields = "face", ("side", "pad")

Add it to `tests/test_selections.py`, including the round trip through `to_dict`.

## A kind of boundary condition

`BC_KINDS` in `0-Common/project.py`. A `BCKind` has a description, a role (`"support"` or
`"load"`), `check(bc)` returning a list of errors, `apply(bc, nodes, mesh, state, where)`
writing into `state["values"]` (dof to imposed value) or `state["force"]`, and `script(bc, on)`
returning the `Project` call that makes it. Add an `add_...` method to `Project` for it.

Add a build test to `tests/test_project.py`; `test_script_round_trip` checks `script`.

## A setting

A new argument of `run_topopt` in `1-PureStructural/MainPureStructural.py` is also added to
`DEFAULT_SETTINGS` (in `common` or `advanced`) and `SETTING_DOCS` in `0-Common/project.py`.
Its default there is the value a new project starts from; files without it are completed
from it.

`test_settings_cover_run_topopt` in `tests/test_run_project.py` fails when an argument is
missing; `utils/reference.py` stops when a setting has no line in `SETTING_DOCS`.

## A material database

An `.xlsx` in `1-PureStructural/MaterialDataPureStructural/`: row 1 the attribute names,
row 2 their units, then one row per material. A project names it by file name without
`.xlsx`, or by a path.

To ship a trained VAE with it, put the `.nt` file beside the `.xlsx` and add the settings it
was trained with to `DATABASE_VAE` in `0-Common/databases.py`. A database not listed there
starts from `VAEParams`.

## A change to the project file format

Raise `SCHEMA_VERSION` in `0-Common/project.py` and add to `_UPGRADES` the function that
turns the previous version into the new one. Older files then load as before. See
`test_schema_upgrades` in `tests/test_project.py`.
