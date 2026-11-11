# Contributing

How changes to LSR MMTO are made. `docs/how_the_code_works.md` describes the code and
`docs/how_to_add_features.md` where each kind of addition goes. `docs/code_map.md` is the map: the code
as nested trees, and a decision tree that leads from what you want to add to its file and its test.

## Where a change goes

Every change is one of three kinds.

| Kind | What it is | Where |
|---|---|---|
| A module | a new physics, or more of one: its solver, supports and loads, quantities, settings, results | its folder in `lsr_mmto/physics/`, plus one line in `MODULES`; no core file changes |
| A widening of the core | a new option the core does not have yet, e.g. an augmented Lagrangian next to the p-norm, or a field solve that is nonlinear in temperature | the one layer it belongs to |
| A new repository | anything every layer would have to check for | not here |

A new physics starts as a copy of `lsr_mmto/physics/template/`, the blank form; `lsr_mmto/physics/thermal/`
is the same form filled in. A physics keeps to four bounds:

1. The core reaches a physics only through its front page (`<name>_module.py`); nothing outside the
   folder imports its other files.
2. More files inside the folder are fine, when the name says the job.
3. A physics may import and reuse another physics' code.
4. Needing to change a file outside the folder means the core lacks something: propose the change to the
   core as a widening (below), never patch it in from the physics.

`tests/test_physics_folders.py` checks that each physics has the form's five files, puts each front-page
entry where the form does, marks each function (`# ESSENTIAL:`, `# OPTIONAL:`, `# HELPER:`), and that the
first bound holds.

A widening of the core follows four rules:

1. It is a new option, kept in one layer; what runs today runs the same with it.
2. It is made on a branch of its own and merged when it is tested.
3. It rests on a published method, cited in the code where it is used.
4. One at a time: the next starts when the last is merged.

## Before writing anything

- **Check PyTO-AD first.** If PyTO-AD has it, import it through `lsr_mmto/pyto_ad_imports.py`, or
  subclass it where it needs a different material in each element. PyTO-AD is pinned to one commit
  and never edited.
- **One way to do each thing.** A project runs only through `run(p)`. A second way of doing
  something that already has one is removed, not kept beside it.
- **Generic code.** Nothing is special-cased by name (`if name == "Mass"`): what differs between
  entries is declared by the entry, in its list.

## Names

A file's name says what it is for. The test: from the name alone, someone new to the code can say
in one line what is in the file. `structural_supports_and_loads.py` passes; `loop.py` and
`utils.py` do not.

## Checking a change

Run the checks in a Miniforge Prompt, from inside this folder, with the environment active:

    conda activate lsr-stress-mmto-ad
    python -m pytest tests -q -n 4

That is every test, spread over 4 processes at once; one after another, without `-n 4`, they take about
six minutes. Each process needs up to about 2.5 GB of memory: on a computer with less than 16 GB, use
`-n 2`. While working, run only the files of the part you changed:

| What you changed | The tests that cover it |
|---|---|
| `lsr_mmto/problem/` (the project, selections, mesh, supports and loads, settings, the project folder) | `test_project.py`, `test_selections.py`, `test_mesh.py`, `test_project_folder.py` |
| `lsr_mmto/materials/` (databases, the material network) | `test_databases.py` |
| `lsr_mmto/formulation/` (quantities, the formula text) | `test_extending.py`, `test_project.py`, `test_ad_quantities.py` |
| `lsr_mmto/physics/` | `test_thermal_physics.py`, `test_physics_folders.py`, `test_final_check.py`, `test_solid_part_analysis.py` |
| `lsr_mmto/optimize/`, `lsr_mmto/design/`, `lsr_mmto/elements/` (the run itself) | `test_run_project.py`, `test_gradcheck.py`, `test_regression.py`, `test_solid_part_analysis.py` |
| `lsr_mmto/results/` (reading runs back, tables, figures, STL export) | `test_stl_export.py`, `test_project_folder.py`, `test_solid_part_analysis.py` |
| `gui/` | `test_gui.py`, `test_gui_look.py`, `test_user_guide.py` |
| a docstring, or anything under `docs/` | `test_reference.py`, `test_user_guide.py`, `test_code_map.py`, `test_docstring_reads_and_sets.py` |
| `setup_environment.py`, `make_shortcut.py`, `uninstall_lsr_mmto.py` | `test_launcher.py` |
| `Python_walkthrough.ipynb` | `test_python_walkthrough.py` |

For example `python -m pytest tests/test_solid_part_analysis.py -q`. The slow gradient checks can be
left out while working, and the tests that failed run again on their own:

    python -m pytest tests -q -m "not slow"
    python -m pytest tests -q --lf

Run every test once before committing. To see which lines of the code no test runs:

    python -m pytest tests -q --cov=lsr_mmto --cov-report=term-missing

- Every test passes, with no warnings. A failing test you did not touch means the change broke
  something that worked before.
- New or changed code has type hints on every function, written from what its callers actually pass, and a
  short docstring saying what it does and its limits. A function that changes an object it is given (the
  run state, an iteration, the model) says what it reads and what it sets, in one line of code names:
  `Reads it.x. Sets it.y.` (`tests/test_docstring_reads_and_sets.py` checks that line against the code). A
  class lists its fields and their types at the top. Names needed only for hints are imported under
  `if TYPE_CHECKING:`.
- A test of something every physics has (a run goes through, its gradients, a run saved and read back,
  the final check) runs for every physics: `@pytest.mark.parametrize` over `available_physics.MODULES`
  and thermal (`registered` in `tests/test_thermal_physics.py`). A new physics is then tested by
  the tests already there, and adds tests only for what is its own: its solver against a case with a known
  answer, its quantities, its supports and loads. A test written for one physics only becomes such a loop
  the next time it is changed.
- A change that moves code or renames things changes no number: the first iterations of the shipped
  projects give the same objective and constraints, to 1 part in a million, before and after
  (`tests/test_regression.py` checks it). Later iterations drift by tiny amounts from one run to the
  next even with no change, because the solver adds numbers in a different order on several threads.
  A change meant to change results (a new formulation, a new PyTO-AD pin) fails that test: store the
  new values on purpose with `python tests/test_regression.py --update`, and say why in the commit
  message.
- Text the GUI shows is written for someone who has never seen the code: plain words, the labels
  the form itself shows, and the details behind an (i).
- After a change to a list (settings, quantities, supports and loads, selection types, material
  attributes) or to a docstring the reference reads, run `python docs/make_reference.py` and
  `python -m gui.user_guide_window --write`; the reference page and the user guide's lists are made
  from the code.
- The versions in requirements.txt are the ones every test passes with. A newer version of a package
  is a change like any other: change its pin, run every test, and commit both.

## What is not committed

The setup and the runs make these on your computer; `.gitignore` leaves them out, and they are never
committed:

- `lsr_mmto.egg-info/`, made by the setup (`pip install -e .`);
- `__pycache__/` folders, made by Python;
- the `results/` folder of each project, with its runs and analyses;
- `python_walkthrough_project/`, the project `Python_walkthrough.ipynb` makes.

## Commit messages

Plain sentences saying what changed and why, in the present tense, ending with "No number
changes." when that is so.
