# Contributing

How changes to LSR MMTO are made. `docs/how_the_code_works.md` describes the code and
`docs/how_to_add_features.md` where each kind of addition goes. `docs/code_map.md` is the map: the code
as nested trees, and a decision tree that leads from what you want to add to its file and its test.

## Where a change goes

Every change belongs to one of three categories.

| Kind | What it is | Where |
|---|---|---|
| A module | a new physics, or more of one: its solver, supports and loads, quantities, settings, results | its folder in `lsr_mmto/physics/`, plus a single line in `MODULES`; no core file changes |
| A widening of the core | a feature that the core does not currently support, e.g. an augmented Lagrangian next to the p-norm, or a new filter on the densities | the specific layer responsible for it |
| A new repository | anything every layer would have to check for | not here |

To add new physics, start by copying `lsr_mmto/physics/template/`, the blank form; `lsr_mmto/physics/thermal/`
is the same form filled in. Each physics module must follow four rules:

1. The core reaches a physics only through its front page (`<name>_module.py`); nothing outside the
   folder imports its other files.
2. More files inside the folder are fine, when the name says the job.
3. A physics may import and reuse another physics' code.
4. Needing to change a file outside the folder means the core lacks something: propose the change to the
   core as a widening (below), never patch it in from the physics.

`tests/code_rules/test_physics_folders.py` checks that each physics has the five files required by the template, puts each front-page
entry where the form does, marks each function (`# ESSENTIAL:`, `# OPTIONAL:`, `# HELPER:`), and that the
first rule holds.

When extending the core, follow these four rules:

1. It is a new option, kept in one layer; what runs today runs the same with it.
2. It is made on a branch of its own and merged when it is tested.
3. It must be based on a published method, cited in the code where it is used.
4. One at a time: the next starts when the last is merged.

## Before writing anything

- **Check PyTO-AD first.** If PyTO-AD already provides what you need, import it through `lsr_mmto/pyto_ad_imports.py`, or
  subclass it where that code needs a different material in each element. PyTO-AD is pinned to one commit
  and never edited. What LSR MMTO takes from it is LSR MMTO's own code all the same: a bug found there is
  fixed here, by subclassing or wrapping it, never left as PyTO-AD's.
- **One way to do each thing.** A project runs only through `run(p)`. A second way of doing
  something that already has one is removed, not kept beside it.
- **Generic code.** Nothing is special-cased by name (`if name == "Mass"`): what differs between
  entries is declared by the entry, in its list. The GUI names no physics: what a panel shows comes from
  the physics' entries (`tests/gui/test_new_physics_in_the_gui.py` opens each panel that shows a physics' pieces with a
  made-up one).

## Names

A filename should tell you what the file does. The test: from the name alone, a person who has never seen the code can explain
in one line what is in the file. `structural_supports_and_loads.py` passes; `loop.py` and
`utils.py` do not.

## Writing

These hold for everything committed: code, comments, docstrings, docs, tests and commit messages.

- No em dashes. Use a comma, a colon, or two sentences.
- No mention of any tool that helped write a change.
- The GUI shows only what a user needs at that moment; anything longer goes behind an (i). Its words are
  plain, use the labels the form itself shows, and never name PyTO-AD or a name from the code.

## Checking a change

Run these checks in a Miniforge Prompt, while you are in this folder, with the environment active:

    conda activate lsr-stress-mmto-ad
    python -m pytest tests -q -n 4

This runs the entire test suite, spread over 4 processes in parallel; one at a time, without `-n 4`, they take about
six minutes. Each process can use up to about 2.5 GB of memory: on a computer with less than 16 GB, use
`-n 2`. The tests are in folders by what they check (physics, gradients, problem, materials, runs, gui,
docs, code_rules); `docs/testing_the_codebase.md` lists every test, folder by folder, with what it checks.
It is generated from the tests' docstrings: after adding, renaming or moving a test, run
`python docs/make_testing_doc.py` (`tests/docs/test_testing_doc.py` fails until you do). A new folder of
tests needs its line in `FOLDERS` there. While developing, run only the tests for the part you changed:

| What you changed | The tests that cover it |
|---|---|
| `lsr_mmto/problem/` (the project, selections, mesh, supports and loads, settings, the project folder) | `problem/test_project.py`, `problem/test_selections.py`, `problem/test_mesh.py`, `problem/test_project_folder.py` |
| `lsr_mmto/materials/` (databases, the material network) | `materials/test_databases.py` |
| `lsr_mmto/formulation/` (quantities, the formula text) | `code_rules/test_extending.py`, `problem/test_project.py`, `gradients/test_ad_quantities.py` |
| `lsr_mmto/physics/` | `physics/test_thermal_physics.py`, `physics/test_thermoelastic_physics.py`, `code_rules/test_physics_folders.py`, `gui/test_thermal_walk_through.py`, `runs/test_final_check.py`, `runs/test_solid_part_analysis.py` |
| `lsr_mmto/optimize/`, `lsr_mmto/design/`, `lsr_mmto/elements/` (what happens during an optimization run) | `runs/test_run_project.py`, `gradients/test_gradcheck.py`, `runs/test_regression.py`, `runs/test_solid_part_analysis.py` |
| `lsr_mmto/results/` (reading runs back, tables, figures, STL export) | `runs/test_stl_export.py`, `problem/test_project_folder.py`, `runs/test_solid_part_analysis.py` |
| `gui/` | `gui/test_gui.py`, `gui/test_gui_look.py`, `docs/test_user_guide.py`, `gui/test_thermal_walk_through.py` |
| a docstring, or anything under `docs/` | `docs/test_reference.py`, `docs/test_user_guide.py`, `docs/test_code_map.py`, `docs/test_testing_doc.py`, `code_rules/test_docstring_reads_and_sets.py` |
| a test | `docs/test_testing_doc.py` |
| `setup_environment.py`, `make_shortcut.py`, `uninstall_lsr_mmto.py` | `gui/test_launcher.py` |
| `Python_walkthrough.ipynb` | `docs/test_python_walkthrough.py` |

For example `python -m pytest tests/runs/test_solid_part_analysis.py -q`. The slow gradient checks can be
left out while working, and the tests that failed run again on their own:

    python -m pytest tests -q -m "not slow"
    python -m pytest tests -q --lf

Run every test once before committing. To find the lines of code that are not covered by tests:

    python -m pytest tests -q --cov=lsr_mmto --cov-report=term-missing

- Every test passes, with no warnings. If a test you did not change now fails, the change broke
  something that worked before.
- Every function in new or changed code has type hints, written from what its callers actually pass, and a
  short docstring saying what it does and its limits. A function that changes an object it is given (the
  run state, an iteration, the model) says what it reads and what it sets, in one line of code names:
  `Reads it.x. Sets it.y.` (`tests/code_rules/test_docstring_reads_and_sets.py` checks that line against the code). A
  class lists its fields and their types at the top. Names needed only for hints are imported under
  `if TYPE_CHECKING:`.
- A test of something all physics modules share (a run goes through, its gradients, a run saved and read back,
  the final check) runs for every physics: `@pytest.mark.parametrize` over `available_physics.MODULES`.
  The existing tests will then also test any newly added physics; the new physics adds tests only for what is its own: its
  solver against a case with a known answer, its quantities, its supports and loads. A test written for
  one physics only becomes such a loop the next time it is changed.
- Moving or renaming code must not change any numerical result: the first iterations of the shipped
  projects give the same objective and constraints, to 1 part in a million, before and after
  (`tests/runs/test_regression.py` checks it). Results from later iterations can differ slightly from one run to the
  next even with no change, because the solver adds numbers in a different order on several threads.
  A change meant to change results (a new formulation, a new PyTO-AD pin) fails that test: store the
  new values on purpose with `python tests/runs/test_regression.py --update`, and say why in the commit
  message.
- A name a test makes up (a unit kind, a property, a setting, a penalty) starts with `test_`, so it never
  clashes with a real one a physics adds later: a test's made-up unit kind `heat` clashed with thermal's
  own once thermal was in the list of physics.
- Registering a physics (its line in `MODULES`) adds its settings, quantities, kinds of support and load
  and material properties to every list the app makes: regenerate the reference and the user guide's lists
  (the two commands in `docs/how_to_add_features.md`), and expect a test that types such a list out by hand
  to fail. Such a test should work its list out from the physics instead.
- Text the GUI shows is written for someone who has never seen the code: plain words, the labels
  the form itself shows, and the details behind an (i) (see Writing).
- After a change to a list (settings, quantities, supports and loads, selection types, material
  attributes) or to a docstring the reference reads, run `python docs/make_reference.py` and
  `python -m gui.user_guide_window --write`; the reference page and the user guide's lists are made
  from the code.
- The versions in requirements.txt are the ones every test passes with. A newer version of a package
  is a change like any other: change its pin, run every test, and commit both.

## How a change gets in

1. Make the change on a branch of its own.
2. Run every test on your computer (`python -m pytest tests -q -n 4`). Every one passes, with no warnings.
3. Only then merge it and push. A change is never pushed with a test failing.
4. Delete the branch once it is merged, so no stray branches are left.

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
