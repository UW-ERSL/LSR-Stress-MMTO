# The map of the code

Two parts. The first is the code as nested trees: each part says what it is and what slots into it.
The second is a decision tree for a researcher with something to add: answer its questions from the
top, and the leaf you reach names the file, the test to copy and the section of
[how_to_add_features.md](how_to_add_features.md) to follow. The rules for any change are in
[CONTRIBUTING.md](../CONTRIBUTING.md).

## 1. What is where

```
LSR MMTO
├── PyTO-AD                       pinned to one commit, never edited
│     the hexahedral mesher, the finite element solvers, the sparse solve, SIMP, MMA, the GUI it all
│     started from. Everything taken from it comes through one file per side:
│     lsr_mmto/pyto_ad_imports.py and gui/pyto_ad_imports.py
│
├── the core (lsr_mmto/)          no physics in it: it asks the physics for what it needs
│   ├── problem/                  what the user sets up
│   │   ├── project.py            the project: load, save, check, build, run(p)
│   │   ├── settings.py           slot: a setting every physics has (one entry of SETTINGS)
│   │   ├── selections.py         slot: a selection type (one class)
│   │   ├── boundary_conditions.py   the kinds of support and load, made from every physics' kinds
│   │   ├── mesh.py               the voxel mesh, its size from the physics' unknowns, its surface faces
│   │   └── project_folder.py     a project's folder: its part, materials and results
│   ├── materials/                material databases and the material network
│   │   ├── databases.py          slot: a material attribute (ATTRIBUTES), a kind of quantity every
│   │   │                         problem has (UNITS); the database check
│   │   ├── encoder.py            reads a database, trains or loads the network
│   │   ├── networks.py           the network
│   │   └── pca.py                principal component analysis: the database's effective dimension,
│   │                             the 2D latent views
│   ├── elements/hex_element.py   the hexahedral element: strain matrix, Lame parameters
│   ├── design/                   how the optimizer's numbers become what the physics uses
│   │   ├── density_filter_and_projection.py   densities to physical densities
│   │   ├── latent_to_properties.py            each element's latent point to its material properties
│   │   └── latent_penalties.py                slot: a penalty every physics has (PENALTIES):
│   │                                          latent smoothing, the distance penalty
│   ├── formulation/              what is minimized and limited
│   │   ├── quantities.py         slot: a quantity for every physics (Mass, Cost, the volume fractions)
│   │   └── formulation_text.py   reads the objective and constraints as written in a project
│   ├── optimize/                 one run
│   │   ├── optimization_loop.py  setup, then optimize: each design goes through the steps of
│   │   │                         ITERATION_STEPS (penalties, keep solid, continuation, record);
│   │   │                         slot: a constraint method (CONSTRAINT_METHODS, MMA today)
│   │   ├── design_model.py       one design: filter, projection, properties, then the physics' analysis
│   │   ├── history.py            what each iteration records and prints
│   │   ├── final_design_analysis.py   the final design analysed, the results kept
│   │   ├── solid_part_analysis.py     Analyse: the solid part solved once in one material, no optimizer
│   │   ├── save_results.py       a finished run on disk
│   │   ├── gradient_check.py     a run's gradient against finite differences
│   │   ├── filters.py            the filter matrices
│   │   └── mmto_to_params.py     the parameters the loop reads
│   └── results/                  a finished run read back: tables, plots, 3D, STL export, comparison
│
├── the physics (lsr_mmto/physics/)   one folder each; available_physics.py lists them (MODULES)
│   ├── <name>/<name>_module.py   the front page: every slot of a physics (MODULE_ENTRIES)
│   │     FIELDS (what is solved for), PROPERTIES, OPTIONAL_PROPERTIES, UNIT_KINDS, NEEDS, SETTINGS;
│   │     the optional entries may be left out (OPTIONAL_ENTRIES gives each a default)
│   ├── <name>/<name>_physics.py  slot: the solver (make_solver) and one design's analysis (analyse)
│   ├── <name>/<name>_supports_and_loads.py   slot: a kind of support or load of this physics
│   ├── <name>/<name>_quantities.py   slot: a quantity of this physics (QUANTITIES)
│   ├── <name>/<name>_results.py  slot: what each iteration records, the final check, the results a run
│   │                             saves (RESULT_FIELDS) and its own Summary rows (SUMMARY_ROWS)
│   ├── structural/               the one physics users have today
│   └── template/                 a small working physics to copy; not offered in the GUI
│
├── the GUI (gui/)                reads everything above; no physics written in it
│   ├── main_window.py            the window, File and Help, New project (with the physics), the 3D view
│   ├── materials_panel.py, database_editor.py, formulation_panel.py, settings_panel.py,
│   │   supports_and_loads_panel.py, run_panel.py, results_panel.py   one panel each
│   ├── draw_design.py            a design in the 3D view, with the views its run saved
│   ├── background_work.py        runs and analyses on one background thread
│   ├── crash_report.py           crashes, errors, freezes and the last actions, in crash_report.log
│   ├── dark_and_light_3d_view.py, panel_widgets.py, user_guide_window.py, assets/
│   └── pyto_ad_imports.py
│
├── docs/                         this map, how the code works, how to add features, the reference
│                                 page (generated), the user guide
└── tests/                        one file per part; the template physics' test is the one to copy
```

## 2. Where does my feature go?

```
Is it about one physics only?
├── yes → in that physics' folder, lsr_mmto/physics/<name>/
│   ├── a new physics altogether?
│   │     copy physics/template/ and follow its FILL lines; one line in available_physics.py
│   │     test: tests/test_template_physics.py · docs: "A physics"
│   ├── something to solve for, or a different solver?
│   │     <name>_physics.py (make_solver, analyse), FIELDS on the front page
│   │     test: a case whose answer is known, as the template's bar test · docs: "A physics"
│   ├── something that holds or pushes the part?
│   │     <name>_supports_and_loads.py, one entry of SUPPORT_AND_LOAD_KINDS
│   │     test: tests/test_project.py · docs: "A kind of support or load"
│   ├── something to minimize or limit?
│   │     <name>_quantities.py, one entry of QUANTITIES
│   │     test: tests/test_ad_quantities.py · docs: "A quantity"
│   ├── a result to show after a run?
│   │     RESULT_FIELDS, its values returned by final_summary (<name>_results.py)
│   │     test: tests/test_template_physics.py · docs: "A physics"
│   ├── a number of its own in a run's Summary and Compare?
│   │     SUMMARY_ROWS, one entry naming a history key (<name>_results.py)
│   │     test: tests/test_extending.py · docs: "A physics"
│   ├── a term added to the objective (a penalty) of this physics only?
│   │     PENALTIES on the front page, one entry
│   │     test: tests/test_template_physics.py · docs: "A penalty"
│   ├── a setting of this physics only?
│   │     SETTINGS on the front page
│   │     test: tests/test_run_project.py · docs: "A setting"
│   └── a unit its values or quantities are measured in?
│         UNIT_KINDS on the front page
│         test: tests/test_databases.py · docs: "A material attribute"
└── no → would every physics use it?
    ├── a material property?
    │     one entry of ATTRIBUTES, lsr_mmto/materials/databases.py
    │     test: tests/test_databases.py · docs: "A material attribute"
    ├── a material database?
    │     an .xlsx in data/databases/, or made in the GUI
    │     test: tests/test_databases.py · docs: "A material database"
    ├── a way to pick part of the mesh?
    │     one class in lsr_mmto/problem/selections.py
    │     test: tests/test_selections.py · docs: "A selection type"
    ├── a quantity for every physics (made from the design and the materials alone)?
    │     one entry of QUANTITIES, lsr_mmto/formulation/quantities.py
    │     test: tests/test_ad_quantities.py · docs: "A quantity"
    ├── a setting for every physics?
    │     one entry of SETTINGS, lsr_mmto/problem/settings.py
    │     test: tests/test_run_project.py · docs: "A setting"
    ├── a new way to treat the constraints?
    │     one entry of CONSTRAINT_METHODS, lsr_mmto/optimize/optimization_loop.py
    │     test: tests/test_regression.py · docs: "A constraint method"
    ├── a term added to the objective (a penalty) every physics has?
    │     one entry of PENALTIES, lsr_mmto/design/latent_penalties.py
    │     test: tests/test_template_physics.py, tests/test_regression.py · docs: "A penalty"
    ├── another step of each iteration (a continuation)?
    │     one step of ITERATION_STEPS, lsr_mmto/optimize/optimization_loop.py
    │     test: tests/test_regression.py · docs: CONTRIBUTING, "Where a change goes"
    ├── something in the project file itself?
    │     lsr_mmto/problem/project.py, with an upgrade for older files
    │     test: tests/test_project.py · docs: "A change to the project file format"
    └── none of these: it changes what the core assumes
          a widening of the core: CONTRIBUTING, "Where a change goes", its four rules
          test: tests/test_regression.py, and a test of its own
```

Something that is both a new physics and a widening of the core (a property that depends on the
temperature, a nonlinear solve) is two changes: the widening first, tested with a physics that
already exists, then the physics that uses it.
