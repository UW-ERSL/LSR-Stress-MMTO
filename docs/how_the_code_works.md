# How the code works

## Layers

| Layer | Where | What it is |
|---|---|---|
| PyTO-AD | `../PyTO-AD`, pinned (see the README) | finite element analysis, filters, MMA, the differentiable solve |
| Core | `lsr_mmto/`, except `physics/<name>/` | the project, the materials and their network, the design, the optimization loop, the results |
| Physics modules | `lsr_mmto/physics/<name>/` | one physics each: its solver, supports and loads, quantities, results |
| GUI | `gui/` | the window; everything it does goes through the core |

PyTO-AD is never edited. Everything taken from it is imported in one place,
`lsr_mmto/pyto_ad_imports.py` (and `gui/pyto_ad_imports.py` for the GUI), and where it needs a
different material in each element, a subclass does that (`MultiMaterialHexFEA` in
`physics/structural/structural_physics.py`).

## The core, folder by folder

| Folder | What it holds |
|---|---|
| `problem/` | the project (`Project`, `run(p)`), its settings, selections, supports and loads, the mesh, the project folder |
| `materials/` | material databases, the material network (VAE), training and loading it |
| `elements/` | the hexahedral element: strain matrix and Lame parameters |
| `design/` | density filter and Heaviside projection, latent points to material properties (`decode`), the penalties added to the objective (`PENALTIES`) |
| `formulation/` | the quantities an objective or a constraint can name, and the formula text |
| `physics/` | `available_physics.py`, the list of physics, and one folder per physics |
| `optimize/` | the optimization loop, one iteration's model, the history, the final design analysis, saving a run |
| `results/` | a saved run read back: tables, plots, 3D views, STL export, two runs compared |

## One run

`run(p)` is the only way a project runs; the GUI's Run and a setup script both call it.

1. `Project.build()` (`problem/project.py`) makes the mesh from the STL, one boundary condition per
   field of the project's physics from its supports and loads, and the optimization parameters.
2. The loop's `setup` (`optimize/optimization_loop.py`) reads the material database, loads or trains
   the network, asks the physics' front page for its solver (`make_solver`), and makes the start design
   and the bounds. `optimize` then runs the constraint method (MMA), which asks `evaluate` about each
   design; `evaluate` is the list `ITERATION_STEPS`, in order.
3. For each design, `MMTOModelAD` (`optimize/design_model.py`) builds one torch tree:

        x (density)  -> density filter -> Heaviside projection -> x_phys
        z (latents)  -> network decoder -> props (every column of the database)
        x_phys, props -> physics.analyse   (structural: E, nu -> solve -> u -> von Mises)
        -> objective and constraints (QUANTITIES), by name

   autograd gives their gradients with respect to `x_phys` and `z`. The loop adds the penalties
   (`PENALTIES`), carries the gradient back through the projection and the filter to `x`, and the
   physics records what each iteration measures (`iteration_diagnostics`).
4. After the last iteration, `optimize/final_design_analysis.py` binarizes the design and snaps
   the materials, the physics prints its numbers for each and returns its results (`final_summary`), and
   `optimize/save_results.py` saves the run with the physics' results by name (`RESULT_FIELDS`).
5. `results/load_results.py` reads a saved run back; the Results panel, the tables and the plots
   use it.

## A physics' front page

A physics is a folder in `physics/` whose `<name>_module.py` gives, by name, everything the rest of
the code takes from it. `MODULE_ENTRIES` in `physics/available_physics.py` lists them:

| Entry | What it is |
|---|---|
| `NAME`, `DOC` | the name a project file uses, and what the physics is |
| `FIELDS` | each field it solves for and its unknowns per node (structural: `{"u": 3}`) |
| `PROPERTIES` | the database columns every problem of it reads |
| `make_solver` | its finite element solver, from the mesh and one boundary condition per field |
| `analyse` | one iteration's solve, inside the torch tree |
| `SUPPORT_AND_LOAD_KINDS` | its kinds of support and load, each naming its field |
| `NEEDS` | what every project of it must have: a support, and a load if a problem without one has no meaning |
| `deflation_space` | for DPCG on problems above the direct solver's limit; `None` keeps the direct solver |
| `supports_and_loads_error` | why the supports and loads leave no solution, or None |
| `QUANTITIES` | its objectives and constraints |
| `PENALTIES` | the terms it adds to the objective besides those every physics has (may be empty) |
| `SETTINGS` | its own settings |
| `RESULT_FIELDS` | the per-element results a run saves: their names, views and units |
| `iteration_diagnostics` | what each iteration records and prints |
| `final_summary` | the final check's numbers, printed; it returns the values of `RESULT_FIELDS` for the last design |

The core's lists are made from the modules: `QUANTITIES` (`formulation/quantities.py`),
`PENALTIES` (`design/latent_penalties.py`), `BC_KINDS` (`problem/boundary_conditions.py`) and
`SETTINGS` (`problem/settings.py`). No core file imports a physics by name. `docs/how_to_add_features.md` says how to add one.
