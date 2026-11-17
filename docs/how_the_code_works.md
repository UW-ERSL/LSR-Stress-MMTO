# How the code works

## Layers

| Layer | Where | What it is |
|---|---|---|
| PyTO-AD | `../PyTO-AD`, pinned (see the README) | finite element analysis, filters, MMA, the differentiable solve |
| Core | `lsr_mmto/`, except `physics/<name>/` | the project, materials, material network, design, optimization loop, and results |
| Physics modules | `lsr_mmto/physics/<name>/` | a separate module for each physics, containing its solver, supports, loads, quantities, and results |
| GUI | `gui/` | the application window; every action goes through the core |

PyTO-AD is never edited. All imports from it are collected in one place,
`lsr_mmto/pyto_ad_imports.py` (and `gui/pyto_ad_imports.py` for the GUI). Where PyTO-AD's code needs a
different material in each element, a subclass handles that (`MultiMaterialHexFEA` in
`physics/structural/structural_physics.py`).

## The core, folder by folder

| Folder | What it holds |
|---|---|
| `problem/` | the project (`Project`, `run(p)`), its settings, selections, supports and loads, the mesh, the project folder |
| `materials/` | material databases and the material network (VAE), including how the network is trained and loaded |
| `elements/` | the hexahedral element, including its strain matrix and Lame parameters |
| `design/` | density filtering and Heaviside projection, latent points to material properties (`decode`), the penalties added to the objective (`PENALTIES`) |
| `formulation/` | the available objectives and constraints, and the reader of the formula text |
| `physics/` | `available_physics.py`, the available physics modules, with a separate folder for each one |
| `optimize/` | the optimization loop, the model used in each iteration, history, final design analysis, and saving results, plus Analyse (the solid part solved once in one material) |
| `results/` | loading a saved run, showing tables, plots, and 3D views, exporting STL files, and comparing two runs |

## One run

`run(p)` is the single entry point for running a project; the Run button in the GUI and a project script both call it.

1. `Project.build()` (`problem/project.py`) builds the mesh from the STL, a boundary condition for each
   field of the project's physics from its supports and loads, and the optimization parameters.
2. The loop's `setup` (`optimize/optimization_loop.py`) reads the material database. Through
   `material_source`, it either loads the existing network or trains a new one (none for a single material, whose database values
   are used), gets the solver from the physics module's front page (`make_solver`), and makes the initial design
   and the bounds. `optimize` then uses the constraint method (MMA), which asks `evaluate` about each
   design; `evaluate` follows the list `ITERATION_STEPS`, in order.
3. For each design, `MMTOModelAD` (`optimize/design_model.py`) builds one connected torch tree:

        x (density)  -> density filter -> Heaviside projection -> x_phys
        z (latents)  -> material source (enc.decode) -> props (every column of the database)
        x_phys, props -> physics.analyse   (structural: E, nu -> solve -> u -> von Mises)
        -> objective and constraints (QUANTITIES), by name

   autograd calculates their gradients with respect to `x_phys` and `z`. The loop adds the penalties
   (`PENALTIES`), passes the gradient backward through the projection and the filter to `x`, and the
   physics records what each iteration measures (`iteration_diagnostics`).
4. When the final iteration is complete, `optimize/final_design_analysis.py` binarizes the design and snaps
   the materials, the physics prints its numbers for each and returns its results (`final_summary`), and
   `optimize/save_results.py` saves the run, with the physics' results under their names (`RESULT_FIELDS`).
5. `results/load_results.py` loads a previously saved run; the Results panel, the tables and the plots
   use it.

## A physics' front page

Each physics has a folder under `physics/`. Its `<name>_module.py` provides the named entries that
the rest of the code needs. `MODULE_ENTRIES` in `physics/available_physics.py` lists them:

| Entry | What it is |
|---|---|
| `NAME`, `DOC` | the name a project file uses, and what the physics is |
| `FIELDS` | the fields the physics solves, with the number of unknowns at each node (structural: `{"u": 3}`) |
| `PROPERTIES` | the database columns required by every problem using this physics |
| `make_solver` | its finite element solver, from the mesh and a boundary condition for each field; the core also reads its `.bc`, `.mesh` and `.plot_mesh` |
| `analyse` | the solve performed during one iteration, inside the torch tree |
| `SUPPORT_AND_LOAD_KINDS` | the types of supports and loads supported by this physics, with the field for each type |
| `NEEDS` | the supports and loads that every project using this physics must provide: a support, and a load if a problem without one has no meaning |
| `deflation_space` | for DPCG on problems above the direct solver's limit, a space for each field (thermal's for a field of one unknown per node); `None` keeps the direct solver |
| `supports_and_loads_error` | an explanation of why the supports and loads make the problem unsolvable, or None if they are valid |
| `QUANTITIES` | its objectives and constraints |
| `PENALTIES` | any extra terms this physics adds to the objective, beyond the common terms (may be empty) |
| `SETTINGS` | settings specific to that physics |
| `RESULT_FIELDS` | the results saved for each run, including names, views, and units, and where each lives (element or node) and its shape (scalar or vector) |
| `iteration_diagnostics` | the information recorded and printed at every iteration, from the model after its solve (read with `.detach()` only) |
| `final_summary` | the numerical results printed by the final check; it returns the values of `RESULT_FIELDS` for the final design |

The core builds these lists from the physics modules: `QUANTITIES` (`formulation/quantities.py`),
`PENALTIES` (`design/latent_penalties.py`), `BC_KINDS` (`problem/boundary_conditions.py`) and
`SETTINGS` (`problem/settings.py`). No file in the core imports a particular physics module by name. `docs/how_to_add_features.md` says how to add one.
