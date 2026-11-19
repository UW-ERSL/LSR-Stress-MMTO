# What LSR MMTO takes from PyTO-AD

Every part of PyTO-AD that LSR MMTO runs, what each one assumes, where LSR MMTO goes beyond that, what
guards it, and what to do if it breaks. PyTO-AD is pinned to one commit (790d935) and never edited: a
problem in a part taken from it is fixed in LSR MMTO, by subclassing or wrapping it.

## How this list was made

1. **Every way in.** Only two files import PyTO-AD to run it: `lsr_mmto/pyto_ad_imports.py` (the solvers,
   the mesh, the optimizer) and `gui/pyto_ad_imports.py` (the GUI's windows). Every other mention of
   PyTO-AD in LSR MMTO is a type hint, which runs nothing. This was checked by reading every import in
   `lsr_mmto/` and `gui/` with a program, not by eye.
2. **What actually runs.** The tests of runs, of each physics, of Analyse, of STL export, of the final check,
   of single material and of the GUI (`tests/runs/test_run_project.py`, `tests/physics/`,
   `tests/runs/test_solid_part_analysis.py`, `tests/runs/test_stl_export.py`, `tests/runs/test_final_check.py`,
   `tests/runs/test_single_material.py`, `tests/gui/test_thermal_walk_through.py`, `tests/gui/test_gui.py`)
   were run with coverage measured over PyTO-AD's own code. They run **171 of PyTO-AD's 1,059 functions**.
   The rest of PyTO-AD (its own examples, its own optimizer driver, its tetrahedral mesher, most of its GUI)
   never runs in LSR MMTO.
3. **What each part assumes,** read from its code and from where LSR MMTO calls it.

This list is complete for what enters through the two files and for what those tests run. A path no test
takes is marked **untested** below: it may work, but nothing shows it does.

## At a glance

| Status | Meaning | Count |
|---|---|---|
| **Problem** | known to give wrong numbers or break in a case LSR MMTO allows | 4 (P2, P3, P4 fixed; P1 kept on purpose, flagged) |
| **Fragile** | works today, but depends on something easy to break by a later change | 2 (both fixed) |
| **Untested** | reachable from the GUI or a project, run by no test | 2 (U1 tested, U2 removed) |
| **Safe** | used within what it was made for, or already adapted, and tested | the rest |

The four problems, worst first:

| # | Part | What goes wrong | Level |
|---|---|---|---|
| P1 | DPCG, the iterative solver for large parts | Does not converge on the final check's binarized designs (Airplane bracket at 30,000 dofs: an error about 600 times the force). The stress and displacement a large run saves are not reliable. | Numbers |
| P2 | Plain CG, the fallback inside PyTO-AD's solve | Was used for thermoelastic's temperature on large parts, unrecorded. **Fixed**: the temperature has its own deflation space. | Numbers |
| P3 | The SIMP penalty, PyTO-AD's global setting | Was: the run loop named two physics ("structural", "thermal") and raised both. **Fixed**: each physics declares its own. | Generality |
| P4 | The Pick window ("Click on the part...") | Was: offered only structural supports and loads, whatever the physics. **Fixed**: it offers what the physics' kinds take. | GUI |

## The solve

| Part | What LSR MMTO uses it for | Where | Status |
|---|---|---|---|
| `hex_structural_fea.HexStructuralFEA` | the structural solve; LSR MMTO's `MultiMaterialHexFEA` subclasses it with a different E and nu in each element | `lsr_mmto/physics/structural/structural_physics.py` | Safe |
| `hex_thermal_fea.HexThermalFEA` | the thermal solve; `ConductionHexFEA` subclasses it with a conductivity in each element | `lsr_mmto/physics/thermal/thermal_physics.py` | Safe |
| `hex_element_stiffness` | the element matrices of one unit material, scaled per element | both solvers above | Safe |
| `autodiff/sparse_solve` (reached through the solvers) | the solve and its gradient: PARDISO for small parts, DPCG for large ones (plain CG never: a test fails if it runs) | inside PyTO-AD's `solve` | see P1; P2 fixed |
| `deflation.DeflationSolver` | DPCG for large parts; LSR MMTO's `CheckedDeflationSolver` records whether each solve converged | `lsr_mmto/optimize/iterative_solve_check.py`, `optimization_loop.problem_solver` (line 261) | **P1** |
| `solvers` (`Solvers.PARDISO`, `Solvers.DPCG`) | which solver to use | `optimization_loop.problem_solver` | Safe |
| `DIRECT_SOLVER_DOF_CUTOFF` (100,000) | the size above which DPCG is used | `optimization_loop.py`, line 282 | Safe |

**P1. DPCG on binarized designs.**
- *Assumes:* a design whose stiffness varies smoothly, as during the iterations. A binarized design mixes
  solid elements with nearly empty ones, a contrast of about a million to one, and DPCG's preconditioner and
  coarse space are not strong enough for that.
- *Guard today:* the run records every solve that did not converge, and the Summary, Compare and Analyse say
  which numbers are not reliable (`iterative_solve_check.py`); a run or analysis that ends with one says so in red
  under Current run and in the messages panel, and the .npz keeps the record too. The wrong numbers are flagged,
  not prevented. Kept this way on purpose: the direct solver is not used on very large parts.
- *If you fix it:* in `optimization_loop.problem_solver`, or in a solve of the final check alone, use the
  direct solver (PARDISO) for the final check when the part fits in memory; or give DPCG a stronger
  preconditioner. Check: `tests/runs/test_run_project.py::test_a_solve_that_does_not_converge_is_marked`,
  and a run of the Airplane bracket at 30,000 dofs with the solver limit set to 20,000 whose solver check
  comes back empty.

**P2. Plain CG, unrecorded. Fixed.**
- *Was:* thermoelastic's temperature has one unknown per node, which the structural deflation space (three per
  node) does not fit, so PyTO-AD's solve fell back to plain CG (`autodiff/sparse_solve.py`,
  `_IterativeSolve.solve`), which only warns. PyTO-AD's own thermoelastic solver does the same; its examples
  run at 25,000 dofs, on the direct solver, so they never reach it.
- *Now:* thermal has a deflation space of its own, a constant on each group of nodes
  (`thermal_physics.deflation_space`, `iterative_solve_check.one_value_per_node_space`); thermoelastic uses
  structural's for its displacement and thermal's for its temperature, and thermal-only uses DPCG above the
  limit too. `CheckedDeflationSolver` keeps one space per size of system, so both solves are checked and
  recorded. On the MBB beam (108k nodes, a near-binary design) the temperature took 78 iterations, against
  250 for plain CG.
- *Found on the way:* PyTO-AD's coarse solve adds 1e-10 to the diagonal of the coarse matrix
  (`solve/deflation.py`, `_coarse_space`). That is nothing beside structural's coarse matrix, but a
  temperature's can be small (0.003 to 13 on the U-bracket of conductivity 3), and with it DPCG diverged; the
  temperature's coarse solve leaves it out (`SpaceSolver._coarse_space`). Structural's is unchanged. It is not
  the cause of P1: a binarized structural design fails with the exact coarse matrix too.
- *Guard:* above the limit a run is refused when a physics' deflation_space leaves a field without a space
  (`optimization_loop.problem_solver`); every test fails if plain CG runs (`tests/conftest.py`); and
  `tests/runs/test_run_project.py::test_every_field_of_every_physics_has_a_deflation_space` checks that each
  physics' deflation_space covers each of its fields.

## The material model

| Part | What LSR MMTO uses it for | Where | Status |
|---|---|---|---|
| `MaterialModel.SIMP` | which density scaling each solve uses | every physics' `analyse` | Safe |
| `get_structural_material_model_scaling_torch`, `get_thermal_material_model_scaling_torch` | the density scaling, inside the solve | PyTO-AD's `solve`; thermoelastic's coupling | Safe |
| `get_stress_relaxation_factor`, `get_pNorm_exponent` | read once, as the structural solver is made, as the start values of its own `stress_relaxation` and `pnorm_exponent`, which the run's settings then replace | `structural_physics.MultiMaterialHexFEA` | Safe (F2 fixed) |
| `set_SIMP_STRUCTURAL_PENALTY` | the stiffness's SIMP exponent, at the start and at each step of continuation | `structural_physics.set_simp_penalties`, called by the run loop | Safe (P3 fixed, U1 tested) |

**P3. The SIMP penalty was two named physics in the core. Fixed.**
- *Was:* PyTO-AD keeps two penalties, one for "structural" and one for "thermal", as global settings of its
  material model, and the run loop started and raised both, whatever the physics.
- *Now:* each physics declares its own exponents (`SIMP_PENALTIES`: start, start with continuation, largest)
  and puts them where its solver reads them (`set_simp_penalties`). The run loop starts and raises them
  (`optimization_loop.simp_penalties_at_start`, `raised_simp_penalties`) and names no physics. Thermal's goes
  on its own solver (`conductivity_penalty`); structural's still goes into PyTO-AD's global, because PyTO-AD's
  structural solve reads only that (`hex_structural_fea.py`, line 196), but only structural's own folder
  sets it. The numbers did not move (`tests/runs/test_regression.py`).
- *Guard:* `tests/runs/test_run_project.py::test_simp_continuation_raises_each_penalty_of_every_physics`;
  `tests/code_rules/test_layout.py::test_shared_settings_stay_in_the_physics_folders` fails if code outside a
  physics' folder names one of PyTO-AD's SIMP or p-norm globals.
- *When the pin changes:* if PyTO-AD's structural solve ever takes an exponent of its own, structural's
  `set_simp_penalties` can put it on the solver as thermal's does, and the global goes.

**F2. Global settings shared by every run in the program. Fixed.** The p-norm exponent was written into
PyTO-AD's global at every build, and the stress relaxation read from its global at each use. Now the stress
quantities read the exponent from the run's settings, and the structural solver keeps its own `pnorm_exponent`
and `stress_relaxation` (both set by `analyse` and by the final check, from the run's settings); nothing writes
PyTO-AD's p-norm global any more. Guard: `test_the_pnorm_exponent_is_the_runs_own` and the code rule above.

## The mesh, the supports, the filters, the optimizer

| Part | What LSR MMTO uses it for | Where | Status |
|---|---|---|---|
| `hex_mesher.HexMesher` | the voxel mesh from the STL, its dof tables, boundary nodes | `lsr_mmto/problem/mesh.py` (`make_mesh`, line 22), selections | Safe (F1 fixed) |
| `bc.BC`, `check_rigid_body_constraint` | supports and loads as a solver reads them; the check that a part is held | `project.py` (`supports_and_loads`), `structural_supports_and_loads.py` (line 63) | Safe |
| `mat_lib.create_material_with_defaults` | a placeholder material PyTO-AD's constructors need; the real values are set per element afterwards | the physics' `make_solver` | Safe |
| `createFilters` | the density filter matrix, and the one-layer (extrude) filter | `lsr_mmto/optimize/filters.py` | Safe |
| `TOParams` | the parameter container LSR MMTO's `MMTOParams` extends | `mmto_to_params.py`, `settings.py` | Safe |
| `runMMA` | the optimizer | `optimization_loop.mma` | Safe |
| `topopt_stl_recovery` | the smooth surfaces of Export STL | `lsr_mmto/results/stl_export.py` | Safe |

**F1. The mesh's shared dof table. Fixed.** PyTO-AD's structural and thermal solvers each point the
mesh's `edofMat` at their own table when they are made, so with two solvers on one mesh (thermoelastic) the
one made last won. LSR MMTO's solvers keep their own tables (`.edof`) and nothing in LSR MMTO reads the
mesh's; thermoelastic sets it to the displacement's after making both, so the order no longer matters.
Guard: `tests/physics/test_thermoelastic_physics.py::test_the_shared_dof_table_is_the_displacements_whatever_the_order`,
and the code rule above fails if any LSR MMTO code reads `mesh.edofMat`.

**MMA** keeps Svanberg's usual settings (`a0 = 1`, `c = 100` for every constraint). They assume each
constraint is of a size near 1, which LSR MMTO's constraints are: each is divided by its limit, at most or
at least alike.

## The GUI

| Part | What LSR MMTO uses it for | Where | Status |
|---|---|---|---|
| `MainWindow` | the window LSR MMTO's `MMTOWindow` subclasses: the 3D view, the sidebar, the messages | `gui/main_window.py`, line 972 | Safe |
| `StructuralLoadsWindow`, `ThermalLoadsWindow` | the Pick windows ("Click on the part..."): supports and forces, temperatures and heat | `gui/main_window.py`, `PickWindow`, `ThermalPickWindow`, `pick_windows` and `adopt_picks` | Safe (P4 fixed) |
| `StructuralTopOptWindow` | the Current run panel's worker thread, signals and log; LSR MMTO replaces its layout and plots | `gui/run_panel.py`, line 196 | Safe |
| `DisplayOptionsWindow` | View: cuts, the part's triangles, see-through | `gui/main_window.py`, line 629 | Safe |
| `UnitsWindow`, `Settings` | the units dialog | `gui/main_window.py` | Safe |
| `ProjectsWindow` | drawing the STL and turning picking on | `gui/main_window.py` | Safe |
| `STLGeom` | the STL read in: triangles, normals, neighbours | `gui/main_window.py` | Safe |
| `MATERIAL_COLOR`, `FILE_DIALOG_OPTIONS` | a colour, and the file dialogs' options | several panels | Safe |

**P4. The Pick window was structural only. Fixed.**
- *Was:* PyTO-AD's Structural Loads window was the only picking window, so a thermal project could pick only
  structural supports, which its check then refused.
- *Now:* a kind of support or load declares the types of pick it takes (`picks`); thermal's take PyTO-AD's
  Thermal Loads window's Temperature and Total Heat (not Heat Flux, which no kind takes). Each picking window
  (`PICK_WINDOWS` in `gui/main_window.py`: `PickWindow`, `ThermalPickWindow`) says which types it makes, and
  Click on the part offers the windows whose types the project's physics takes (`pick_windows`): one, as a
  line; several (thermoelastic), one line each; none, greyed, saying why. Each window's Type offers only
  those types, and each Apply goes into the project (`adopt_picks`), as a kind acting on the project's own
  fields. Clicks on the part pick for whichever picking window is open in front (`picking_window`); a
  project switch or a change of units closes them. No GUI code names a physics.
- *Guard:* `tests/gui/test_thermal_walk_through.py::test_temperatures_and_heat_are_picked_on_the_part`
  (clicks through the window's own mouse handling, both types applied), and
  `tests/gui/test_new_physics_in_the_gui.py`, where a made-up physics with nothing to pick greys it.
- *A new physics:* its kinds declare `picks` for a type an existing window makes; a kind of a new sort
  (say a fluid's inflow) needs a picking window of its own, added to `PICK_WINDOWS`.

## Untested paths

| # | Path | Reached from | Now |
|---|---|---|---|
| U1 | SIMP continuation | Settings > SIMP continuation | **Tested**, for every physics: the schedule of each exponent, and a U-bracket run with it on that raises the stiffness's to 1.75 in 11 iterations. |
| U2 | Keep support elements (`find_elements_with_fixedDOF`) | the project's Keep support elements | **Removed**: Keep solid on each support's selection does nearly the same (on the Airplane bracket at 3,000 dofs, 346 elements against 354; the same on the other shipped projects). An older file with it on opens, and the project's check says so until it is saved; an older script's line turning it on is an error. |

## When PyTO-AD's pin changes

A new pin can change any part above. In that order:

1. Run every test. `gui/pyto_ad_imports.py` and `lsr_mmto/pyto_ad_imports.py` fail first if a name was
   renamed or removed.
2. `tests/runs/test_regression.py` fails if the numbers moved. If they moved on purpose, store them again
   (`python tests/runs/test_regression.py --update`) and say why in the commit message.
3. Measure again what runs (the coverage command below) and update this file where a part changed.
4. Check the two places LSR MMTO still follows PyTO-AD's own code rather than calling it: the temperature's
   coarse solve (`iterative_solve_check.SpaceSolver._coarse_space`, PyTO-AD's `deflation._coarse_space`
   without its 1e-10), and structural's SIMP exponent, which goes into PyTO-AD's global because its
   structural solve (`hex_structural_fea.py`, `solve`) reads only that.

The measurement, in a Miniforge Prompt in this folder:

    python -m pytest tests/runs tests/physics tests/gui -q --cov=../PyTO-AD/src/pyto --cov-report=term
