# Testing the codebase

Every test of LSR MMTO, folder by folder: what each folder checks, what each file in it checks, and each test in it, 368 in all. Generated from the tests' own docstrings by `python docs/make_testing_doc.py`; do not edit it by hand.

Run them all from the repository folder with `python -m pytest tests -q -n 4` (`-n 2` on a computer with less than 16 GB of memory), one folder with `python -m pytest tests/<folder> -q`, or one file with `python -m pytest tests/<folder>/<file> -q`.

## physics

Each physics gives the right answers: its solver on cases with a known answer, its gradients, its runs.

### `tests/physics/test_structural_physics.py`

The structural physics (lsr_mmto/physics/structural/): its tiny example, which tests/runs/test_every_physics_runs.py checks and runs. Structural is the physics the core's own tests use, so its other tests are with the core's: a run and its record in tests/runs/, its gradients in tests/gradients/, the paper's examples in tests/runs/test_regression.py. The smallest test file a physics can have: start from this one for a new physics.


### `tests/physics/test_thermal_physics.py`

The thermal physics (lsr_mmto/physics/thermal/) keeps working: its solver gives the right answer on a bar, and short U-bracket runs go through with it and its example database, with no core file changed. Copy this file for a physics of your own. A physics not yet in available_physics.MODULES can be tested before it is: wrap each test in registered(module) (tests/physics/core_lists_for_a_test.py), which adds it to the core's lists as its line in MODULES would.

- `test_a_project_sees_only_its_own_physics_settings`: The Settings panel shows the settings every project has and those of the project's own physics, each group once. Thermal has none of its own, so one is given to it here, added to the core's settings as registering adds a front page's.
- `test_thermal_solver_gives_a_straight_line_along_a_bar`: T held at 0 at one end of a bar and 1 at the other, solid everywhere, one conductivity: T rises in a straight line from end to end, at every node. Trilinear elements give this exactly.
- `test_thermal_gradients_agree_with_finite_differences`: ThermalCompliance, f . T, on the bar held at 0 on one end with 1 of heat put in at the other, at an uneven design (densities and conductivities differ by element). Its autograd gradient for the density and the conductivity of a few elements, each where the gradient is large, agrees with central differences: the quantity's own function after the physics' analyse, the thermal solve included.
- `test_thermal_physics_runs_with_no_core_file_changed`: A thermal project passes its check and a structural kind on it is refused; built, it holds the temperature and spreads the heat source over its nodes. Its tiny example's run (tiny_run, run once per session for every test that reads it) saved the temperature at each node, as thermal's own result, and recorded thermal's own numbers and none of structural's.
- `test_a_thermal_summary_shows_its_own_row_and_units`: A run in mmKS: its Summary has thermal's own row, the max temperature in K, and its Units row the units of the thermal kinds; nothing of structural's (no failure factor, no force or stress).
- `test_a_thermal_run_with_held_temperatures_only`: The physics says what a project must have (NEEDS): thermal needs a support and no load, so a run with two held temperatures and no source goes through, and its save gives no RuntimeWarning (no load nodes, so no load centroid in the record); the structural physics still needs a load.
- `test_a_thermal_run_of_one_material`: Thermal with one material in the design: the network fixed to one of the example database's materials, and a database of that one material at its database values. Each runs through to the save, every element is that material, with the conductivity the network or the database gives, and the objective is finite.
- `test_the_saved_run_keeps_the_properties_the_physics_reads`: A run saves each element's value of every property its physics reads (PROPERTIES): the conductivity, and nothing of the structural physics, which its database does not have.
- `test_thermal_is_analysed_with_no_file_of_it_changed`: Analyse (optimize/solid_part_analysis.py) on a thermal project: its analyse and final_summary run on the solid part in one material, and the analysis gives its objective and its RESULT_FIELDS. Its iteration_diagnostics runs once too, so the analysis has thermal's Summary row, and Compare shows it beside a run's.
- `test_result_fields_say_where_and_shape`: A RESULT_FIELDS entry may say where its values live and their shape; one that does not is a number at each element. The final check refuses values of another length or shape than declared.
- `test_a_result_the_physics_does_not_return_stops_the_save_with_a_message`: final_summary returns the value of every RESULT_FIELDS entry; one left out is a clear error, not a result silently missing or taken from an earlier solve.
- `test_the_simp_exponent_is_the_solvers_own`: Thermal's SIMP exponent goes on its solver (set_simp_penalties), not in PyTO-AD's global: on a grey design the temperature at exponent 3 is that of a solver built with 3, differs from exponent 1, and PyTO-AD's global is left as it was.
- `test_simp_continuation_raises_the_conductivitys_exponent_on_its_solver`: With SIMP continuation on, a thermal run starts the conductivity's exponent at 1 and raises it to 1.25 at iteration 10, each time on the run's own solver, which its solves read.
- `test_a_large_thermal_problem_uses_dpcg_and_gives_the_direct_answer`: Above the direct solver's limit thermal uses DPCG on its own deflation space (a constant on each group of nodes). On a near-binary design the temperature and its gradient (the adjoint solve) are the direct solver's, and every iterative solve, the adjoint's too, is kept by the run's checked solver as converged.
- `test_a_large_structural_problem_still_uses_dpcg`: The structural physics builds its deflation space, so above the limit it runs DPCG as before.
- `test_a_physics_penalty_is_added_with_its_gradient`: A penalty of thermal is added after the two every physics has, divided by its first value and times its weight, and the gradient of the objective with it agrees with central differences (the two every physics has are switched off, so the objective is mass plus this penalty).
- `test_a_physics_penalty_keeps_its_history_and_its_schedule`: A short thermal run with the test penalty on for its first iteration only: the saved run's history has its key, 0.5 at the first iteration (its first value, times 0.5) and 0 from then on; the history keys of the two every physics has are kept.
- `test_a_quantity_of_another_physics_is_refused`: The project's check refuses a quantity of another physics, as the objective or as a constraint (the structural Compliance in a thermal project), and keeps the quantities every physics has and thermal's own; a structural project may still name Compliance.
- `test_a_penalty_must_state_its_scale`: A penalty entry without its scale, or with one that is not "raw" or "first value", is refused when the core's lists are made (at start-up, where each front page's entry becomes a Penalty), and nothing of it is added.
- `test_a_penalty_must_say_whether_it_reads_the_latent_points`: A penalty entry without reads_latent_points, or with one that is not True or False, is refused when the core's lists are made; the two every physics has both read the latent points.
- `test_the_mesh_is_sized_by_the_physics_unknowns_per_node`: Mesh dofs counts every unknown at every node: thermal (one unknown per node) gets about three times the elements the structural physics (three per node) gets for the same number.
- `test_one_name_one_meaning_across_physics`: The core's lists (material properties, quantities, kinds of support and load) join the core's entries and every physics': an entry a second physics gives exactly as the structural one does is one entry, tagged with both; a different entry under the same name, or a property no one defines, stops LSR MMTO with a message naming the places. Thermal, which brings its own conductivity, joins with its own entries.

### `tests/physics/test_thermoelastic_physics.py`

The thermoelastic physics (lsr_mmto/physics/thermoelastic/) gives the right answers: a bar heated evenly and held only against rigid motion grows freely by its thermal strain. Its solver is checked on its own (make_solver, then analyse), so no material database is needed; its tiny example, which tests/runs/test_every_physics_runs.py runs, writes a database of one material.

- `test_a_bar_heated_evenly_grows_freely`: The bar held at T_ref + dT on two opposite faces with no heat source, so dT everywhere; held only in x on xmin, y on ymin and z on zmin, so nothing stops it growing. Each node moves by alpha * dT times its distance from those three faces, in each direction: the thermal strain with no stress. Trilinear elements give this linear field exactly. A bar that shrinks has H's sign wrong; one that grows (1 - 2 nu) times too little is missing the 1 / (1 - 2 nu).
- `test_no_heating_gives_the_structural_compliance`: The bar held at T_ref at both ends, so no thermal strain anywhere: clamped on xmin and pulled down on xmax, its thermoelastic compliance (f + f_th) . u equals the structural solver's f . u for the same mesh, supports, load and material, and the thermal force is zero.
- `test_each_simp_exponent_reaches_its_solver`: Thermoelastic's two SIMP exponents (set_simp_penalties): the stiffness's where PyTO-AD's structural solve reads it (its global), the conductivity's on the temperature's own solver.
- `test_the_shared_dof_table_is_the_displacements_whatever_the_order`: Each solver keeps its own dof table (.edof); the mesh's shared one, which PyTO-AD's constructors point at their own, is the displacement's once thermoelastic's solver is made.
- `test_gradients_agree_with_finite_differences`: J = (f + f_th) . u on the bar clamped and held at T_ref on xmin and heated on xmax (no applied force, so all of J is the heating's, and finite differences can see each element's part), at an uneven design (densities, expansion coefficients and conductivities all differ by element). Its autograd gradient for the density, the expansion coefficient and the conductivity of a few elements, each where the gradient is large, agrees with central differences: the whole chain T -> f_th -> u, the thermal solve included. The expansion coefficient and the conductivity reach J only through the heating.
- `test_a_large_problem_uses_dpcg_and_gives_the_direct_answer`: Above the direct solver's limit the core uses DPCG for a physics with a deflation space; thermoelastic takes structural's for its displacement and thermal's for its temperature (one unknown per node), both on the same checked solver. The bar, heated and pulled down, solved this way (the limit passed by asking for more unknowns than it) gives the direct solver's temperature and displacement, and both solves are kept as converged.
- `test_a_part_left_free_in_a_rigid_motion_is_refused`: The thermal force pushes in every direction, so the part must be held against all six rigid motions, whatever the applied loads: one left free makes the displacement's matrix singular, and the direct solver then gives a displacement with an arbitrary drift in it. Held in x on xmin and y on ymin only, the bar can still move in z; held in all directions along one edge only, it can still rotate about that edge. Each refusal names the motion left free. Held as in the free-expansion test (x, y and z on three faces), it is held against all six, though it can still grow.

### `tests/physics/core_lists_for_a_test.py` (a script, not collected by pytest)

The core's lists changed for one test, and put back after it: registered(module) adds a physics as its line in available_physics.MODULES would, making every list with the functions start-up makes them with (remake_lists), so whatever start-up would refuse is refused here too; added(entries, ...) adds entries to one list. Not a test file: a physics' own test file imports what it needs from here, and copying a test file never copies these.

## gradients

The autodiff gradients of the objectives and constraints agree with central finite differences.

### `tests/gradients/test_ad_quantities.py` (slow: left out by `-m "not slow"`)

Autodiff gradients of the objectives and constraints against central finite differences, on a coarse U-bracket (n_dof = 3000, about 12,000 DOF; 6-material database with Density and Yield_Strength; Cost on the 20-alloy database, which has a Cost column).

- `test_objective_compliance`: The autodiff gradient of the objective Compliance agrees with central finite differences.
- `test_objective_mass`: The autodiff gradient of the objective Mass agrees with central finite differences.
- `test_objective_failurefactorpnorm`: The autodiff gradient of the objective FailureFactorPNorm agrees with central finite differences.
- `test_constraint_compliance`: The autodiff gradient of the constraint Compliance agrees with central finite differences.
- `test_constraint_mass`: The autodiff gradient of the constraint Mass agrees with central finite differences.
- `test_constraint_mass_at_least`: The autodiff gradient of the constraint Mass, at least agrees with central finite differences.
- `test_constraint_failurefactorpnorm`: The autodiff gradient of the constraint FailureFactorPNorm agrees with central finite differences.
- `test_objective_stresspnorm`: The autodiff gradient of the objective StressPNorm agrees with central finite differences.
- `test_constraint_volumefraction`: The autodiff gradient of the constraint VolumeFraction agrees with central finite differences.
- `test_constraint_materialvolumefraction_material_2`: The autodiff gradient of the constraint MaterialVolumeFraction, material 2 agrees with central finite differences.
- `test_constraint_massfraction_the_lightest`: The autodiff gradient of the constraint MassFraction, the lightest agrees with central finite differences.
- `test_objective_cost`: The autodiff gradient of the objective Cost agrees with central finite differences.
- `test_constraint_cost`: The autodiff gradient of the constraint Cost agrees with central finite differences.
- `test_missing_column_is_named`: Cost on a database without a Cost column says so, instead of a bare KeyError.
- `test_mass_fraction_by_hand`: Mass fraction = mass / (reference density * design domain volume), the reference density the database's. Four elements of volume 0.002 (domain 0.008), x_phys 1, 0.5, 0, 0.25 with decoded densities 2000, 4000, 6000, 8000: mass (2000 + 2000 + 0 + 2000) * 0.002 = 12. Database densities 1500, 7800, 2700: the lightest solid part weighs 1500 * 0.008 = 12 (fraction 1), the heaviest 62.4 (12 / 62.4), the third 21.6 (12 / 21.6). As a constraint it is fraction / limit - 1.
- `test_mass_fraction_on_the_model`: On the U-bracket model at a random design: Mass fraction of the heaviest is the Mass quantity divided by the heaviest density the database file gives (read here on its own) times the number of elements times the element volume; and the constraint's value is that over its limit, minus 1.
- `test_registry`: Names are unique, each finds its entry, and each is an objective, a constraint or both.

### `tests/gradients/test_gradcheck.py`

The gradient check itself: gradient_check.check(p) sets a project up as run(p) does and compares the autodiff gradient with central finite differences. On a coarse U-bracket, for one density and one latent variable, the two must agree closely for at least one step size. About half a minute.

- `test_autodiff_matches_finite_differences`: Autodiff matches finite differences.

## problem

Setting up a project: saving and loading, the checks before a run, the selections, the mesh, the project folder.

### `tests/problem/test_at_most_and_at_least.py`

A constraint is at most its limit (<=) or at least it (>=), for every quantity of every physics, with no code in any physics: each test below runs over available_physics.MODULES, so a new physics is tested as it is added. The optimizer sees g = value / limit - 1 or g = 1 - value / limit, both met at g <= 0; a run's record keeps g and its op, and value / limit comes back from them. A new limit starts by its quantity's rule, 40% of its reference at most and 10% at least, worked out in the design's material when it is one, and a limit still at its start follows a change of materials (restarted_limits). One short U-bracket run with an at-least limit checks the whole way, from the project to the Summary.

- `test_every_quantity_of_every_physics_takes_at_most_and_at_least`: Either op is accepted by the project's check and goes to the run as [name, options, limit, op]; any other op is refused, saying which two there are.
- `test_the_optimizer_sees_g_met_at_or_below_zero_either_way`: The model's g at value v and limit b (the same for every quantity: it reads only the value): (v - b) / |b| at most and (b - v) / |b| at least, v / b - 1 and 1 - v / b for a positive limit, its gradient +1/|b| and -1/|b|; met exactly when the value is within its limit, a negative limit too; value / limit comes back from g, op and limit (share_of_limit).
- `test_every_constraint_of_a_project_goes_to_the_run_with_its_op`: A project of each physics with every one of its constraint quantities, at most and at least in turn: what Project.build gives the optimizer (formulation_params) holds each as [name, options, limit, op].
- `test_a_new_limit_starts_by_its_rule`: A quantity with a reference starts at 40% of it at most and 10% at least, and says both rules in its (i); one with a fixed start keeps it at most and has none at least.
- `test_mass_and_cost_start_in_the_designs_material_and_follow_it`: Mass starts at 40% of the part solid in the design's material (the lightest of a mix), and Cost at 40% of it solid in the design's material (the cheapest of a mix, by density times cost); changing to a single material moves a limit still at its start to the new one (restarted_limits), saying so, and leaves a typed limit as it is.
- `test_a_run_with_an_at_least_limit_is_recorded_and_shown_as_one`: A short U-bracket run with Mass at least its 10% start: the record keeps the op, value / limit comes back from the recorded g, the Summary shows "at least" and no Used for it, met agrees with value >= limit, Compare shows met alone, and the run's lines print ">=".

### `tests/problem/test_mesh.py`

The voxel mesh's surface: surface_faces gives every element face that belongs to one element only, with its outward normal and its area. On a box the faces cover the six sides exactly, their areas add up to the box's surface, the area-weighted normals add up to zero, as on any closed surface, and each side has the faces of its grid.

- `test_surface_faces_of_a_box`: Surface faces of a box.
- `test_a_box_has_the_faces_of_its_grid`: A box meshed as an nx by ny by nz grid of elements has 2 (nx ny + ny nz + nz nx) surface faces, each side's faces with that side's normal.

### `tests/problem/test_project.py`

The Project: saving and loading, the checks made before a run, the translation of objectives and constraints, and the boundary conditions build() puts on the mesh. The meshes are coarse U-brackets, so this runs in well under a minute.

- `test_an_older_file_that_kept_every_support_solid_is_refused_until_saved`: Keeping every element at a support solid is no longer a switch of its own (Keep solid on the support's selection does it). An older file with the switch on still opens, and the project's check refuses it, saying what to do, until it is saved; one with the switch off opens as any other. No file written has the switch. An older project script's line turning it on is an error saying what to use; turning it off changes nothing.
- `test_round_trip`: Round trip.
- `test_expressions`: Expressions.
- `test_validate`: Validate.
- `test_every_error_names_its_step`: Every error validate can give carries the step of the setup it is about, as data (Problem.gui_step): each place in validate, _missing_columns, vae_check and material_source_errors that adds an error makes a Problem with a step of GUI_STEPS, as does every Problem _stamp_errors makes (read from the code, so a new error cannot leave it out), and the errors of broken projects all have one.
- `test_build_coded`: Build coded.
- `test_conflicting_values_are_refused`: A dof given 0 by one support and 0.001 by another is an error, not a silent pick.
- `test_a_load_is_split_evenly_over_its_nodes`: A force on a face of a uniform mesh: the nodal forces add up to the total, and every node of the face carries the same share; nodes off the face carry none. A pressure likewise adds up to pressure times area.
- `test_free_rigid_body_motion_is_refused`: Held at one node and pushed sideways at the top, the bracket is free to rotate: build stops, before the VAE and the run.
- `test_script_round_trip`: Every project in projects/ exported to a script, the script run: the same project.
- `test_project_script_says_where_each_block_is_explained`: Above each block of calls in a project script, a comment names where those calls are explained.
- `test_project_script_names_its_physics_and_material_source`: A project script always says which physics and which material source the project has, the usual ones too (a structural project of several materials), so the script alone tells what problem it is.
- `test_window_picks_become_the_project`: Supports and loads picked on the part in the window (faces, a clicked node, a box) become selections and boundary conditions with the picking tools' own rules: the same fixed dofs and force vector as those rules gave (tests/data/gui_picks). Saved and opened again, the project is the same, and its script shows the whole problem.
- `test_only_mmto_files_load`: A project is a .mmto file with an mmto section; anything else is refused with the reason.
- `test_gui_sections_survive_a_save`: Saving from code keeps what the window wrote in the same file (its settings section).
- `test_schema_upgrades`: A file of an older schema is brought up to date on loading; a newer one is refused.
- `test_bc_kinds`: Every kind of boundary condition is a support or a load, says what it does, and labels its values.
- `test_every_kind_round_trips_through_a_project_script`: Each kind of support and load, added with add_bc, written into a project script and the script run: the same support or load. A wrong value is refused by add_bc as the project's check refuses it.

### `tests/problem/test_project_folder.py`

A project folder holds everything its project uses: create makes one with the STL copied in; pack copies an example's STL, database and network into a new folder, leaving the example as it was, and the copy checks and builds exactly as the example does; copy_in keeps a different file of the same name and never copies the same file twice; runs are listed newest first and only a run's folder can be deleted; a save from a script makes the same folder as the window, and leaves the project script alone. A few seconds.

- `test_create`: Create.
- `test_pack_an_example`: Save As: the STL into part/, the shipped database and its network into materials/; the example's file is unchanged; the copy has the same settings, the same checks and builds the same problem. Saved As again, its database's versions go with it.
- `test_copy_in`: Copy in.
- `test_runs_listed_and_deleted`: Runs listed and deleted.
- `test_a_save_brings_in_the_files_regions_name`: A region that names a file of its own (the airplane bracket's collar, inside_stl) has the file copied into part/ with the part, and points at the copy, so the saved folder runs on its own.
- `test_a_script_save_makes_the_same_folder`: Saving from a script copies the part and the database with its network into the project's folder, exactly as the window does at File > New and when a database is chosen; a second save copies nothing more; a script's save writes no project script.
- `test_project_script_left_alone_by_a_script_save`: The project script the window wrote is not rewritten by a save from a script (a script must not rewrite itself), and it rebuilds the project it was written from.
- `test_scale_part`: Scale part: every coordinate of the project's own part times the factor, exactly, written over the same file (one STL in part/ after three scales; by 2 then by 3 is by 6); the STL it came from is untouched; a project not in a folder of its own, and a factor that is not a number above 0, are refused.
- `test_new_project_has_the_physics_chosen`: New project lists every physics, the first chosen; the project is made with the one chosen.
- `test_runs_survive_moving_the_project_folder`: A run's record holds its paths relative to the project folder, so after the folder is moved (or renamed, or zipped to someone else) the run still rebuilds its mesh, redraws and exports. An older record, with absolute paths (written on Windows, as C:\...), finds them at the same place in the folder where it is now.
- `test_new_project_dialog_offers_every_physics`: New project dialog offers every physics.

### `tests/problem/test_selections.py`

Each selection type on a 10 x 10 x 10 grid of unit cubes, where the answer is known by hand: nodes at integer coordinates 0..10, element centers at 0.5..9.5. No solver is needed, so this runs in seconds.

- `test_every_field_says_how_it_is_given`: Each selection type declares a Field for each of its fields: the GUI's form is made from them.
- `test_every_selection_type_can_be_imported_from_lsr_mmto`: A project script (Project.save_as_script) imports each selection type it uses from lsr_mmto, by its class name in lower case, and lsr_mmto/__init__.py lists them by hand: a type missing from that list would break the script of every project that uses it.
- `test_box`: Box.
- `test_face_plane_near_face`: Face plane near face.
- `test_selections_by_name`: Selections by name.
- `test_cylinder`: Cylinder.
- `test_sphere_nearest_ids`: Sphere nearest ids.
- `test_element_size_lengths_and_rows`: '2h' is two element sizes; rows() takes the outermost node rows of a face.
- `test_lengths_in_element_sizes`: A length may be arithmetic in numbers, pi and h, the largest element size: kept as text when it has h and worked out on each mesh; without h it is a number at once. "1.5h" means what it always did.
- `test_stl_faces`: stl_faces asks the mesh for the nodes on those triangles (PyTO-AD's own lookup).
- `test_gui_box_and_point`: A box dragged in the GUI (six planes) and a picked point, on boundary nodes only.
- `test_combinations`: Combinations.
- `test_typed_lengths`: The selections whose coordinates or lengths are typed in model units (a scaled part leaves them as they are): points and lengths of field_kinds, not lengths in element sizes h, nor triangle numbers.
- `test_json_round_trip`: Json round trip.
- `test_function_and_inside_stl`: Function and inside stl.

## materials

Material databases: their attributes and units, the check of a database file, the latent dimension recommended for one.

### `tests/materials/test_database_versions.py`

The versions of a material database (materials/database_versions.py): numbered v1, v2, ... in a folder of the database's own beside it, each with the network trained on it; a state no version has kept (record), the one in use found by its contents, put back in the database's place, a note given, deleted (never the one in use, and no number given twice), renamed with the database, copied with it, and the dated versions of earlier releases numbered.

- `test_kept_numbered_in_use_restored_and_deleted`: The first record keeps v1 with its network; the same state again keeps nothing; a change made outside (Excel) is kept as v2, without the network trained on other numbers; the one in use is found by its contents; restoring v1 puts back its six materials and its network and makes it the one in use, every version staying; the version in use can't be deleted; a deleted number is not given again; the folder's list of databases does not show the versions.
- `test_a_colour_alone_is_a_new_version_and_a_network_trained_on_other_numbers_is_not_kept`: Every change, a colour too, is a version of its own; the network beside the database goes with a version unless its record says it was trained on other numbers, and a restored version with no network sets aside one that does not fit it.
- `test_a_note_a_version_from_another_file_and_an_included_database`: A note is kept in the version's name and can be taken away, its number staying; keep with source keeps that file (the included database a copy was made from) as the database's version; a database that comes with LSR MMTO gets no versions.
- `test_renamed_with_its_network_and_versions_and_a_copy_takes_them_along`: rename_database renames the database, its network and its versions together, and refuses a name another database has or had; copy_versions gives a copy (Save As) the versions and the last number given.
- `test_the_dated_versions_of_earlier_releases_are_numbered`: Versions kept by an earlier LSR MMTO (versions/steels <date>[ first].xlsx) are moved into the database's folder and numbered oldest first, with their networks.
- `test_nothing_is_lost_by_a_restore_a_retraining_a_broken_file_or_a_failed_rename`: A restore first keeps the database as it is now when no version has it (a change made in Excel while Versions was open); a network trained again on the same contents replaces the older one kept with the version in use, so a restore never brings back an older network over a newer; a database that cannot be read keeps no version and breaks nothing; a rename that fails part way is undone, the database staying with its network and versions; a network that fits no version is set aside under a name of its own.

### `tests/materials/test_databases.py`

Material databases: the attribute registry, the check of a database file, the latent dimension recommended for it, the Excel template, and what a Project does with them. Small files written on the fly; a few seconds.

- `test_shipped_databases`: The four databases pass with no errors and no warnings, and the recommended latent dimension is the one each network was trained with.
- `test_decode_undoes_normalize_by_the_attributes_flag`: The network's output is turned back into values by the same flag normalize scales by: a column whose attribute is log_scaled comes back through 10 **, a linear one directly; each round trip gives the database's values to 1e-10. Nothing is named by hand: a column named E0 is linear like any other.
- `test_errors_name_the_cell`: Errors name the cell.
- `test_warnings_and_aliases`: Warnings and aliases.
- `test_display`: The Display sheet: each material's colour and label, matched by name; empty means the default colour and the material's name. Two materials shown the same way are an error.
- `test_shipped_display`: BM1 draws its materials in the colours their names say; the airplane bracket's alloys are labelled A to T, as in the paper.
- `test_template`: Template.
- `test_units`: Units.
- `test_benchmark_unit_warnings`: The benchmarks declare the units they are set up in; the paper's databases mix units, and each mismatch is reported with what it does to the results.
- `test_a_mass_limit_far_below_the_lightest_solid_part_is_warned`: A Mass limit below 1 % of the part solid in the lightest material is a warning (not an error), with the share it allows to 2 significant digits: the U-bracket (volume 0.00875) with Beam_8Mats (lightest density 1500) and Mass <= 0.0035 kg allows 0.0035 / (1500 * 0.00875) = 0.027 %.
- `test_project_uses_the_check`: Project uses the check.
- `test_vae_check`: A database's network is the .nt next to it with the same name. A trained network must fit the settings and the database; a run never trains into the shipped folder; replacing a network warns.
- `test_a_trained_network_sets_its_own_shape`: A database chosen with a trained network of another shape than the settings say: the settings that give the network its shape follow the network, the change is said, and validate finds nothing wrong. A later explicit change is still checked; a project read from its file keeps its settings as saved.
- `test_a_hand_set_network_is_cleared_on_another_database`: A VAE file set by hand (settings.vae.saveNet) belongs to the database it was set for. The checks refuse a network of another number of columns, but not one trained on another database of as many columns, so choosing another database clears it, and says so; the network next to the new database is then used. Setting the same database again keeps it.
- `test_read_once_per_file_version`: A database is read once for each version of its file: looked at again it is not read again, a copy is handed back each time, and once the file is saved again it is read again.
- `test_import_table`: A table made elsewhere: the first column the names, row 1 the properties under any of their names (with a unit in brackets or not), an optional units row as in a database of LSR MMTO; columns that are not a known property, a second column of the same property and units a property does not take are reported, not read.
- `test_unit_kinds_of_a_physics`: A physics' own kinds of quantity (UNIT_KINDS on its front page) are in the unit tables; a kind given differently by two physics (other units, or the same units but another unit in a unit system), or a unit system's unit that is not one of the kind's units, is refused.
- `test_columns_found_by_what_they_measure`: The mass of a design is made from the column that measures density, and the per-iteration materials line prints the columns whose attribute has a label there, in their order, whatever the columns are named.
- `test_a_blank_unit_is_no_unit_for_the_network_too`: A blank unit cell (BM1_3Mats_2Attr's Poisson's ratio, a ratio) is no unit, None, for the material network as for the database's own reader, so the network's error table shows nothing there, not "nan".

### `tests/materials/test_network_stamp.py`

The record a trained network keeps of the database it learned from (materials/network_stamp.py): written when a network is trained, and checked before a run. Any change to a cell of the first sheet, the 15th significant digit of a number too, stops the run until a new network is trained, and so does a record that can't be read; a change of colours or labels only, a conversion of units, and a network saved before networks kept a record are used after one yes (Project.confirm_network), and a script's run refuses until then; the networks that come with LSR MMTO match their databases and are never stamped again.

- `test_every_network_that_comes_with_lsr_mmto_matches_its_database`: The four included databases' networks and the thermal example's carry a record that matches their database.
- `test_a_trained_network_keeps_the_database_it_learned_from`: Training writes the database as it was read into the network's file; loading reads the weights back.
- `test_the_smallest_change_of_a_number_stops_the_run`: Saved again unchanged: the same. The 15th significant digit of one number, or spaces around a name: no run, the cell named, and nothing to confirm.
- `test_colours_or_labels_and_a_network_with_no_record_are_used_after_one_yes`: No error in the project's check, but a script's run refuses until confirm_network, which stamps the network (its weights unchanged); confirming a network that matches does nothing.
- `test_a_changed_database_that_comes_with_lsr_mmto_is_not_confirmed`: Its network is never stamped again: the file is put back, or copied and trained as the user's own.
- `test_converted_units_are_asked_about_once_and_a_changed_database_stays_changed`: Convert to ... units on the Airplane bracket (mmKS, its database in kg/m^3, Pa and USD/kg) in a folder of its own: the network keeps its weights and asks once, naming the conversion. Converted after a number changed, the network still refuses.
- `test_single_material_and_a_network_to_train_are_not_checked`: No trained network in use (single material, or Use the trained network unticked): nothing to check.

### `tests/materials/test_pca.py`

lsr_mmto/materials/pca.py gives what scikit-learn's PCA (1.9.1) gave: the reference numbers below were worked out with it, before it was taken out of the requirements.

- `test_every_shipped_database_as_before`: Every shipped database as before.
- `test_the_2d_projection_as_before`: The 2d projection as before.

## runs

A run or an analysis goes through and saves what it should, and its numbers do not drift.

### `tests/runs/test_benchmark_projects.py`

The four benchmarks (projects/*.mmto) against what their coded versions gave before they were retired: tests/data/benchmarks.npz holds the fixed dofs, imposed values, forces and keep-solid elements, tests/data/benchmarks.json the mesh size, the optimization parameters, the VAE settings and the run settings.

- `test_compare_problem_l_bracket`: The L-bracket project builds the same held dofs, imposed values, forces, keep-solid elements and mesh as its retired coded version did.
- `test_compare_problem_u_bracket`: The U-bracket project builds the same held dofs, imposed values, forces, keep-solid elements and mesh as its retired coded version did.
- `test_compare_problem_beam`: The beam project builds the same held dofs, imposed values, forces, keep-solid elements and mesh as its retired coded version did.
- `test_compare_problem_airplane_bracket`: The Airplane bracket project builds the same held dofs, imposed values, forces, keep-solid elements and mesh as its retired coded version did.

### `tests/runs/test_every_physics_runs.py`

Every physics in available_physics.MODULES runs: its tiny example passes the project's check, goes through a short run of 2 iterations, is saved, reads back with its final check's results, and its objective is a number at every iteration. This file names no physics: each gives its own example, as tiny_example in its own test file, tests/physics/test_<physics>_physics.py, where everything about testing that physics lives. A physics added to MODULES is run here at once; until its test file gives a tiny_example, this test fails saying what to add.

- `test_every_physics_runs_its_tiny_example`: The physics' tiny example passes the project's check, so a failure of its run is the run's, not the setup's. Its run (tiny_run) was saved; read back, it is a run of that physics with every result its RESULT_FIELDS lists, each a finite number, and a finite objective at the start and at each of 2 iterations.
- `test_a_physics_with_no_test_file_or_no_tiny_example_is_told_what_to_add`: A physics whose test file is missing fails here with the file to add and what to put in it; one whose test file has no tiny_example fails saying to add one.

### `tests/runs/test_final_check.py`

The final check at the end of a run (compliance, stresses, failure factor of the continuous, binarized and snapped designs) must analyse the design with the same materials as the optimization: each element's Young's modulus from the network, and its Poisson's ratio either from the network or, when the project says so, the constant one. The run saves the last check's displacement, x, y and z at each node, as the solver gave it.

- `test_constant_poissons_ratio_is_used`: Constant poissons ratio is used.
- `test_poissons_ratio_from_the_network_is_used`: Poissons ratio from the network is used.

### `tests/runs/test_no_latent_points.py`

A run whose material source has no latent points (latent dimension 0): the design is the densities only, and every step of the run takes it, from the start design and its bounds through the iterations, the final check and the save. The source here is the database's own values with one material for every element (materials/material_source.py, DatabaseMaterials), put in where the run makes its source (optimization_loop.material_source).

- `test_a_run_with_no_latent_points_goes_through`: The U-bracket on a coarse mesh, its distance penalty and latent smoothing switched on as usual: the run asks about designs of the densities only (one number per element), leaves out both penalties (they read the latent points) and says so, goes through and saves; each element has MATERIAL's database values, the saved run has latent dimension 0 and no history of either penalty, and every element's material is MATERIAL.

### `tests/runs/test_regression.py`

The numbers a run makes must not change by accident. The first iterations of every shipped project are compared with stored values: the materials' latent points, and the objective and every constraint at each iteration, to 1 part in a million. The L-bracket, the U-bracket and the rectangular beam run exactly as their projects define them; the beam and the airplane bracket on a coarser mesh (30000 dofs) to keep the test short, and the airplane bracket a second time on the iterative solver (DPCG with deflation), which its own mesh would use, by lowering the direct solver's dof limit for the test.

- `test_first_iterations_unchanged`: First iterations unchanged.

### `tests/runs/test_run_project.py`

Running a Project: its settings cover every setting of _run_topopt; the run record keeps the settings as they were given, says when a setting was switched off during the run, and stores the project so it can be rebuilt. A run from a script goes to the project folder's results, as one from the window does, and its folder holds exactly the run's three files.

- `test_settings_cover_run_topopt`: The loop reads each setting as s.<name>: every name it reads (anywhere in the loop's file: its setup, the steps of an iteration, the optimizer; and in the penalties' weights) is a setting of SETTINGS, the run gets every one, and _run_topopt has no setting of its own (no default that could differ from SETTINGS).
- `test_progress_and_stop`: progress() hears every iteration; should_stop() ends the run there, and the design reached so far is still post-processed (the GUI's Run and Stop).
- `test_simp_continuation_raises_each_penalty_of_every_physics`: Each SIMP exponent a physics declares (SIMP_PENALTIES) starts at its start, or at its start with continuation when SIMP continuation is on, and each step of continuation raises it by 0.25 up to its largest; nothing in the core names a physics. A U-bracket run with continuation on for 11 iterations raises the stiffness's once, 1.5 to 1.75, where the structural solve reads it as well as in its record, and records no exponent structural does not use.
- `test_the_pnorm_exponent_is_the_runs_own`: The p-norm exponent is the run's setting: the stress quantities read it from the run's settings, and building a project never writes it into PyTO-AD's global, so two projects in one program keep their own.
- `test_optimizer_failure_names_the_constraint_over_its_limit`: When the optimizer fails while a constraint is far over its limit (a limit that cannot be met from the starting design), the error says which constraint, by how much, and keeps the optimizer's own message.
- `test_a_run_with_a_mass_fraction_constraint`: A short run with Mass fraction of the lightest <= 0.4 as its constraint goes through: every iteration's constraint is a finite number, the run keeps the constraint with its material word, names it as the app does, and the project it rebuilds from the record is the one that ran.
- `test_unknown_and_missing_settings`: Unknown and missing settings.
- `test_run_finishes`: Run finishes.
- `test_record_keeps_settings_as_given`: z smoothing was on at the start and switched off at iteration 2: the record says both.
- `test_record_has_provenance`: The run record says which code made it.
- `test_record_rebuilds_the_project`: Record rebuilds the project.
- `test_record_stores_display`: The run takes its colours and labels from the database; the record stores them.
- `test_script_run_lands_in_the_project_results`: run(p) from a script, with no folder given, writes the run where the window does: results/<date and time> in the project's folder, listed by the window's Results, and returns that folder; nothing goes to the current folder. The run's folder holds its three files and no copy of the project; moved with its project folder (zipped and sent), the run still finds its part.
- `test_vae_training_guards`: Training never writes into the shipped folder; a network being replaced is kept as <name>.previous.nt; a network that reproduces the database less closely than the project's network_error_warning is kept and the run warns, by property, and goes on (on_network_error, the GUI's question, can end it); the setting moves the limit.
- `test_network_check_leaves_the_encoder_as_it_was`: Checking a network read from its file leaves its encoder giving each material's own latent point (not a random sample around it): the same points every time, each nearest its own material.
- `test_shipped_networks_are_usable`: Every shipped network reproduces its database well within the default warning limit.
- `test_training_stop`: The material network's training: unchanged when nothing asks it to stop (the same seed gives the same network, to the last bit, with or without a should_stop that says no); asked to stop, it raises TrainingStopped and saves no network.
- `test_a_solve_converged_unless_it_used_up_its_iterations`: CheckedDeflationSolver's verdict is DPCG's own: a solve converged unless it used up its iterations, and one that used them up but landed within its tolerance on the last step converged too. Its error is kept either way, a NaN one as NaN (a breakdown is never shown as exact).
- `test_every_field_of_every_physics_has_a_deflation_space`: Above the direct solver's limit every system a physics solves is solved by DPCG, checked, only if its deflation space fits it: PyTO-AD solves a system that no space fits by plain CG, which is not checked. So the space a physics builds (its deflation_space) covers each of its fields, one system of (unknowns per node) x nodes each: the solver's own W or one of its added spaces (CheckedDeflationSolver.space_for). A physics whose space leaves a field out (here a made-up one: structural's space, and a temperature too) is refused before its solver is made, naming the field.
- `test_a_solve_that_does_not_converge_is_marked`: The iterative solver (DPCG, forced here by lowering the direct solver's limit) records whether each solve converged: a run whose solves all converge records none that failed and its Summary says nothing of it. When the final check's solves of the binarized designs stop at their limit (here: returning zero), their numbers are still printed and saved, but the run record names them with their error, the log says they and the results the run saves are not reliable, the Summary has a Final check row naming them, its (i) gives each error, and Compare keeps the row beside a run without it. An analysis whose solve stops short has a Solver row. The .npz keeps the same record (its meta_json), and the lines said in red at the end name the same solves. The solve inside is replaced by an exact one, so the test checks the record, not DPCG itself.

### `tests/runs/test_single_material.py`

A project of a single material, every element the same material, the design the densities only: one database material (material_source "one_database_material": pinned_material, or the database's only material, at its database values, no network read or trained), or the network pinned to one database material (pinned_material, that material's properties as the network gives them). Each runs through to the save; the project saves and reads back its choice, an older file reads as the network with no pin, and the project's check refuses what cannot run.

- `test_one_database_material_runs_with_no_network`: A database of one material and material_source "one_database_material": the project passes its check, the run reads and trains no network (no .nt is written beside the database), leaves out the penalties of the latent points, and every element has the material's database values; the saved run has latent dimension 0.
- `test_one_material_of_a_database_of_several`: Single material with the 6-material database itself: the material named (pinned_material) is every element's, at its database values, with no network read or trained; nothing needs deleting from the database.
- `test_the_settings_of_the_latent_points_change_nothing_with_one_material`: Every setting that says it reads the latent points (Setting.reads_latent_points, which the Settings panel greys out with one material) changed from its default: a run of one database material, and a run of the network fixed to one material, are the same, to the last digit.
- `test_a_material_name_with_spaces_around_it`: A database whose material's name has spaces around it (a cell typed "C "): the network reads the name as the project's check does, without them, so the network fixed to "C" runs and gives that row; and the record of a run of one material says it left out the penalties and, with no network, has no network settings.
- `test_the_network_pinned_to_one_material`: pinned_material names a material of the 6-material database: every element has that material's properties as the shipped network gives them at the material's own latent point, the design is the densities only, and every element's material is that one, in the saved run and in what each iteration tells the window.
- `test_the_choice_is_saved_and_an_older_file_reads_as_the_network`: material_source and pinned_material are saved and read back; a file saved before them reads as the network with no pin; the project script always sets the material source, and the pin when there is one.
- `test_what_cannot_run_is_refused`: The project's check refuses: a material source it does not know; a pin that names no material of the database, with either source; single material with no material chosen from a database of several (one material, none chosen, is that one); the network with a database of one material; and, for a single material, a quantity that reads the latent points (material volume fraction).

### `tests/runs/test_solid_part_analysis.py`

Analyse (optimize/solid_part_analysis.py, the button beside Run): the solid part solved once in one material of the database, with that material's database values.

- `test_compliance_is_a_direct_solve_of_the_solid_part`: The analysis in each of two materials gives the compliance F . u of a solve with x_phys = 1 in every element and that material's Young's modulus and Poisson's ratio from the database, to 1e-10, and that solve's displacement u at each node, x, y and z, as its result.
- `test_a_project_that_cannot_run_is_not_analysed`: A project that cannot run is not analysed.
- `test_not_saved_writes_nothing_saved_is_one_entry_that_loads_back`: Not saved writes nothing saved is one entry that loads back.
- `test_stop_ends_it_and_nothing_is_written`: Stop ends it and nothing is written.
- `test_a_quantity_of_the_latent_points_is_not_given`: The L-bracket's constraints are material volume fractions, worked out from the latent points: the analysis gives the objective; its record holds no value for the constraints, and its Summary and Compare leave them out. Compared with a run that gives them, only the rows both have are shown: the constraints and what only an optimization has (grey elements, latent dimension, iterations) are left out.
- `test_analyse_in_the_window`: Analyse beside Run, with the same look; its window; not saved: shown first in Results, nothing written; saved: listed among the runs with its label; a run and an analysis compared: no convergence or latent space to plot, said in their place. Analyse this design in Results, for a run only: the analysis of its design is saved, shown, and ticked with its run, ready for Compare.
- `test_analyse_this_design`: Analyse this design: a finished run's design as built (density above 0.5 solid, each element in its nearest database material), solved once under the project's supports and loads now. With the design the final check analysed (its solid elements) and the materials it used (the network's values of the real materials, written into the run's database), it gives the run's own final check: the stress and the displacement of every element and node, to 1 part in 10,000 of the largest. Twice the load gives twice the displacement; a project whose mesh is not the run's is refused, saying what differs, and so is a database changed since the run, naming the cell; saved, its record says which run it analysed. A database renamed since the run (Rename... in Materials) is found under its new name when its contents are the run's.

### `tests/runs/test_stl_export.py`

The per-material STL export (lsr_mmto/results/stl_export.py) on shapes whose answer is known: three slabs around a spherical hole, a plate one element thick (as the L-bracket and the U-bracket), a sheet of one material one element thick between two others, scattered specks, a round cylinder. Every piece is closed, touches no other body at a single point, and the pieces add up exactly to their whole; volumes are near the design's, and the cylinder comes out round. Contacts along an edge or at a corner are always resolved. The whole part (design.stl) is the material regions' surface where the smooth one cannot be closed: a plate two elements thick, an analysis. About half a minute.

- `test_slabs_and_a_hole`: Slabs and a hole.
- `test_plate_one_element_thick`: Plate one element thick.
- `test_sheet_one_element_thick`: Sheet one element thick.
- `test_specks_and_contacts`: Specks and contacts.
- `test_round_cylinder`: Round cylinder.
- `test_whole_part_from_the_material_regions`: A plate two elements thick whose void surface cannot be closed (a hole of 6 x 8 elements and a hole of one element two elements from its corner, as in a U-bracket run): design.stl is the material regions' outer surface, closed and near the design's volume, and the report says so. A design whose smooth surface closes keeps it, as written before.
- `test_whole_part_of_an_analysis`: An analysis has every element solid, so there is no void surface to take away: design.stl is the material regions' outer surface, closed, with the part's volume.
- `test_a_design_with_no_material_says_so_and_writes_nothing`: A design with no material anywhere (a run stopped after a few iterations): a plain message, nothing written. VTK's surface of an all-void grid ended the whole program; the export runs in its own process here, so a return of that crash fails this test only.

## gui

The window and the desktop start, off-screen: the panels, the buttons, a project taken through the window the way a user would.

### `tests/gui/test_background_work.py`

gui/background_work.py: jobs run one after another on the same thread; an error in one is reported as a thread's error would be, and the next job still runs.

- `test_jobs_share_one_thread_and_an_error_does_not_stop_the_next`: Jobs share one thread and an error does not stop the next.

### `tests/gui/test_crash_report.py`

The crash report (gui/crash_report.py): what normal use, an error in a button, an error in a background thread and a freeze each leave in the file, and that the file stays short.

- `test_normal_use_writes_only_actions`: Normal use writes only actions.
- `test_an_error_in_a_button_is_written`: An error in a button is written.
- `test_an_error_in_a_thread_is_written`: An error in a thread is written.
- `test_a_freeze_writes_every_stack_then_how_long`: A freeze writes every stack then how long.
- `test_the_file_stays_short`: The file stays short.

### `tests/gui/test_database_versions_gui.py`

The versions of a database in the window, off-screen: the editor's (modified) title and its one question (changes not saved, on closing); every Save kept as a version without asking, for a database made, imported or copied from an included one too; Versions... with "(in use)", Restore without a question, Rename... of a note and Delete (never the one in use); and in Materials: Rename..., Import copying a database in, and a change made in Excel kept as Materials opens. The questions are answered by the test (database_editor.ask), never shown.

- `test_changes_are_marked_and_closing_with_them_asks`: Opened: no (modified). A value changed: (modified) in the title. Closing then asks: Cancel keeps the editor open, Discard changes closes it with the file as it was; Save... saves, keeping v1 (as it was) and v2 (as saved).
- `test_every_save_is_a_version_without_asking`: Each Save that changes the database keeps it as saved as the next version, asking nothing; the state it had before is v1. A Save with nothing changed keeps nothing new.
- `test_a_new_database_and_a_copy_of_an_included_one_get_their_versions`: Saved as new (imported here), its v1 is itself; an included database edited, saved as a copy in the folder: the copy's v1 is the included database as it came, with its network, and v2 the copy as saved.
- `test_versions_marks_the_one_in_use_restores_renames_and_deletes`: Versions...: the versions newest first, "(in use)" beside the database's own. Restore asks nothing, keeps every version and closes the window, the restored one now in use; the one in use can't be deleted or restored; Rename... gives a note; Delete asks first, and No keeps the version.
- `test_a_save_that_changes_the_file_keeps_it_as_it_was_first`: A database with a column LSR MMTO does not know: the editor shows (modified) from the start, since Save writes it without that column, and that Save keeps the file as it was (v1, with the column) and as saved (v2). Save... from the close question saves at once.
- `test_an_import_asks_before_it_is_thrown_away_and_a_new_name_skips_old_versions`: An imported table not yet saved: closing asks first. A new database's name skips one whose versions are still there after its file was deleted, so it does not take them on.
- `test_materials_keeps_versions_renames_imports_and_says_what_was_restored`: In Materials: Versions... and Rename... only for a database of the project's own; opening it keeps the database as a version, and a change made in Excel since as the next; Rename... renames the database, its network and versions, and saves the project with the new name; Import copies a database of LSR MMTO into the project's folder; a restore says which version is now used.

### `tests/gui/test_gui.py`

The LSR MMTO window, off-screen: the names it takes from PyTO-AD exist; each project opened and saved with no edits is written back unchanged (line endings aside: Windows writes CRLF, which git does not count as a change); only .mmto projects open; New, Save and Project.load agree; the title marks unsaved changes and closing asks; the unit system goes both ways between the project and the Units dialog; selections made of triangle numbers are found before the STL is changed. The Materials, Formulation and Settings panels: OK with no edits changes nothing, Cancel drops edits, every setting has a row, quantities the database cannot give are greyed out with the reason, a new database brings its VAE settings. Supports and loads: a region typed or filled in is the selection a script makes, a code line is never run, picks applied in PyTO-AD's Structural Loads window are adopted, the preview counts the run's nodes and runs the rigid-body check. About a minute.

- `test_what_it_takes_from_pyto_ad`: What it takes from pyto ad.
- `test_open_and_save_unchanged`: A project opened and saved with no edits is written back unchanged. A copy of an example takes into its folder, at its first save, what it uses (the part, the database and its network), as every save does; from then on it is written back unchanged.
- `test_a_setting_missing_from_the_file_is_no_change`: A project file written before a setting existed opens with that setting at its default and counts as unchanged; the next save writes the setting into the file.
- `test_only_mmto_projects_open`: Only mmto projects open.
- `test_new_save_load`: New makes a project folder with the STL copied in, saved at once; Save As copies everything into another folder and takes the new name.
- `test_save_as_brings_in_region_files`: Save As of a project whose region names a file of its own (the airplane's collar, inside_stl): the file is copied into part/ of the new folder, and the saved project points at that copy.
- `test_project_script_follows_window_saves`: File > New writes <name>_setup.py in the project folder; every save in the window writes it again to match the project; a save from a script leaves it alone.
- `test_unsaved_changes`: Unsaved changes.
- `test_units_both_ways`: Units go from the project to the dialog and back; Apply writes the part's extent, the ruler and the Snap steps again in the new length unit.
- `test_triangle_selections`: Triangle selections.
- `test_replace_part`: File > Replace part changes the project's STL (copied into the part folder of a project of its own), draws the new part and drops the design of the old one. There is no Geometry button.
- `test_scale_part`: File > Scale part: the window shows the size the part will have; the selections typed for the old size (a Box, a Cylinder) are named first, not picked faces or a length in element sizes; a project not in a folder of its own is saved as one first and only its part/ is written, every coordinate exactly times the factor; scaling again multiplies again, one STL in part/; the extent follows; Cancel changes nothing.
- `test_start_page`: With no project open the start page covers the 3D view, and the side buttons, Run, Export STL and Part / Design are greyed with a tip; opening a project hides the page and they work. On a laptop screen (1366 x 768) the page stays whole: nothing in it cut short, overlapping or outside the 3D view.
- `test_side_marks`: Units, Materials, Supports and loads and Formulation: a tick once set up and the run's check finds nothing wrong there, a red mark for an error of that step, plain while not set up (never red then); a warning marks nothing; the other buttons have no mark. The marks follow every change.
- `test_the_materials_window_works_out_the_database_pca_once_and_prints_nothing`: Opening the Materials window: the PCA of the database (its effective dimension) is worked out once, for the recommended latent dimension and the network's check alike, and nothing goes to the terminal. It was worked out by each of them and printed by the encoder, once more for each project check.
- `test_panels_with_no_edits_change_nothing`: OK with no edits, on every project and every panel: the project is as it was, to the last digit (the U-bracket's mass bound is 0.0035000000000000005).
- `test_cancel_changes_nothing`: Cancel changes nothing.
- `test_settings_panel_covers_every_setting`: Settings panel covers every setting.
- `test_settings_panel_edits`: Settings panel edits.
- `test_formulation_panel`: Formulation panel.
- `test_formulation_mass_fraction_row`: A Mass fraction row reads 'Mass fraction of [material] <= [limit] of a solid part in that material': its choices are the materials' labels, then the heaviest and the lightest; it is saved with the material word, opens again with it chosen, and is named in words in the messages. A material volume fraction row keeps only the materials.
- `test_a_new_constraint_row_starts_at_its_quantity_limit`: A new row starts at the limit its quantity's rule gives: at most 40% of its reference (0.4 for the fractions, and for Mass 0.4 times the part solid in the lightest material, far above the low-mass warning), 1 for the failure factor; a quantity that declares none starts at 0 as before. Switched to at least (>=), a row still at its start moves to 10% of the same reference, and back again; the (i) says the rules, a typed limit too. A limit typed by the user is kept when the quantity or the direction changes, and an existing row keeps its own.
- `test_limits_still_at_their_start_follow_the_materials`: Materials changed with OK: a limit still at its start (40% at most, 10% at least) moves to the start for the design's material now, also when Edit... changed the database's numbers while Materials was open, and the messages say so; a typed limit stays. The window's Compare plot draws an at-least constraint as value / limit from its g, met above the line.
- `test_materials_panel`: The Database list holds the project's own databases (its materials folder, and its database when it lives elsewhere); those that come with LSR MMTO are under Included databases and show as '<name> (included)' once chosen. With no database the report says how to get one and Edit and Open in Excel are off. The database's file saved again while the window is in the background is read again when the window is active again.
- `test_import_button_opens_the_file_window`: Import... clicked asks for the file to import (the click once handed Qt's checked, False, over as the file and the window never opened).
- `test_no_button_hands_its_checked_to_a_method_that_takes_an_argument`: A button's clicked (or an action's triggered) connected straight to a method that takes an argument hands the method Qt's checked (False) as that argument. Every such connection in gui/ goes through a lambda.
- `test_database_editor`: New database and Edit: one row per material (name, optional shown-as, colour, numbers), the units under the headers, bad cells marked and Save off until none; a block pasted from Excel; always saved in the project's folder (no folder to choose); a shipped database saved as a copy; a network set aside when the numbers change, kept when only a colour does.
- `test_typing_a_material_name_types_it_once`: A letter typed in the Material column, which stays in place when the table scrolls sideways, opens one editor and goes into the cell once (B stayed B, not BB), as one typed in any other column; Escape still closes the editor, after asking about the changes not saved (answered Discard changes here).
- `test_database_editor_look`: The New database editor: the units row is part of the headers' band (no "Units" text, as tall as a material row, each unit's drop-down inside it with room for its text, "no unit" for Poisson's ratio); a blank form shows no red, a row typed in does; each column as wide as its title up to 200 px, a longer title cut short with the whole of it as the header's tip; the Material column stays in place when the table scrolls sideways; the four buttons under the table share its width equally. Closed with its changes discarded.
- `test_add_properties_window`: A new database for a project starts with Density, the properties the project needs and its physics' optional ones. Add property opens the Add properties window: with a project, the properties it needs (its physics' and those its objective and constraints read) and every other registered property (worked out from the registry, so a new physics' properties are expected too), each part alphabetical; those in the table ticked and greyed; those ticked are added, and OK says how many; the search box filters. From elsewhere (no project) everything is one list.
- `test_import_database`: Import...: a table of materials made elsewhere (here a CSV) becomes a database. Columns named as a property LSR MMTO knows are read (a unit in brackets after the name too), the others are listed in one message, and the editor opens with what was read, a value that is not a number shown as a cell to correct; once saved in the project's folder the project uses it. A file that is already a database of LSR MMTO is copied into the project's folder with its trained network, under a name no database there has (here BM1_3Mats_2Attr 2: the project's own copy of the included one is there), and no editor opens; the file itself is not changed.
- `test_edit_a_database_with_an_unknown_column`: A database with a column LSR MMTO does not know (E0): the check says so in plain words, and Edit... opens it with that column left out and named in a note, and saves it without the column.
- `test_error_in_the_app_keeps_it_running`: An error in what a button does is shown and written to the log; the app keeps running.
- `test_vae_tab`: With a trained network in use the VAE settings are its own and greyed out; the tab says which network the database has and shows the VAE checks as you edit. (No window has the "?" button: see test_gui_look.test_no_window_has_the_help_button, which turns it off as the app does.)
- `test_code_line`: A code line gives the same selection as a script; anything that is not a selection is refused.
- `test_arithmetic_in_number_boxes`: Every number of the Selection form, the Code line and the support or load form takes plain arithmetic (38/2, 0.5*76, 2*pi*5, inside a point too); the selection or value keeps the number.
- `test_region_and_bc_forms`: Region and bc forms.
- `test_supports_panel`: Supports panel.
- `test_picks_are_adopted`: An Apply in PyTO-AD's Structural Loads window adds to the project what loading a file with the same picks adds; a torque is refused; PyTO-AD's lists are left empty.
- `test_pick_window_clear_picks`: The Pick window has no Box select nodes; Clear picks (and a right click) removes every face and node picked; picks not applied go when the window closes. A project with a Dragged box opens, shows and runs.
- `test_docked`: Supports and loads and the Pick window sit on the right edge of the window, not over the 3D view; a second click raises the open one; Close (or a new project) removes the strip.
- `test_add_flow`: Add offers both ways; an empty list says what to do; New shape makes a selection without leaving the dialog; a pick takes the list's place and Apply brings the list back with the new entry selected; each warning is said once.
- `test_scale`: The Ruler tick shows and hides the scales; the region form says how big the part is; the point under the mouse is read off the part.
- `test_snap_steps_follow_the_part`: Snap offers off and three round steps (1, 2 or 5 times a power of ten, at or below) of a thousandth, a hundredth and a tenth of the part's largest extent, with the length unit; they change with the part, and a point snaps to the step chosen.
- `test_selection_form_extent_follows_the_part_and_units`: The Selection form's "The part spans ..." line is written again when the part is replaced and when the units change, while the form stays open.
- `test_region_preview_clears`: In the region form, Clear takes the preview off the part, any edit does too, and so does closing.
- `test_inside_stl_move_by`: Inside an STL has a Move by row (x, y, z with the length unit and an (i)); the selection keeps the move, the project file and the code line only when it is not zero, and the preview draws the moved shape.
- `test_click_a_hole`: Click a hole in the Cylinder form, then a click on the inside wall of a hole: the wall is grown as a Facet pick grows, and the form gets the hole's centre, axis, radius and ends; Tolerance Automatic takes the wall's nodes half an element outside the radius. A flat face fills nothing.
- `test_filled_numbers_to_the_parts_precision`: The numbers Click a hole fills in keep the part's own precision: under 1e-6 x its largest extent is 0; on a 200 mm part the values are as they were.
- `test_click_a_hole_on_the_rotor`: Click a hole on a brake rotor saved in metres: every triangle of the inside wall of the bore, the six bolt holes and the six rim notches gives its centre and radius (within 1 %); a bolt hole, under two element sizes in radius on the run's mesh, gets the note that it is only a few elements across, the bore does not; Surface nodes only is ticked, and the bore with it has no node off the mesh surface (without it, 12); a click on a flat face says so; the outside of the rim fills a tube about it, which takes the nodes of the rim, not the whole part.
- `test_empty_inner_radius_is_zero`: An empty Inner radius is its default, 0 (a solid cylinder), on the form and the code line; an empty Radius, which has no default, is still refused.
- `test_clear_empties_the_form`: Clear in the Selection form puts every field back to its default and removes the form's preview.
- `test_enter_previews`: Enter in the Selection form and in the support or load form checks what was typed and shows it on the part; the window stays open, a mistake shows in its message line; only OK saves and closes.
- `test_pages`: Forms opened from Supports and loads are pages of its strip, one at a time: Enter by hand, then New selection over it; OK goes back one page each time. Combined selections follow their parts by name.
- `test_keep_solid_on_picked_faces`: Keep solid on a selection that picks nodes only (rows, picked faces): the elements touching it.
- `test_outline_one_row_measure`: A selection's shape is drawn with its nodes; clicking a row shows only that one; Measure gives the distance between two clicks; the mouse lookup never runs while a button is held or one is running.
- `test_a_run_whose_solver_did_not_converge_says_so_in_red`: When a run ends with solves that did not converge, Current run says so in red under its status line, one line each (tables.solver_warning), and the messages panel says it too; a run whose solves all converged, or one that could not be read, adds nothing.
- `test_run`: Run: a real short optimization from the strip; live plot and design; results in a folder of its own with the tabs at the end; Stop ends a run early and keeps its design; nothing closes mid-run.
- `test_network_warning_continue_or_cancel`: A network that reproduces the database less closely than the project's warning limit: before the optimization the window asks, with each property's error; Cancel leaves no run folder behind, Continue runs and keeps the errors with the run.
- `test_export_and_run_script`: File > Export as script writes the project as Python in its folder, after a save when there are changes; the script edited and run with File > Run script opens the project it saved; a script that fails, or saves nothing, says so and leaves the project as it was; the script runs from any folder.
- `test_latent_views`: The latent space in 1, 2, 3 and 4 dimensions: the run's tab, Compare's and the live one draw it; 1D as counts along the line with every name readable, 4D through a fixed projection whose title says for how many elements it shows the right nearest material; the live axes only widen.
- `test_snap_copy_outline_colour_new_database`: Snap rounds Measure and a copied point; a click on the part with a form open copies the point; outlines are magenta whatever the place.
- `test_convert_units`: Convert to <units> units (Materials), on a copy of the Airplane bracket project (mmKS, its database in kg/m^3, Pa and USD/kg): the button shows only while the units differ; the window lists each column with its factor and the limits; OK writes the project's own copy with the values times the factors exactly and the limits converted, saves the project, and leaves no unit warning, the copy as it was before kept as v1 and as converted as v2, the one in use; the network keeps its weights, still gives the materials back as closely, and is asked about once before the next run, the conversion named; the shipped database is untouched. Unticked, the limits stay.
- `test_examples_open_as_templates`: An example opens as a template: Save asks for a project folder of one's own and copies everything into it, and the example's file never changes. In a project folder, a database chosen (a shipped one too) is copied into materials/, and New database writes there.
- `test_part_facts_pca_live_colours_training`: The part's facts behind the (i), not over the 3D view; the PCA breakdown behind the Materials panel's (i), and its Values | Network error switch; the live design by density; the training's epochs in the Run status line.
- `test_results_compare_delete`: Results: the project's runs, newest first; a run shown as each of Material, Density, Stress, Displacement (smooth over the faces, its magnitude or x, y, z, and the deformed shape at a scale that moves it 5 % of the part at most) and Solid; two runs compared side by side; another place's run opened for looking only; Delete, in Results and in the Run strip, removes a run's folder.
- `test_load_arrow_outside_the_part`: A load's arrow is drawn outside the part whichever way it points: pushing in, its tip on the face; pulling away, it starts on the face (it used to sit inside the part, hidden).
- `test_stop_while_training`: Stop while the material network trains: the run ends there, no network is saved and the run's folder is removed; the status says so.
- `test_export_stl`: Export STL of the design shown: design.stl and a file per material used, in the run's folder, all closed; the pieces add up to their whole. Results' Show list has Smooth (exported), greyed with a tip until the run's folder holds the files; after the export it is chosen and the pieces are drawn, and can be pulled apart. What the folder holds decides: a Results panel opened later finds the files, and finds none once they are gone.
- `test_earlier_run_after_scale_part`: A run made before Scale part is drawn and exported at the size it was run at: its mesh is the rebuilt one scaled back (the run's element size, the same nodes), Export STL gives the same extent, and the window's own mesh of the scaled part is left as it is. A part changed in another way still cannot show the run.
- `test_pieces_pulled_apart_in_a_row`: Pull the material pieces apart: the pieces side by side along x, largest first and in place, each next one moved by the part's width and a tenth more, so no two overlap; the legend names them.
- `test_constraints_named_in_the_plots`: The Progress plot of Current run names each constraint in the Formulation panel's words with its limit; the two-run plot of Results names them as the runs do (both names when the runs differ).
- `test_material_legend`: The legend at the top left of the 3D view: each material present with its share, largest first; while a run goes, by density among the elements drawn; on a finished run's Material view, by solid elements; two runs compared, a left and a right column. The database decides whether it shows: a database of several materials keeps it when the design settles on one of them ("Red  100%"), a database of one material has none. Hidden on the Part view and on Density; it follows the light and dark view.
- `test_show_list_is_the_runs_own`: The Results panel's Show list holds the views the shown runs saved, so a run of another physics shows its own results and never another physics' Stress.
- `test_colour_scale_title_never_meets_its_numbers`: The colour scale of a value view: the lowest and highest values under the bar, the title over it as a text of its own (the bar's own title empty), so a long title never runs into the highest value; the title goes with the drawing.
- `test_only_the_latest_meshes_are_kept`: draw_design keeps the meshes of saved runs: the latest KEEP_MESHES only, not every part and size shown.
- `test_only_the_latest_runs_stay_loaded`: Results keeps the latest KEEP_RUNS runs loaded, not every run shown in the session.

### `tests/gui/test_gui_look.py`

The LSR MMTO window's look and speed, off-screen: the part's facts come from one fast sum equal to PyTO-AD's, and a large part opens and draws its ruler in a second or two; plots keep the size they were made at, so the panels holding them do not creep wider.

- `test_part_mass_properties_equal_pyto_ad`: The numpy sum gives PyTO-AD's area, volume, centre and inertia on every shipped part.
- `test_large_part_opens_fast`: A part of about 100 000 triangles: New project and the ruler each take a second or two (they took 43 s and 20 s when the volume was summed triangle by triangle, three times over).
- `test_plot_keeps_its_preferred_size`: A plot's preferred size is the size it was made at, whatever size it is given later: the panels holding plots no longer creep wider.
- `test_current_run_panel_keeps_its_width`: Opening Current run again and again leaves its width where it was.
- `test_faces_picked_only_in_the_pick_strip`: A left click on the part selects faces only while the Pick strip is open, and writes no message line; the count is in the status bar. A right click clears them, again only with Pick open.
- `test_summary_in_sections`: A finished run's Summary: the run, then Result (the objective, each constraint met or not), Design, Materials and Run, as tables; the comparison of two runs in the same sections.
- `test_display_options`: View (PyTO-AD's Display Options) in plain words, without the controls that have nothing to act on, and its cuts work: a cut along x draws the part clipped, and back to 0 draws it whole again.
- `test_view_leaves_the_scene_as_it_was`: Every View option ticked and unticked, twice, and every cut set and set back: the 3D view holds the same drawings as before, each visible as before and over the same space. (Ticking the bounding box once left a box from -1 to 1 on every axis behind for good; the box is no longer offered.)
- `test_no_window_has_the_help_button`: Windows' "?" help button, which has no help to show, is off for every window once the app's style is on (main_window.apply_style), the windows inherited from PyTO-AD too; and no window turns it off on its own, so there is one way of doing it.
- `test_a_control_that_is_off_looks_off`: With the app's style, every kind of control the panels use looks different when it is off (greyed) from when it is on, a ticked checkbox and a chosen round button too (they once looked the same, blue, with dark text): the style greys any control that is off, whatever its kind. The Materials table and the database editor draw every cell and header centred, cells made later too.
- `test_every_look_has_a_greyed_look`: Every rule of the window's style sheet that colours text or a border has a greyed version for a control that is off, in the same states: written by hand, or added by panel_widgets.with_off_look (rules added later too). A widget's own sheet gets the same (panel_widgets.set_style), a bare "color: ..." too.
- `test_every_style_sheet_gets_its_greyed_look`: The window's code sets a style sheet only through panel_widgets.set_style (a widget's) and main_window.apply_style (the app's), which both add the greyed looks: a widget given its own look later gets its greyed look too.
- `test_stop_and_the_info_button_look_off_when_off`: Stop, once pressed, is off and grey, no longer red; an "i" button that is off is grey too. Both have their own look, which the general greyed look alone did not reach.
- `test_top_row_and_bottom_buttons`: One top row: LSR MMTO, File and Help (their menus and shortcuts), Part / Design, the light / dark switch, View, Export STL, Run, Analyse; in a window held at 1225 px the row fits and nothing in it is cut short; while a run goes (View and Analyse step aside), a progress text longer than the room left ends in an ellipsis (a shorter one shows whole). Snap, Measure and Ruler at the bottom drive the window's measuring and scales.
- `test_every_tab_text_fits_its_tab`: The tab shown is marked by a line along its top, not bold text, so Qt's tab sizes (worked out with the plain font) hold its text: in the Settings window and Results (and the strips' own tabs) every tab's text and padding fit inside its tab, whichever tab is shown.
- `test_light_by_default_and_dark_on_the_switch`: The 3D view starts light. In the dark view the window's own drawings are drawn light from the first frame (the Measure line here), and back in the light view they are dark again; the part keeps its colour.
- `test_measure_white_in_the_dark_view`: Measure's line and points are bright white in the dark view and #222222 in the light view, drawn in either view and after a switch both ways; another dark drawing of the same colour is only lightened.
- `test_message_panel_merges_repeats`: A line repeated at once is shown once, with how many times.
- `test_message_panel_problems_then_log`: The message panel: the project's errors (red) and warnings (amber) as they are now at the top, a thin line, then what happened. A problem fixed disappears, and shows again when it comes back; with no problems only the log is there. Clear empties the log and keeps the problems; the panel keeps its height.
- `test_message_panel_resizable`: The divider between the 3D view and the message panel drags: the panel and its log get taller, the problems take at most half of the panel and the log the rest (all of it with no problems); a dragged height comes back in a new window (three lines at first); the start page, the legend and the banner stay in their places on the 3D view as it changes size.
- `test_run_layout`: While a run goes the sidebar steps aside and Current run takes about half the window; afterwards both are back as they were.
- `test_info_card`: An (i) click opens a card with its whole text in Markdown (paragraphs, bold, numbered steps), 420 px wide and wrapped, under the (i) and moved to stay inside the window at each edge, scrolling inside when taller than 60 % of the window; Esc closes it. The mouse resting on it shows only the first sentence.

### `tests/gui/test_gui_soak.py`

tests/gui/gui_soak.py, the soak test, without soaking (a few seconds): pytest never collects it, it drives the app's own lists, and its measurements give numbers here.

- `test_pytest_never_collects_the_soak_test`: Pytest never collects the soak test.
- `test_the_soak_test_uses_the_apps_own_lists`: The soak test uses the apps own lists.
- `test_the_soak_tests_measurements_give_numbers`: The soak tests measurements give numbers.
- `test_the_steps_file_holds_every_measurement`: The steps file holds every measurement.

### `tests/gui/test_launcher.py`

The desktop start: the icon loads at every size Windows asks for, the start-up picture shows, a start with no console writes what the app prints to the log file, the shortcut is written with every path quoted for the desktop or the Start menu (on Windows, a real shortcut in a scratch folder), the setup's one question takes Enter as yes, and Settings > Apps lists LSR MMTO, whose Uninstall removes exactly what it lists (with stand-ins for the registry, the commands and the folders).

- `test_icon_sizes`: Icon sizes.
- `test_splash_screen`: Splash screen.
- `test_start_without_console_writes_the_log`: Start without console writes the log.
- `test_torch_loads_before_qt`: run_gui.py imports torch before anything from PyQt5: the other way round, torch's c10.dll fails to load on Windows (WinError 1114). The start is stopped as soon as PyQt5 is imported.
- `test_shortcut_lines_quote_every_path`: Shortcut lines quote every path.
- `test_start_menu_and_the_question`: Start menu and the question.
- `test_setup_step_lines_count_the_real_steps`: The install prints [n/N] with N the number of steps it really goes through: 6 on Windows (the Start menu step), 5 elsewhere, and main() calls step() once for each of them.
- `test_run_record_starts_git_without_a_window`: A run looks up the code's git commit for its record: with no console window on Windows (started from the shortcut, each git call otherwise flashes a black window up).
- `test_shortcut_made_on_windows`: Shortcut made on windows.
- `test_uninstall_entry`: The entry Settings > Apps lists: its values, the uninstaller run by the conda installation's own python (found from CONDA_EXE, or as the folder holding envs); written twice, still one entry.
- `test_uninstall_removes_exactly_what_it_lists`: Answered no, nothing goes. Answered yes: the two shortcuts, the kernel, the GUI's settings, the environment (conda env remove) and, last, the entry go; everything else stays, the code folder untouched, and the folders kept are named.
- `test_uninstall_refuses_to_run_inside_the_environment`: Uninstall refuses to run inside the environment.
- `test_no_other_way_to_remove`: Uninstall in Settings > Apps is the one way: no remove option in the setup or the shortcut maker.

### `tests/gui/test_network_stamp_gui.py`

The record a trained network keeps of its database (materials/network_stamp.py) in the window, off-screen: a run on a network whose database changed only in colours or labels asks once (No: nothing starts; Yes: the network is stamped and not asked about again); Materials says when the database has changed since its network was trained, and when the next run will ask (for an included database: to put it back or copy it); an analysis, which uses no network, is not asked; the editor keeps every digit of a value it is not asked to change, and sets the network aside after a rename; Open in Excel does not open a database that comes with LSR MMTO. The questions are answered by the test (database_editor.ask), never shown.

- `test_a_run_asks_once_about_a_network_whose_database_changed_only_in_colour`: No: nothing starts, and the status says why. Yes: stamped, and not asked again. An analysis is not asked.
- `test_materials_says_what_the_network_needs`: Nothing when the network matches; the question to come for a label changed; an error naming the cell for a number changed; for an included database, to put it back or make a copy.
- `test_the_editor_keeps_every_digit_and_sets_the_network_aside_after_a_rename`: A value with 16 significant digits shows and saves as it is: a label changed keeps the network (the next run asks once). A material renamed sets the network aside, as a number changed does.
- `test_open_in_excel_leaves_an_included_database_alone`: Nothing is opened; the report says to make a copy with Edit... first.

### `tests/gui/test_new_physics_in_the_gui.py`

A new physics needs no GUI code: a made-up one (one field, phi; one property; one kind of support; one objective; one setting; one result), registered as its line in available_physics.MODULES would register it, shows in each panel that shows a physics' pieces, opened on a project of it: its kinds in Supports and loads, its objective in Formulation, its setting in Settings, its property among a new database's columns, its result in the Results (i); Materials, which shows nothing of a physics, opens on it. It has nothing to pick on the part, so Click on the part is greyed, saying why. Nothing of it is written in gui/: what the panels show comes from its entries alone.

- `test_a_new_physics_shows_in_every_panel_with_no_gui_code`: A new physics shows in every panel with no gui code.

### `tests/gui/test_single_material_gui.py`

The window with one material in the design, off-screen. Materials: Single material or Multi material, the database that fits each (a misfit is an error in the report), Fix every element to with multi material, and OK handing back exactly what was chosen. Settings: every setting of the latent points greyed out with one material, the VAE settings too with single material, their values kept. Formulation: Material volume fraction greyed out, with the reason beside a row that has it. The database editor saves a database of one material. A run of one material: no Latent space tab, None for its latent dimension, and Compare says why it has no latent space to compare.

- `test_materials_window_single_or_multi_material`: Multi material, as every project before this choice: Fix every element to, on the top row, lists None and the database's materials, and the material chosen is the project's pinned material after OK. Single material with the same 6-material database: the list becomes Material:, with every material and no None; the database's first is chosen unless one was chosen for single material before, the report names it, and OK hands back the single material and that material; no Network error tab. Back to multi material: nothing is fixed, and back to single material: its material again. A database of one material works for single material and is an error for multi material; a fixed material that a newly chosen database does not have is no longer fixed, and the report says so.
- `test_materials_window_keeps_each_choice_and_changes_nothing_on_its_own`: Multi material with a fixed material, then single material, then multi again: the fixed material is back. A single material the newly chosen database does not have: its first material is chosen, and the report says so. Single material on a database of one material with no material named: opening and OK change nothing. Analyse starts on the project's material.
- `test_settings_greyed_with_one_material`: With a fixed material, every setting that reads the latent points is greyed out with the reason, the others and the VAE settings are not; with single material, the VAE settings are greyed out too and the VAE line says there is no network. OK changes nothing.
- `test_formulation_greys_material_volume_fraction_with_one_material`: With one material, Material volume fraction is greyed out in the list, resting the mouse on it says why, and a row that already has it shows the reason beside it.
- `test_database_editor_saves_one_material`: A database of one material is a database the editor saves, and its one row cannot be removed.
- `test_the_live_design_of_a_run_of_one_material`: A run of one material started in the window: no live Latent space tab, and the live design draws with this run's material labels, whether it is the session's first run or follows a run of another database (the labels were once set only for a run of several materials).
- `test_a_run_of_one_material_has_no_latent_space`: Its tabs are Convergence and Summary, its latent dimension reads None (one material), its summary names the material every element is, the latent plot says in words that there is none, and Compare puts in the Latent space tab why there is none.

### `tests/gui/test_thermal_walk_through.py`

A thermal project, the physics the window has never shown, taken through the window the way a user would: saved and opened again, Run, its Results views and Summary, Analyse, Analyse this design, Compare, Export STL, and a new database in the editor. A physics that is not structural must go through every step with nothing of structural's showing: no stress, no failure factor, no force or stress units. A run and an analysis whose solver did not converge say so in red under Current run (the lines themselves stood in for here). About ten seconds.

- `test_a_thermal_project_through_the_window`: A thermal project through the window.
- `test_temperatures_and_heat_are_picked_on_the_part`: Click on the part, on a thermal project, opens the Pick window of temperatures and heat: Type offers what thermal's kinds take (Temperature, Total Heat; not Heat Flux, which no kind takes), each value in the project's units. A click on the part selects its face, a right click clears it, and Apply adds the pick to the project at once, saying so in the project's units and leaving no label behind. A thermoelastic project offers both picking windows, one line each, each opening its own; a project switch or a change of units closes them.

### `tests/gui/gui_soak.py` (a script, not collected by pytest)

Soak test: drives the real GUI in a loop for hours, to find what piles up or slows down with use.

## docs

The documents match the code: the generated pages are current, and the notebook and the examples run.

### `tests/docs/test_code_map.py`

docs/code_map.md names files and folders; each one exists, so the map cannot drift from the code when something is renamed or moved. A name with <name> in it must exist for the structural physics and for the template.

- `test_every_name_in_the_code_map_exists`: Every name in the code map exists.

### `tests/docs/test_python_walkthrough.py`

Python_walkthrough.ipynb runs from top to bottom with the settings it uses, in a copy of the LSR MMTO folder that holds the notebook and the U-bracket's part, and leaves the project folder its notes describe; every link in its notes leads to an entry of docs/reference.md.

- `test_every_link_leads_to_a_reference_entry`: Every link leads to a reference entry.
- `test_the_notebook_runs_from_top_to_bottom`: The notebook runs from top to bottom.

### `tests/docs/test_reference.py`

docs/reference.md must be what the code generates: after adding a quantity, a selection type, a boundary-condition kind or a setting, or changing a docstring the Python section is read from, run

- `test_reference_page_is_current`: Reference page is current.
- `test_every_python_example_runs`: The examples, run in their order from a copy of the LSR MMTO folder's U-bracket project and part, make, save and write as a script the project they build, then run the shipped U-bracket and read it back. run(p) is stood in for by Analyse (the solid part solved once and saved as a run is saved), so nothing is optimized; analyse_design, which takes only a run, is stood in for the same way, once its run folder is checked to be one.

### `tests/docs/test_testing_doc.py`

docs/testing_the_codebase.md must be what the tests generate: after adding, removing or renaming a test, changing a test's docstring, or moving a test file, run

- `test_testing_page_is_current`: docs/testing_the_codebase.md is what docs/make_testing_doc.py writes from the tests as they are.
- `test_a_folder_without_its_line_or_a_loose_test_file_stops_the_page`: A folder of tests with no line in FOLDERS, a line for a folder with no tests, and a test file directly in tests/ each stop the page with a message naming them, so a new folder or file cannot go undescribed.

### `tests/docs/test_user_guide.py`

The user guide: every entry of its lists (selection types, supports and loads, material attributes, quantities, settings) has a name and a description; docs/user_guide.md holds the lists as the code makes them now; the guide window opens off-screen, finds a word and marks it on the page; Help > User guide is in the window, on F1; Help > Researcher guide shows the pages for changing the code in the same window, a chapter each.

- `test_every_entry_is_described`: Every entry is described.
- `test_guide_lists_are_current`: Guide lists are current.
- `test_window_finds_and_marks`: Window finds and marks.
- `test_help_menu_opens_the_guide`: Help menu opens the guide.
- `test_researcher_guide`: Help > Researcher guide, between User guide and About: the five pages for changing the code in the guide's window, in their order, each a chapter under its own first heading with its sections as its parts; the search finds a word that only CONTRIBUTING.md has.
- `test_a_page_is_laid_out_for_reading`: A guide page: every table cell on one line (each column as wide as its longest cell), cells of an empty markdown cell kept apart, the header shaded, the text in the middle of each cell; lone asterisks in a generated row shown as written, not as italics; the text in a centred column, as wide as the page's widest table when that is wider, so a table that fits the window needs no scrolling; a menu path of the window in a light box; "section N" a link that opens it, also when it is the part already open; none in a ``` block.

## code_rules

The code keeps its own rules: one entry adds a feature, every physics folder keeps the template's shape, every module imports.

### `tests/code_rules/test_docstring_reads_and_sets.py`

A function that changes an object it is given says so in its docstring, in one line of code names:

- `test_every_docstring_names_what_its_function_reads_and_sets`: Every docstring names what its function reads and sets.
- `test_the_check_catches_a_field_left_out`: The check catches a field left out.

### `tests/code_rules/test_extending.py`

A new setting is one entry of CORE_SETTINGS (lsr_mmto/problem/settings.py), and a new objective or constraint is one entry of CORE_QUANTITIES (lsr_mmto/formulation/quantities.py): nothing else is written by hand. One made-up setting and one made-up quantity are added here, and each must reach every place that uses it: the defaults, the settings page, the app's panel, the settings a run gets. A physics may leave out its optional entries, its Summary rows reach the Summary, and a new database starts with the columns its project's physics reads. Two things of one name are refused at start-up, naming both places, and registering a physics in a test makes the core's lists as start-up does.

- `test_one_entry_makes_a_setting`: One entry makes a setting.
- `test_one_entry_makes_a_quantity`: One entry makes a quantity.
- `test_every_quantity_has_its_gradient_checked`: Every quantity of the registry is in the hand list of tests/gradients/test_ad_quantities.py, which checks its gradient against finite differences (that file is slow; this check is not), or named there as not checked, with the reason. A new quantity left out of the list fails here, by its name.
- `test_a_physics_may_leave_out_its_optional_entries`: _checked gives a physics that leaves out an OPTIONAL_ENTRIES entry its "none" value, and refuses one that leaves out any other entry, naming it, or gives SIMP_PENALTIES without set_simp_penalties.
- `test_a_physics_summary_rows_reach_the_summary`: Each SUMMARY_ROWS entry is a row of the Summary with the last value its history key recorded, and a key the run did not record (or recorded as nan) gives no row.
- `test_a_new_database_starts_with_the_columns_its_physics_reads`: Density, the properties the project needs, then its physics' optional ones, in the property list's order: an L-bracket that limits no stress needs no yield strength.
- `test_two_things_of_one_name_are_refused`: Start-up refuses, naming both places: a physics setting with a name the core already uses (it would take the core setting's place in every project), two different penalties of one name, two penalties that write one history key, two quantities of one name in one physics, a material property's alias that is another property's name, and a result shown as a view every run has (Material, Density or Solid).
- `test_the_reserved_views_are_the_results_panels_own`: RESERVED_VIEWS, the views a physics' result may not take, are exactly the views the Results panel gives every run (gui/draw_design.py, views with no run's results).
- `test_registering_in_a_test_makes_the_lists_as_start_up_does`: remake_lists (tests/physics/core_lists_for_a_test.py), which registered uses, makes from MODULES exactly the lists start-up made: the same physics, quantities, penalties, settings, properties, aliases and kinds, in the same order. A step start-up takes that it does not would show here.

### `tests/code_rules/test_layout.py`

Every module of the package lsr_mmto and of the app (gui) imports, so a file moved or renamed without its imports is caught here, before a run or the app trips over it. Also: the old folders are gone, a script outside the repository folder imports lsr_mmto, and PyTO-AD's settings that every run in the program shares (its SIMP exponents, its p-norm exponent) are named only inside a physics' own folder, while no code reads the mesh's shared dof table (each solver keeps its own).

- `test_every_module_imports`: Every module imports.
- `test_old_folders_gone`: Old folders gone.
- `test_script_imports_from_another_folder`: A script anywhere imports lsr_mmto: installed (pip install -e .), or with the repository on PYTHONPATH.
- `test_shared_settings_stay_in_the_physics_folders`: Shared settings stay in the physics folders.

### `tests/code_rules/test_material_source.py`

A run gets each element's material properties from its material source (lsr_mmto/materials/material_source.py) and from nothing else: outside the material network's own file and the material source's, no code reaches into the network's decoder. Every source gives what the run reads (MaterialSource), and each decodes as it says: the network inside the torch tree, the database at each element's material's own values.

- `test_nothing_but_the_material_source_reaches_into_the_network`: No file of lsr_mmto or gui, other than the network's own (encoder.py) and the material source's, calls the network's decoder or turns its output into properties itself: each asks its material source's decode.
- `test_every_source_gives_what_the_run_reads`: The network, the network pinned to one material and the database's own values each give every member of MaterialSource.
- `test_the_network_decodes_inside_the_tree`: The network's decode gives every column of the database, one value per point, and the gradient of a property reaches the latent points.
- `test_the_database_decodes_to_each_elements_own_material`: The database's own values: each element gets its material's row, whatever its latent point, and there is no network to save.
- `test_every_quantity_and_penalty_says_truly_whether_it_reads_the_latent_points`: A quantity or a penalty whose own function reads the model's latent points (m.z, m.z_real) says so (reads_latent_points), and one that does not says it does not: a single-material project refuses the first kind of quantity, and a run with no latent points leaves out the first kind of penalty, so a wrong word would let a value of no meaning into a run. Read from the names the function's code uses, and the code of every function of LSR MMTO it calls, however deep (reaches_the_latent_points), so a quantity of any physics that reads them through a helper is caught too.
- `test_the_check_follows_a_function_into_its_helpers`: reaches_the_latent_points finds the latent points read two helpers down, in a lambda, through a module's function, a closure, a partial, a decorator, a callable object and getattr(m, "z"); a quantity's own function that reads none (Mass) gives None.

### `tests/code_rules/test_physics_folders.py`

Every physics folder keeps the shape of the form (lsr_mmto/physics/template/):

- `test_every_physics_has_the_five_files`: Every physics has the five files.
- `test_the_form_lists_every_entry_and_each_physics_puts_them_where_it_does`: The form lists every entry and each physics puts them where it does.
- `test_every_function_of_a_physics_is_marked`: Every function of a physics is marked.
- `test_outside_a_physics_only_its_front_page_is_imported`: Outside a physics only its front page is imported.
- `test_the_final_check_reads_its_designs_one_by_one`: The final check reads its designs one by one.
- `test_the_checks_catch_what_they_are_for`: The checks catch what they are for.
