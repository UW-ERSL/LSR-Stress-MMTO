# Project reference

Generated from the code by `python docs/make_project_file_reference.py`; do not edit by hand.

A project is an LSR MMTO project file (.mmto): the layout of a PyTO-AD project file, with an `mmto` section, which holds the whole problem. Supports and loads picked on the part in the window become selections (`gui_support1`, `gui_load1`, ...) and boundary conditions of the mmto section as soon as they are applied; the rest of what the window saves in the file (its settings) is kept.

## The mmto section

| key | holds |
|---|---|
| `schema_version` | version of the project format |
| `name, comment` | the project's name (also the run's name) and a description |
| `physics` | "structural" for now |
| `mesh` | n_dof (degrees of freedom asked for) and single_layer (one element through the thickness) |
| `selections` | named regions of the mesh, see Selection types |
| `bcs` | boundary conditions, each on a named selection, see Boundary conditions |
| `keep_solid` | names of element selections kept solid |
| `keep_support_elements` | also keep solid the elements touching fixed dofs |
| `database` | a database name in data/databases, or a path to an .xlsx |
| `formulation` | objective and constraints, see Quantities |
| `settings` | common, advanced and vae, see Settings |

## Selection types

Lengths are numbers in model units, or `"<k>h"` for k times the largest element size. `pad` widens every bound; `"auto"` is half an element for nodes and 0 for element centres (faces and planes always half an element).

| type | fields | selects |
|---|---|---|
| `box` | lo, hi, pad | Points with lo <= p <= hi; the corners may come in any order. By default pad is half the element size along each axis, so a box thinner than an element still catches the nearest layer. |
| `cylinder` | center, axis, radius, inner_radius, along, pad | Points whose distance to the axis is in [inner_radius, radius]; along = (t0, t1) limits (p - center) . axis. |
| `difference` | parts | Points in the first part but not the second. |
| `face` | side, pad | The plane at the mesh's extreme on one side: xmin, xmax, ymin, ymax, zmin or zmax. |
| `function` | ref | Whatever a Python function name(mesh, on) in file.py returns. |
| `halfspaces` | normals, origins, boundary | Points on the inner side of every plane: n . (p - o) <= 1e-12 for each normal n and origin o. A box dragged in the PyTO-AD GUI is six such planes. boundary=True keeps only nodes on the mesh boundary, as the GUI does. |
| `ids` | indices | Explicit node or element numbers; tied to one mesh. |
| `inside_stl` | path | Points inside a closed STL surface. |
| `intersect` | parts | Points in all of the parts. |
| `near_face` | side, distance | Points strictly within distance of the mesh's extreme on one side. |
| `nearest` | point, boundary | The single node or element closest to point; boundary=True only looks at nodes on the mesh boundary, as the PyTO-AD GUI does for a picked point. |
| `plane` | axis, value, pad, limits | Points with \|p_axis - value\| <= pad; by default pad is half the element size along axis. limits, e.g. {"y": [100, 120]}, keeps only the part of the cut between those values of the other axes (each widened by pad along its own axis). |
| `rows` | face, axis, n, end | The n outermost node rows of a face, counted along an axis: e.g. the 8 highest rows of the right face. Rows are node layers, so their height is n element sizes. |
| `selection` | name | Another selection of the project, by name. Only a project can evaluate it: Project.evaluate puts the named selection in (resolved) each time, so a combination follows later edits of its parts. |
| `sphere` | center, radius, pad | Points within radius of center. |
| `stl_faces` | triangles, tol | Boundary nodes on triangles of the STL the mesh was made from: within tol times the smallest element size of a triangle. tol None is PyTO-AD's own lookup (0.707); the PyTO-AD GUI uses 0.8. |
| `union` | parts | Points in any of the parts. |

## Boundary conditions

| kind | means |
|---|---|
| `fixed` | the given dofs (x, y, z) held at 0 |
| `displacement` | the given dofs held at a nonzero value |
| `force` | a total force [Fx, Fy, Fz], split equally over the nodes |
| `traction` | pressure * area along a direction, split equally over the nodes; area a number or 'bbox' |

## Material database attributes

A database is an .xlsx file: attribute names in row 1, units in row 2, then one row per material with its name in column A. Every column must be one of these attributes (or one of its other names); the Materials panel of the app shows the errors and warnings of a file.

| attribute | other names | meaning | allowed range | units |
|---|---|---|---|---|
| `Density` |  | mass per unit volume | 0 < value | kg/m^3, g/cm^3, t/mm^3, lb/in^3, lbf*s^2/in^4 |
| `Youngs_Modulus` | Youngs Modulus, Young_Modulus, E | Young's modulus, the stiffness | 0 < value | Pa, kPa, MPa, GPa, psi, ksi |
| `Poissons_Ratio` | Poisson_Ratio, Poissons Ratio, nu | Poisson's ratio | -1 < value < 0.5 | none (may be blank) |
| `Yield_Strength` | Yield Strength, Yield | yield strength, the stress at which the material yields | 0 < value | Pa, kPa, MPa, GPa, psi, ksi |
| `Cost` |  | price per unit mass | 0 <= value | USD/kg, USD/g, USD/t, USD/lb, USD/(lbf*s^2/in) |

An optional sheet `Display` gives each material the colour and the label it is shown with: columns Material, Color (#rrggbb) and Label, one row per material, matched by name. An empty Color is the default colour, an empty Label the material's name. Two materials with the same colour or the same label are an error. The run record stores both.

## Unit systems

A project declares the units its numbers are in, as the PyTO-AD GUI does (`settings.unit_system` in the project file; `p.units` in code). Nothing is converted: forces, lengths, bounds and the database are used as written. A database column in other units is reported before a run, with what it does to the results. `none` (no unit system) checks nothing.

| system | length | force | stress | density | mass | cost per mass |
|---|---|---|---|---|---|---|
| `MKS` | m | N | Pa | kg/m^3 | kg | USD/kg |
| `mmKS` | mm | N | MPa | t/mm^3 | t | USD/t |
| `IPS` | in | lbf | psi | lbf*s^2/in^4 | lbf*s^2/in | USD/(lbf*s^2/in) |

## Physics

The `physics` of a project names the model its optimization uses.

| physics | model | needs columns |
|---|---|---|
| `structural` | linear elasticity, small strain | Youngs_Modulus |

## Quantities

An objective or a constraint is one quantity, e.g. `Mass()` or `MaterialVolumeFraction(2, tau=0.5)`. A constraint is value / bound - 1 <= 0.

| name | shown as | objective | constraint | needs columns | value | meaning |
|---|---|---|---|---|---|---|
| `Compliance` | Compliance | yes | yes |  | F . u | How much the part gives under the loads. Lower is stiffer. |
| `StressPNorm` | Stress (p-norm) | yes |  |  | p-norm of the relaxed von Mises stress, no 1/N | One value close to the highest von Mises stress in the part. |
| `FailureFactorPNorm` | Failure factor (p-norm) | yes | yes | Yield_Strength | mean p-norm of von Mises / yield strength, per element | One value close to the highest stress / yield strength in the part. Above 1 means some element yields. |
| `Mass` | Mass | yes | yes | Density | sum of density * x_phys * element volume | The mass of the part, from each material's density. |
| `Cost` | Cost |  | yes | Density, Cost | sum of density * cost * x_phys * element volume | The cost of the part, from each material's density and cost per mass. |
| `VolumeFraction` | Volume fraction |  | yes |  | mean of x_phys | The share of the design space filled with material, 0 to 1. |
| `MaterialVolumeFraction` | Material volume fraction |  | yes |  | volume fraction of one database material, softmax weights | The share of the design space filled with one chosen material, 0 to 1. |

## Settings

Every setting a run takes, under its own name. A new project starts from these values (the settings of the paper's examples); a file missing some is completed from them, and the run says which. The VAE's own settings start from the database's known values. The mesh settings are the file's `mesh` section; the others are `settings.<section>`.

### mesh

| setting | default | meaning |
|---|---|---|
| `n_dof` | 30000 | How fine the mesh is: the number of unknowns it is built for. More is finer, and slower. |
| `single_layer` | False | One element through the thickness, for a flat (2D) part. |

### common

| setting | default | meaning |
|---|---|---|
| `maxIterations` | 300 | The optimization stops after this many iterations. |
| `relative_filter_radius` | 2.5 | The smallest feature the design may have, in element sizes. Larger gives smoother, simpler designs. |
| `material_filter_radius` | None | How far material choices are smoothed between neighbouring elements, in element sizes. Ticked: the same as the density filter radius. Empty (None): same as density filter. |
| `beta_schedule_scheme` | 'PAPER' (one of PAPER, PAPER_BM1) | How fast the design is pushed to clear solid and empty: the sharpness starts at 1 and grows 1.5 times every 5 iterations after iteration 50, up to 32 (PAPER) or 16 (PAPER_BM1). |
| `pnorm_exponent` | 6.0 | How closely the single stress value follows the highest stress in the part. Larger is closer, but harder to optimize. |

### advanced

| setting | default | meaning |
|---|---|---|
| `timeLimit` | 36000 | The optimization stops after this many seconds. |
| `rel_conv_tol` | 1e-07 | The optimization stops early once the objective changes by less than this, relatively, between iterations, and every constraint is met to within this. |
| `use_heaviside_projection` | True | Pushes each element towards fully solid or fully empty. |
| `eta_proj` | 0.5 | The density at which that push splits solid from empty. |
| `use_penalization` | True | Pushes each element's material towards one of the real materials in the database. |
| `gamma_init` | 0.001 | How strong that push is at the start. |
| `gamma_max` | 10.0 | How strong it can get. |
| `gamma_factor` | 1.1 | How much stronger it gets each iteration while it ramps up. |
| `penalty_start_iter` | 0 | The iteration it starts at, when Start the ramp when the objective plateaus is off. |
| `distance_penalty_disable_after_iter` | 250 | The iteration it stops at, when Start the ramp when the objective plateaus is off. |
| `use_dynamic_gamma_scheduling` | True | Start the push when the design stops improving, instead of at a set iteration. |
| `gamma_plateau_window` | 10 | The objective has plateaued when it changed by less than the plateau tolerance over this many iterations. |
| `gamma_plateau_tolerance` | 0.001 | How little the objective may change over the plateau window, relatively, to count as plateaued. |
| `gamma_plateau_earliest_iter` | 20 | The push never starts before this iteration. |
| `gamma_ramp_end_margin` | 30 | If the objective never plateaus, the push starts in time to reach its largest weight this many iterations before Max iterations. |
| `use_z_smoothing` | True | Discourages neighbouring elements from flickering between materials. |
| `z_smoothing_weight` | 0.0001 | How strong that smoothing is. |
| `z_smoothing_disable_after_iter` | 250 | The iteration it stops at. |
| `use_continuation` | False | Raises the SIMP penalty by 0.25 every 10 iterations during the run. |
| `z0_init_method` | 'ORIGIN' (one of LIGHTEST, HEAVIEST, ORIGIN, UNIFORM) | The material every element starts with: ORIGIN (the middle of the material map), the LIGHTEST or HEAVIEST material, or UNIFORM (random). |
| `snap_to_real_material` | False | In the final design, give each element the nearest real material. |
| `binarize_topology` | True | In the final design, make each element fully solid or empty, keeping the volume. |
| `use_constant_poissons_ratio` | False | One Poisson's ratio for every material, instead of each material's own. |
| `constant_poissons_ratio` | 0.3 | The Poisson's ratio used when Constant Poisson's ratio is ticked. |

### vae

| setting | default | meaning |
|---|---|---|
| `use_pretrained_vae` | True | Use the network already trained for this database. Unticked, the next run trains a new one. |
| `saveNet` | None | The trained network: always the .nt file next to the database, with the same name. Empty (None): next to the database. |
| `network_error_warning` | 5.0 | Before a run, you are warned if the network reproduces any material property less closely than this, in % of the property's range in the database. You can still run. |
| `latentDim` | from the database | How many numbers describe one material in the material map. |
| `learningRate` | from the database | The step size of the network's training. |
| `vae_hiddenDim` | from the database | The size of the network's inner layers. |
| `maxAttributeErrorPercent` | from the database | Training stops once the network reproduces every material property within this percent. |
| `klFactor` | from the database | How strongly training keeps the material map compact and smooth. |
| `numEpochs` | from the database | Training stops after this many passes at the latest. |
