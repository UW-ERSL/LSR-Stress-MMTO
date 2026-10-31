# Project reference

Generated from the code by `python utils/reference.py`; do not edit by hand.

A project is a PyTO-AD project file (.pyto) with an `mmto` section, which holds the whole problem. Supports and loads picked in the PyTO-AD GUI become selections (`gui_support1`, `gui_load1`, ...) and boundary conditions of the mmto section when the project is loaded, and are removed from the GUI's own section; the rest of what the GUI saved is kept.

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
| `database` | a database name in 1-PureStructural/MaterialDataPureStructural, or a path to an .xlsx |
| `formulation` | objective and constraints, see Quantities |
| `settings` | common, advanced and vae, see Settings |

## Selection types

Lengths are numbers in model units, or `"<k>h"` for k times the largest element size. `pad` widens every bound; `"auto"` is half an element for nodes and 0 for element centres (faces and planes always half an element).

| type | fields | selects |
|---|---|---|
| `box` | lo, hi, pad | Points with lo <= p <= hi. |
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
| `plane` | axis, value, pad | Points with \|p_axis - value\| < pad. |
| `rows` | face, axis, n, end | The n outermost node rows of a face, counted along an axis: e.g. the 8 highest rows of the right face. Rows are node layers, so their height is n element sizes. |
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

A database is an .xlsx file: attribute names in row 1, units in row 2, then one row per material with its name in column A. Every column must be one of these attributes (or one of its other names); `python utils/database.py check <file>` checks a file.

| attribute | other names | meaning | allowed range | units |
|---|---|---|---|---|
| `Density` |  | mass per unit volume | 0 < value | kg/m^3, g/cm^3, t/mm^3, lb/in^3, lbf*s^2/in^4 |
| `Youngs_Modulus` | Youngs Modulus, Young_Modulus, E | Young's modulus, the stiffness | 0 < value | Pa, kPa, MPa, GPa, psi, ksi |
| `Poissons_Ratio` | Poisson_Ratio, Poissons Ratio, nu | Poisson's ratio | -1 < value < 0.5 | none (may be blank) |
| `Yield_Strength` | Yield Strength, Yield | yield strength, the stress at which the material yields | 0 < value | Pa, kPa, MPa, GPa, psi, ksi |
| `Cost` |  | price per unit mass | 0 <= value | USD/kg, USD/g, USD/t, USD/lb, USD/(lbf*s^2/in) |

## Unit systems

A project declares the units its numbers are in, as the PyTO-AD GUI does (`settings.unit_system` in the .pyto file; `p.units` in code). Nothing is converted: forces, lengths, bounds and the database are used as written. A database column in other units is reported before a run, with what it does to the results. `none` (no unit system) checks nothing.

| system | length | force | stress | density | mass | cost per mass |
|---|---|---|---|---|---|---|
| `MKS` | m | N | Pa | kg/m^3 | kg | USD/kg |
| `mmKS` | mm | N | MPa | t/mm^3 | t | USD/t |
| `IPS` | in | lbf | psi | lbf*s^2/in^4 | lbf*s^2/in | USD/(lbf*s^2/in) |

## Quantities

An objective or a constraint is one quantity, e.g. `Mass()` or `MaterialVolumeFraction(2, tau=0.5)`. A constraint is value / bound - 1 <= 0.

| name | objective | constraint | needs columns | value |
|---|---|---|---|---|
| `Compliance` | yes | yes |  | F . u |
| `Mass` | yes | yes | Density | sum of density * x_phys * element volume |
| `Cost` |  | yes | Density, Cost | sum of density * cost * x_phys * element volume |
| `StressPNorm` | yes |  |  | p-norm of the relaxed von Mises stress, no 1/N |
| `FailureFactorPNorm` | yes | yes | Yield_Strength | mean p-norm of von Mises / yield strength, per element |
| `VolumeFraction` |  | yes |  | mean of x_phys |
| `MaterialVolumeFraction` |  | yes |  | volume fraction of one database material, softmax weights |

## Settings

Every setting of `run_topopt`, under its own name. A new project starts from these values (the settings of the paper's examples); a file missing some is completed from them, and the run says which. The VAE's own settings start from the database's known values.

### common

| setting | default | meaning |
|---|---|---|
| `maxIterations` | 300 | largest number of MMA iterations |
| `relative_filter_radius` | 2.5 | density filter radius, in element sizes |
| `material_filter_radius` | None | radius of the latent smoothing neighbourhood, in element sizes; None = relative_filter_radius |
| `pnorm_exponent` | 6.0 | p of the p-norm that aggregates stress or failure factor |
| `beta_schedule_scheme` | 'PAPER' | Heaviside sharpness schedule: beta = 1, then x1.5 every 5 iterations after 50, up to 32 (PAPER) or 16 (PAPER_BM1) |

### advanced

| setting | default | meaning |
|---|---|---|
| `timeLimit` | 36000 | wall-clock limit of the optimization, seconds |
| `rel_conv_tol` | 1e-07 | stop when the objective and constraints change by less than this, relatively |
| `use_penalization` | True | penalty on the distance from each solid element's latent point to the nearest real material |
| `gamma_init` | 0.001 | starting weight of that penalty |
| `gamma_max` | 10.0 | largest weight of that penalty |
| `gamma_factor` | 1.1 | the weight is multiplied by this each iteration while it ramps up |
| `penalty_start_iter` | 0 | first iteration of the penalty, without dynamic scheduling |
| `distance_penalty_disable_after_iter` | 250 | iteration at which the penalty is switched off, without dynamic scheduling |
| `use_dynamic_gamma_scheduling` | True | start the weight's ramp when the objective plateaus, instead of at fixed iterations |
| `use_z_smoothing` | True | penalty on latent points that differ from their neighbours' (material flicker) |
| `z_smoothing_weight` | 0.0001 | weight of that penalty |
| `z_smoothing_disable_after_iter` | 250 | iteration at which that penalty is switched off |
| `use_heaviside_projection` | True | push the filtered density toward 0 or 1 |
| `eta_proj` | 0.5 | threshold of that projection |
| `use_continuation` | False | raise the SIMP penalty by 0.25 every 10 iterations |
| `use_constant_poissons_ratio` | False | use constant_poissons_ratio instead of the decoded Poisson's ratio |
| `constant_poissons_ratio` | 0.3 | the Poisson's ratio used when it is held constant |
| `snap_to_real_material` | False | in the reported design, replace each latent point by the nearest real material's |
| `z0_init_method` | 'ORIGIN' | starting latent point: ORIGIN (z = 0), LIGHTEST, HEAVIEST material, or UNIFORM random |
| `binarize_topology` | True | in the reported design, round the density to 0 or 1, keeping the volume |

### vae

| setting | default | meaning |
|---|---|---|
| `use_pretrained_vae` | True | load the trained VAE from saveNet if it exists; otherwise train one |
| `saveNet` | None | the VAE file; None = the .nt file next to the database's .xlsx |
| `latentDim` | from the database | latent dimension of the VAE |
| `learningRate` | from the database | VAE training learning rate |
| `vae_hiddenDim` | from the database | width of the VAE's hidden layers |
| `maxAttributeErrorPercent` | from the database | VAE training stops once every attribute is reproduced within this percent |
| `klFactor` | from the database | weight of the KL term in VAE training |
| `numEpochs` | from the database | largest number of VAE training epochs |
