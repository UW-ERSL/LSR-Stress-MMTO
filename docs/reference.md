# Reference

Generated from the code by `python docs/make_reference.py`; do not edit by hand.

A project is an LSR MMTO project file (.mmto): the layout of a PyTO-AD project file, with an `mmto` section, which holds the whole problem. Supports and loads picked on the part in the window become selections (`gui_support1`, `gui_load1`, ...) and boundary conditions of the mmto section as soon as they are applied; the rest of what the window saves in the file (its settings) is kept.

## The mmto section

| key | holds |
|---|---|
| `schema_version` | version of the project format |
| `name, comment` | the project's name (also the run's name) and a description |
| `physics` | the physics the project uses, see Physics |
| `mesh` | n_dof (degrees of freedom asked for) and single_layer (one element through the thickness) |
| `selections` | named regions of the mesh, see Selection types |
| `bcs` | boundary conditions, each on a named selection, see Boundary conditions |
| `keep_solid` | names of element selections kept solid |
| `keep_support_elements` | also keep solid the elements touching fixed dofs |
| `database` | a database name in data/databases, or a path to an .xlsx |
| `formulation` | objective and constraints, see Quantities |
| `settings` | common, advanced and vae, see Settings |

## Selection types

Lengths are numbers in model units, or text in numbers, pi and h, the largest element size: `"<k>h"` for k times it, or arithmetic such as `"5.64 + 1h"`, worked out on each mesh. `pad` widens every bound; `"auto"` is half an element for nodes and 0 for element centres (faces and planes always half an element).

| type | fields | selects |
|---|---|---|
| `box` | lo, hi, pad | Points with lo <= p <= hi; the corners may come in any order. By default pad is half the element size along each axis, so a box thinner than an element still catches the nearest layer. |
| `cylinder` | center, axis, radius, inner_radius, along, pad, boundary | Points whose distance to the axis is in [inner_radius, radius]; along = (t0, t1) limits (p - center) . axis. boundary=True keeps only nodes on the mesh boundary; it is written only when on. |
| `difference` | parts | Points in the first part but not the second. |
| `face` | side, pad | The plane at the mesh's extreme on one side: xmin, xmax, ymin, ymax, zmin or zmax. |
| `function` | ref | Whatever a Python function name(mesh, on) in file.py returns. |
| `halfspaces` | normals, origins, boundary | Points on the inner side of every plane: n . (p - o) <= 1e-12 for each normal n and origin o. A box dragged in the PyTO-AD GUI is six such planes. boundary=True keeps only nodes on the mesh boundary, as the GUI does. |
| `ids` | indices | Explicit node or element numbers; tied to one mesh. |
| `inside_stl` | path, shift | Points inside a closed STL surface, the surface first moved by shift (x, y, z in model units), e.g. when the STL was saved shifted to 0, 0, 0. shift is written only when it is not zero. |
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

A project file keeps each as {"kind": kind, "on": a selection's name, ...its values}; a setup script adds one with p.add_bc(kind, on, **values).

| kind | physics | values | means |
|---|---|---|---|
| `fixed` | `structural`, `thermoelastic` | `dofs` (the directions held) | Holds the selection still in the ticked directions. |
| `displacement` | `structural`, `thermoelastic` | `dofs` (the directions moved), `value` (the displacement) | Moves the selection by a set distance in the ticked directions. |
| `force` | `structural`, `thermoelastic` | `total` (the total force, split equally over the nodes) | A total force on the selection, shared equally by its nodes. |
| `traction` | `structural`, `thermoelastic` | `pressure` (the pressure), `direction` (the direction it pushes), `area` (the area it acts on, or bbox: the selection's extent across the direction) | A pressure on the selection, turned into a force: Pressure times Area, pushing along Direction, shared equally by its nodes. Tick Area from the selection to use the selection's size across the direction (Direction must then be along x, y or z). |
| `fixed_temperature` | `thermal`, `thermoelastic` | `value` (the temperature held) | Holds the selection at a set temperature. |
| `heat_source` | `thermal`, `thermoelastic` | `total` (the heat put in, split equally over the nodes) | Puts heat into the selection, shared equally by its nodes. |

## Material database attributes

A database is an .xlsx file: attribute names in row 1, units in row 2, then one row per material with its name in column A. Every column must be one of these attributes (or one of its other names); the Materials panel of the app shows the errors and warnings of a file.

| attribute | other names | meaning | allowed range | units |
|---|---|---|---|---|
| `Density` |  | mass per unit volume | 0 < value | kg/m^3, g/cm^3, t/mm^3, lb/in^3, lbf*s^2/in^4 |
| `Cost` |  | price per unit mass | 0 <= value | USD/kg, USD/g, USD/t, USD/lb, USD/(lbf*s^2/in) |
| `Youngs_Modulus` | Youngs Modulus, Young_Modulus, E | Young's modulus, the stiffness | 0 < value | Pa, kPa, MPa, GPa, psi, ksi |
| `Poissons_Ratio` | Poisson_Ratio, Poissons Ratio, nu | Poisson's ratio | -1 < value < 0.5 | none (may be blank) |
| `Yield_Strength` | Yield Strength, Yield | yield strength, the stress at which the material yields | 0 < value | Pa, kPa, MPa, GPa, psi, ksi |
| `Conductivity` | Thermal_Conductivity, Thermal Conductivity, k | thermal conductivity, how well heat flows through it | 0 < value | W/(m*K), mW/(mm*K), BTU/(s*in*°R) |
| `Thermal_Expansion` | Thermal Expansion, CTE, alpha | coefficient of thermal expansion, the strain per degree of temperature rise | 0 < value | 1/K, 1/°R |

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
| `thermal` | steady heat conduction, one temperature per node | Conductivity |
| `thermoelastic` | linear isotropic thermoelasticity, one-way coupled: the temperature drives the displacement | Youngs_Modulus, Conductivity, Thermal_Expansion |

## Quantities

An objective or a constraint is one quantity, e.g. `Mass()` or `MaterialVolumeFraction(2, tau=0.5)`. A material is its row in the database, counted from 0. `MassFraction` also takes `heaviest` or `lightest` in its place, e.g. `MassFraction(lightest)`. A constraint is value / bound - 1 <= 0. An empty physics column: every physics has it.

| name | shown as | physics | objective | constraint | needs columns | value | meaning |
|---|---|---|---|---|---|---|---|
| `Compliance` | Compliance | `structural` | yes | yes |  | F . u | How much the part gives under the loads. Lower is stiffer. |
| `StressPNorm` | Stress (p-norm) | `structural` | yes |  |  | p-norm of the relaxed von Mises stress, no 1/N | One value close to the highest von Mises stress in the part. |
| `FailureFactorPNorm` | Failure factor (p-norm) | `structural` | yes | yes | Yield_Strength | mean p-norm of von Mises / yield strength, per element | One value close to the highest stress / yield strength in the part. Above 1 means some element yields. |
| `ThermalCompliance` | Thermal compliance | `thermal` | yes | yes | Conductivity | f . T | How hot the part gets where the heat goes in. Lower carries the heat away better. |
| `ThermoelasticCompliance` | Thermoelastic compliance | `thermoelastic` | yes | yes |  | (F + F_th) . u, the total load including the thermal force | How much the part gives under the loads and the heating. Lower is stiffer. |
| `Mass` | Mass |  | yes | yes | Density | sum of density * x_phys * element volume | The mass of the part, from each material's density. |
| `Cost` | Cost |  |  | yes | Density, Cost | sum of density * cost * x_phys * element volume | The cost of the part, from each material's density and cost per mass. |
| `VolumeFraction` | Volume fraction |  |  | yes |  | mean of x_phys | The share of the design space filled with material, 0 to 1. |
| `MaterialVolumeFraction` | Material volume fraction |  |  | yes |  | volume fraction of one database material, softmax weights | The share of the design space filled with one chosen material, 0 to 1. |
| `MassFraction` | Mass fraction |  |  | yes | Density | mass / (reference density * design domain volume), reference from the database | The part's mass compared with the whole part made solid in one material: 0.4 allows 40 % of that mass. Choose the material, or the heaviest or the lightest of the database. |

## Settings

Every setting a run takes, under its own name. A new project starts from these values (the settings of the paper's examples); a file missing some is completed from them, and the run says which. The VAE's own settings start from the database's known values. The mesh settings are the file's `mesh` section; the others are `settings.<section>`. An empty physics column: every project has it.

### mesh

| setting | physics | default | meaning |
|---|---|---|---|
| `n_dof` |  | 30000 | How fine the mesh is: the number of unknowns it is built for, counting each unknown at each node (three per node for a structural problem: the movement in x, y and z). More is finer, and slower. |
| `single_layer` |  | False | One element through the thickness, for a flat (2D) part. |

### common

| setting | physics | default | meaning |
|---|---|---|---|
| `maxIterations` |  | 300 | The optimization stops after this many iterations. |
| `relative_filter_radius` |  | 2.5 | The smallest feature the design may have, in element sizes. Larger gives smoother, simpler designs. |
| `material_filter_radius` |  | None | How far material choices are smoothed between neighbouring elements, in element sizes. Ticked: the same as the density filter radius. Empty (None): same as density filter. |
| `beta_schedule_scheme` |  | 'PAPER' (one of PAPER, PAPER_BM1) | How fast the design is pushed to clear solid and empty: the sharpness starts at 1 and grows 1.5 times every 5 iterations after iteration 50, up to 32 (PAPER) or 16 (PAPER_BM1). |
| `pnorm_exponent` | `structural` | 6.0 | How closely the single stress value follows the highest stress in the part. Larger is closer, but harder to optimize. |
| `reference_temperature` | `thermoelastic` | 293.15 | The temperature at which the part has no thermal strain, in the project's temperature unit. |

### advanced

| setting | physics | default | meaning |
|---|---|---|---|
| `timeLimit` |  | 36000 | The optimization stops after this many seconds. |
| `rel_conv_tol` |  | 1e-07 | The optimization stops early once the objective changes by less than this, relatively, between iterations, and every constraint is met to within this. |
| `use_heaviside_projection` |  | True | Pushes each element towards fully solid or fully empty. |
| `eta_proj` |  | 0.5 | The density at which that push splits solid from empty. |
| `use_penalization` |  | True | Pushes each element's material towards one of the real materials in the database. |
| `gamma_init` |  | 0.001 | How strong that push is at the start. |
| `gamma_max` |  | 10.0 | How strong it can get. |
| `gamma_factor` |  | 1.1 | How much stronger it gets each iteration while it ramps up. |
| `penalty_start_iter` |  | 0 | The iteration it starts at, when Start the ramp when the objective plateaus is off. |
| `distance_penalty_disable_after_iter` |  | 250 | The iteration it stops at, when Start the ramp when the objective plateaus is off. |
| `use_dynamic_gamma_scheduling` |  | True | Start the push when the design stops improving, instead of at a set iteration. |
| `gamma_plateau_window` |  | 10 | The objective has plateaued when it changed by less than the plateau tolerance over this many iterations. |
| `gamma_plateau_tolerance` |  | 0.001 | How little the objective may change over the plateau window, relatively, to count as plateaued. |
| `gamma_plateau_earliest_iter` |  | 20 | The push never starts before this iteration. |
| `gamma_ramp_end_margin` |  | 30 | If the objective never plateaus, the push starts in time to reach its largest weight this many iterations before Max iterations. |
| `use_z_smoothing` |  | True | Discourages neighbouring elements from flickering between materials. |
| `z_smoothing_weight` |  | 0.0001 | How strong that smoothing is. |
| `z_smoothing_disable_after_iter` |  | 250 | The iteration it stops at. |
| `use_continuation` |  | False | Raises the SIMP penalty by 0.25 every 10 iterations during the run. |
| `z0_init_method` |  | 'ORIGIN' (one of LIGHTEST, HEAVIEST, ORIGIN, UNIFORM) | The material every element starts with: ORIGIN (the middle of the material map), the LIGHTEST or HEAVIEST material, or UNIFORM (random). |
| `snap_to_real_material` |  | False | In the final design, give each element the nearest real material. |
| `binarize_topology` |  | True | In the final design, make each element fully solid or empty, keeping the volume. |
| `use_constant_poissons_ratio` | `structural` | False | One Poisson's ratio for every material, instead of each material's own. |
| `constant_poissons_ratio` | `structural` | 0.3 | The Poisson's ratio used when Constant Poisson's ratio is ticked. |
| `te_use_constant_poissons_ratio` | `thermoelastic` | False | One Poisson's ratio for every material, instead of each material's own. |
| `te_constant_poissons_ratio` | `thermoelastic` | 0.3 | The Poisson's ratio used when Constant Poisson's ratio is ticked. |

### vae

| setting | physics | default | meaning |
|---|---|---|---|
| `use_pretrained_vae` |  | True | Use the network already trained for this database. Unticked, the next run trains a new one. |
| `saveNet` |  | None | The trained network: always the .nt file next to the database, with the same name. Empty (None): next to the database. |
| `network_error_warning` |  | 5.0 | Before a run, you are warned if the network reproduces any material property less closely than this, in % of the property's range in the database. You can still run. |
| `latentDim` |  | from the database | How many numbers describe one material in the material map. |
| `learningRate` |  | from the database | The step size of the network's training. |
| `vae_hiddenDim` |  | from the database | The size of the network's inner layers. |
| `maxAttributeErrorPercent` |  | from the database | Training stops once the network reproduces every material property within this percent. |
| `klFactor` |  | from the database | How strongly training keeps the material map compact and smooth. |
| `numEpochs` |  | from the database | Training stops after this many passes at the latest. |


## Python

In the examples, `p` is your project: `p = Project(...)` for a new one, or `p = Project.load('my project.mmto')` for a saved one.

Every call a setup script uses, with its arguments, read from the code's own docstrings; `Python_walkthrough.ipynb` uses them on one whole project. Model units are the project's units (`p.units`). A length is a number in model units, or text in numbers, pi and h, the largest element size, such as `"1.5h"` or `"5.64 + 1h"`. A selection is made with one of the selection types first and then added to a project under a name. The examples follow on from one another, run from the LSR MMTO folder: the selections, then a new project that the next examples build up, save and write as a script, then the shipped U-bracket, run and read back.

### box

`box(lo, hi, pad='auto')`, after `from lsr_mmto import box`

Points with lo <= p <= hi; the corners may come in any order. By default pad is half the element size along each axis, so a box thinner than an element still catches the nearest layer.

| argument | format | default |
|---|---|---|
| `lo` | one corner, [x, y, z] in model units | required |
| `hi` | the opposite corner, [x, y, z] in model units | required |
| `pad` | how far every bound is widened: "auto" (see Selection types), "half" (half an element), or a length, a number in model units or text in element sizes h such as "0.5h"; 0 is exact | `'auto'` |

```python
tip = box([0.95, 0.9, 0], [1, 1, 0.01])
```

### cylinder

`cylinder(center, axis, radius, inner_radius=0.0, along=None, pad='auto', boundary=False)`, after `from lsr_mmto import cylinder`

Points whose distance to the axis is in [inner_radius, radius]; along = (t0, t1) limits (p - center) . axis. boundary=True keeps only nodes on the mesh boundary; it is written only when on.

| argument | format | default |
|---|---|---|
| `center` | a point on the axis, [x, y, z] in model units | required |
| `axis` | its direction: "x", "y", "z", or a vector such as [1, 1, 0] | required |
| `radius` | the outer radius, a length: a number in model units or text such as "1.5h" | required |
| `inner_radius` | the inner radius, a length; 0 is a full cylinder, more is a tube | `0.0` |
| `along` | where it starts and ends along the axis, measured from center: [start, end], two lengths; None is endless | `None` |
| `pad` | how far every bound is widened: "auto" (see Selection types), "half" (half an element), or a length, a number in model units or text in element sizes h such as "0.5h"; 0 is exact | `'auto'` |
| `boundary` | True keeps only the nodes on the outside of the mesh, False all | `False` |

```python
boss = cylinder(center=[0.875, 0.25, 0], axis="z", radius=0.05)
```

### sphere

`sphere(center, radius, pad='auto')`, after `from lsr_mmto import sphere`

Points within radius of center.

| argument | format | default |
|---|---|---|
| `center` | [x, y, z] in model units | required |
| `radius` | a length: a number in model units or text such as "1.5h" | required |
| `pad` | how far every bound is widened: "auto" (see Selection types), "half" (half an element), or a length, a number in model units or text in element sizes h such as "0.5h"; 0 is exact | `'auto'` |

```python
ball = sphere([0.25, 0.25, 0.005], 0.1)
```

### plane

`plane(axis, value, pad='auto', limits=None)`, after `from lsr_mmto import plane`

Points with \|p_axis - value\| <= pad; by default pad is half the element size along axis. limits, e.g. {"y": [100, 120]}, keeps only the part of the cut between those values of the other axes (each widened by pad along its own axis).

| argument | format | default |
|---|---|---|
| `axis` | the axis across the plane: "x", "y" or "z" | required |
| `value` | where the plane cuts that axis, a number in model units | required |
| `pad` | how far every bound is widened: "auto" (see Selection types), "half" (half an element), or a length, a number in model units or text in element sizes h such as "0.5h"; 0 is exact | `'auto'` |
| `limits` | None for the whole cut, or {other axis: [from, to]} in model units, for one or both other axes, e.g. {"x": [0.75, 1]} | `None` |

```python
top = plane("y", 1.0, limits={"x": [0.75, 1]})
```

### face

`face(side, pad='auto')`, after `from lsr_mmto import face`

The plane at the mesh's extreme on one side: xmin, xmax, ymin, ymax, zmin or zmax.

| argument | format | default |
|---|---|---|
| `side` | "xmin", "xmax", "ymin", "ymax", "zmin" or "zmax" | required |
| `pad` | how far every bound is widened: "auto" (see Selection types), "half" (half an element), or a length, a number in model units or text in element sizes h such as "0.5h"; 0 is exact | `'auto'` |

```python
left = face("xmin")
```

### near_face

`near_face(side, distance)`, after `from lsr_mmto import near_face`

Points strictly within distance of the mesh's extreme on one side.

| argument | format | default |
|---|---|---|
| `side` | "xmin", "xmax", "ymin", "ymax", "zmin" or "zmax" | required |
| `distance` | how far in from that side, a length: a number in model units or text such as "1.5h" | required |

```python
bottom = near_face("ymin", "1.5h")
```

### nearest

`nearest(point, boundary=False)`, after `from lsr_mmto import nearest`

The single node or element closest to point; boundary=True only looks at nodes on the mesh boundary, as the PyTO-AD GUI does for a picked point.

| argument | format | default |
|---|---|---|
| `point` | [x, y, z] in model units | required |
| `boundary` | True looks only at the nodes on the outside of the mesh | `False` |

```python
corner = nearest([1, 1, 0])
```

### ids

`ids(indices)`, after `from lsr_mmto import ids`

Explicit node or element numbers; tied to one mesh.

| argument | format | default |
|---|---|---|
| `indices` | a list of node or element numbers, counted from 0 | required |

```python
first = ids([0, 1, 2])
```

### stl_faces

`stl_faces(triangles, tol=None)`, after `from lsr_mmto import stl_faces`

Boundary nodes on triangles of the STL the mesh was made from: within tol times the smallest element size of a triangle. tol None is PyTO-AD's own lookup (0.707); the PyTO-AD GUI uses 0.8.

| argument | format | default |
|---|---|---|
| `triangles` | a list of triangle numbers of the STL, counted from 0 | required |
| `tol` | how close a node must be, in smallest element sizes; None is 0.707 | `None` |

```python
faces = stl_faces([0, 1])
```

### halfspaces

`halfspaces(normals, origins, boundary=True)`, after `from lsr_mmto import halfspaces`

Points on the inner side of every plane: n . (p - o) <= 1e-12 for each normal n and origin o. A box dragged in the PyTO-AD GUI is six such planes. boundary=True keeps only nodes on the mesh boundary, as the GUI does.

| argument | format | default |
|---|---|---|
| `normals` | a list of vectors [a, b, c], each pointing out of its plane | required |
| `origins` | a list of points [x, y, z] in model units, one on each plane, in the same order | required |
| `boundary` | True keeps only the nodes on the outside of the mesh, False all | `True` |

```python
right = halfspaces([[-1, 0, 0]], [[0.9, 0, 0]])
```

### rows

`rows(face, axis, n, end='max')`, after `from lsr_mmto import rows`

The n outermost node rows of a face, counted along an axis: e.g. the 8 highest rows of the right face. Rows are node layers, so their height is n element sizes.

| argument | format | default |
|---|---|---|
| `face` | the side the rows are on: "xmin", "xmax", "ymin", "ymax", "zmin" or "zmax" | required |
| `axis` | the axis the rows are counted along: "x", "y" or "z" | required |
| `n` | how many rows, a whole number | required |
| `end` | "max" counts from the high end of axis, "min" from the low end | `'max'` |

```python
upper = rows("xmax", "y", 8)
```

### inside_stl

`inside_stl(path, shift=(0.0, 0.0, 0.0))`, after `from lsr_mmto import inside_stl`

Points inside a closed STL surface, the surface first moved by shift (x, y, z in model units), e.g. when the STL was saved shifted to 0, 0, 0. shift is written only when it is not zero.

| argument | format | default |
|---|---|---|
| `path` | the STL file, absolute or relative to the project's folder | required |
| `shift` | how far the surface is moved first, [x, y, z] in model units | `(0.0, 0.0, 0.0)` |

```python
inside = inside_stl("data/models/UBracket/UBracket.STL")
```

### function

`function(ref)`, after `from lsr_mmto import function`

Whatever a Python function name(mesh, on) in file.py returns.

| argument | format | default |
|---|---|---|
| `ref` | "file.py:name": the Python file, absolute or relative to the project's folder, and the function in it | required |

```python
picked = function("my_selection.py:pick")
```

### union

`union(*parts)`, after `from lsr_mmto import union`

Points in any of the parts.

| argument | format | default |
|---|---|---|
| `parts` | two or more selections, each a selection type above or a name in selection("name") | required |

```python
ends = union(left, tip)
```

### intersect

`intersect(*parts)`, after `from lsr_mmto import intersect`

Points in all of the parts.

| argument | format | default |
|---|---|---|
| `parts` | two or more selections, each a selection type above or a name in selection("name") | required |

```python
both = intersect(bottom, left)
```

### difference

`difference(a, b)`, after `from lsr_mmto import difference`

Points in the first part but not the second.

| argument | format | default |
|---|---|---|
| `a` | the selection to start from | required |
| `b` | the selection taken away from it | required |

```python
rest = difference(left, bottom)
```

### selection

`selection(name)`, after `from lsr_mmto import selection`

Another selection of the project, by name. Only a project can evaluate it: Project.evaluate puts the named selection in (resolved) each time, so a combination follows later edits of its parts.

| argument | format | default |
|---|---|---|
| `name` | the name the other selection was added under | required |

```python
link = selection("left")
```

### Project

`Project(name, stl, n_dof=None, single_layer=None, base_dir=None)`, after `from lsr_mmto import Project`

One problem: the part, its selections, supports and loads, keep-solid regions, material database, objective and constraints, and settings.

| argument | format | default |
|---|---|---|
| `name` | the project's name, text; its runs are named after it | required |
| `stl` | the part, an STL file: its path, absolute or relative to base_dir | required |
| `n_dof` | how fine the mesh is, a whole number of unknowns (see Settings, mesh); None is the default there | `None` |
| `single_layer` | True for one element through the thickness (a flat part such as a plate), False for a solid part; None is the default under Settings, mesh | `None` |
| `base_dir` | the folder relative paths start from; None is the folder Python runs in | `None` |

```python
p = Project("my bracket", stl="data/models/UBracket/UBracket.STL", n_dof=8000)
```

### p.units

`p.units = value`

The unit system: "MKS", "mmKS", "IPS" (those of the PyTO-AD GUI, kept where the GUI keeps it, settings.unit_system) or "none". Numbers are never converted; the unit system says what they are meant to be in, so that a database in other units is reported.

| argument | format | default |
|---|---|---|
| `value` | "MKS", "mmKS", "IPS" or "none" (see Unit systems) | `'none'` |

```python
p.units = "MKS"
```

### p.comment

`p.comment = value`

| argument | format | default |
|---|---|---|
| `value` | a description, text | `''` |

```python
p.comment = "U-bracket, fixed on the left"
```

### p.physics

`p.physics = value`

| argument | format | default |
|---|---|---|
| `value` | the physics the optimization uses, one of those under Physics | `'structural'` |

```python
p.physics = "structural"
```

### p.add_selection

`p.add_selection(name, selection)`

Adds a selection under a name, which supports, loads and keep solid can then use; returns the name.

| argument | format | default |
|---|---|---|
| `name` | the selection's name, text | required |
| `selection` | one of the selection types above, e.g. face("xmin") | required |

```python
p.add_selection("left", left)
```

### p.mesh

`p.mesh()`

The voxel mesh of the part, made from the mesh settings (n_dof, single_layer) as a run makes it.

```python
mesh = p.mesh()
```

### p.evaluate

`p.evaluate(selection, mesh, on='nodes')`

Indices of a selection (or of the selection with that name) on mesh, with every selection(name) in it looked up now, so a combination follows later edits of its parts. Elements of a selection that picks only nodes (stl_faces, rows, ...) are the elements touching those nodes.

| argument | format | default |
|---|---|---|
| `selection` | a selection's name, or a selection type above | required |
| `mesh` | the mesh, from p.mesh() | required |
| `on` | "nodes" for node numbers, "elements" for element numbers | `'nodes'` |

```python
print(len(p.evaluate("left", mesh)))
```

### p.add_bc

`p.add_bc(kind, on, **values)`

A support or load of a kind of BC_KINDS (the structural ones: fixed, displacement, force, traction) on a selection, its name or a Selection (stored under a new name), with the kind's values, e.g. p.add_bc("fixed", "clamp", dofs="xyz"), p.add_bc("force", "corner", total=[0, -4, 0]). The values are checked as the project's check checks them; a wrong one is an error here.

| argument | format | default |
|---|---|---|
| `kind` | one of the kinds under Boundary conditions, such as "fixed" or "force" | required |
| `on` | the selection it acts on: its name, or a selection type above (added under a new name) | required |
| `values` | the kind's values, as keywords, e.g. dofs="xyz" or total=[0, -1000, 0]; each kind's values are under Boundary conditions | required |

```python
p.add_bc("fixed", "left", dofs="xyz")
```

### p.keep_solid

`p.keep_solid(*names)`

Keeps these selections solid: their elements are always material.

| argument | format | default |
|---|---|---|
| `names` | one or more selections: their names, or selection types above (each added under a new name) | required |

```python
p.keep_solid(boss)
```

### p.keep_support_elements

`p.keep_support_elements = value`

| argument | format | default |
|---|---|---|
| `value` | True also keeps solid every element touching a fixed node | `False` |

```python
p.keep_support_elements = True
```

### p.database

`p.database = value`

The material database. When the database chosen has a trained network, and the trained network is used (the vae setting use_pretrained_vae), the settings that give the network its shape (latentDim, vae_hiddenDim) are set from the network itself, and a printed line says which changed.

| argument | format | default |
|---|---|---|
| `value` | the name of one in data/databases, such as "6Materials_AF_3Attr", or the path of an .xlsx file, absolute or relative to base_dir | `None` |

```python
p.database = "6Materials_AF_3Attr"
```

### p.network_error

`p.network_error()`

How closely the database's trained network gives back its materials: (rows, line). rows: one for each material and property, the database's value and the network's; line: the same in one sentence, as the Materials panel shows it. (None, None) when the database has no trained network yet, or the network cannot be read for it (vae_check says why).

```python
rows, line = p.network_error()
```

### p.vae_check

`p.vae_check()`

(errors, warnings, status) about the VAE: a run that would train into the shipped folder, a trained network that does not fit the settings or the database, a network a run would replace, a latent dimension other than the recommended one; status says which network the database has. For a database that passes its check. Each error says how to put it right.

```python
errors, warnings, status = p.vae_check()
```

### p.objective

`p.objective(expression, sense='minimize')`

What the optimization makes as small as it can.

| argument | format | default |
|---|---|---|
| `expression` | one quantity, text, such as "Mass()" or "FailureFactorPNorm()" (see Quantities) | required |
| `sense` | "minimize", the only one for now | `'minimize'` |

```python
p.objective("FailureFactorPNorm()")
```

### p.constraint

`p.constraint(expression, op, bound)`

A limit on a quantity, expression op bound; each call adds one more constraint.

| argument | format | default |
|---|---|---|
| `expression` | one quantity, text, such as "Mass()" or "MaterialVolumeFraction(2, tau=0.5)" (see Quantities) | required |
| `op` | "<=", the only one for now | required |
| `bound` | the limit, a number in the project's units | required |

```python
p.constraint("Mass()", "<=", 0.0035)
```

### p.settings

`p.settings[section][name] = value`

| argument | format | default |
|---|---|---|
| `value` | one setting: section is "common", "advanced" or "vae", name a setting of that section; every setting, its default and what it does are under Settings | the values under Settings |

```python
p.settings["common"]["maxIterations"] = 50
```

### p.save

`p.save(path)`

Write the project, every setting included, with the other sections it was loaded with. Always as .mmto. Its folder becomes the project's folder, as the window keeps it: the part and the database with its network are copied into part/ and materials/ there (project_folder.pack), unless they are in it already.

| argument | format | default |
|---|---|---|
| `path` | the .mmto file, absolute or relative to the folder Python runs in | required |

```python
p.save("my bracket/my bracket.mmto")
```

### p.to_script

`p.to_script(path)`

Write a Python script that rebuilds this project with the same calls, next to it. Settings that differ from a new project's are written; the rest are the defaults.

| argument | format | default |
|---|---|---|
| `path` | the .py file, in the project's folder | required |

```python
p.to_script("my bracket/my bracket_setup.py")
```

### Project.load

`Project.load(path)`

A project read from its .mmto file, as the GUI and p.save write it.

| argument | format | default |
|---|---|---|
| `path` | the .mmto file, absolute or relative to the folder Python runs in | required |

```python
p = Project.load("projects/U-bracket.mmto")
```

### run

`run(project, save_dir=None, **run_options)`, after `from lsr_mmto import run`

Optimizes the project. The run's record keeps the project as it was run (Project.from_record rebuilds it). Returns the folder the results went to; None when nothing was saved.

| argument | format | default |
|---|---|---|
| `project` | the project, a Project | required |
| `save_dir` | the folder the results go into; None is a new folder in results in the project's folder, named after the date and time | `None` |
| `run_options` | any of the options below, as keywords | none |
| `verbose` | True prints each iteration's numbers | `True` |
| `plot_progress` | True draws the design in a window as it changes | `False` |
| `plot_bcs` | True opens a window with the supports and loads before the optimization starts | `False` |
| `plotter` | a PyVista plotter the design is drawn into instead of a new window, or None | `None` |
| `progress` | a function called after each iteration with a dict of its numbers (iteration, objective, constraints, x_phys, z, z_real), or None | `None` |
| `should_stop` | a function the run asks as it goes; when it returns True the run stops and saves the design reached so far; or None | `None` |
| `on_network_error` | a function called with the warning when the trained network is further from the database than the vae setting network_error_warning; when it returns False nothing is optimized; or None | `None` |

```python
folder = run(p)
```

### load

`load(path)`, after `from lsr_mmto.results.load_results import load`

A saved run read back, as a Run.

| argument | format | default |
|---|---|---|
| `path` | the run's folder (one of those in results in the project's folder), or the run_data.npz in it; the .npz may be left off | required |

```python
r = load(folder)
```

### Run fields

One optimization run. Per-element arrays are in solver element order.

| field | what it is |
|---|---|
| `path` | the run_data.npz file it was read from |
| `n_elems` | how many elements the mesh has |
| `solid` | one True or False per element: material or empty, once the design is made all or nothing |
| `x_cont` | each element's density, 0 to 1, before that |
| `material_indices` | each element's material, its row in the database, counted from 0 |
| `material_labels` | the materials' names, as tables and legends show them |
| `results` | {name: values}, the physics' results, e.g. results["von_mises"]: one value per element, or as result_info[name] says (where: "element" or "node"; shape: "scalar", or "vector" with x, y, z) |
| `history` | {name: one value per iteration}, e.g. history["J_phys"], the objective |
| `iterations` | how many iterations the optimizer made |
| `final_objective` | the objective at the last iteration |
| `objective_label` | the objective's name as the GUI shows it |
| `constraint(i)` | constraint i (counted from 0) at the last iteration, as value / bound - 1: 0 or less is met |

```python
print(r.iterations, r.final_objective)
```

### summary

`summary(run)`, after `from lsr_mmto.results.tables import summary`

One run at a glance, in sections: the result (objective and each constraint, met or not), the design, the materials and the run. An analysis: the same for its one solve, without what only an optimization has (grey elements, the latent space, iterations) and the constraints it cannot give. A notebook shows it as tables.

| argument | format | default |
|---|---|---|
| `run` | a Run, from load | required |

```python
summary(r)
```

### design

`design(run, out=None, *, field='material', size=(1200, 950), view='iso', **kw)`, after `from lsr_mmto.results.run_plots import design`

The optimized design, drawn the way this example wants. 2D examples get a matplotlib panel; 3D examples get a mirrored voxel render placed on an axes, so both return a figure and display the same way. field applies to the 2D path only: material \| topology \| density \| a result's name (von_mises).

| argument | format | default |
|---|---|---|
| `run` | a Run, from load | required |
| `out` | None returns the figure; a path without its extension writes a .png and a .pdf there | `None` |
| `field` | what colours the design of a flat part: "material", "topology", "density" or a result's name, such as "von_mises"; a solid part shows "material" only | `'material'` |
| `size` | (width, height) in pixels of a solid part's picture | `(1200, 950)` |
| `view` | where a solid part is seen from: "iso", "xy", "xz" or "yz", with a "-" in front for the other side | `'iso'` |
| `kw` | angles=(azimuth, elevation, roll, zoom) turns a solid part's view further, in degrees, and zooms by a factor | none |

```python
design(r)
```

### mesh_for

`mesh_for(run)`, after `from lsr_mmto.results.design_3d import mesh_for`

The voxel mesh this run used, rebuilt and cached. nDOFDesired comes from the run's own config, so the rebuild matches the grid the run solved on rather than the example's current default.

| argument | format | default |
|---|---|---|
| `run` | a Run, from load | required |

```python
mesh = mesh_for(r)
```

### export

`export(run, mesh, folder, materials=True)`, after `from lsr_mmto.results.stl_export import export`

design.stl (and design_\<material>.stl) into folder, as Export STL in the GUI writes them; returns the Report, whose lines() say what was written and whether each file is closed.

| argument | format | default |
|---|---|---|
| `run` | a Run, from load | required |
| `mesh` | the mesh it was solved on, from mesh_for(run) | required |
| `folder` | the folder the files go into, usually the run's own | required |
| `materials` | True also writes one file for each material, False only the whole part | `True` |

```python
report = export(r, mesh, folder)
```

### analyse_design

`analyse_design(project, run_folder, save_dir=None, should_stop=None)`, after `from lsr_mmto.optimize.solid_part_analysis import analyse_design`

Analyses a saved run's design as built, solved once under the project's supports and loads as they are now, with its settings, checked as a run is checked. Each element with a density above 0.5 is solid, in its nearest material of the database the run used (the run's material_indices, as Export STL takes them), with that material's values as the database gives them; the others are empty. The run's mesh must be the project's now: the same mesh dofs, single element layer and part, else a ValueError says what differs. Returns the analysis as a Run (load), its record saying which run it analysed (Run.analysed_design).

| argument | format | default |
|---|---|---|
| `project` | the project, a Project, with the supports and loads to analyse the design under | required |
| `run_folder` | the run's folder (run returns it), one of those in results in the project's folder | required |
| `save_dir` | the folder the analysis goes into, kept with the runs; None saves nothing | `None` |
| `should_stop` | a function asked between its steps; when it returns True the analysis stops and saves nothing; or None | `None` |

```python
a = analyse_design(p, folder, save_dir=folder + " design")
```
