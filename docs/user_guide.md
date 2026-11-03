# LSR MMTO user guide

This guide walks through every part of the LSR MMTO window. Open it in the window with Help > User guide, or press F1.

Next to most fields there is a small (i). Click it, or rest the mouse on it, to see what that field means. This guide gives the bigger picture: what each part is for, how the parts fit together, and what to watch out for.

## 1. Start here

To be written.

## 2. The window

To be written.

## 3. File menu

To be written.

## 4. Units

To be written.

## 5. Geometry

To be written.

## 6. Materials

To be written.

The attributes a material database can hold:

<!-- list: material_attributes -->

| Attribute | What it is | Units |
| --- | --- | --- |
| Density | Mass per unit volume | kg/m^3, g/cm^3, t/mm^3, lb/in^3, lbf*s^2/in^4 |
| Young's modulus | Young's modulus, the stiffness | Pa, kPa, MPa, GPa, psi, ksi |
| Poisson's ratio | Poisson's ratio | - |
| Yield strength | Yield strength, the stress at which the material yields | Pa, kPa, MPa, GPa, psi, ksi |
| Cost | Price per unit mass | USD/kg, USD/g, USD/t, USD/lb, USD/(lbf*s^2/in) |

<!-- end of list -->

## 7. Supports and loads

A part needs two things before it can be optimized: something that holds it, and something that pushes on it. In LSR MMTO these are **supports** and **loads**. Each one acts on a **selection**: a named piece of the part, such as one face, a hole, or a few faces you clicked.

Open **Supports and loads** from the buttons on the right of the window. The panel shows:

- **Add**, to put a new support or load on the part, and **Preview** and **Clear**.
- A list of the supports and loads, with their **Type**, **Where** (the selection they act on) and **Value**. Double-click one to change it. **Delete** removes the one you clicked.
- **Selections**, closed at first. Click it to see every selection of the project, with its **Name**, its **Type**, and a **Keep solid** tick.

Changes go into the project at once. Use File > Save to keep them.

### 7.1 Two ways to add a support or load

Click **Add** and choose one of:

- **Click on the part...** opens the **Pick** window. You click faces on the part, choose what they are, and press **Apply**. This is the quickest way for a face you can see.
- **Enter by hand...** opens a form. You choose the **Type**, choose the selection under **Where** (or make one there with **New selection...**), and fill in the values. Use this when a face is hard to click, when you need exact numbers, or when the place is not a face at all (a hole, a patch, a point).

Both end up in the same list, and you can change either one later by double-clicking it.

### 7.2 Picking faces on the part

Click **Add**, then **Click on the part...**. The Pick window opens beside the part.

1. Under **Selection**, choose how a click selects:
   - **Facet**: the face you click and every face joined to it with no sharp edge in between (an edge that bends more than about 15 degrees stops it). One click selects a whole flat side, or a whole smoothly curved surface such as the inside of a hole.
   - **Triangle**: only the single triangle you click. The edges of the triangles show, so you can see them. Use it for a small area.
   - **Node**: the point of the part nearest to your click. **Box select nodes** then lets you drag a rectangle: every node inside it is selected, through the whole depth of the part.
2. Click the part. Each click adds to what is selected; clicking a selected face again takes it off. A right click clears everything selected so far.
3. Under **Type**, choose **Fixed XYZ**, **Fixed X**, **Fixed Y** or **Fixed Z** for a support, or **Force** for a load. For a force, give its X, Y and Z under **Force Components**: this is the total force on everything you selected.
4. Press **Apply**.

The pick becomes a support or load in the list, on a new selection named gui_support1, gui_load1 and so on. Rename the selection under Selections if you like.

Faces are only picked while the Pick window is open. With it closed, clicking and dragging on the part just turns the view.

A picked selection remembers the triangles of the STL by their numbers. If you load a different STL for the part, the same numbers are different faces, so pick them again.

### 7.3 Selections

A selection is a named piece of the part. Several supports and loads can share one selection, and the name shows under **Where**. To make one, open **Selections** and click **Add**, or click **New selection...** in the support or load form.

The Selection form has a **Name** (letters, digits and _, not starting with a digit), a type, and the type's own fields. The (i) next to the type says what it selects. The part's size is shown under the type, so you can see which numbers make sense. Two buttons help you check a selection before you keep it:

- **Preview** draws the selection on the part in orange, and says how many nodes it has on the mesh the run will use.
- **Clear** removes that drawing again.

While the form is open, rest the mouse on the part to read the coordinates under it. Click the part to copy that point; paste it into a field with Ctrl+V.

The types of selection:

- **Face of the part**: every node on one outer side of the part, such as Right (x max) or Bottom (y min). The quickest way to hold or load a whole side of a box-like part. *Example: Side Left (x min) to clamp the left end of a beam.*
- **Rows on a face**: a few rows of nodes at one end of a face. Use it to put a load on a small patch instead of a whole face. *Example: the top 8 rows of the right face: Face Right (x max), Counted along y, Number of rows 8, From High end.* A row is one layer of nodes, so the patch is as tall as that many elements.
- **Near a face**: every node within a set **Distance** of one outer side. Use it when the side is not perfectly flat, or when you want a thicker band than a single face.
- **Flat cut**: the nodes on a flat cut through the part, across x, y or z. *Example: every node where y is 38: Cut across y, At 38.* To keep only part of the cut, fill in the from and to of the other directions, for example x from 0 to 40. Leave them empty for the whole cut.
- **Box**: every node inside a box, given by two opposite corners (**Corner 1** and **Corner 2**, each as x, y, z). The corners can be in any order. A box thinner than an element still catches the nearest layer of nodes.
- **Sphere**: every node within a **Radius** of a **Centre**.
- **Cylinder**: every node inside a cylinder, or inside a tube if you give an **Inner radius**. Good for holes and bolt bores. Give a **Point on the axis**, the **Axis direction** (x, y, z, or a direction such as 1, 1, 0) and the **Radius**. Leave the two ends empty for a cylinder through the whole part. With the axis along x, y or z, the ends are coordinates, for example x from 22 to 61. Along a tilted axis they are distances from the point on the axis. *Example: the inside of a bolt hole of radius 5 through z at x 20, y 30: Point on the axis 20, 30, 0, Axis direction z, Radius 5.5, Inner radius 4.5.*
- **Closest node**: the one node nearest to a **Point**. Tick **Surface nodes only** to stay on the outside of the part. Use it for a load at one point.
- **Inside an STL**: every node inside a closed surface loaded from another STL file (**File**, with **Browse...**). Use it for a shape no other type describes, such as a collar around a bore.
- **Combined selections**: two selections of the project joined, overlapped, or one taken from the other. **How** is one of **Join: in A or in B**, **Overlap: in both A and B**, or **A minus B: in A, not in B**. It follows them: change Selection A or Selection B later and this selection changes too. It appears once the project has at least two selections. *Example: the left face without its bottom rows: A minus B, with A a face and B the rows.*

Three more types are made with the mouse, and show up when you open such a selection: **Picked faces** (faces you clicked), **Dragged box** (a box you dragged in Node mode) and **Node numbers** (single nodes, by number; they stay right only while the mesh does not change).

Most types have a **Tolerance**: how far outside the selection a node may sit and still count. **Automatic** is right almost always.

The **Code line** tab shows the same selection as one line of text, for example `rows(face="xmax", axis="y", n=8, end="max")`. You can type a selection there instead of filling in the form.

Every selection type at a glance:

<!-- list: selection_types -->

| Selection | What it selects |
| --- | --- |
| Face of the part | All nodes on one outer face of the part. |
| Rows on a face | A few rows of nodes at one end of a face. For example, the top 8 rows of the right face: Face Right, Counted along y, Number of rows 8, From High end. Use it to spread a load over a small patch. |
| Near a face | Nodes within a set distance of one outer face. |
| Flat cut | Nodes on a flat cut through the part. For example, every node where y is 38: Cut across y, At 38. To keep only part of the cut, fill in the from and to of the other directions, e.g. x from 0 to 40; leave them empty for the whole cut. |
| Box | Nodes inside a box given by two opposite corners. |
| Sphere | Nodes within a radius of a point. |
| Cylinder | Nodes inside a cylinder, or a tube if you give an inner radius. Good for holes and bolt bores. Leave the ends empty for a cylinder through the whole part. With the axis along x, y or z the ends are coordinates, e.g. x from 22 to 61; along a tilted axis they are distances from the point on the axis. |
| Closest node | The one node closest to a point. |
| Inside an STL | Nodes inside a closed surface loaded from an STL file. |
| Combined selections | Two selections of this project joined, overlapped or one taken from the other. It follows them: edit Selection A or Selection B later and this selection changes with it. |
| Picked faces | Faces you clicked with the mouse. Snap distance is how close a node must be to a face to count, in element sizes; leave it empty for the usual 0.7. Made with the mouse; open it from Selections to change it. |
| Dragged box | A box you dragged with the mouse, as the planes that bound it. Made with the mouse; open it from Selections to change it. |
| Node numbers | Specific node numbers. They stay right only while the mesh does not change. Made with the mouse; open it from Selections to change it. |

<!-- end of list -->

### 7.4 Supports

A support holds the part. Without one, the part would float away under the loads.

- **Fixed support**: holds the selection still. Under **Held in**, tick the directions it is held in: x, y and z for a clamp, or only some of them for a support that lets the part slide in the others (for example, held in y only, like a part resting on a floor).
- **Moved support**: moves the selection by a set **Distance** in the ticked directions (**Moved in**), instead of holding it still. Use it to push or pull on a part by a known amount rather than with a known force.

A support given in the Pick window is a Fixed support: Fixed XYZ holds x, y and z; Fixed X, Fixed Y and Fixed Z hold one direction each.

### 7.5 Loads

A load pushes on the part.

- **Force**: a **Total force** on the selection, given as its x, y and z parts. The total is shared equally between the nodes of the selection: a force of 0, -100, 0 on a selection of 50 nodes puts 0, -2, 0 on each node. The force does not grow when the selection does, so a bigger selection only spreads the same force wider.
- **Pressure**: a **Pressure** on the selection, turned into one force: Pressure times **Area**, pushing along **Direction**, shared equally between the nodes. Give the area as a number, or tick **from the selection** to use the size of the selection across the direction. With from the selection, the Direction must be along x, y or z, and the area is that of the rectangle around the selection, seen along the Direction. On a round or uneven face that rectangle is larger than the face, so type the area instead.

The force in the Pick window is a Force: its X, Y and Z are the total over everything you selected.

The units show beside each field, from the project's units (for example N and MPa for mmKS). See Units, section 4.

### 7.6 Keep solid

Tick **Keep solid** next to a selection, under Selections, and the optimization keeps material there: it favours those elements above every other, so they end the run solid. Use it for the places the part must keep whatever happens: where bolts go, where it is clamped, a mounting face.

A selection can be kept solid without any support or load on it.

### 7.7 The preview

Press **Preview** at the top of the panel to see everything on the mesh the run will use:

- supports as blue points,
- loads as red points, with an arrow for each load,
- other selections as orange points,
- kept-solid elements as grey blocks.

Under the buttons, one line gives the verdict:

- **The supports hold the part against the loads.** All is well.
- **Error: the supports leave the part free to move or rotate in a direction the loads push...** Add supports, or tick more directions on one, until the part cannot move or turn.

A selection with no nodes on the mesh shows in red: *name has no nodes on this mesh*. The (i) next to the verdict lists the size of the mesh, how many nodes each selection has, and how many elements are kept solid.

Click a row of the list, or a selection under Selections, to see only that one on the part. **Clear** removes the preview, and any faces picked in the Pick window but not applied.

The preview uses the run's mesh, not the STL, so it shows exactly the nodes the run will use. A finer mesh (Settings) gives more nodes, and can change which nodes a small selection catches.

### 7.8 Common mistakes

- **The part is not held in every direction.** A part held only in y can still slide in x and z, or spin. The preview says so. Hold it in more directions, or add a second support. A clamp on one face (Fixed support, held in x, y and z) is the usual fix.
- **A load or support on a selection with no nodes.** A box or a flat cut that misses the part, or a Rows on a face that counts from the wrong end, selects nothing. The preview shows the selection in red. Press Preview in the Selection form to check its node count before you keep it.
- **Two supports on the same node in the same direction, with different values.** For example a Fixed support and a Moved support whose selections overlap. The preview and the run stop with a message that a direction of a node is already given another value. Make the selections not overlap, or tick that direction on only one of them.
- **A force that is too big or too small by a factor of 1000 or more.** The values are in the project's units. A force of 1000 means 1000 N in mmKS, and 1000 lbf in IPS. Check Units before you type the numbers.
- **A pressure with the wrong area.** Pressure times Area is the force. With from the selection, the area is the rectangle around the selection; on a round face, type the area instead.
- **Picked faces after changing the STL.** Picked faces are stored by triangle number, so they belong to one STL. Pick them again on the new one.

Every type of support and load at a glance:

<!-- list: support_and_load_kinds -->

| Type | Support or load | What it does | Values |
| --- | --- | --- | --- |
| Fixed support | Support | Holds the selection still in the ticked directions. | Held in |
| Moved support | Support | Moves the selection by a set distance in the ticked directions. | Moved in, Distance |
| Force | Load | A total force on the selection, shared equally by its nodes. | Total force |
| Pressure | Load | A pressure on the selection, turned into a force: Pressure times Area, pushing along Direction, shared equally by its nodes. Tick Area from the selection to use the selection's size across the direction (Direction must then be along x, y or z). | Pressure, Direction, Area |

<!-- end of list -->

## 8. Formulation

To be written.

What the objective and the constraints can be:

<!-- list: quantities -->

| Quantity | Can be | What it is |
| --- | --- | --- |
| Compliance | Objective and constraint | How much the part gives under the loads. Lower is stiffer. |
| Stress (p-norm) | Objective | One value close to the highest von Mises stress in the part. |
| Failure factor (p-norm) | Objective and constraint | One value close to the highest stress / yield strength in the part. Above 1 means some element yields. |
| Mass | Objective and constraint | The mass of the part, from each material's density. |
| Cost | Constraint | The cost of the part, from each material's density and cost per mass. |
| Volume fraction | Constraint | The share of the design space filled with material, 0 to 1. |
| Material volume fraction | Constraint | The share of the design space filled with one chosen material, 0 to 1. |

<!-- end of list -->

## 9. Settings

To be written.

Every setting, in the groups of the Settings panel:

<!-- list: settings -->

| Group | Setting | What it does |
| --- | --- | --- |
| Main | Mesh dofs | How fine the mesh is: the number of unknowns it is built for. More is finer, and slower. |
| Main | Single element layer (2D) | One element through the thickness, for a flat (2D) part. |
| Main | Max iterations | The optimization stops after this many iterations. |
| Main | Density filter radius | The smallest feature the design may have, in element sizes. Larger gives smoother, simpler designs. |
| Main | Latent smoothing radius | How far material choices are smoothed between neighbouring elements, in element sizes. Ticked: the same as the density filter radius. |
| Main | Projection sharpness schedule | How fast the design is pushed to clear solid and empty: the sharpness starts at 1 and grows 1.5 times every 5 iterations after iteration 50, up to 32 (PAPER) or 16 (PAPER_BM1). |
| Run length | Time limit [s] | The optimization stops after this many seconds. |
| Run length | Convergence tolerance | The optimization stops early once the objective changes by less than this, relatively, between iterations, and every constraint is met to within this. |
| Projection | Heaviside projection | Pushes each element towards fully solid or fully empty. |
| Projection | Projection threshold | The density at which that push splits solid from empty. |
| Distance penalty | Distance penalty | Pushes each element's material towards one of the real materials in the database. |
| Distance penalty | Starting weight | How strong that push is at the start. |
| Distance penalty | Largest weight | How strong it can get. |
| Distance penalty | Weight growth per iteration | How much stronger it gets each iteration while it ramps up. |
| Distance penalty | First iteration | The iteration it starts at, when Start the ramp when the objective plateaus is off. |
| Distance penalty | Off after iteration | The iteration it stops at, when Start the ramp when the objective plateaus is off. |
| Distance penalty | Start the ramp when the objective plateaus | Start the push when the design stops improving, instead of at a set iteration. |
| Distance penalty | Plateau window | The objective has plateaued when it changed by less than the plateau tolerance over this many iterations. |
| Distance penalty | Plateau tolerance | How little the objective may change over the plateau window, relatively, to count as plateaued. |
| Distance penalty | Earliest start | The push never starts before this iteration. |
| Distance penalty | Iterations left at full weight | If the objective never plateaus, the push starts in time to reach its largest weight this many iterations before Max iterations. |
| Latent smoothing | Latent smoothing | Discourages neighbouring elements from flickering between materials. |
| Latent smoothing | Weight | How strong that smoothing is. |
| Latent smoothing | Off after iteration | The iteration it stops at. |
| Material model | SIMP continuation | Raises the SIMP penalty by 0.25 every 10 iterations during the run. |
| Start and reported design | Starting latent point | The material every element starts with: ORIGIN (the middle of the material map), the LIGHTEST or HEAVIEST material, or UNIFORM (random). |
| Start and reported design | Snap to real materials | In the final design, give each element the nearest real material. |
| Start and reported design | Binarize the topology | In the final design, make each element fully solid or empty, keeping the volume. |
| VAE | Use the trained network | Use the network already trained for this database. Unticked, the next run trains a new one. |
| VAE | VAE file | The trained network: always the .nt file next to the database, with the same name. |
| VAE | Warn above network error [%] | Before a run, you are warned if the network reproduces any material property less closely than this, in % of the property's range in the database. You can still run. |
| VAE | Latent dimension | How many numbers describe one material in the material map. |
| VAE | Learning rate | The step size of the network's training. |
| VAE | Hidden layer width | The size of the network's inner layers. |
| VAE | Stop at attribute error [%] | Training stops once the network reproduces every material property within this percent. |
| VAE | KL weight | How strongly training keeps the material map compact and smooth. |
| VAE | Max epochs | Training stops after this many passes at the latest. |
| Main | p-norm exponent | How closely the single stress value follows the highest stress in the part. Larger is closer, but harder to optimize. |
| Material model | Constant Poisson's ratio | One Poisson's ratio for every material, instead of each material's own. |
| Material model | Poisson's ratio value | The Poisson's ratio used when Constant Poisson's ratio is ticked. |

<!-- end of list -->

## 10. Running

To be written.

## 11. Results

To be written.

## 12. Export STL

To be written.

## 13. Display options

To be written.

## 14. When something goes wrong

To be written.
