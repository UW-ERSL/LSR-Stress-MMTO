# LSR MMTO user guide

This guide walks through every part of the LSR MMTO window. Open it in the window with Help > User guide, or press F1.

Next to most fields there is a small (i). Click it, or rest the mouse on it, to see what that field means. This guide gives the bigger picture: what each part is for, how the parts fit together, and what to watch out for.

## 1. Start here

LSR MMTO designs a part made of several materials. You give it the shape the part may fill, where the part is held and pushed, and a list of materials. It works out where material should go, and which material goes where, so that the part does what you ask (for example, as stiff as it can be for a given weight).

The work always goes in the same order, and the buttons on the right of the window follow it from top to bottom:

1. **The part.** File > New makes a project from an STL file of the space the part may fill. See The part, section 5.
2. **The units.** The unit system all your numbers are in. See Units, section 4.
3. **The materials.** A material database: one row per material, with its properties. See Materials, section 6.
4. **Supports and loads.** Where the part is held, and where and how hard it is pushed. See section 7.
5. **The objective and the constraints.** What to make as small as possible, and the limits it must keep to. See Formulation, section 8.
6. **Run.** The blue button at the top. The design grows on the screen while it runs. See Running, section 10.
7. **The results.** Each run's design, its numbers and its plots. See Results, section 11.
8. **Export STL.** The design as STL files, one for the whole part and one per material. See section 12.

Settings (section 9) has everything else, and most projects can leave it as it is.

Next to most fields there is a small (i). Click it, or rest the mouse on it, to see what that field means and what its numbers are. When a field puzzles you, the (i) is the quickest answer. This guide gives the bigger picture.

To try LSR MMTO before making a project of your own, open one of the examples: click it under **Try an example** on the start page, or use File > Open. The examples are the L-bracket, the U-bracket, the beam and the airplane bracket. They are complete, so you can press Run at once. An example never changes: when you save it, or run it, LSR MMTO asks for a folder of your own and saves a copy there.

## 2. The window

While no project is open, the 3D view shows the start page: **New project from an STL...** (the same as File > New), **Open a project...** (File > Open), four examples under **Try an example**, and under **Recent** the last five projects you opened or saved, each with its folder under its name. Click one to open it. Until a project is open, the buttons on the right, Run, Export STL and Part / Design are greyed.

### 2.1 The top row

From left to right:

- **LSR MMTO**, and the **File** and **Help** menus. File is described in section 3. Help has this **User guide** (also F1) and **About LSR MMTO**.
- **Part** and **Design**, a two-way switch for the 3D view. **Part** shows the part, with its supports, loads and previews. **Design** shows the design of the last run you made or opened. Design is greyed until there is a design to show.
- The **moon** button switches the 3D view to dark; it then shows a **sun**, which switches it back to light. The 3D view starts light. Only the background and the window's own drawings (the ruler, Measure, labels) change colour; the materials keep the colours the database gives them.
- **Export STL** writes the design on the screen as STL files. It is greyed while there is no finished design on the screen. See section 12.
- **Run** starts a run of the project. While the run goes, the same button reads **Stop**, and the run's progress shows beside it, for example *Iteration 29 / 40 · 0:33 so far · about 0:12 left*. See section 10.

### 2.2 The buttons on the right

**Units**, **Materials**, **Supports and loads**, **Formulation**, **Settings**, **Current run**, **Results** and **Display Options**. Units, Materials, Supports and loads and Formulation show how far each step is: a green tick once it is set up and nothing in it stops a run, a red mark when something in it would stop a run (the messages say what, as an **Error**), and the button's own picture while it is not set up yet. A warning does not change the mark.

Units, Materials, Formulation, Settings and Display Options open as windows with OK and Cancel (or Apply and Close): OK puts your changes into the project, and Cancel leaves the project as it was.

Supports and loads, Current run and Results open as panels beside the 3D view, so you can work on the part while they are open. When more than one is open they share the space as tabs. A panel can be made wider or narrower by dragging its edge.

While a run goes, the buttons on the right step aside, and the Current run panel takes half the window beside the design growing in the 3D view. You can still drag the divider between them. When the run ends, or you stop it, the buttons come back and the panel goes back to its width.

### 2.3 The 3D view

- Drag with the left mouse button to turn the part.
- Turn the mouse wheel to zoom.
- Drag with the middle button, or with Shift and the left button, to move the part across the view.

A click on the part selects nothing, unless one of these is open:

- the **Pick** window (section 7.2), where a click picks faces;
- **Measure** (below), where a click is a point to measure;
- a form under Supports and loads, where a click copies the point under the mouse, to paste into a field with Ctrl+V.

### 2.4 The messages

Under the 3D view, three lines of messages: what was opened and saved, what a run is doing, and any errors and warnings about the project. A message repeated at once is shown once, with how many times, for example *(×3)*. Scroll up to read earlier ones.

An **Error** stops a run from starting until it is put right. A **Warning** lets the run go, but says something is probably not what you meant. Section 14 lists the common ones.

### 2.5 The bottom bar

- On the left, the part's size, for example *Part: x 0 to 200 · y 0 to 200 · z 0 to 1 mm*. The (i) next to it gives the part's file, volume, size and material database.
- In the middle, the point under the mouse on the part, and what Measure has measured.
- **Snap**: rounds the point under the mouse, a copied point and Measure to a step. The steps follow the part's size: a round number near a thousandth, a hundredth and a tenth of its largest extent, in the project's length unit (on a 200 mm part 0.2, 2 and 20 mm). With Snap: 2 mm, a point at x 37.62 reads 38.
- **Measure**: click it, then click two points on the part to see the distance between them, and its x, y and z parts. A third click starts again. The line and its points are dark in the light 3D view and white in the dark one. Press Esc, or click Measure again, to stop.
- **Ruler**: shows scales along the part, in the project's length unit. Click it again to hide them.

## 3. File menu

- **New...** makes a new project. Give it a **Name**, a **Location**, the **Part (STL)** and the **Physics** (Structural is the one there is today). LSR MMTO makes a folder of that name at that location, copies the STL into it and saves the project there. Every setting starts at its usual value.
- **Open...** opens a project, a file ending in .mmto. Only LSR MMTO projects open here.
- **Save** saves the project in its file. A project never saved, or an example, is saved with Save As instead.
- **Save As...** saves the project as a new one, in a new folder, with everything it uses copied in. The new project takes the new name; the runs made so far stay with the old one.
- **Replace part...** changes the project's STL for another one. See section 5.
- **Export as script** writes the project as a short Python script, *name*_setup.py, in the project's folder. Every save writes it again, so it always matches the project; this entry saves first if needed and says where the script is. Open it in any text editor to see the whole project written out, or to change it.
- **Run script...** runs a Python script, such as one written by Export as script and then changed. What the script prints goes to the messages. When the script saves a project in its folder, that project opens in the window.
- **Exit** closes LSR MMTO.

The window's title shows the project's name, with *(example)* for an example and a * when there are changes not saved yet. Before closing, or opening another project, LSR MMTO asks whether to save them.

### 3.1 The project folder

A project keeps everything it uses in its own folder:

- the project file, *name*.mmto, and its script, *name*_setup.py;
- **part**: the STL, and any other STL a selection uses;
- **materials**: the material database and its trained network, when the database is one of your own;
- **results**: one folder per run, named by when it was made.

Zip the folder to send the whole project to someone else.

## 4. Units

**Units** sets the unit system every number of the project is in. Choose it under **Unit System**:

| Unit system | Length | Force | Stress | Density | Mass |
| --- | --- | --- | --- | --- | --- |
| MKS | m | N | Pa | kg/m^3 | kg |
| mmKS | mm | N | MPa | t/mm^3 | t |
| IPS | in | lbf | psi | lbf*s^2/in^4 | lbf*s^2/in |

Press **Apply**. The messages say *Project units: mmKS. Numbers are not converted.*

Nothing is converted. The numbers you typed stay as they are, and only their unit changes. Choose the unit system first, then type the numbers in it: the part's size, forces, pressures and distances, and the bounds of the constraints. The units show beside every field that has one.

The material database has its own units, in its second row. When they do not match the project's (a Young's modulus in Pa in an mmKS project, for example), the messages say so and by how much, and what it changes in the results. See section 14.

**Temperature Unit** and **Angle Unit** are in the same window; LSR MMTO does not use them yet.

## 5. The part

The part is an STL file of the space the part may fill: the design domain. The optimization decides where, inside it, material goes. The surface of the STL is the outside of that space.

The part comes in when you make the project (File > New). To use a different STL, choose File > **Replace part...** and choose the new STL. If some selections are faces you picked on the old STL, LSR MMTO says so first: picked faces are stored by triangle number, and on another STL the same numbers are other faces. Pick them again afterwards. In a project folder of its own, the new STL is copied into the part folder.

The STL should be closed (watertight) and in the project's length unit. The size in the bottom bar (section 2.5) is a quick check: a part that should be 200 mm long and reads *x 0 to 0.2 mm* was drawn in metres.

### 5.1 The mesh

A run does not work on the STL itself. It fills the part with a grid of small equal blocks, the elements, and decides the material of each one. The corners of the blocks are the nodes, which supports and loads act on.

How fine the grid is comes from **Mesh dofs** in Settings: the number of unknowns the mesh is built for, three at each node for a structural problem (the movement in x, y and z). More is finer, and slower. A finer mesh shows more detail, and makes smoother STL files when you export.

For a flat part, tick **Single element layer (2D)** in Settings: the grid then has one element through the thickness, and the run is much faster.

The preview under Supports and loads (section 7.7) builds the same mesh the run will use, so you can see how many elements and nodes the run will have.

## 6. Materials

The materials the optimization chooses from come from a **material database**: a spreadsheet with one row per material and one column per property (density, Young's modulus, and so on). LSR MMTO learns a map of the materials from it, the material network, so that the optimization can move smoothly from one material to another. Each element of the design ends up with one of the database's materials.

Open **Materials** from the buttons on the right. The window shows:

- **Database:** the database the project uses. The list has the databases that come with LSR MMTO, and the project's own.
- A table of the database's materials, each in the colour and with the label it is drawn with, and its properties with their units.
- Under the table, a short report: how many materials there are, the latent dimension recommended for them, how closely the trained network gives them back, and any errors or warnings. The (i) next to it explains the recommended latent dimension.

Choose a database and press **OK**. When you change the database, the network settings of the new one come with it (see Settings, section 9), and the messages say what changed. In a project folder of its own, the database you choose is copied into the project's materials folder, with its trained network, so the project keeps everything it uses.

### 6.1 Where a database comes from

The buttons next to the list:

- **New database...** opens the database editor (section 6.3) on an empty database, to type or paste your materials in.
- **Import...** makes a database from a table of materials you already have, in Excel (.xlsx or .xls) or CSV. The first column holds the material names and the first row the properties. A property can be written under any of the names LSR MMTO knows it by (for example Young's modulus as Youngs_Modulus, Youngs Modulus or E), and a unit may follow in brackets, as in *Density [kg/m^3]*. A second row of units, as in an LSR MMTO database, is read too. Columns that are not a property LSR MMTO knows are left out, and one message lists them. The database editor then opens with what was read, for you to check and save.
- **Edit...** opens the database in the editor. One of your own is changed in place. One that comes with LSR MMTO is saved as a copy in your project's folder, and the project then uses the copy. A column that is not a property LSR MMTO knows is left out, and a note over the table names it; Save writes the database without it.
- **Browse...** chooses a database file (.xlsx) from anywhere.
- **Open in Excel** opens the database file in Excel, or whatever opens .xlsx files on your computer.
- **Check again** reads the database again, for example after you changed it in Excel.

New database and Import save into the project's folder, so the project must be saved first; LSR MMTO says so if it is not.

### 6.2 Values and Network error

Above the table, two buttons choose what the table shows:

- **Values**: each material's properties, as the database gives them.
- **Network error**: for each value, how far the trained network is from it when it turns the material into its point on the material map and back, in % of that property's range over the database's materials. Rest the mouse on a cell to see both values and the difference in the property's unit.

Network error is greyed until the database has a trained network. The first run with a new database trains one.

A small error (well under 1 %) means the run works with properties very close to the database's. If an error is large, the run's materials differ from the real ones by that much; see the network warning in section 14.

### 6.3 The database editor

The editor shows the database as a table:

- **Name**: the database's name, which is also its file name.
- One row per material: its name, **Shown as** (an optional shorter name for legends and plots) and **Colour** (click it to choose another).
- One column per property. The row under the headers holds each property's unit.
- **Add material** and **Remove material** add and remove rows. **Add property...** adds a column for one of the properties below; **Remove property** removes the column you are in.

You can paste a block copied from Excel into the table. Problems are listed in red under the table as you type (a missing value, a value outside what the property allows, two materials with the same name). **Save** keeps the database in the project's folder and the project then uses it.

If you change the numbers of a database that already has a trained network, the network no longer fits it: LSR MMTO sets the old network aside, and the next run trains a new one.

### 6.4 The database file

A database is an Excel file (.xlsx): row 1 holds the property names, row 2 their units, and then one row per material, its name in the first column. An optional second sheet, **Display**, holds the colour (#rrggbb) and the shorter label each material is drawn with; the editor writes it for you.

The trained network is a file next to the database with the same name and the ending .nt (my_materials.xlsx and my_materials.nt). A database that comes with LSR MMTO has its network already, and the one in LSR MMTO's own folder is never trained again. Its copy in your project's folder can be: see Use the trained network, in Settings.

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
- **Enter by hand...** opens a form. You choose the **Type**, choose the selection under **Where** (or make one there with **New selection...**), and fill in the values. **Preview**, or Enter, checks the values and shows the selection on the part (with an arrow for a load), and the form stays open; **OK** keeps it. Use this when a face is hard to click, when you need exact numbers, or when the place is not a face at all (a hole, a patch, a point).

Both end up in the same list, and you can change either one later by double-clicking it.

### 7.2 Picking faces on the part

Click **Add**, then **Click on the part...**. The Pick window opens beside the part.

1. Under **Selection**, choose how a click selects:
   - **Facet**: the face you click and every face joined to it with no sharp edge in between (an edge that bends more than about 15 degrees stops it). One click selects a whole flat side, or a whole smoothly curved surface such as the inside of a hole.
   - **Triangle**: only the single triangle you click. The edges of the triangles show, so you can see them. Use it for a small area.
   - **Node**: the point of the part nearest to your click. Click again for each node you want.
2. Click the part. Each click adds to what is selected; clicking a selected face again takes it off. **Clear picks**, or a right click on the part, clears everything selected so far.
3. Under **Type**, choose **Fixed XYZ**, **Fixed X**, **Fixed Y** or **Fixed Z** for a support, or **Force** for a load. For a force, give its X, Y and Z under **Force Components**: this is the total force on everything you selected.
4. Press **Apply**.

The pick becomes a support or load in the list, on a new selection named gui_support1, gui_load1 and so on. Rename the selection under Selections if you like.

Faces are only picked while the Pick window is open. Closing it clears whatever was picked and not applied. With it closed, clicking and dragging on the part just turns the view.

A picked selection remembers the triangles of the STL by their numbers. If you load a different STL for the part, the same numbers are different faces, so pick them again.

### 7.3 Selections

A selection is a named piece of the part. Several supports and loads can share one selection, and the name shows under **Where**. To make one, open **Selections** and click **Add**, or click **New selection...** in the support or load form.

The Selection form has a **Name** (letters, digits and _, not starting with a digit), a type, and the type's own fields. The (i) next to the type says what it selects. The part's size is shown under the type, so you can see which numbers make sense. Two buttons help you check a selection before you keep it:

- **Preview** draws the selection on the part in orange, and says how many nodes it has on the mesh the run will use.
- **Clear** removes that drawing again.

Pressing Enter in the form does the same as **Preview**: it checks what you typed and shows the selection on the part, and the form stays open. A mistake shows in red under the form. Only **OK** keeps the selection and closes the form.

While the form is open, rest the mouse on the part to read the coordinates under it. Click the part to copy that point; paste it into a field with Ctrl+V.

The types of selection:

- **Face of the part**: every node on one outer side of the part, such as Right (x max) or Bottom (y min). The quickest way to hold or load a whole side of a box-like part. *Example: Side Left (x min) to clamp the left end of a beam.*
- **Rows on a face**: a few rows of nodes at one end of a face. Use it to put a load on a small patch instead of a whole face. *Example: the top 8 rows of the right face: Face Right (x max), Counted along y, Number of rows 8, From High end.* A row is one layer of nodes, so the patch is as tall as that many elements.
- **Near a face**: every node within a set **Distance** of one outer side. Use it when the side is not perfectly flat, or when you want a thicker band than a single face.
- **Flat cut**: the nodes on a flat cut through the part, across x, y or z. *Example: every node where y is 38: Cut across y, At 38.* To keep only part of the cut, fill in the from and to of the other directions, for example x from 0 to 40. Leave them empty for the whole cut.
- **Box**: every node inside a box, given by two opposite corners (**Corner 1** and **Corner 2**, each as x, y, z). The corners can be in any order. A box thinner than an element still catches the nearest layer of nodes.
- **Sphere**: every node within a **Radius** of a **Centre**.
- **Cylinder**: every node inside a cylinder, or inside a tube if you give an **Inner radius**. Good for holes and bolt bores. Give a **Point on the axis**, the **Axis direction** (x, y, z, or a direction such as 1, 1, 0) and the **Radius**. Leave the two ends empty for a cylinder through the whole part. With the axis along x, y or z, the ends are coordinates, for example x from 22 to 61. Along a tilted axis they are distances from the point on the axis. *Example: the inside of a bolt hole of radius 5 through z at x 20, y 30: Point on the axis 20, 30, 0, Axis direction z, Radius 5.5, Inner radius 4.5.* For a round hole in the part, click **Click a hole** and then click the inside wall of the hole: the form is filled in with the point on its axis, its axis, its ends, and a Radius of the hole's radius + 1h, one element size larger, so the nodes of the wall are in on any mesh. If the wall is not round, the form says *This doesn't look like a round hole* and fills nothing.
- **Closest node**: the one node nearest to a **Point**. Tick **Surface nodes only** to stay on the outside of the part. Use it for a load at one point.
- **Inside an STL**: every node inside a closed surface loaded from another STL file (**File**, with **Browse...**). Use it for a shape no other type describes, such as a collar around a bore. **Move by** (x, y, z, default 0, 0, 0) moves the shape before nodes are selected, for example when the STL was saved shifted to 0, 0, 0. Preview draws the shape where it ends up.
- **Combined selections**: two selections of the project joined, overlapped, or one taken from the other. **How** is one of **Join: in A or in B**, **Overlap: in both A and B**, or **A minus B: in A, not in B**. It follows them: change Selection A or Selection B later and this selection changes too. It appears once the project has at least two selections. *Example: the left face without its bottom rows: A minus B, with A a face and B the rows.*

Three more types are made with the mouse, and show up when you open such a selection: **Picked faces** (faces you clicked), **Node numbers** (single nodes, by number; they stay right only while the mesh does not change) and **Dragged box** (a box dragged over the part in earlier versions; a project that has one still opens and runs as before).

Most types have a **Tolerance**: how far outside the selection a node may sit and still count. **Automatic** is right almost always.

The **Code line** tab shows the same selection as one line of text, for example `rows(face="xmax", axis="y", n=8, end="max")`. You can type a selection there instead of filling in the form.

A number can be typed as plain arithmetic, in the Selection form, on the Code line and in the support or load form: for example 38/2, 0.5*76 or 2*pi*5, also inside a point (0, 38/2, 0). You can use + - * / ** ( ) and pi. The project keeps the number it comes to.

A length of a selection (a radius, a distance, the ends of a cylinder, a tolerance) can also be given in element sizes with h: 1.5h is one and a half element sizes, and 5 + 1h is 5 plus one element size. Such a length is kept as you typed it and follows the mesh: a finer mesh gives a smaller h.

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
| Inside an STL | Nodes inside a closed surface loaded from an STL file. Move by moves the shape first. |
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

**Formulation** says what the optimization is for: one quantity to make as small as possible (the objective), and limits other quantities must keep to (the constraints).

- **Minimize**: choose the objective. For example **Compliance**, to make the part as stiff as it can be, or **Mass**, to make it as light as it can be.
- **Constraints**: one row per constraint: the quantity, then **<=** and its limit, with the limit's unit beside it. For example *Mass <= 0.002 t*, or *Volume fraction <= 0.3* (at most 30 % of the design space filled). The (i) at the end of the row says what the quantity is. **Add constraint** adds a row; **Remove** takes one away. A new row starts at a limit that suits its quantity (0.4 for a fraction, 1 for the failure factor, and for Mass 40 % of the part solid in the lightest material); change it to the one you want.
- A **Material volume fraction** constraint also names its material, chosen from the database's materials in their colours: *Material volume fraction of Mat1_Red <= 0.15* keeps that material to at most 15 % of the design space. With one or more of these, **Material sharpness (tau)** appears above the rows: how sharply an element counts as its nearest material when the volumes are measured. One value is used for all of them; the usual value is right in most cases.
- A **Mass fraction** constraint compares the part's mass with the mass of the whole part made solid in one material: *Mass fraction of the lightest <= 0.4 of a solid part in that material* allows 40 % of what the part would weigh solid in the lightest material of the database. The material is one of the database's, or **the heaviest** or **the lightest** of them. The limit has no unit, so it stays right when the units or the part's size change.

A quantity the database cannot give is greyed out in the list, and resting the mouse on it says why, for example *needs a Yield_Strength column in the database*. Choose another database, or add that property to yours (section 6.3).

A constraint keeps its quantity at or below its limit. To ask for at least some amount, constrain the opposite: instead of a minimum stiffness, minimize compliance with a limit on mass, or minimize mass with a limit on compliance.

Some pairs that work well:

- **Stiffest part for its weight**: Minimize Compliance, with Mass (or Volume fraction) <= a limit.
- **Lightest part that does not yield**: Minimize Mass, with Failure factor (p-norm) <= 1.
- **Lowest stress**: Minimize Stress (p-norm), with Volume fraction <= a limit.
- **Cheapest part**: Minimize Mass or Compliance, with Cost <= a limit (the database needs a Cost column).

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
| Mass fraction | Constraint | The part's mass compared with the whole part made solid in one material: 0.4 allows 40 % of that mass. Choose the material, or the heaviest or the lightest of the database. |

<!-- end of list -->

## 9. Settings

**Settings** has everything about how the optimization runs. Most projects keep the values a new project starts with.

The window has two tabs:

- **Optimization**: how fine the mesh is, how long the run is, and how the design is pushed towards clear solid and empty. The **Main** settings are at the top. The others are under **Advanced**, closed at first; click it to open them, in their groups.
- **VAE**: the material network. At the top, a line says which network the project uses and whether it fits the database, with its errors and warnings under it. With **Use the trained network** ticked and a trained network there, the network's own settings are greyed: they are the ones it was trained with. Untick it to train a new network on the next run; the settings can then be changed.

Every setting has an (i) that says what it does. A setting with a value other than the usual one shows **↺** next to it: click it to put the usual value back.

Press **OK** to keep the changes, or **Cancel** to leave everything as it was.

The settings that matter most often:

- **Mesh dofs**: how fine the mesh is. More gives finer detail and smoother STL files, and takes longer. Start coarse while you set a problem up, and go finer for the design you keep.
- **Single element layer (2D)**: for a flat part, one element through its thickness. Much faster.
- **Max iterations**: the run stops after this many iterations at the latest. It can stop earlier, once the design has settled (Convergence tolerance, under Advanced).

Every setting, in the groups of the Settings panel:

<!-- list: settings -->

| Group | Setting | What it does |
| --- | --- | --- |
| Main | Mesh dofs | How fine the mesh is: the number of unknowns it is built for, counting each unknown at each node (three per node for a structural problem: the movement in x, y and z). More is finer, and slower. |
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

Press **Run** at the top. Before it starts:

- LSR MMTO checks the project. If something is missing or wrong, the run does not start, and the Current run panel lists why under *Cannot start:*. See section 14.
- The project is saved, so the run matches the file. If there are unsaved changes, LSR MMTO asks first. An example is saved as a project of your own first.

The run keeps its own copy of the project, so you can go on editing while it runs; the changes go into the next run.

### 10.1 While it runs

The buttons on the right step aside, and the **Current run** panel takes half the window. The run goes through these steps, and the line at the top of the panel (and beside the Stop button) says which:

1. *Starting: building the mesh and the material network...*
2. *Training the material network: epoch 4,000 / 30,000*, only when the database has no trained network yet. This can take a few minutes.
3. *Iteration 12 / 300 · 0:41 so far · about 16:20 left*: the optimization itself.

The 3D view shows the design as it grows: each element in its material's colour, pale while it is half solid and in full colour once solid. Elements that are almost empty are not drawn. At the top left, a legend gives each material's colour and its share of the design, largest first (counted by how solid each element is). It shows whenever the database has more than one material, also while the design uses only one of them. **Part**, at the top, shows the part again without stopping the run; **Design** brings the growing design back. Untick **Show the design while it runs** to leave the 3D view alone.

The panel's tabs:

- **Progress**: the objective, as a solid line, and each constraint, as a dotted line named as in Formulation with its limit, for example *Mass <= 0.4 t*. A constraint is met at or below the grey line at 0.
- **Latent space**: where the elements' materials are heading on the material map. Red points are the elements, black stars the database's materials. As the run goes on, the points usually gather on the stars.
- **Log**: everything the run writes, line by line.

### 10.2 Stopping

Press **Stop** (the Run button, while a run goes). The run ends after the iteration under way and keeps the design reached so far; the line says *Stopping after the iteration under way; the design so far is kept...* while it finishes. Stopped while the material network is still training, nothing is saved.

You cannot open another project, or close LSR MMTO, while a run goes. Stop it first.

### 10.3 When it ends

The run stops after Max iterations, or earlier once the design has settled. The line then says, for example, *Run finished after 40 iterations, 0:47*. The final design shows on the part, and three more tabs appear: **Convergence**, **Latent space** and **Summary**, the same as in Results (section 11).

Under the tabs:

- **Open results folder** opens the run's folder.
- **Open in Results** shows the run in the Results panel.
- **Delete this run** removes the run and its folder, after asking. This cannot be undone.

Every run has its own folder in the project's results folder, named by when it started.

How long a run takes depends mostly on Mesh dofs and Max iterations: a few minutes for a coarse flat part, up to hours for a fine 3D part.

## 11. Results

**Results** lists the runs of the project, newest first: when each ran, how many iterations it made, and *stopped* for a run that was stopped. Click one to see it:

- its design on the part, in the 3D view;
- **Convergence**: the objective and the constraints, iteration by iteration;
- **Latent space**: where the final design's materials sit on the material map;
- **Summary**: the run's numbers (below).

**Show** chooses what the part shows:

- **Material**: each element in its material's colour, with a legend at the top left: each material and its share of the solid elements, largest first. Two runs compared get a column each, left and right. The legend shows whenever the database has more than one material (a design of one of them reads 100%), and not for a database of one material.
- **Density**: how solid each element is, from empty to solid.
- **Stress**: the stress in each element.
- **Solid**: only which elements are solid, in one colour.
- **Smooth (exported)**: the STL files of the run, once it has been exported (section 12). Greyed until then.

**Open folder** opens the run's folder. **Delete this run** removes it, after asking.

### 11.1 The Summary

The Summary is in sections:

- **Result**: the objective, with its value and unit. Then a table of the constraints: each one's name, its **Value**, its **Limit**, how much of the limit it **Used**, and **✓ met** in green or **✗ not met** in red. A constraint counts as met when it is at most 0.1 % over its limit.
- **Design**: the number of **Elements**, how many are **Solid** (density 0.5 or more) and what share of the elements that is, and **Grey (in between)**: how much of the design is neither solid nor empty. 0 % is a clean design.
- **Materials**: how many of the database's materials the design **Used**, and the **Latent dimension**.
- **Run**: the number of **Iterations**, and the **Units**.

The (i) at the top of the Summary says what each word means, and gives each constraint's exact value and limit.

### 11.2 Comparing two runs

Tick two runs and press **Compare**. A Compare window opens with their convergence on one plot (each constraint by its name in the legend), their latent spaces side by side, and their summaries in two columns. The 3D view shows both designs side by side, the first on the left. Close the Compare window, or click a run, to stop comparing.

### 11.3 Results from elsewhere

**Open other results...** shows a run from any folder, another project's too, to look at. It changes nothing in your project. A run's folder is one that holds run_data.npz.

## 12. Export STL

**Export STL**, at the top, writes the design shown in the 3D view as STL files, for printing or for CAD. It works on a finished run's design: the last run, or the run shown in Results. It is greyed while the 3D view shows the part, while a run still goes, while two runs are compared, and for a run of another project.

The files go into the run's folder:

- **design.stl**: the whole part, in one piece, for printing in one material. Faces of your STL the optimization did not touch keep their exact shape.
- **design_*material*.stl**: one file per material the design uses. Together they make up the part, with no gaps and no overlaps between them.

The files are smoothed, so they do not show the staircase of the elements. Smoothing cannot add detail finer than the mesh: features one element across are dropped, and square corners come out rounded. For a part to print, run on a finer mesh.

While it works, a note at the top of the 3D view says it is exporting. When it is done, the note says how many files were saved, with a link, **Open folder**. The messages list every file with its volume next to the design's, and say whether each file is closed. *NOT closed* after a file means it has holes, which some printers and CAD programs refuse; a finer mesh usually helps.

After an export, Results shows the smooth pieces on the part, each in its material's colour: **Show** turns to **Smooth (exported)**. It is there for every run whose folder holds the exported files, also later, and greyed for a run not exported yet. Beside it, tick **Pull the material pieces apart** to set the pieces side by side in a row along x, the largest first, so each can be seen on its own. The legend at the top left names each piece, with its share of the volume.

## 13. Display options

**Display Options** changes how the part is drawn:

- **Cut along x (%)**, **Cut along y (%)** and **Cut along z (%)**: cut away that share of the part, along x, y or z, to see inside it. 0 shows it whole. The cut works on the design too.
- **Show bounding box**: the box around the part.
- **Show the part's triangles**: the edges of the STL's triangles.
- **See-through part**: the part drawn half transparent, to see supports, loads and selections behind it.
- **Hide Selected Part**: takes the highlight of picked faces off the view.
- **Save Image**: saves the 3D view as a picture (PNG or JPEG).

## 14. When something goes wrong

Each message below is written as LSR MMTO shows it, with what it means and what to do. Names such as corner, clamp or my_materials stand for your own selections and files. Some messages use the names a project file stores; this section says which field each one is.

### 14.1 Before a run

These appear in the messages as **Error:** lines, and under *Cannot start:* in the Current run panel. The run starts once they are put right.

- **no supports** / **no loads**: the project has no support, or no load. Add them under Supports and loads (section 7).
- **no material database**: choose one under Materials (section 6).
- **no objective**: choose one under Formulation (section 8).
- **boundary condition 2 (force): no selection named 'corner'**: a support or load acts on a selection that no longer exists. Edit it under Supports and loads and choose a selection under Where.
- **selection clamp: ...**: a selection cannot be made as it is written, for example a file it needs is missing. Open it under Selections and correct it.
- **material database: the structural model needs a Youngs_Modulus column**, or **material database: FailureFactorPNorm needs a Yield_Strength column**: the database lacks a property the problem needs (FailureFactorPNorm is the Failure factor (p-norm); Youngs_Modulus is Young's modulus). Add the column in the database editor, or choose another database or objective.
- **material database my_materials.xlsx: D1: 'E0' is not a property LSR MMTO knows ...**: a column of the database has a name LSR MMTO does not know. Rename the column to one of the properties listed, or open the database with Edit..., which leaves the column out, and save it.
- **material database my_materials.xlsx: ...**: a problem in the database itself, with the cell it is in, for example *C5 (Steel, Density): 'abc' is not a number*, *... is outside the allowed range*, *no units for Density*, or *Steel, Steel2 have the same value in every attribute, so the VAE cannot tell them apart*. Correct it in the database editor.
- **constraint 1: the bound must be a nonzero number**: a constraint's limit is 0 or empty. Give it a limit under Formulation.
- Warning **constraint 1 (Mass <= 0.0035): This limit allows at most 0.027% of the part even in the lightest material.**: the mass limit is less than 1 % of what the whole part weighs solid in the lightest material of the database, so almost no material can be used. Often the limit is in another unit than meant (g for kg, say). Check it under Formulation, and the project's units (section 4).
- **the material volume fraction constraints use different tau (0.5, 1)**: give them all the same Material sharpness.
- **STL file ... does not exist**: the part's file has moved or been deleted. Choose it again with File > Replace part... (section 5).

### 14.2 The material network

- **my_materials.nt takes 3 attributes, but my_materials.xlsx has 4: it was trained on another database**: the trained network does not fit the database. In Settings, on the VAE tab, untick **Use the trained network**; the next run trains a new one.
- **latentDim is 3, but my_materials.nt was trained with 2: set it to 2, or switch off use_pretrained_vae to train a new network**: a network setting differs from the trained network's. latentDim is **Latent dimension**, and use_pretrained_vae is **Use the trained network**, both in Settings on the VAE tab.
- **a run would train a new VAE and save it over the shipped ...**: a database that comes with LSR MMTO would have its network replaced. Copy the database into your project with Edit... (section 6.1) and use the copy.
- Warning **my_materials.nt exists: the next run trains a new network and replaces it**: Use the trained network is off, so the next run trains again. The old network is kept beside it, ending in .previous.nt.
- Warning **latentDim is 3; the database's effective dimension is ... so 2 is recommended**: the latent dimension is larger than the materials need, which has made runs fail to settle. Set Latent dimension to the recommended value, unless you have a reason not to.
- When a run starts, a window **Material network** can say *The material network reproduces the database less closely than the 5% warning limit: ...*, with the error of each property. The run would work with properties that far from your database. **Continue** runs anyway; **Cancel** stops before anything is made. To do better, train the network again (untick Use the trained network), or raise **Warn above network error [%]** in Settings if the error is acceptable to you.

### 14.3 Units

- Warning **the database gives Young's modulus in Pa, but the project's units (mmKS) expect MPa: its numbers are 10^6 times what they should be. Displacement and compliance come out 10^6 times too small; stress and failure factor are unaffected.** The database and the project use different unit systems. The message says what it changes, sometimes including what a constraint's limit then really means. Either change the project's units (section 4) to match the database, or make a database in the project's units.

### 14.4 Supports and loads

- **Error: the supports leave the part free to move or rotate in a direction the loads push...**, after Preview: the part is not held well enough. See section 7.8.
- **corner has no nodes on this mesh**, in red after Preview: the selection catches no node. Check it with Preview in the Selection form.
- **Check the selections gui_support1: they are triangle numbers of the old STL**, after changing the STL: pick those faces again.

### 14.5 During and after a run

- **The run failed: ...**: the run stopped with an error. The **Log** tab of the Current run panel has the details; the last lines usually say what went wrong.
- **Stopped while training the material network. Nothing was saved.**: Stop was pressed before the first iteration. Run again and let the training finish.
- **Not run: cancelled at the material network warning. No design was made.**: Cancel was chosen at the network warning.
- **This run cannot be shown: ... was saved by an earlier version of LSR MMTO; run the project again to see it here**: the run is from an older version. Run the project again.
- **The STL export failed: ...**: the export stopped with an error, given after the colon.

### 14.6 Opening and saving

- **STL not found: ...**, when a project opens: the part's file is not where the project says. Choose it again with File > Replace part... (section 5).
- **3 settings missing from the file were set to their defaults; Save writes them.**: the project was made by an older version. Nothing to do; save to keep the values.
- **Not saved: ...**: the project could not be written, for example the folder is read-only or the file is open elsewhere. Save it somewhere else with Save As.
- **Start a project first: File > New or File > Open.**: the panel needs a project. Make or open one.
- **A run is going. Stop it first...**: another project cannot open, and LSR MMTO cannot close, while a run goes.

### 14.7 Something went wrong

A message **Something went wrong**, with a short technical line under it, means LSR MMTO met a problem it did not expect. LSR MMTO keeps running. Save the project; if anything looks wrong afterwards, close LSR MMTO and start it again.
