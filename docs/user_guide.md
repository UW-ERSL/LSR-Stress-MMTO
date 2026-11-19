# LSR MMTO user guide

This guide explains every part of the LSR MMTO window. To open it, go to Help > User guide, or press F1.

Most fields have a small (i) next to them. Click it to find out what the field means. This guide explains how the different parts work, why they are needed, how they connect to each other, and what problems to look out for.

## 1. Start here

LSR MMTO helps you design a part using several materials. First, you tell it the space the part is allowed to occupy, where the part is supported, where forces act on it, and which materials it can use. The software then decides where to put material and which material to use in each place. It tries to produce a design that meets your goal. For example, you might want the stiffest possible part while keeping its weight below a certain limit.

You always follow the same steps. The buttons on the right side of the window are arranged in that order, from top to bottom:

1. **The part.** Use **Geometry**, the first button, to give the project its part: an STL file made in a CAD program, or a part made in LSR MMTO from shapes. The part is the space the design is allowed to occupy. See The part, section 5.
2. **The units.** Choose the unit system used for all the numbers in your project. See Units, section 4.
3. **The materials.** Choose a material database. It has one row for each material, along with its properties. See Materials, section 6.
4. **Supports and loads.** Tell the software where the part is held in place, where forces act, and how large those forces are. See section 7.
5. **The objective and the constraints.** Choose what the software should minimize and what limits the design must satisfy. See Formulation, section 8.
6. **Run.** Press the blue button at the top to start the optimization. You can watch the design develop on the screen as it runs. See Running, section 10.
7. **The results.** Look at the design, numerical results, and plots from each run. See Results, section 11.
8. **Export STL.** Export the design as STL files. You get one file for the whole part and a separate file for each material. See section 12.

Settings (section 9) contains the remaining options. For most projects, you can leave them as they are.

Most fields have a small (i) next to them. Move your mouse over it to read the first sentence of its explanation. Click it to open a card with the full explanation, including what the numbers mean. To close the card, click anywhere outside it or press Esc. If you do not understand a field, checking its (i) is usually the fastest way to find out what it does. This guide explains how everything fits together.

If you want to try LSR MMTO before creating your own project, start with one of the examples. You can click an example under **Try an example** on the start page, or open one through File > Open. There are four examples: the L-bracket, the U-bracket, the beam and the airplane bracket. Each example is already set up, so you can press Run immediately. The original examples are never changed. If you run or save one, LSR MMTO asks you to choose your own folder and saves a copy there.

## 2. The window

When no project is open, the 3D view shows the start page. It has **New project...**, which does the same thing as File > New, and **Open a project...**, which does the same thing as File > Open. It also shows four examples under **Try an example**. Until you open a project, the buttons on the right are greyed out, and so are Run, Analyse, Export STL and Part / Design.

### 2.1 The top row

The top row contains the following controls, from left to right:

- **LSR MMTO**, followed by the **File** and **Help** menus. The File menu is explained in section 3. The Help menu contains the **User guide** (which you can also open with F1), the **Researcher guide** (which explains how the code works and how to extend or change LSR MMTO), and **About LSR MMTO**.
- **Part** and **Design** are two options for what you see in the 3D view. **Part** shows the original part, along with its supports, loads and previews. **Design** shows the optimized design from the most recent run you made or opened. You cannot select Design until there is a design available to show.
- The **moon** button changes the 3D view to dark mode. Once you click it, the button becomes a **sun**, which changes the view back to light mode. The 3D view starts in light mode. This only changes the background and the colours of the window's own drawings, such as the ruler, Measure and labels. It does not change the material colours, which come from the material database.
- **View** controls how the part appears in the 3D view. You can change how its triangles are shown, make it see-through, or cut through it to look inside. See section 13. While a run or an analysis is going, View is temporarily hidden, just like Analyse.
- **Export STL** saves the design currently shown on the screen as STL files. It is greyed out while no finished design is shown on the screen. See section 12.
- **Run** starts optimizing the project. While the optimization is running, this button changes to **Stop**. If the Current run panel is hidden because you selected Results or closed the panel, the run's progress appears next to the button. For example, it might show *Iteration 29 / 40 · 0:33 so far · about 0:12 left*. See section 10.
- **Analyse** solves the project once, treating the entire part as solid and made from one material. It does not optimize the design. This lets you check whether the project is set up correctly before starting an optimization. While a run is in progress, Analyse is temporarily replaced by the run's progress. See section 10.4.

### 2.2 The buttons on the right

The buttons on the right are **Geometry**, **Units**, **Materials**, **Supports and loads**, **Formulation**, **Settings**, **Current run** and **Results**. They are arranged in the order you normally use them. Supports and loads is greyed out until the project has a part, since there is nothing yet to put supports and loads on.

The first five buttons, Geometry, Units, Materials, Supports and loads, and Formulation, also show whether each step is ready:

- A green tick means that step has been set up and has no problems that would prevent a run.
- A red mark means there is a problem that would prevent a run. The messages explain what is wrong and label it as an **Error**.
- If the step has not been set up yet, the button shows its normal picture.

Warnings do not change these marks.

Units, Materials, Formulation and Settings open in separate windows. These windows have OK and Cancel buttons, or Apply and Close. When you press OK, your changes are applied to the project. When you press Cancel, the project stays as it was before you opened the window.

Geometry, Supports and loads, Current run and Results work differently. They open as panels beside the 3D view, so you can still interact with the part while using them. If you open more than one panel, they share the same space using tabs. You can drag the edge of a panel to make it wider or narrower.

While a run or an analysis is going, the buttons on the right are temporarily hidden. The Current run panel then takes up half the window, beside the 3D view where you can watch the design develop. You can still drag the divider to change how much space each side uses. When the run finishes or you stop it, the buttons return and the panel goes back to its previous width.

### 2.3 The 3D view

You can move around the 3D view using the mouse:

- Hold the left mouse button and drag to rotate the part.
- Turn the mouse wheel to zoom in or out.
- Hold the middle mouse button and drag to move the part across the screen. You can also hold Shift and drag with the left mouse button.

Normally, clicking on the part does not select anything. There are three exceptions:

- When the **Pick** window is open (section 7.2), clicking on the part selects faces.
- When **Measure** is active (explained below), each click chooses a point for measuring.
- When a form under Supports and loads is open, clicking on the part copies the coordinates of the point under your mouse. You can then paste those coordinates into a field using Ctrl+V.

### 2.4 The messages

The messages area is below the 3D view. It starts out three lines tall. You can drag the divider between the messages and the 3D view to make it taller or shorter. LSR MMTO remembers the height you choose and uses it the next time.

The top of this area shows any current problems with your project. Errors appear in red and warnings in amber. A thin line separates them from the other messages.

These errors and warnings are kept up to date. When you fix a problem, its message disappears immediately. If the same problem comes back, the message appears again. They can take up at most half the height of the messages area. If there are no errors or warnings, they take up no space.

Below them is a record of what the software has done. This includes what was opened and saved and what is happening during a run.

If the same message appears several times in a row, it is shown only once, with a count of how many times it occurred. For example, *(×3)* means the message appeared three times.

You can scroll up to read older messages. The bin button at the top right clears the history of what happened. It does not clear the current errors or warnings.

An **Error** means you cannot start a run until you fix the problem. A **Warning** means you can still start the run, but something in the setup may not be what you intended. See section 14 for the common errors and warnings.

### 2.5 The bottom bar

The bottom bar shows information about the part and provides tools for measuring it.

- On the left, you can see the part's size. For example, *Part: x 0 to 200 · y 0 to 200 · z 0 to 1 mm*. Click the (i) beside it to see the part's file, volume, dimensions and material database.
- In the middle, you can see the coordinates of the point under your mouse when it is over the part. This area also shows the measurements made using Measure.
- **Snap** rounds coordinates to a chosen step size. It affects the point under your mouse, points you copy, and measurements made with Measure. The available steps depend on the size of the part. They are rounded values near one-thousandth, one-hundredth and one-tenth of the part's largest dimension, using the project's length unit. For a part that is 200 mm long, the steps are 0.2, 2 and 20 mm. If you choose Snap: 2 mm, a point at x 37.62 is shown as 38.
- **Measure** lets you measure distances on the part. Click Measure, then click two points on the part. It shows the distance between them, along with the x, y and z components of that distance. If you click a third point, you start a new measurement. In the light 3D view, the measurement line and its points are dark. In the dark view, they are white. To stop measuring, press Esc or click Measure again.
- **Ruler** shows scales along the part, using the project's length unit. Click it again to hide the scales.

## 3. File menu

The File menu contains the following options:

- **New...** creates a new project. You enter a **Name**, choose a **Location**, and choose the **Physics**. LSR MMTO then creates a folder with the project name at the location you chose and saves the project there, with no part yet. All settings start with their default values. Geometry opens, so you can give the project its part (section 5).
- **Open...** opens an existing project. Project files have the .mmto extension. This option only opens LSR MMTO projects.
- **Save** saves the project to its existing file. If the project has never been saved, or if it is an example, LSR MMTO uses Save As instead.
- **Save As...** creates a new copy of the project in a new folder. Everything the project uses is copied into that folder. The new project gets the new name, but any runs you already completed remain with the original project.
- **Export as script** creates a short Python script representing the project. It is named *name*_setup.py and saved in the project's folder. Every time you save the project, the script is written again so it stays consistent with the project. If necessary, this menu option saves the project first. It also tells you where the script was saved. You can open the script in any text editor to see the entire project written out or to make changes to it.
- **Run script...** runs a Python script. For example, you can run a script created using Export as script after making changes to it. Anything the script prints appears in the messages area. If the script saves a project in its folder, LSR MMTO opens that project in the window.
- **Exit** closes LSR MMTO.

The title of the window shows the project name. If you are using an example, it also shows *(example)*. If you have made changes that have not been saved, an * appears in the title.

If you try to close the window or open a different project while you have unsaved changes, LSR MMTO asks whether you want to save them.

### 3.1 The project folder

Every project has its own folder, which contains everything the project needs:

- The project file, *name*.mmto, and its Python script, *name*_setup.py.
- **part**: a folder containing the part's STL file and any additional STL files used by selections.
- **materials**: a folder containing the material database and its trained network, if you are using a database of your own.
- **results**: a folder containing a separate folder for each run, named according to when that run was made.

If you want to send the entire project to someone else, you can zip its folder.

Each run remembers where its files are located relative to the project folder. This means you can move the entire folder, rename it, or unzip it somewhere else without breaking the results. The runs will still appear in Results, and you can still export them.

## 4. Units

**Units** controls the unit system used for every number in the project. Open Units and choose a system under **Unit System**:

| Unit system | Length | Force | Stress | Density | Mass |
| --- | --- | --- | --- | --- | --- |
| MKS | m | N | Pa | kg/m^3 | kg |
| mmKS | mm | N | MPa | t/mm^3 | t |
| IPS | in | lbf | psi | lbf*s^2/in^4 | lbf*s^2/in |

Press **Apply** to use the selected unit system. The messages show *Project units: mmKS. Numbers are not converted.* The part's size at the bottom left, the ruler and the Snap steps immediately start showing the new length unit.

Changing the unit system does not convert any numbers. Every number you entered stays exactly the same. Only the unit assigned to it changes.

For this reason, choose your unit system before entering the numbers for your project. This includes the part's dimensions, forces, pressures, distances and constraint limits. Enter all these values using the unit system you selected. Fields that have units show them next to the value.

The material database has its own units, shown in its second row. These may not match the units of your project. For example, the database might give Young's modulus in Pa while your project uses mmKS. If this happens, the messages explain the mismatch, how large the difference is, and how it affects the results. See section 14.

**Temperature Unit** and **Angle Unit** are also available in the Units window, but LSR MMTO does not use them yet.

## 5. The part

The part is the space the design is allowed to occupy, called the design domain. The optimization decides where to put material inside it. You give the project its part in **Geometry**, the first button on the right. Geometry opens as a panel beside the 3D view. At its top you choose **From an STL file** or **From shapes**.

A new project has no part yet. Until it has one, the 3D view is empty and Supports and loads is greyed out.

### 5.1 From an STL file

An STL file made in a CAD program. Press **Browse...** and choose the file. In a project folder of its own, the file is copied into the part folder. If you choose another STL later and you picked faces on the old one to make selections, LSR MMTO warns you first. Picked faces are saved using their triangle numbers. Those numbers can refer to different faces in another STL, so you need to pick the faces again afterwards. If the part was made of shapes, LSR MMTO asks first, since the shapes are removed.

The STL should describe a closed (watertight) surface and use the project's length unit. You can check the part's size in the bottom bar (section 2.5): a part that should be 200 mm long and reads *x 0 to 0.2 mm* was drawn in metres.

To fix that, press **Scale...** and type in **Scale by** the number every coordinate is multiplied by: 1000 from metres to millimetres, 1/1000 back, 25.4 from inches to millimetres. The window previews the new size of the part. OK changes the project's own copy of the part, in its part folder, so scaling again multiplies again (by 2, then by 3, is by 6). An example, or a project whose part is not in a folder of its own, is first saved as a project of your own: the STL the part came from never changes. Scaling changes only the part. It does not change the units, loads, limits or materials. Faces you picked stay picked. Selections with coordinates or lengths you typed (a Box, a Cylinder, ...) are not scaled, so LSR MMTO names them first: check them afterwards. A length in element sizes, such as 1.5h, follows the mesh by itself. Earlier runs keep their original size: Results shows them, and Export STL exports them, at the size they were run at.

### 5.2 From shapes

A part made in LSR MMTO from simple shapes: boxes, cylinders, spheres, and outlines pushed up (an **Extrusion**) or spun around (a **Revolution**). The part is a list of steps, done in order. Each step **Adds** a shape to the part, **Cuts** it out of the part, or keeps only the **Overlap** of the two. The first step adds.

- **New step** shows the shapes as tiles. Click one to add a step with that shape. For a step after the first, choose Add, Cut or Overlap above the tiles. In an example, LSR MMTO first asks to save it as a project of your own.
- Click a step in the list to see its boxes below. Every change goes into the project at once. Its shape is drawn see-through in blue over the part, so you can see where it is, even a cut that misses.
- Click the shape's name (for example *Cylinder: change*) to give the step another shape.
- **At** is where the shape's origin goes: a box's lowest corner, the centre of a cylinder's base, a sphere's centre.
- A **Box** has its **Size** along x, y and z. A **Cylinder** has a **Radius** and a **Height** and stands along z; **+ Cone** gives it a **Top radius** (0 for a point), and **Remove cone** makes it a cylinder again. A **Sphere** has a **Radius**.
- An **Extrusion** is an **Outline** drawn flat and pushed up along z by its **Height**. A **Revolution** is an outline spun around the z axis by its **Angle** (360 for a full turn): for a Rectangle, **Width** is how far out from the axis it reaches and **Depth** is its height.
- The outline is a **Rectangle** (from its corner at the origin), a **Circle** (centred on the origin; not for a Revolution), or a **Polygon**, whose corner points are in a table: **Add point** and **Remove point** change it, and **Paste points** takes two columns of numbers (x and y) copied from Excel or a text file. **Corner radius** rounds the corners of a Rectangle or a Polygon.
- A **Wrap** is the tightest shape around two or more shapes, like a sheet stretched over them: two cylinders make a lug. **Shapes wrapped** lists them; **+ Shape** and **Remove shape** change the list, and the shape chosen in it has its own boxes, **At** and **Turn**.
- **+ Turn** turns the shape about the x, y and z axes through its origin, in that order, before it is moved.
- **+ Repeat** makes copies of the shape. **Along a line**: **Count** copies (the first one included), each moved by the **Spacing** from the one before. **Around a circle**: Count copies spread evenly over a full turn about the line through the **Centre** along the **Axis** (x, y or z), for a ring of bolt holes. **Remove repeat** goes back to one.
- **Up**, **Down**, **Copy** and **Delete** change the order of the steps, copy the step chosen (for a second hole like the first), or remove it. The first step must stay an Add.

The 3D view does not change while you edit, so typing never slows down. **Update part** builds the part from the steps and draws it. Until you press it, the button is highlighted and Geometry says *The view shows the part before your last changes.* Closing Geometry, opening Supports and loads, Analyse and Run update the part first if needed.

If a step cannot be made (a size of 0, a cut that removes the whole part, a polygon whose edges cross), Update part says which step and why, in red, and the view keeps the last part built. If the part can be made but may not come out as meant, Geometry lists it: a cut that misses the part, a part in separate pieces, or a feature less than 2 elements across, which may vanish in the mesh (the message says what Mesh dofs would show it).

A changed part numbers its triangles anew, so faces picked on it may land elsewhere: after Update part, LSR MMTO names the selections to check. **Scale...** multiplies every length in the steps and asks first when you have picked faces.

The steps are saved in the project file and written in the project's script, so the same part can also be written in a script: docs/reference.md, in the LSR MMTO folder, shows how under Making the part from shapes.

### 5.3 The mesh

A run does not work directly on the STL. Instead, it fills the part with a grid of equally sized blocks called elements. It then decides what material each element should contain. The corners of these blocks are called nodes. Supports and loads act on these nodes.

The fineness of the grid is controlled by **Mesh dofs** in Settings: the number of unknowns the mesh is built for, three at each node for a structural problem (the movement in x, y and z). A larger value gives a finer mesh but also makes the computation slower. A finer mesh captures more detail and produces smoother STL files when you export the design.

If your part is flat, tick **Single element layer (2D)** in Settings: the grid then has one element through the thickness, and the run is much faster.

The preview under Supports and loads (section 7.7) builds the same mesh the run will use, so you can check how many nodes and elements the run will use.

## 6. Materials

The optimization chooses materials from a **material database**: a spreadsheet. Each row represents one material, and each column represents a property (density, Young's modulus, and so on). LSR MMTO uses this database to train a material network. The network creates a map of the materials so the optimization can move smoothly between different material choices. In the final design, each element is assigned one of the materials in the database.

Open **Materials** from the buttons on the right. The window contains:

- **Materials in the design:** **Single material** or **Multi material**, and on the same row the material to use:
  - With single material, **Material:** lists the database's materials. Every element is the one you choose, with its values from the database; no network is trained, and the optimization only chooses where the part is solid. Any database works, so you never need to delete materials to run one of them.
  - With multi material, the database holds two or more materials and the network learns them. **Fix every element to:** is *None (the optimizer chooses)*, or one of the database's materials. Fixed, every element is that material, with its values as the network gives them, and the optimization only chooses where the part is solid. Use it to compare a design of one material with a design of several, both through the same network.
- **Database:** the database the project uses. The list holds the project's own databases: every database in its materials folder, and the one it uses when that one is kept elsewhere. **Included databases** beside it lists the databases that come with LSR MMTO; choosing one makes it the project's database, shown in the list as *its name (included)*.
- With no database chosen yet, the report says so; make one with **New database...**, or bring one in with **Import...**.
- A table of the database's materials, each in the colour and with the label it is drawn with, and its properties with their units.
- Under the table, a short report: how many materials there are, the latent dimension recommended for them, how closely the trained network gives them back, and any errors or warnings. The (i) next to it explains the recommended latent dimension. For a single-material design, the report names the material, which the run uses at its database values. A database that does not fit the choice is an error there, for example *Multi material: steel.xlsx has one material, and the network needs at least two to learn from*.

When you use only one material, there is no material map. Because of this, some options elsewhere in the software are greyed out: **Material volume fraction** in Formulation, and the settings that act on the material map (the Distance penalty and Latent smoothing groups, Latent smoothing radius, Starting latent point and Snap to real materials). With single material, the VAE settings are greyed out too. A run of one material has no **Latent space** tab.

Select a database and press **OK**. When you select a different database, its network settings are loaded as well (see Settings, section 9), and the messages explain the changes. A database with a trained network sets the network's shape (its latent dimension and hidden layer width) from the network itself, so they always match. In a project folder of its own, the database you choose is copied into the project's materials folder, with its trained network, so the project keeps everything it uses.

### 6.1 Where a database comes from

The following buttons appear beside the database list:

- **New database...** opens the database editor (section 6.3) on an empty database, to type or paste your materials in.
- **Edit...** opens the database in the editor. One of your own is changed in place. One that comes with LSR MMTO is saved as a copy in your project's folder, and the project then uses the copy. A column that is not a property LSR MMTO knows is left out, and a note over the table names it; Save writes the database without it.
- **Open in Excel** opens the database file in Excel, or whatever opens .xlsx files on your computer. When you come back to the Materials window after saving it there, the window reads the database again by itself, and keeps the changed database as a new version (section 6.5). A database that comes with LSR MMTO is not opened: a change there would change it for every project, so make your own copy with **Edit...** first.
- **Versions...** lists the versions of one of your own databases (section 6.5).
- **Rename...** renames one of your own databases, in your project's folder. Its trained network and all its versions are renamed with it, and the project is saved with the new name (with any other change you made to it). Earlier runs still find it for **Analyse this design**. A name another database in the folder has, or had, is refused. A database renamed outside LSR MMTO (in File Explorer) is a new database to it, with a new v1.
- **Import...**, at the right, brings in a table of materials you already have, in Excel (.xlsx or .xls) or CSV. A database made by LSR MMTO (one whose check finds nothing wrong) is copied into your project's folder as it is, with its trained network if it has one next to it, and the project uses the copy. Your original file is never changed. Any other table becomes a new database: the first column holds the material names and the first row the properties. A property can be written under any of the names LSR MMTO knows it by (for example Young's modulus as Youngs_Modulus, Youngs Modulus or E), and a unit may follow in brackets, as in *Density [kg/m^3]*. A second row of units, as in an LSR MMTO database, is read too. Columns that are not a property LSR MMTO knows are left out, and one message lists them. The database editor then opens with what was read, for you to check and save.

Creating a new database or importing a table saves it in the project folder. Because of this, you must save the project first; LSR MMTO says so if it is not.

When some columns of the database are in other units than the project's (section 4), a button **Convert to mmKS units...** (with the project's unit system) shows under the report. It opens a window that lists each of those columns with what its numbers are multiplied by, for example *Young's modulus: Pa -> MPa, x 1e-06*, and the limits of the formulation written for the old numbers with what they become, for example *Mass <= 4e+07 -> 4e-05 t*. **Convert these limits too** is ticked, so each limit keeps its meaning. **OK** writes the converted database as the project's own copy, in its materials folder (one that comes with LSR MMTO is never changed), keeps the database as it was and as converted as versions (section 6.5), keeps its trained network, which still fits up to rounding in the last digit (the next run asks once before using it, section 6.4), converts the ticked limits, saves the project and says in the messages what was converted.

### 6.2 Values and Network error

Use the two tabs above the table to choose what information you see:

- **Values**: each material's properties, as the database gives them.
- **Network error**: for each value, how far the trained network is from it when it turns the material into its point on the material map and back, in % of that property's range over the database's materials. Rest the mouse on a cell to see both values and the difference in the property's unit.

Network error is greyed until the database has a trained network. The first run with a new database trains one. With single material there is no network, and no Network error tab.

A small error (well under 1 %) means the properties used during the run are very close to the values in the database. If the error is large, the materials used by the run differ from the actual database values by that amount; see the network warning in section 14.

### 6.3 The database editor

The database editor displays the materials in a table:

- **Database name**: the database's name, which is also its file name. The grey words beside it say where it is saved (rest the mouse on them for the whole path).
- One row per material: its name under **Material**, **Shown as** (an optional shorter name for legends and plots) and **Colour** (click it to choose another). The Material column stays in place when the table scrolls sideways.
- One column per property, as wide as its title (a long title is cut short with ...; rest the mouse on it for the whole name). The row under the titles holds each property's unit; a property without a unit says *no unit*.
- **Add material** and **Remove material** add and remove rows. **Add property** opens the **Add properties** window: every property below with a tick box, those this project needs first (the ones its physics, its objective and its constraints read), then the others. Those already in the table are ticked and greyed. Tick as many as you want (the search box finds one by any of its names); **Add 2 properties** adds a column for each. **Remove property** removes the column you are in.

You can copy several cells from Excel and paste them directly into the table. Problems show in red once you have typed in a row, and are listed under the table (a missing value, a value outside what the property allows, two materials with the same name); a new, empty form shows none. **Save** keeps the database in the project's folder and the project then uses it.

While there are changes not saved, the editor's title says *(modified)*. **Save** keeps the database as saved as a new version (section 6.5), without asking. Closing the editor with changes not saved asks first: **Save...** saves them, **Discard changes** closes without them, **Cancel** goes back to editing. A database with columns LSR MMTO does not know shows *(modified)* from the start, since Save writes it without them.

If you change the materials (a name, or a material added or taken away) or the numbers of a database that already has a trained network, the trained network no longer matches the database: LSR MMTO sets the old network aside, and the next run trains a new one.

### 6.4 The database file

A database is an Excel file (.xlsx): row 1 holds the property names, row 2 their units, and then one row per material, its name in the first column. An optional second sheet, **Display**, holds the colour (#rrggbb) and the shorter label each material is drawn with; the editor writes it for you.

The trained network is stored in a file beside the database with the same name and the ending .nt (my_materials.xlsx and my_materials.nt). A database that comes with LSR MMTO has its network already, and the one in LSR MMTO's own folder is never trained again. Its copy in your project's folder can be: see Use the trained network, in Settings.

A trained network keeps a record of the database it learned from: every name, unit and number, and each material's colour and label. Before each run, LSR MMTO checks the current database against this saved record:

- A number, a unit, a name, or a material added or taken away, however small the change: the run doesn't start until a new network is trained (untick **Use the trained network** in Settings) or the version the network was trained on is brought back with **Versions...** (section 6.5). The report under the table names the cells that changed.
- Only colours or labels changed, the units converted (section 6.1), or a network trained before networks kept this record: the next run asks once, *Use it for my_materials.xlsx as it is now?* **Yes** uses the network and records the database as it is now, so you aren't asked again; **No** starts nothing. The report under the table says when the next run will ask.

LSR MMTO writes a number into the file with up to 16 significant digits, so a difference beyond that is not kept.

A material database can contain these attributes:

<!-- list: material_attributes -->

| Attribute | What it is | Units |
| --- | --- | --- |
| Density | Mass per unit volume | kg/m^3, g/cm^3, t/mm^3, lb/in^3, lbf\*s^2/in^4 |
| Cost | Price per unit mass | USD/kg, USD/g, USD/t, USD/lb, USD/(lbf\*s^2/in) |
| Young's modulus | Young's modulus, the stiffness | Pa, kPa, MPa, GPa, psi, ksi |
| Poisson's ratio | Poisson's ratio | - |
| Yield strength | Yield strength, the stress at which the material yields | Pa, kPa, MPa, GPa, psi, ksi |
| Thermal conductivity | Thermal conductivity, how well heat flows through it | W/(m\*K), mW/(mm\*K), BTU/(s\*in\*°R) |
| Thermal expansion coefficient | Coefficient of thermal expansion, the strain per degree of temperature rise | 1/K, 1/°R |

<!-- end of list -->

### 6.5 Versions

Every state of one of your own databases (one in your project's folder) is kept as a version, numbered **v1**, **v2**, **v3**, and so on, with nothing to answer. A new version is kept whenever the database has changed in any way, even only a colour or a label:

- when you save it in the editor, make it, import it or convert its units;
- when you change it outside LSR MMTO, in Excel for example. LSR MMTO finds the change and keeps it when the project opens, when the Materials window opens or comes back into view, and when a run starts.

The first version, **v1**, is the database as the project first used it. For an included database you edit, v1 is the included database as it came.

Each version also keeps the trained network that goes with it, when it has one, so you can restore a version without training a new network.

**Versions...** in the Materials window lists them, newest first, each with when it was saved and how many materials it has. **(in use)** marks the version the database is now.

- **Restore** puts the chosen version back as the database, without asking. The database as it was is kept first if no version has it yet (a change you made in Excel while the window was open, say), and every version stays, so a restore can always be undone by restoring again. The window closes and the report says which version is now used, for example *Now using steels v2*.
- **Rename...** gives the chosen version a note after its number, for example *steels v3 softer*. Leave it empty to take the note away. The name before the note never changes.
- **Delete** deletes the chosen version, after you confirm; that cannot be undone. The version in use can't be deleted, since it is the database as it is now. A deleted version's number is never given again.

The versions are kept in a **versions** folder beside the database (for your own databases, inside the project's materials folder), out of the Database list, and **File > Save As** takes them with the project. An included database has none: it is never changed.

## 7. Supports and loads

Before you can optimize a part, you need to define what holds it in place and what forces act on it. In LSR MMTO, these are called **supports** and **loads**. Each support or load is applied to a **selection**: a named piece of the part, such as one face, a hole, or a few faces you clicked.

Open **Supports and loads** from the buttons on the right of the window. The panel contains:

- **Add**, to put a new support or load on the part, and **Preview** and **Clear**.
- A list of the supports and loads, with their **Type**, **Where** (the selection they act on) and **Value**. Double-click one to change it. **Delete** removes the one you clicked.
- **Selections**, closed at first. Click it to see every selection of the project, with its **Name**, its **Type**, and a **Keep solid** tick.

Your changes are applied to the project immediately. Use File > Save to keep them.

### 7.1 Two ways to add a support or load

Click **Add** and choose one of:

- **Click on the part...** opens the **Pick** window. You click faces on the part, choose what they are, and press **Apply**. This is the quickest way for a face you can see. A project with both supports and forces and temperatures and heat (thermoelastic) has one line for each. When nothing of the project's physics can be picked, it is greyed, and resting the mouse on it says why.
- **Enter by hand...** opens a form. You choose the **Type**, choose the selection under **Where** (or make one there with **New selection...**), and fill in the values. **Preview**, or Enter, checks the values and shows the selection on the part (with arrows for a load), and the form stays open; **OK** keeps it. Use this when a face is hard to click, when you need exact numbers, or when the place is not a face at all (a hole, a patch, a point).

Both methods add a support or load to the same list. You can edit either one later by double-clicking it.

### 7.2 Picking faces on the part

Click **Add**, then **Click on the part...**. The Pick window opens beside the part.

1. Under **Selection**, choose how a click selects:
   - **Facet**: the face you click and every face joined to it with no sharp edge in between (an edge that bends more than about 15 degrees stops it). One click selects a whole flat side, or a whole smoothly curved surface such as the inside of a hole.
   - **Triangle**: only the single triangle you click. The edges of the triangles show, so you can see them. Use it for a small area.
   - **Node**: the point of the part nearest to your click. Click again for each node you want.
2. Click the part. Each click adds to what is selected; clicking a selected face again takes it off. **Clear picks**, or a right click on the part, clears everything selected so far.
3. Under **Type**, choose what it is. Type offers only what the project's physics takes:
   - supports and forces: **Fixed XYZ**, **Fixed X**, **Fixed Y** or **Fixed Z** for a support, or **Force** for a load. For a force, give its X, Y and Z under **Force Components**: this is the total force on everything you selected.
   - temperatures and heat (Facet and Triangle only): **Temperature** holds the faces at a temperature (a Fixed temperature), and **Total Heat** puts heat in, shared by everything you selected (a Heat source). Each value is in the project's units.
4. Press **Apply**.

The pick becomes a support or load in the list, on a new selection named gui_support1, gui_load1 and so on. Rename the selection under Selections if you like.

You can select faces only while the Pick window is open. Closing it clears whatever was picked and not applied. With it closed, clicking and dragging on the part just turns the view.

A picked selection remembers the triangles of the STL by their numbers. If you load a different STL for the part, the same numbers are different faces, so pick them again.

### 7.3 Selections

A selection identifies a particular region of the part and gives that region a name. You can use the same selection for several supports or loads. Its name appears under **Where**. To make one, open **Selections** and click **Add**, or click **New selection...** in the support or load form.

The Selection form has a **Name** (letters, digits and _, not starting with a digit), a type, and the type's own fields. The (i) next to the type says what it selects. The part's size is shown under the type, so you can see which numbers make sense. You can check the selection before saving it using two buttons:

- **Preview** draws the selection on the part in orange, and says how many nodes it has on the mesh the run will use.
- **Clear** empties the form, every field back to its default, and removes that drawing.

Pressing Enter in the form does the same as **Preview**: it checks what you typed and shows the selection on the part, and the form stays open. A mistake shows in red under the form. Only **OK** keeps the selection and closes the form.

While the form is open, rest the mouse on the part to read the coordinates under it. Click the part to copy that point; paste it into a field with Ctrl+V.

The available selection types are:

- **Face of the part**: every node on one outer side of the part, such as Right (x max) or Bottom (y min). The quickest way to hold or load a whole side of a box-like part. *Example: Side Left (x min) to clamp the left end of a beam.*
- **Rows on a face**: a few rows of nodes at one end of a face. Use it to put a load on a small patch instead of a whole face. *Example: the top 8 rows of the right face: Face Right (x max), Counted along y, Number of rows 8, From High end.* A row is one layer of nodes, so the patch is as tall as that many elements.
- **Near a face**: every node within a set **Distance** of one outer side. Use it when the side is not perfectly flat, or when you want a thicker band than a single face.
- **Flat cut**: the nodes on a flat cut through the part, across x, y or z. *Example: every node where y is 38: Cut across y, At 38.* To keep only part of the cut, fill in the from and to of the other directions, for example x from 0 to 40. Leave them empty for the whole cut.
- **Box**: every node inside a box, given by two opposite corners (**Corner 1** and **Corner 2**, each as x, y, z). The corners can be in any order. A box thinner than an element still catches the nearest layer of nodes.
- **Sphere**: every node within a **Radius** of a **Centre**.
- **Cylinder**: every node inside a cylinder, or inside a tube if you give an **Inner radius**. Good for holes and bolt bores. Give a **Point on the axis**, the **Axis direction** (x, y, z, or a direction such as 1, 1, 0) and the **Radius**. Leave the two ends empty for a cylinder through the whole part. With the axis along x, y or z, the ends are coordinates, for example x from 22 to 61. Along a tilted axis they are distances from the point on the axis. *Example: the inside of a bolt hole of radius 5 through z at x 20, y 30: Point on the axis 20, 30, 0, Axis direction z, Radius 5.5, Inner radius 4.5.* An empty Inner radius is 0, a solid cylinder. Tick **Surface nodes only** to keep only the nodes on the outside of the part, for example the wall of a hole without the nodes inside the material around it. For a round hole in the part, click **Click a hole** and then click the inside wall of the hole: the form is filled in with the point on its axis, its axis, its ends, a Radius of the hole's own radius (Tolerance Automatic already takes the nodes of the wall, up to half an element outside it), Inner radius 0, and Surface nodes only ticked. When the hole is less than four elements across on the run's mesh, the form says so: some of its nodes may be missed, so check with Preview, or use more dofs. A click on the outside of a round face, such as the rim of a disc, fills a tube one element size each side of that face (Radius r + 1h, Inner radius r - 1h), so it takes the nodes of that face, not the whole part. A click on a flat face fills nothing and says so; turn the part so the wall of the hole shows and click on it. If the wall is not round, the form says *This doesn't look like a round hole* and fills nothing.
- **Closest node**: the one node nearest to a **Point**. Tick **Surface nodes only** to stay on the outside of the part. Use it for a load at one point.
- **Inside an STL**: every node inside a closed surface loaded from another STL file (**File**, with **Browse...**). Use it for a shape no other type describes, such as a collar around a bore. **Move by** (x, y, z, default 0, 0, 0) moves the shape before nodes are selected, for example when the STL was saved shifted to 0, 0, 0. Preview draws the shape where it ends up.
- **Combined selections**: two selections of the project joined, overlapped, or one taken from the other. **How** is one of **Join: in A or in B**, **Overlap: in both A and B**, or **A minus B: in A, not in B**. It follows them: change Selection A or Selection B later and this selection changes too. It appears once the project has at least two selections. *Example: the left face without its bottom rows: A minus B, with A a face and B the rows.*

Three additional types are created using the mouse. They appear when you open one of these selections: **Picked faces** (faces you clicked), **Node numbers** (single nodes, by number; they stay right only while the mesh does not change) and **Dragged box** (a box dragged over the part in earlier versions; a project that has one still opens and runs as before).

Most selection types have a **Tolerance**: how far outside the selection a node may sit and still count. **Automatic** is right almost always.

The **Code line** tab shows the same selection as one line of text, for example `rows(face="xmax", axis="y", n=8, end="max")`. You can type a selection there instead of filling in the form.

A number can be typed as plain arithmetic, in the Selection form, on the Code line and in the support or load form: for example 38/2, 0.5*76 or 2*pi*5, also inside a point (0, 38/2, 0). You can use + - * / ** ( ) and pi. The project keeps the number it comes to.

A length of a selection (a radius, a distance, the ends of a cylinder, a tolerance) can also be given in element sizes with h: 1.5h is one and a half element sizes, and 5 + 1h is 5 plus one element size. Such a length is kept as you typed it and follows the mesh: a finer mesh gives a smaller h.

Here is a summary of all selection types:

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

A support restricts how the part can move. Without enough support, the applied loads could make the whole part move instead of deforming it.

- **Fixed support**: holds the selection still. Under **Held in**, tick the directions it is held in: x, y and z for a clamp, or only some of them for a support that lets the part slide in the others (for example, held in y only, like a part resting on a floor).
- **Moved support**: moves the selection by a set **Distance** in the ticked directions (**Moved in**), instead of holding it still. Use it to push or pull on a part by a known amount rather than with a known force.

A support given in the Pick window is a Fixed support: Fixed XYZ holds x, y and z; Fixed X, Fixed Y and Fixed Z hold one direction each.

### 7.5 Loads

A load represents a force or pressure acting on the part.

- **Force**: a **Total force** on the selection, given as its x, y and z parts. The total is shared equally between the nodes of the selection: a force of 0, -100, 0 on a selection of 50 nodes puts 0, -2, 0 on each node. The force does not grow when the selection does, so a bigger selection only spreads the same force wider.
- **Pressure**: a **Pressure** on the selection, turned into one force: Pressure times **Area**, pushing along **Direction**, shared equally between the nodes. Give the area as a number, or tick **from the selection** to use the size of the selection across the direction. With from the selection, the Direction must be along x, y or z, and the area is that of the rectangle around the selection, seen along the Direction. On a round or uneven face that rectangle is larger than the face, so type the area instead.

The force in the Pick window is a Force: its X, Y and Z are the total over everything you selected.

The units show beside each field, from the project's units (for example N and MPa for mmKS). See Units, section 4.

### 7.6 Keep solid

Tick **Keep solid** next to a selection, under Selections, and those elements stay fully solid for the whole run, whatever the optimization does elsewhere. Use it for the places the part must keep whatever happens: where bolts go, where it is clamped, a mounting face.

You can mark a selection to remain solid even if no support or load acts on that selection.

### 7.7 The preview

Press **Preview** at the top of the panel to see everything on the mesh the run will use. The part is drawn as that mesh, its elements as small blocks, so you see exactly what the run solves:

- supports as blue points,
- loads as red points, with arrows on their nodes showing the direction (on a few nodes spread over the load, when it has many),
- other selections as orange points,
- kept-solid elements as grey blocks.

A line below the buttons gives the verdict, whether the supports hold the part:

- **The supports hold the part against the loads.** All is well.
- **Error: the supports leave the part free to move or rotate in a direction the loads push...** Add supports, or tick more directions on one, until the part cannot move or turn.

A selection with no nodes on the mesh shows in red: *name has no nodes on this mesh*. The (i) next to the verdict lists the size of the mesh, how many nodes each selection has, and how many elements are kept solid.

Click a row of the list, or a selection under Selections, to see only that one on the mesh. **Clear** removes the preview and brings back the part, and removes any faces picked in the Pick window but not applied. A form, the Pick window and **Measure** show the part instead of the mesh while they are open, since they read the mouse on the part. While the mesh is shown, the bottom bar does not show the point under the mouse.

The preview uses the same mesh as the optimization, rather than the STL surface. This means the nodes shown are the nodes the run will actually use. A finer mesh (Settings) gives more nodes, and can change which nodes a small selection catches.

### 7.8 Common mistakes

- **The part is not held in every direction.** A part held only in y can still slide in x and z, or spin. The preview says so. Hold it in more directions, or add a second support. A clamp on one face (Fixed support, held in x, y and z) is the usual fix.
- **A load or support on a selection with no nodes.** A box or a flat cut that misses the part, or a Rows on a face that counts from the wrong end, selects nothing. The preview shows the selection in red. Press Preview in the Selection form to check its node count before you keep it.
- **Two supports on the same node in the same direction, with different values.** For example a Fixed support and a Moved support whose selections overlap. The preview and the run stop with a message that a direction of a node is already given another value. Make the selections not overlap, or tick that direction on only one of them.
- **A force that is too big or too small by a factor of 1000 or more.** The values are in the project's units. A force of 1000 means 1000 N in mmKS, and 1000 lbf in IPS. Check Units before you type the numbers.
- **A pressure with the wrong area.** Pressure times Area is the force. With from the selection, the area is the rectangle around the selection; on a round face, type the area instead.
- **Picked faces after changing the STL.** Picked faces are stored by triangle number, so they belong to one STL. Pick them again on the new one.

Here is a summary of all support and load types:

<!-- list: support_and_load_kinds -->

| Type | Support or load | What it does | Values |
| --- | --- | --- | --- |
| Fixed support | Support | Holds the selection still in the ticked directions. | Held in |
| Moved support | Support | Moves the selection by a set distance in the ticked directions. | Moved in, Distance |
| Force | Load | A total force on the selection, shared equally by its nodes. | Total force |
| Pressure | Load | A pressure on the selection, turned into a force: Pressure times Area, pushing along Direction, shared equally by its nodes. Tick Area from the selection to use the selection's size across the direction (Direction must then be along x, y or z). | Pressure, Direction, Area |
| Fixed temperature | Support | Holds the selection at a set temperature. | Temperature |
| Heat source | Load | Puts heat into the selection, shared equally by its nodes. | Total heat |

<!-- end of list -->

## 8. Formulation

**Formulation** defines the optimization problem: you choose one quantity to minimize (the objective) and set limits that other quantities must satisfy (the constraints).

- **Minimize**: choose the objective. For example **Compliance**, to make the part as stiff as it can be, or **Mass**, to make it as light as it can be.
- **Constraints**: one row per constraint: the quantity, then **<=** (at most) or **>=** (at least) and its limit, with the limit's unit beside it. For example *Mass <= 0.002 t*, *Volume fraction <= 0.3* (at most 30 % of the design space filled), or *Volume fraction >= 0.1* (at least 10 %). Any quantity can take either. The (i) at the end of the row says what the quantity is. **Add constraint** adds a row; **Remove** takes one away.
- **Starting limits**: a new row starts at a limit worked out from its quantity:
  - **The 40% rule**, for **<=**: 40% of a reference. For a fraction, the reference is the whole design space, so the start is 0.4. For **Mass**, it is the mass of the whole part made solid in the design's material: the single material with **Single material**, the fixed material with **Fix every element to**, and otherwise the lightest material of the database. For **Cost**, it is the cost of the whole part made solid in the design's material, or the cheapest material (least density times cost) otherwise.
  - **The 10% rule**, for **>=**: 10% of the same reference. Switching a row from **<=** to **>=**, or back, moves a limit still at its start to the other rule's; a limit you typed stays.
  - The failure factor starts at 1 for **<=** (stress up to the yield strength) and keeps its limit for **>=**.
  - When you change the materials of the design (Materials), a limit still at its start moves to the start for the new materials, and the messages say so; a limit you typed stays.
  - The row's (i) always gives the start for the materials chosen now and how it is worked out, for example *Starting limit 0.0035 kg: 40% of the mass of the whole part made solid in its lightest material (0.00875 kg)*. Resting the mouse on a limit still at its start says the same. Change it to the one you want.
- A **Material volume fraction** constraint also names its material, chosen from the database's materials in their colours: *Material volume fraction of Mat1_Red <= 0.15* keeps that material to at most 15 % of the design space. With one or more of these, **Material sharpness (tau)** appears above the rows: how sharply an element counts as its nearest material when the volumes are measured. One value is used for all of them; the usual value is right in most cases.
- A **Mass fraction** constraint compares the part's mass with the mass of the whole part made solid in one material: *Mass fraction of the lightest <= 0.4 of a solid part in that material* allows 40 % of what the part would weigh solid in the lightest material of the database. The material is one of the database's, or **the heaviest** or **the lightest** of them. The limit has no unit, so it stays right when the units or the part's size change.

A quantity the database cannot give is greyed out in the list, and resting the mouse on it says why, for example *needs a Yield_Strength column in the database*. Choose another database, or add that property to yours (section 6.3). With one material in the design (section 6), Material volume fraction is greyed out: *one material fills the design*. A row that already has a greyed-out quantity keeps it, with the reason beside it in red, and a run is refused until you remove it.

A constraint keeps its quantity at or below its limit. To ask for at least some amount, constrain the opposite: instead of a minimum stiffness, minimize compliance with a limit on mass, or minimize mass with a limit on compliance.

Here are some useful combinations of objectives and constraints:

- **Stiffest part for its weight**: Minimize Compliance, with Mass (or Volume fraction) <= a limit.
- **Lightest part that does not yield**: Minimize Mass, with Failure factor (p-norm) <= 1.
- **Lowest stress**: Minimize Stress (p-norm), with Volume fraction <= a limit.
- **Cheapest part**: Minimize Mass or Compliance, with Cost <= a limit (the database needs a Cost column).

The following quantities can be used as objectives or constraints:

<!-- list: quantities -->

| Quantity | Can be | What it is |
| --- | --- | --- |
| Compliance | Objective and constraint | How much the part gives under the loads. Lower is stiffer. |
| Stress (p-norm) | Objective | One value close to the highest von Mises stress in the part. |
| Failure factor (p-norm) | Objective and constraint | An average of stress / yield strength over the whole part. It sits below the highest value, often well below, so 1 here does not mean no element yields. |
| Failure factor (highest) | Constraint | The highest stress / yield strength in the part. At most 1 means no element yields. |
| Thermal compliance | Objective and constraint | How hot the part gets where the heat goes in. Lower carries the heat away better. |
| Thermoelastic compliance | Objective and constraint | How much the part gives under the loads and the heating. Lower is stiffer. |
| Mass | Objective and constraint | The mass of the part, from each material's density. |
| Cost | Constraint | The cost of the part, from each material's density and cost per mass. |
| Volume fraction | Constraint | The share of the design space filled with material, 0 to 1. |
| Material volume fraction | Constraint | The share of the design space filled with one chosen material, 0 to 1. |
| Mass fraction | Constraint | The part's mass compared with the whole part made solid in one material: 0.4 allows 40 % of that mass. Choose the material, or the heaviest or the lightest of the database. |

<!-- end of list -->

## 9. Settings

**Settings** contains the options that control how the optimization runs. For most projects, you can keep the default values.

The window has two tabs:

- **Optimization**: how fine the mesh is, how long the run is, and how the design is pushed towards clear solid and empty. The **Main** settings are at the top. The others are under **Advanced**, closed at first; click it to open them, in their groups.
- **VAE**: the material network. At the top, a line says which network the project uses and whether it fits the database, with its errors and warnings under it. With **Use the trained network** ticked and a trained network there, the network's own settings are greyed: they are the ones it was trained with. Untick it to train a new network on the next run; the settings can then be changed. With single material (section 6) there is no network, and every VAE setting is greyed out.

With one material in the design, the settings that act only on the material map are greyed out, and resting the mouse on one says *Not used with one material.* Their values are kept for when you go back to several materials.

Every setting has an (i) that says what it does. A setting with a value other than the usual one shows **↺** next to it: click it to put the usual value back.

Press **OK** to keep the changes, or **Cancel** to leave everything as it was.

The settings you will use most often are:

- **Mesh dofs**: how fine the mesh is. More gives finer detail and smoother STL files, and takes longer. Start coarse while you set a problem up, and go finer for the design you keep.
- **Single element layer (2D)**: for a flat part, one element through its thickness. Much faster.
- **Max iterations**: the run stops after this many iterations at the latest. It can stop earlier, once the design has settled (Convergence tolerance, under Advanced).

The following table lists every setting, organized by its group in the Settings panel:

<!-- list: settings -->

| Group | Setting | What it does |
| --- | --- | --- |
| Main | Mesh dofs | How fine the mesh is: the number of unknowns it is built for, counting each unknown at each node: three for structural (the movement in x, y and z), one for thermal (the temperature), four for thermoelastic. More is finer, and slower. |
| Main | Single element layer (2D) | One element through the thickness, for a flat (2D) part. |
| Main | Max iterations | The optimization stops after this many iterations. |
| Main | Density filter radius | The smallest feature the design may have, in element sizes. Larger gives smoother, simpler designs. |
| Main | Latent smoothing radius | How far material choices are smoothed between neighbouring elements, in element sizes. |
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
| Main | Stress scale damping | How fast Failure factor (highest) follows the design. 1 follows it at once; smaller is steadier but slower. |
| Material model | Constant Poisson's ratio | One Poisson's ratio for every material, instead of each material's own. |
| Material model | Poisson's ratio value | The Poisson's ratio used when Constant Poisson's ratio is ticked. |
| Main | Reference temperature | The temperature at which the part has no thermal strain, in the project's temperature unit. |
| Material model | Constant Poisson's ratio | One Poisson's ratio for every material, instead of each material's own. |
| Material model | Poisson's ratio value | The Poisson's ratio used when Constant Poisson's ratio is ticked. |

<!-- end of list -->

## 10. Running

Press **Run** at the top. Before the run begins:

- LSR MMTO checks the project. If something is missing or wrong, the run does not start, and the Current run panel lists why under *Cannot start:*. See section 14.
- The project is saved, so the run matches the file. If there are unsaved changes, LSR MMTO asks first. An example is saved as a project of your own first.

Each run uses its own copy of the project. You can continue editing the project while a run is in progress, but those changes will only affect the next run.

### 10.1 While it runs

The buttons on the right step aside, and the **Current run** panel takes half the window. The run goes through these steps, and the line at the top of the panel (and beside the Stop button) says which:

1. *Starting: building the mesh and the material network...*
2. *Training the material network: epoch 4,000 / 30,000*, only when the database has no trained network yet. This can take a few minutes.
3. *Iteration 12 / 300 · 0:41 so far · about 16:20 left*: the optimization itself.

The 3D view shows the design as it grows: each element in its material's colour, pale while it is half solid and in full colour once solid. Elements that are almost empty are not drawn. At the top left, a legend gives each material's colour and its share of the design, largest first (counted by how solid each element is). It shows whenever the database has more than one material, also while the design uses only one of them. **Part**, at the top, shows the part again without stopping the run; **Design** brings the growing design back. Untick **Show the design while it runs** to leave the 3D view alone.

The panel has these tabs:

- **Progress**: the objective, as a solid line, and each constraint, as a dotted line named as in Formulation with its limit, for example *Mass <= 0.4 t* or *Mass >= 0.1 t*. Either way, a constraint is met at or below the grey line at 0.
- **Latent space**: where the elements' materials are heading on the material map. Red points are the elements, black stars the database's materials. As the run goes on, the points usually gather on the stars. A design of one material has no Latent space tab.
- **Log**: everything the run writes, line by line.

### 10.2 Stopping

Press **Stop** (the Run button, while a run goes). The run ends after the iteration under way and keeps the design reached so far; the line says *Stopping after the iteration under way; the design so far is kept...* while it finishes. Stopped while the material network is still training, nothing is saved.

You cannot open another project or close LSR MMTO while a run is active. You must stop the run first.

### 10.3 When it ends

The run stops after Max iterations, or earlier once the design has settled. The line then says, for example, *Run finished after 40 iterations, 0:47*. The final design shows on the part, and three more tabs appear: **Convergence**, **Latent space** and **Summary**, the same as in Results (section 11); a design of one material has no Latent space.

Under the tabs:

- **Open results folder** opens the run's folder.
- **Open in Results** shows the run in the Results panel.
- **Delete this run** removes the run and its folder, after asking. This cannot be undone.

Each run is saved in a separate folder inside the project's results folder. The folder name records when the run started.

The run time mainly depends on Mesh dofs and Max iterations: a few minutes for a coarse flat part, up to hours for a fine 3D part.

### 10.4 Analyse

**Analyse**, beside Run, solves the project once with every element of the part solid, in one material of the database. It uses that material's values as the database gives them, the project's supports and loads, and the same mesh as a run. This does not perform optimization or use the material network, so it is faster than a full run. Use it to check the supports, the loads and the materials before a run: a stress far from what you expect usually means a load or a unit is wrong.

Press **Analyse**. A small window asks for:

- **Material**: the material the whole part is made of, named as in the Materials panel.
- **Save in the results**: unticked, the analysis is only shown, and nothing is written, not even the project. Ticked, it is kept in the project's results folder like a run; the project is saved first, as before a run.

Before starting, LSR MMTO performs the same checks that it uses for an optimization run: if something stops a run, it stops the analysis too, and the Current run panel says why under *Cannot start:*. While it goes, the line beside the buttons says *Analysing the solid part in* and the material, and **Stop** ends it; nothing is saved then.

When it is done, Results shows it the way it shows a run, titled **Analysis:** and the material: the part in the 3D view (Material, Density, Stress, Displacement, Solid) and its **Summary**, with the objective and each constraint for the solid part. An analysis has no **Convergence** and no **Latent space**. A material volume fraction is worked out from the material network, which an analysis does not use, so the Summary and Compare leave it out.

## 11. Results

**Results** lists the runs of the project, newest first: when each ran, how many iterations it made, and *stopped* for a run that was stopped. An analysis (section 10.4) is listed among them as **Analysis:** and its material; one that was not saved is listed first, as *not saved*, until the next analysis, or until Results or the project is closed. Click one to see it:

- its design on the part, in the 3D view;
- **Convergence**: the objective and the constraints, iteration by iteration;
- **Latent space**: where the final design's materials sit on the material map (not for a run of one material, which has none);
- **Summary**: the run's numbers (below).

**Show** chooses what the part shows:

- **Material**: each element in its material's colour, with a legend at the top left: each material and its share of the solid elements, largest first. Two runs compared get a column each, left and right. The legend shows whenever the database has more than one material (a design of one of them reads 100%), and not for a database of one material.
- **Density**: how solid each element is, from empty to solid.
- **Stress**: the stress in each element.
- **Displacement**: how far each point of the part moves under the loads. The colours blend smoothly over the faces. Beside it, choose **magnitude** (the distance) or **x**, **y** or **z** (the movement along one axis, below zero for a movement the other way). Tick **Deformed shape** to see the part moved, enlarged by **Scale**: at first the largest movement is drawn as about 5% of the part's largest size, since the real movement is usually too small to see. The (i) beside them explains.
- **Solid**: only which elements are solid, in one colour.
- **Smooth (exported)**: the STL files of the run, once it has been exported (section 12). Greyed until then.

**Open folder** opens the run's folder. **Delete this run** removes it, after asking.

**Analyse this design** solves the run's design once more as it would be built, under the supports and loads the project has now: each element with a density above 0.5 solid, in its nearest material with that material's values from the database, the others empty. Nothing is optimized. Use it to see how a design holds up under other loads: change them in Supports and loads, then come back and press it. It goes like Analyse (section 10.4), with the same checks and Stop, and is always saved in the results, listed as **Analysis: design of** and the run's date. The mesh must be the run's (the same Mesh dofs, Single element layer and part), and the database the run used must be as the run read it; if not, the messages say what changed (for the database, which cells), and **Versions...** in Materials can bring back the version the run used. When it is done, the analysis and its run are ticked, so **Compare** shows them side by side.

### 11.1 The Summary

The Summary is divided into these sections:

- **Result**: the objective, with its value and unit. Then a table of the constraints: each one's name, its **Value**, its **Limit**, how much of the limit it **Used**, and **✓ met** in green or **✗ not met** in red. A constraint counts as met when it is at most 0.1 % over its limit. For an at-least constraint (**>=**) the limit reads *at least ...*, **Used** is left empty, and it counts as met when it is at most 0.1 % under its limit.
- **Design**: the number of **Elements**, how many are **Solid** (density 0.5 or more) and what share of the elements that is, and **Grey (in between)**: how much of the design is neither solid nor empty. 0 % is a clean design.
- **Materials**: how many of the database's materials the design **Used**, and the **Latent dimension**.
- **Run**: the number of **Iterations**, and the **Units**. On a large part the solve is iterative; if some of its solves did not converge, one or two more rows say so: **Solver**, for solves during the iterations, and **Final check**, naming the designs whose numbers are not reliable (section 14.5). The final check is what a run does at its end: it solves the finished design three ways, **Continuous**, **Binarized topology** and **Binarized + snapped materials**, and the **Log** tab of the Current run panel gives each one's numbers while the run is open there. For an analysis, a **Solver** row says its numbers are not reliable. Compare shows these rows too.

The (i) at the top of the Summary says what each word means, and gives each constraint's exact value and limit.

### 11.2 Comparing two runs

Tick two runs and press **Compare**. A Compare window opens with their convergence on one plot (each constraint by its name in the legend), their latent spaces side by side (unless one of them is a run of one material), and their summaries in two columns. With an analysis, the summaries show only the rows both have. The 3D view shows both designs side by side, the first on the left. Close the Compare window, or click a run, to stop comparing.

### 11.3 Results from elsewhere

**Open other results...** shows a run from any folder, another project's too, to look at. It changes nothing in your project. A run's folder is one that holds run_data.npz.

## 12. Export STL

**Export STL**, at the top, writes the design shown in the 3D view as STL files, for printing or for CAD. It works on a finished run's design: the last run, or the run shown in Results. It is greyed while the 3D view shows the part, while a run still goes, while two runs are compared, and for a run of another project.

The exported STL files are saved in the run's folder:

- **design.stl**: the whole part, in one piece, for printing in one material. Faces of your STL the optimization did not touch keep their exact shape. When that smooth surface cannot be closed (a plate only two elements thick can do this, and an analysis has no empty elements to take away), design.stl is built from the material regions instead, the same surface the material files share, and the messages say so.
- **design_*material*.stl**: one file per material the design uses. Together they make up the part, with no gaps and no overlaps between them.

The exported files have smoothed surfaces, so the individual mesh elements do not appear as staircase-shaped steps. Smoothing cannot add detail finer than the mesh: features one element across are dropped, and square corners come out rounded. For a part to print, run on a finer mesh.

While it works, a note at the top of the 3D view says it is exporting. When it is done, the note says how many files were saved, with a link, **Open folder**. The messages show each file, compare its volume with the design's volume, and report whether the file has a closed surface. *NOT closed* after a file means it has holes, which some printers and CAD programs refuse; a finer mesh usually helps.

After an export, Results shows the smooth pieces on the part, each in its material's colour: **Show** turns to **Smooth (exported)**. It is there for every run whose folder holds the exported files, also later, and greyed for a run not exported yet. Beside it, tick **Pull the material pieces apart** to set the pieces side by side in a row along x, the largest first, so each can be seen on its own. The legend at the top left names each piece, with its share of the volume.

## 13. View

**View**, in the top row, controls how the part is displayed:

- **Cut along x (%)**, **Cut along y (%)** and **Cut along z (%)**: cut away that share of the part, along x, y or z, to see inside it. 0 shows it whole. The cut works on the design too.
- **Show the part's triangles**: the edges of the STL's triangles.
- **See-through part**: the part drawn half transparent, to see supports, loads and selections behind it.
- **Save Image**: saves the 3D view as a picture (PNG or JPEG).

## 14. When something goes wrong

The messages below are written as LSR MMTO shows them. Each one is followed by an explanation of what it means and how to fix the problem. Names such as corner, clamp or my_materials stand for your own selections and files. Some messages use the names a project file stores; this section says which field each one is.

### 14.1 Before a run

These appear in the messages as **Error:** lines, and under *Cannot start:* in the Current run panel. You can start the run once you fix these problems.

- **no supports** / **no loads**: the project has no support, or no load. Add them under Supports and loads (section 7).
- **no material database**: choose one under Materials (section 6).
- **no objective**: choose one under Formulation (section 8).
- **boundary condition 2 (force): no selection named 'corner'**: a support or load acts on a selection that no longer exists. Edit it under Supports and loads and choose a selection under Where.
- **selection clamp: ...**: a selection cannot be made as it is written, for example a file it needs is missing. Open it under Selections and correct it.
- **material database: the structural model needs a Youngs_Modulus column**, or **material database: FailureFactorPNorm needs a Yield_Strength column**: the database lacks a property the problem needs (FailureFactorPNorm is the Failure factor (p-norm); Youngs_Modulus is Young's modulus). Add the column in the database editor, or choose another database or objective.
- **material database my_materials.xlsx: D1: 'E0' is not a property LSR MMTO knows ...**: a column of the database has a name LSR MMTO does not know. Rename the column to one of the properties listed, or open the database with Edit..., which leaves the column out, and save it.
- **material database my_materials.xlsx: ...**: a problem in the database itself, with the cell it is in, for example *C5 (Steel, Density): 'abc' is not a number*, *... is outside the allowed range*, *no units for Density*, or *Steel, Steel2 have the same value in every attribute, so the VAE cannot tell them apart*. Correct it in the database editor.
- **constraint 1: the bound must be a nonzero number**: a constraint's limit is 0 or empty. Give it a limit under Formulation.
- Warning **constraint 1 (Mass <= 0.0035): This limit allows at most 0.027% of the part even in the lightest material.**: the mass limit is less than 1 % of what the whole part weighs solid in the design's material (with a mix, the lightest material of the database), so almost no material can be used. Often the limit is in another unit than meant (g for kg, say). Check it under Formulation, and the project's units (section 4).
- **the material volume fraction constraints use different tau (0.5, 1)**: give them all the same Material sharpness.
- **STL file ... does not exist**: the part's file has moved or been deleted. Choose it again in Geometry with Browse... (section 5.1).
- **no part yet: give the project its part in Geometry**: the project has no part. Give it one in Geometry (section 5).
- **the part's shapes: step 3: ...**: a step of a part made of shapes cannot be made, for the reason given. Correct it in Geometry (section 5.2).

### 14.2 The material network

- **my_materials.nt takes 3 attributes, but my_materials.xlsx has 4: it was trained on another database**: the trained network does not fit the database. In Settings, on the VAE tab, untick **Use the trained network**; the next run trains a new one.
- **latentDim is 3, but my_materials.nt was trained with 2: set it to 2, or switch off use_pretrained_vae to train a new network**: a network setting differs from the trained network's. latentDim is **Latent dimension**, and use_pretrained_vae is **Use the trained network**, both in Settings on the VAE tab.
- **my_materials.xlsx: the database has changed since its network was trained: cell B4 is 210000.0, was 200000.0**: the network learned from other numbers. Train a new one (untick **Use the trained network** in Settings), or bring back the version it was trained on with **Versions...** in Materials (section 6.5).
- **... which comes with the app, does not match ...**: a database that comes with LSR MMTO was changed outside LSR MMTO. Put the file back as it came, or make your own copy with **Edit...** and let its first run train a network.
- **a run would train a new VAE and save it over the shipped ...**: a database that comes with LSR MMTO would have its network replaced. Copy the database into your project with Edit... (section 6.1) and use the copy.
- Warning **my_materials.nt exists: the next run trains a new network and replaces it**: Use the trained network is off, so the next run trains again. The old network is kept beside it, ending in .previous.nt.
- Warning **latentDim is 3; the database's effective dimension is ... so 2 is recommended**: the latent dimension is larger than the materials need, which has made runs fail to settle. Set Latent dimension to the recommended value, unless you have a reason not to.
- When a run starts, a window **Material network** can say *The material network reproduces the database less closely than the 5% warning limit: ...*, with the error of each property. The run would work with properties that far from your database. **Continue** runs anyway; **Cancel** stops before anything is made. To do better, train the network again (untick Use the trained network), or raise **Warn above network error [%]** in Settings if the error is acceptable to you.

### 14.3 Units

- Warning **the database gives Young's modulus in Pa, but the project's units (mmKS) expect MPa: its numbers are 10^6 times what they should be. Displacement and compliance come out 10^6 times too small; stress and failure factor are unaffected.** The database and the project use different unit systems. The message says what it changes, sometimes including what a constraint's limit then really means. Either change the project's units (section 4) to match the database, or convert the database into the project's units with **Convert to ... units...** in Materials (section 6.1).

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
- **Final check: solver did not converge for ...: not reliable**, in the Summary: on a large part the solve is iterative, and for the designs named it stopped before reaching the answer. Their numbers are still shown, but do not trust them. If **Binarized + snapped materials** is named, the results the run shows (Stress and Displacement, for a structural run) are not reliable either, since they come from it. The (i) gives each solve's error. **Solver: did not converge in N solves of the iterations** means the same for solves during the optimization: the design it reached may not be the best one. **Solver: did not converge: not reliable**, for an analysis, means none of its numbers can be trusted. When a run or an analysis ends with any of these, the Current run panel says so in red under its status line, and the messages panel says it too.

### 14.6 Opening and saving

- **STL not found: ...**, when a project opens: the part's file is not where the project says. Choose it again in Geometry with Browse... (section 5.1).
- **3 settings missing from the file were set to their defaults; Save writes them.**: the project was made by an older version. You do not need to fix anything. Save the project to keep these values.
- **Not saved: ...**: the project could not be written, for example the folder is read-only or the file is open elsewhere. Save it somewhere else with Save As.
- **Start a project first: File > New or File > Open.**: the panel needs a project. Make or open one.
- **A run is going. Stop it first...**: another project cannot open, and LSR MMTO cannot close, while a run goes.

### 14.7 Something went wrong

A message **Something went wrong**, with a short technical line under it, means LSR MMTO met a problem it did not expect. LSR MMTO keeps running. Save the project; if anything looks wrong afterwards, close LSR MMTO and start it again.
