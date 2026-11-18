# Making a part from shapes

A project's part can come from an STL file made in a CAD program, or it can be made in LSR MMTO from simple shapes: boxes, cylinders, spheres, outlines pushed up or spun around, joined and cut out of each other. This section shows how, in a script. In the GUI, the same steps are made in Geometry, the first step (user guide section 5.2). Every block of Python in this section is run by the tests, so it works as written (all but the CSV example in section 4, which needs a file of your own).

Lengths are in the project's length unit and angles are in degrees.

## 1. A first part

An L-bracket 100 by 100 and 10 thick, with a hole near the end of each arm:

```python
from lsr_mmto import Project, shapes as S

part = [S.add(S.extrusion(S.polygon([(0, 0), (100, 0), (100, 30), (30, 30), (30, 100), (0, 100)]), 10)),
        S.cut(S.cylinder(radius=6, height=10), at=(80, 15, 0)),
        S.cut(S.cylinder(radius=6, height=10), at=(15, 80, 0))]
p = Project("my L-bracket", part_shapes=part)
```

A part is a list of steps, done in order. The first step adds the L. The next two cut a hole each.

`Project` takes either `stl=` (an STL file) or `part_shapes=` (a list of steps), never both. Once made, the steps are `p.part_shapes`. Everything else about the project (units, supports, loads, materials, the run) is the same as for a part from an STL.

## 2. Steps

| Step | What it does |
|---|---|
| `S.add(shape)` | joins the shape to the part |
| `S.cut(shape)` | removes the shape from the part |
| `S.overlap(shape)` | keeps only the part inside the shape |

The first step must be an add. Each step also takes:

- `at=(x, y, z)`: where the shape's origin goes. The default is (0, 0, 0).
- `rotate=(rx, ry, rz)`: turns the shape before it is moved to `at`: first by rx degrees about the x axis through the shape's origin, then by ry about the y axis, then by rz about the z axis. The axes stay fixed while it turns.
- `repeat=`: makes copies of the shape (section 5).

A cylinder lying along x, for a hole through the side of a part, is a cylinder turned 90 degrees about y:

```python
part = [S.add(S.box(40, 40, 40)),
        S.cut(S.cylinder(radius=8, height=40), at=(0, 20, 20), rotate=(0, 90, 0))]
```

## 3. Shapes

| Shape | Its origin (where `at` puts it) |
|---|---|
| `S.box(x, y, z)` | its lowest corner |
| `S.cylinder(radius, height)` | the centre of its base; it stands along z |
| `S.cylinder(radius, height, top_radius=r)` | a cone: the top has radius r (0 for a point) |
| `S.sphere(radius)` | its centre |
| `S.extrusion(outline, height)` | the outline is drawn flat on the x-y plane and pushed up z by height |
| `S.revolution(outline, angle)` | the outline's x is the distance from the z axis and its y the height; it is spun around z, a full turn unless angle says less (a part turn starts at +x and turns towards +y) |
| `S.wrap(S.place(...), S.place(...))` | the convex hull of the placed shapes: the shape a sheet stretched tight around them would make |

`S.place(shape, at=..., rotate=...)` puts a shape somewhere for a wrap, the same way a step does. Two cylinders wrapped make a lug:

```python
part = [S.add(S.wrap(S.place(S.cylinder(25, 10)), S.place(S.cylinder(12, 10), at=(90, 0, 0)))),
        S.cut(S.cylinder(10, 10)),
        S.cut(S.cylinder(5, 10), at=(90, 0, 0))]
```

A flange is an outline spun around z, with six bolt holes:

```python
part = [S.add(S.revolution(S.polygon([(15, 0), (60, 0), (60, 10), (25, 10), (25, 40), (15, 40)]))),
        S.cut(S.cylinder(4, 10), at=(45, 0, 0), repeat=S.around_circle(6))]
```

## 4. Outlines

Extrusions and revolutions are made from an outline:

| Outline | Drawn on the x-y plane |
|---|---|
| `S.rectangle(width, depth)` | from (0, 0) to (width, depth) |
| `S.circle(radius)` | centred on (0, 0) |
| `S.polygon(points)` | through its corner points, in order around the edge (either way round) |

A rectangle and a polygon take `round=r`, which rounds every corner by radius r: the straight edges stay where they are and each corner is cut in to an arc between them. For a rectangle, r is at most half its narrower side (exactly half makes a slot). For a polygon, r must be less than half its narrowest width, and each edge long enough for the rounding at its two ends; rounding that would remove or fill in more than its corners is refused. A circle cannot be spun by `S.revolution`, since it is centred on the axis it would spin around. A plate with rounded corners:

```python
part = [S.add(S.extrusion(S.rectangle(120, 120, round=8), 6))]
```

A polygon can have as many points as needed, so an outline with many points (an airfoil, a cam) is a polygon too. To read the points from a CSV file, with x in the first column and y in the second (a header row is skipped):

```
part = [S.add(S.extrusion(S.polygon(S.points_from_csv("profile.csv")), 20))]
```

The points are copied into the project, so the CSV file is not needed after that.

## 5. Repeats

| Repeat | What it makes |
|---|---|
| `S.along_line(count, step=(dx, dy, dz))` | count copies, each moved by the step from the one before |
| `S.around_circle(count, centre=(x, y, z), axis="z")` | count copies spread evenly over a full turn about the line through centre along the axis |

The first copy stays where `at` puts it. A plate with a ring of eight bolt holes and a large hole in the middle:

```python
part = [S.add(S.extrusion(S.rectangle(120, 120, round=8), 6)),
        S.cut(S.cylinder(4, 6), at=(100, 60, 0), repeat=S.around_circle(8, centre=(60, 60, 0))),
        S.cut(S.cylinder(20, 6), at=(60, 60, 0))]
```

A beam with four rounded windows cut through it:

```python
part = [S.add(S.box(200, 20, 40)),
        S.cut(S.extrusion(S.rectangle(25, 20, round=5), 20), at=(20, 20, 10), rotate=(90, 0, 0),
              repeat=S.along_line(4, step=(45, 0, 0)))]
```

A channel: a base and two walls, with a hole through both walls:

```python
part = [S.add(S.box(120, 60, 8)),
        S.add(S.box(10, 60, 60)),
        S.add(S.box(10, 60, 60), at=(110, 0, 0)),
        S.cut(S.cylinder(8, 140), at=(-10, 30, 40), rotate=(0, 90, 0))]
```

## 6. The examples, made from shapes

Three of the five projects in the projects folder can be made from shapes, and their mesh comes out exactly the same as from their STL files (the tests check this):

```python
l_bracket = [S.add(S.extrusion(S.polygon([(0, 0), (200, 0), (200, 80), (80, 80), (80, 200), (0, 200)]), 1))]
u_bracket = [S.add(S.extrusion(S.polygon([(0, 0), (1, 0), (1, 1), (0.75, 1), (0.75, 0.5), (0.5, 0.5),
                                          (0.5, 1), (0, 1)]), 0.01))]
rectangular_beam = [S.add(S.box(1000, 100, 50))]
```

The beam and the airplane bracket have curved free-form surfaces, so they need their STL files.

## 7. What happens when the part is used

When the part is first needed (to mesh it, to show it, or to run), LSR MMTO builds it into an STL file in the `part` folder of the project's folder. A project not saved yet uses the folder Python runs in, so give `base_dir=` or save the project first to keep its files together. Keep one project made of shapes to a folder (as the GUI does): two in one folder remove each other's built file each time they build.

The file is named `shapes_` and 6 characters, for example `shapes_3f9a1c.stl`. The 6 characters are a fingerprint of the steps: the same steps always give the same name, and any change gives a new one. When the project builds its part, the files built before are removed. A run keeps its own copy of the steps, so the design of an older run can still be shown after the part has changed: its part is built again from those steps, beside the project's, and is removed the next time the project builds.

Curves are drawn as flat pieces that stay within a ten-thousandth of the part's size (the diagonal of the box around it) of the true curve, at most 1024 pieces to a circle. This is far finer than any element. The pieces depend only on the steps, not on Mesh dofs, so changing Mesh dofs never moves the supports and loads picked on the part.

Some parts cannot be made. Then the project's checks (and so a run) say which step is wrong and why:

- the first step is not an add;
- a step leaves nothing of the part;
- a polygon's edges cross, a size is 0 or less, or rounding is too large (section 4).

Some parts can be made, but may not come out as meant. These are warnings: they are among the project's warnings (`p.warnings()`), which a run prints before it starts, and `p.part_warnings()` gives them alone.

- a cut, or some copies of a repeated cut, misses the part, so it cuts nothing;
- the part is in separate pieces;
- a feature is less than 2 elements across at the project's Mesh dofs, so it may vanish or come out square in the mesh. The warning says what Mesh dofs would show it. For a single layer, only sizes across the plate count, since the plate is one element thick by choice.

```python
p = Project("plate", part_shapes=[S.add(S.box(100, 100, 10)),
                                  S.cut(S.cylinder(0.5, 10), at=(50, 50, 0)),
                                  S.cut(S.box(5, 5, 5), at=(500, 0, 0))], n_dof=3000)
for warning in p.part_warnings():
    print(warning)
```

## 8. Saving

The steps are saved in the project file under `part_shapes`. `p.save_as_script(...)` writes them as the same `S.` calls used in this section, so a saved script rebuilds the same part.

Scaling the part (Scale... in the GUI's Geometry step, or `scale_part` in `lsr_mmto.problem.project_folder`) multiplies every length in the steps (sizes, radii, positions, points, repeats) and builds the part again. The project file changes when the project is next saved. The part is made again rather than stretched, so its triangles may be numbered differently: supports and loads picked as triangles may need picking again (the GUI asks first when there are any).

Choosing an STL file in the GUI's Geometry step (Browse...) removes the steps from the project, after asking; they are gone from the file at the next save. To keep them, save the project script first.

## 9. When CAD is still needed

Shapes cover parts made of flat faces, round holes, cylinders, cones, spheres and outlines. A CAD program is still needed for:

- free-form curved surfaces, like the beam and the airplane bracket;
- rounding the edge where two faces meet (`round` only rounds the corners of an outline);
- a shape swept along a curved path, or blended between two different outlines;
- importing a STEP file. Export it as an STL from the CAD program instead.
