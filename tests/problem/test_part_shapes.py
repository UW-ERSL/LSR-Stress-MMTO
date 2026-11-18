"""
test_part_shapes.py
===================
A part made of shapes (lsr_mmto/problem/shapes.py, built by part_from_shapes.py): every shape comes out
closed with the volume it should have; every block of Python in docs/part_shapes.md runs and its parts build;
the L-bracket, U-bracket and rectangular beam made from shapes mesh exactly as their STL files do; the element
size worked out for the warnings is the mesher's; the STL's triangles depend on the steps alone, not on Mesh
dofs; parts that cannot be made are refused naming the step, and parts that may not come out as meant are
warned about; each change of the steps is built into a file of its own, so an older run's part and the
project's never share one; a project made of shapes saves, loads, writes a script that rebuilds it, and
scales. A few seconds.

    python tests/problem/test_part_shapes.py        (or: python -m pytest tests)
"""
import math
import os
import re
import shutil
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from lsr_mmto import shapes as S  # noqa: E402
from lsr_mmto.optimize.solid_part_analysis import _same_part  # noqa: E402
from lsr_mmto.problem import project_folder as F  # noqa: E402
from lsr_mmto.problem.part_from_shapes import _pieces, build_surface, element_size  # noqa: E402
from lsr_mmto.problem.project import Project  # noqa: E402
from lsr_mmto.problem.shapes import ShapeError  # noqa: E402

DOCS = os.path.join(ROOT, "docs", "part_shapes.md")
PROJECTS = os.path.join(ROOT, "projects")


def _scratch():
    return tempfile.mkdtemp(prefix="_test_part_shapes_")


def _volume(steps):
    surface, _ = build_surface(steps)
    assert surface.is_watertight
    return surface.volume


def _warnings(steps, n_dof=3000, single_layer=False):
    folder = _scratch()
    try:
        return Project("p", part_shapes=steps, n_dof=n_dof, single_layer=single_layer, base_dir=folder).part_warnings()
    finally:
        shutil.rmtree(folder, ignore_errors=True)


def _run_docs():
    """Every python block of the docs page run in turn in one namespace, as a reader would, in a folder of its
    own; each part a block makes, built. Returns the namespace."""
    blocks = re.findall(r"```python\n(.*?)```", open(DOCS, encoding="utf-8").read(), re.S)
    names, here, folder = {}, os.getcwd(), _scratch()
    try:
        os.chdir(folder)                                    # where a block's project builds its part
        for block in blocks:
            names.pop("part", None)
            exec(block, names)
            for key, value in names.items():
                if key == "part" or key.endswith(("_bracket", "_beam")):
                    assert build_surface(value)[0].is_watertight, key
    finally:
        os.chdir(here)
        shutil.rmtree(folder, ignore_errors=True)
    return names


def test_every_shape_has_the_volume_it_should():
    near = lambda a, b: abs(a - b) < 0.002 * b       # noqa: E731   curves are drawn in flat pieces
    assert near(_volume([S.add(S.box(2, 3, 4))]), 24)
    assert near(_volume([S.add(S.cylinder(5, 10))]), math.pi * 25 * 10)
    assert near(_volume([S.add(S.cylinder(5, 10, top_radius=0))]), math.pi * 25 * 10 / 3)
    assert near(_volume([S.add(S.sphere(5))]), 4 / 3 * math.pi * 125)
    assert near(_volume([S.add(S.extrusion(S.rectangle(20, 10, round=2), 3))]), (200 - (4 - math.pi) * 4) * 3)
    assert near(_volume([S.add(S.extrusion(S.rectangle(20, 4, round=2), 1))]), 16 * 4 + math.pi * 4)   # a slot
    assert near(_volume([S.add(S.extrusion(S.circle(4), 2))]), math.pi * 16 * 2)
    assert near(_volume([S.add(S.revolution(S.rectangle(2, 5), angle=180))]), math.pi * 4 * 5 / 2)
    assert near(_volume([S.add(S.revolution(S.polygon([(3, 0), (5, 0), (5, 4), (3, 4)])))]), math.pi * 16 * 4)
    assert near(_volume([S.add(S.wrap(S.place(S.cylinder(10, 2)), S.place(S.cylinder(10, 2), at=(30, 0, 0))))]),
                (math.pi * 100 + 20 * 30) * 2)
    hexagon = [(20 * math.cos(k * math.pi / 3), 20 * math.sin(k * math.pi / 3)) for k in range(6)]
    for points, r in ((hexagon, 5), ([(0, 10), (-5, 0), (0, -10), (5, 0)], 2), ([(0, 0), (30, 0), (10, 20)], 2),
                      ([(0, 0), (100, 0), (100, 30), (30, 30), (30, 100), (0, 100)], 14.9)):
        assert _volume([S.add(S.extrusion(S.polygon(points, round=r), 1))]) < \
            _volume([S.add(S.extrusion(S.polygon(points), 1))])                # rounded, not refused
    assert near(_volume([S.add(S.box(10, 10, 10)), S.overlap(S.box(10, 10, 10), at=(5, 0, 0))]), 500)
    assert near(_volume([S.add(S.box(100, 10, 10)), S.cut(S.box(2, 10, 10), at=(10, 0, 0),
                                                            repeat=S.along_line(3, (20, 0, 0)))]), 10000 - 600)
    ring = [S.add(S.cylinder(50, 5)), S.cut(S.cylinder(5, 5), at=(30, 0, 0), repeat=S.around_circle(6))]
    assert near(_volume(ring), math.pi * (2500 - 6 * 25) * 5)
    turned = [S.add(S.box(10, 10, 10)), S.cut(S.cylinder(2, 10), at=(0, 5, 5), rotate=(0, 90, 0))]
    assert near(_volume(turned), 1000 - math.pi * 4 * 10)                   # a hole along x, through the side


def test_every_block_of_the_docs_page_runs():
    assert {"l_bracket", "u_bracket", "rectangular_beam"} <= set(_run_docs())


def test_three_examples_made_of_shapes_mesh_as_their_stl_files():
    names, folder = _run_docs(), _scratch()
    try:
        for name, part in (("L-bracket", names["l_bracket"]), ("U-bracket", names["u_bracket"]),
                           ("rectangular beam", names["rectangular_beam"])):
            example = Project.load(os.path.join(PROJECTS, f"{name}.mmto"))
            made = Project(name, part_shapes=part, n_dof=example.n_dof, single_layer=example.single_layer,
                           base_dir=os.path.join(folder, name))
            a, b = example.mesh(), made.mesh()
            assert a.num_elems == b.num_elems and (a.node_xyz == b.node_xyz).all(), name
            _, facts = build_surface(part)
            assert np.allclose(element_size(facts["bounds"], facts["volume"], made.n_dof, made.single_layer,
                                            made.unknowns_per_node()), b.elem_size, rtol=1e-12), name
    finally:
        shutil.rmtree(folder, ignore_errors=True)


def test_the_triangles_depend_on_the_steps_alone():
    plate = [S.add(S.box(100, 40, 10)), S.cut(S.cylinder(15, 10), at=(50, 20, 0))]
    folder = _scratch()
    try:
        p = Project("p", part_shapes=plate, n_dof=3000, base_dir=folder)
        first = open(p.stl_path(), "rb").read()
        p.n_dof = 300000                                    # a finer mesh: the same file, the same triangles
        assert open(p.stl_path(), "rb").read() == first
    finally:
        shutil.rmtree(folder, ignore_errors=True)
    assert (_pieces(1, 0.5), _pieces(400, 0.04), _pieces(1e6, 1e-6)) == (16, 223, 1024)     # within the limits


def test_parts_that_cannot_be_made_are_refused_naming_the_step():
    for steps, says in (([S.cut(S.box(1, 1, 1))], "step 1: the first step adds"),
                        ([S.add(S.box(10, 10, 10)), S.cut(S.box(20, 20, 20), at=(-5, -5, -5))],
                         "step 2 leaves nothing"),
                        ([S.add(S.extrusion(S.polygon([(0, 0), (10, 10), (10, 0), (0, 10)]), 5))],
                         "edges cross"),
                        ([S.add(S.box(0, 1, 1))], "step 1"),
                        ([S.add(S.extrusion(S.rectangle(10, 4, round=2.5), 1))], "at most half its narrower side"),
                        ([S.add(S.extrusion(S.polygon([(0, 0), (40, 0), (40, 4), (22, 4), (22, 100), (18, 100),
                                                       (18, 4), (0, 4)], round=2.5), 1))], "narrower than 5"),
                        ([S.add(S.extrusion(S.polygon([(0, 0), (30, 0), (30, 5), (5, 5), (5, 22), (20, 22),
                                                       (20, 23), (5, 23), (5, 40), (30, 40), (30, 45), (0, 45)],
                                                      round=1), 1))], "is lost or filled in"),       # a thin prong
                        ([S.add(S.extrusion(S.polygon([(0, 0), (60, 0), (60, 20), (30.5, 20), (30.5, 5), (29.5, 5),
                                                       (29.5, 20), (0, 20)], round=1), 1))], "is lost or filled in"),
                        ([S.add(S.revolution(S.circle(2)))], "a circle is centred on the axis"),
                        ([S.add(S.extrusion({"kind": "circle", "radius": 2, "round": 1}, 1))], "no corners to round"),
                        ([{"action": "add", "shape": {"kind": "box", "size": np.ones(3)}, "at": [0, 0, 0],
                           "rotate": [0, 0, 0]}], "three numbers (x, y, z), in a list"),
                        ([S.add(S.extrusion(S.rectangle(1, 1), 0))], "an extrusion's height must be above 0"),
                        ([S.add(S.wrap(S.place(S.box(1, 1, 1)), S.box(1, 1, 1)))], "S.place(shape"),
                        ([S.add(S.box(1, 1, 1), repeat=S.along_line(2.5, (1, 0, 0)))], "whole number"),
                        ((S.add(S.box(1, 1, 1)),), "a list of steps"),
                        ([{"action": "add", "shape": {"kind": "torus"}, "at": [0, 0, 0], "rotate": [0, 0, 0],
                           "repeat": None}], "step 1")):
        try:
            build_surface(steps)
        except ShapeError as e:
            assert says in str(e), (says, str(e))
        else:
            raise AssertionError(f"built, but should say: {says}")
    try:
        Project("p", stl="a.stl", part_shapes=[S.add(S.box(1, 1, 1))])
    except ValueError as e:
        assert "give one of the two" in str(e)
    else:
        raise AssertionError("an STL and shapes both")
    assert not Project("p").has_part()                      # neither: the part comes later
    folder = _scratch()
    try:
        p = Project("p", part_shapes=[S.add(S.box(1, 1, 1))], base_dir=folder)
        p.part_shapes = [S.add(S.box(10, 10, 10)), S.cut(S.box(20, 20, 20), at=(-5, -5, -5))]
        assert any(e.startswith("the part's shapes: step 2 leaves nothing") for e in p.validate(False))
        assert not os.path.exists(p.stl_path()) and p.warnings() == []
    finally:
        shutil.rmtree(folder, ignore_errors=True)


def test_parts_that_may_not_come_out_as_meant_are_warned_about():
    assert _warnings([S.add(S.box(10, 10, 10)), S.cut(S.box(2, 2, 10)), S.cut(S.box(3, 3, 3), at=(50, 50, 50))]) \
        == ["step 3 cuts nothing: it misses the part"]
    assert _warnings([S.add(S.box(10, 10, 10)), S.cut(S.box(2, 2, 10), repeat=S.along_line(3, (5, 0, 0)))]) \
        == ["step 2: 1 of its 3 copies cut nothing: they miss the part"]
    assert _warnings([S.add(S.box(10, 10, 10)), S.add(S.box(5, 5, 5), at=(30, 0, 0))]) \
        == ["the part is in 2 separate pieces; a run treats them as one part"]
    small = _warnings([S.add(S.box(100, 100, 10)), S.cut(S.cylinder(0.5, 10), at=(50, 50, 0))])
    assert len(small) == 1 and small[0].startswith("step 2 (") and "Mesh dofs of about" in small[0]
    flat = [S.add(S.extrusion(S.polygon([(0, 0), (200, 0), (200, 80), (80, 80), (80, 200), (0, 200)]), 1))]
    assert _warnings(flat, n_dof=30000, single_layer=True) == []         # a single layer is one element thick
    assert _warnings([S.add(S.revolution(S.rectangle(40, 2)))], n_dof=30000, single_layer=True) == []
    assert _warnings([S.add(S.box(10, 10, 10)), S.cut(S.box(2, 2, 10))]) == []
    folder = _scratch()
    try:
        p = Project("p", part_shapes=[S.add(S.box(10, 10, 10)), S.add(S.box(5, 5, 5), at=(30, 0, 0))], base_dir=folder)
        assert "part: the part is in 2 separate pieces; a run treats them as one part" in p.warnings()
    finally:
        shutil.rmtree(folder, ignore_errors=True)


def test_each_change_of_the_steps_is_built_into_a_file_of_its_own():
    folder = _scratch()
    try:
        p = Project("p", part_shapes=[S.add(S.box(10, 10, 10))], base_dir=folder)
        run = Project.from_record(p.to_record())            # a run's project, as its record keeps it
        first = p.stl_path()
        assert re.fullmatch(r"shapes_[0-9a-f]{6}\.stl", os.path.basename(first))
        p.part_shapes = [S.add(S.box(10, 10, 10)), S.cut(S.box(2, 2, 10))]
        second = p.stl_path()
        assert second != first and sorted(os.listdir(os.path.dirname(second))) == \
            sorted([os.path.basename(second), os.path.basename(second)[:-4] + ".json"])     # only the newest kept
        assert run.stl_path() == first and os.path.isfile(first)            # the run's own part, built again
        assert os.path.isfile(second)                                       # beside the project's, kept
        os.remove(second)
        assert p.stl_path() == second and not os.path.exists(first)         # built again; the run's removed
        assert not _same_part(run, p) and _same_part(run, run.copy())
        by_hand = [{"action": "add", "shape": {"kind": "box", "size": [10, 10, 10]}, "at": (0, 0, 0),
                    "rotate": [0, 0, 0]}]                                   # no repeat key, whole numbers
        assert _same_part(run, Project("h", part_shapes=by_hand, base_dir=folder))
        mine = os.path.join(folder, "part", "shapes_mine.stl")
        with open(mine, "w") as fh:
            fh.write("an STL of the user's own")
        p.part_shapes = [S.add(S.box(5, 5, 5))]
        p.stl_path()
        assert open(mine).read() == "an STL of the user's own"              # only built names are removed
        stl = os.path.join(folder, "part", "part.stl")
        with open(stl, "w") as fh:
            fh.write("an STL of the user's own")
        assert open(Project("q", stl=stl, base_dir=folder).stl_path()).read() == "an STL of the user's own"
        q = Project("q", stl=stl, base_dir=folder)
        q.part_shapes = [S.add(S.box(1, 1, 1))]             # shapes set on a project of an STL
        assert q.stl_path() != stl and open(stl).read() == "an STL of the user's own"
    finally:
        shutil.rmtree(folder, ignore_errors=True)


def test_a_project_made_of_shapes_saves_loads_scripts_and_scales():
    part = [S.add(S.box(100, 40, 10)),
            S.cut(S.cylinder(5, 20), at=(20, 20, -5), repeat=S.along_line(3, (30, 0, 0)))]
    first, folder = _scratch(), _scratch()
    try:
        p = Project("plate", part_shapes=part, base_dir=first)
        p.save(os.path.join(folder, "plate.mmto"))
        assert os.listdir(first) == []                          # saved elsewhere: nothing built where it was
        assert os.path.dirname(p.stl_path()) == os.path.join(folder, "part") and F.has_own_folder(p)
        assert not F.has_own_folder(Project("other", part_shapes=part, base_dir=folder))   # its file is not there
        q = Project.load(os.path.join(folder, "plate.mmto"))
        assert q.part_shapes == part and q.to_dict() == p.to_dict() and q.stl is None
        r = Project.from_record(q.to_record())
        assert r.part_shapes == part and r.stl_path() == q.stl_path()
        F.scale_part(q, np.float64(2))
        assert q.part_shapes[0]["shape"]["size"] == [200, 80, 20] and q.part_shapes[1]["at"] == [40, 40, -10]
        assert abs(q.mesh().node_xyz.max(axis=0)[0] - 200) < 1e-9
        script = os.path.join(folder, "plate_setup.py")
        q.save_as_script(script)
        assert "np." not in open(script).read()
        os.remove(os.path.join(folder, "plate.mmto"))
        exec(compile(open(script).read(), script, "exec"), {"__file__": script, "__name__": "__main__"})
        assert Project.load(os.path.join(folder, "plate.mmto")).part_shapes == q.part_shapes
        q.part_shapes = [S.cut(S.box(1, 1, 1))]
        try:
            q.save_as_script(script)
        except ShapeError as e:
            assert "step 1" in str(e)
        else:
            raise AssertionError("a script of steps that cannot be built was written")
        assert Project("a", stl="a.stl").part_warnings() == []
    finally:
        shutil.rmtree(first, ignore_errors=True)
        shutil.rmtree(folder, ignore_errors=True)


if __name__ == "__main__":
    test_every_shape_has_the_volume_it_should()
    test_every_block_of_the_docs_page_runs()
    test_three_examples_made_of_shapes_mesh_as_their_stl_files()
    test_the_triangles_depend_on_the_steps_alone()
    test_parts_that_cannot_be_made_are_refused_naming_the_step()
    test_parts_that_may_not_come_out_as_meant_are_warned_about()
    test_each_change_of_the_steps_is_built_into_a_file_of_its_own()
    test_a_project_made_of_shapes_saves_loads_scripts_and_scales()
    print("ok")
