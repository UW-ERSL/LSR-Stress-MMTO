"""
test_project_folder.py
======================
A project folder holds everything its project uses: create makes one with the STL copied in;
pack copies an example's STL, database and network into a new folder, leaving the example as
it was, and the copy checks and builds exactly as the example does; copy_in keeps a different
file of the same name and never copies the same file twice; runs are listed newest first and
only a run's folder can be deleted; a save from a script makes the same folder as the window,
and leaves the setup script alone. A few seconds.

    python tests/test_project_folder.py        (or: python -m pytest tests)
"""
import filecmp
import os
import shutil
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from lsr_mmto.problem import project_folder as F  # noqa: E402
from lsr_mmto.problem.project import Project  # noqa: E402

PROJECTS = os.path.join(ROOT, "projects")
STL = os.path.join(ROOT, "data", "models", "LBracket_BM1", "LBracket_BM1.STL")


def _scratch():
    return tempfile.mkdtemp(prefix="_test_folder_")


def test_create():
    d = _scratch()
    try:
        p, path = F.create(os.path.join(d, "Bracket study"), STL)
        assert path == os.path.join(d, "Bracket study", "Bracket study.mmto") and os.path.isfile(path)
        assert p.name == "Bracket study" and p.stl == "part/LBracket_BM1.STL"
        assert filecmp.cmp(p.stl_path(), STL, shallow=False) and F.has_own_folder(p)
        assert Project.load(path).to_dict() == p.to_dict()
        assert "cannot contain" in F.name_problem("a:b")       # the name alone: on Windows, "a:b" joined to a
        for bad, says in ((os.path.join(d, "Bracket study"), "already exists"),      # folder is a path on drive A:
                          (os.path.join(d, "no", "such"), "is not a folder")):
            try:
                F.create(bad, STL)
                raise AssertionError(bad)
            except ValueError as e:
                assert says in str(e), e
        os.makedirs(os.path.join(d, "empty"))
        assert F.create(os.path.join(d, "empty"), STL)[0].name == "empty"          # an empty folder is fine
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_pack_an_example():
    """Save As: the STL into part/, the shipped database and its network into materials/; the example's
    file is unchanged; the copy has the same settings, the same checks and builds the same problem."""
    d = _scratch()
    original = os.path.join(PROJECTS, "L-bracket.mmto")
    before = open(original).read()
    try:
        example = Project.load(original)
        example.complete_settings()
        p = F.pack(Project.load(original), os.path.join(d, "Mine"))
        p.save(os.path.join(d, "Mine", "Mine.mmto"))
        assert open(original).read() == before
        assert p.stl == "part/LBracket_BM1.STL" and p.database == "materials/BM1_3Mats_2Attr.xlsx"
        assert os.path.isfile(os.path.join(d, "Mine", "materials", "BM1_3Mats_2Attr.nt"))
        q = Project.load(os.path.join(d, "Mine", "Mine.mmto"))
        q.complete_settings()
        assert q.settings == example.settings                       # the shipped network's settings kept
        assert q.validate() == [] and q.vae_check()[0] == []
        assert os.path.normpath(q.network_path()) == os.path.join(d, "Mine", "materials", "BM1_3Mats_2Attr.nt")
        assert F.has_own_folder(q) and F.results_root(q) == os.path.join(d, "Mine", "results")
        assert F.bring_in_database(q) is None                       # already in the folder
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_copy_in():
    d = _scratch()
    try:
        a, b = os.path.join(d, "a", "x.txt"), os.path.join(d, "b", "x.txt")
        for path, text in ((a, "one"), (b, "two")):
            os.makedirs(os.path.dirname(path))
            open(path, "w").write(text)
        folder = os.path.join(d, "project")
        assert F.copy_in(a, folder, "part") == "part/x.txt"
        assert F.copy_in(a, folder, "part") == "part/x.txt"                 # the same file: not again
        assert F.copy_in(b, folder, "part") == "part/x_2.txt"               # another file: kept apart
        assert open(os.path.join(folder, "part", "x_2.txt")).read() == "two"
        assert F.copy_in(os.path.join(folder, "part", "x.txt"), folder, "part") == "part/x.txt"
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_runs_listed_and_deleted():
    d = _scratch()
    try:
        p, _ = F.create(os.path.join(d, "P"), STL)
        made = []
        for _ in range(2):
            folder = F.new_run_folder(p)
            os.makedirs(folder)
            open(os.path.join(folder, F.RUN_FILE), "wb").write(b"")
            made.append(folder)
            time.sleep(0.05)
        assert made[0] != made[1] and F.runs(p) == made[::-1]               # newest first
        when, iterations, stopped = F.run_facts(made[0])
        assert len(when) == 19 and iterations is None and not stopped
        try:
            F.delete_run(os.path.join(d, "P", "part"))
            raise AssertionError("deleted a folder that holds no run")
        except ValueError:
            pass
        assert F.run_name(made[0]) is None and F.run_name(made[0] + "_2") is None        # names a run is given
        renamed = F.rename_run(made[0], "  Low stress try ")
        assert renamed == os.path.join(os.path.dirname(made[0]), "Low stress try")
        assert F.run_name(renamed) == "Low stress try" and set(F.runs(p)) == {made[1], renamed}
        for bad in ("", "a/b", "x:y", os.path.basename(made[1])):              # empty, not a name, taken
            try:
                F.rename_run(renamed, bad)
                raise AssertionError(f"renamed to {bad!r}")
            except ValueError:
                pass
        assert F.rename_run(renamed, "Low stress try") == renamed               # the same name: nothing to do
        made[0] = F.rename_run(renamed, os.path.basename(made[0]))             # back to its date: unnamed again
        assert F.run_name(made[0]) is None
        F.delete_run(made[0])
        assert F.runs(p) == [made[1]] and os.path.isdir(os.path.join(d, "P", "part"))
        old = Project.load(os.path.join(PROJECTS, "beam.mmto"))              # not in a folder of its own:
        assert F.results_root(old) == os.path.join(PROJECTS, "results", "beam")   # results as before
    finally:
        shutil.rmtree(d, ignore_errors=True)


def _files(folder):
    """The files in a project folder, relative, without the project file and its setup script."""
    out = []
    for root, _, names in os.walk(folder):
        out += [os.path.relpath(os.path.join(root, n), folder).replace("\\", "/") for n in names
                if not n.endswith((".mmto", "_setup.py"))]
    return sorted(out)


def test_a_save_brings_in_the_files_regions_name():
    """A region that names a file of its own (the airplane bracket's collar, inside_stl) has the file
    copied into part/ with the part, and points at the copy, so the saved folder runs on its own."""
    from lsr_mmto.problem import selections as S
    p = Project.load(os.path.join(PROJECTS, "Airplane bracket.mmto"))
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "Plane", "Plane.mmto")
        p.save(path)
        q = Project.load(path)
        found = []

        def walk(s):
            for part in getattr(s, "parts", ()):
                walk(part)
            if isinstance(s, S.inside_stl):
                found.append(s.path)
        for s in q.selections.values():
            walk(s)
        assert found == ["part/AirplaneBracket_NonDesign.stl"], found
        assert os.path.isfile(os.path.join(d, "Plane", found[0]))


def test_a_script_save_makes_the_same_folder():
    """Saving from a script copies the part and the database with its network into the project's folder,
    exactly as the window does at File > New and when a database is chosen; a second save copies
    nothing more; a script's save writes no setup script."""
    d = _scratch()
    try:
        app, _ = F.create(os.path.join(d, "App"), STL)                       # the window's way
        app.database = "BM1_3Mats_2Attr"
        F.bring_in_database(app)
        mine = Project("Mine", stl=STL, base_dir=d)                           # a script's way
        mine.database = "BM1_3Mats_2Attr"
        path = os.path.join(d, "Mine", "Mine.mmto")
        mine.save(path)
        assert _files(os.path.join(d, "Mine")) == _files(os.path.join(d, "App")) == [
            "materials/BM1_3Mats_2Attr.nt", "materials/BM1_3Mats_2Attr.xlsx", "part/LBracket_BM1.STL"]
        assert (mine.stl, mine.database) == (app.stl, app.database) and F.has_own_folder(mine)
        assert not os.path.exists(F.setup_script(mine))
        mine.save(path)
        assert _files(os.path.join(d, "Mine")) == _files(os.path.join(d, "App"))
        assert Project.load(path).to_dict() == mine.to_dict()
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_setup_script_left_alone_by_a_script_save():
    """The setup script the window wrote is not rewritten by a save from a script (a script must not
    rewrite itself), and it rebuilds the project it was written from."""
    d = _scratch()
    try:
        p, path = F.create(os.path.join(d, "P"), STL)
        script = F.write_setup_script(p)
        assert script == os.path.join(d, "P", "P_setup.py") and "from lsr_mmto import Project" in open(script).read()
        written = open(script).read()
        p.n_dof = 4321
        p.save(path)
        assert open(script).read() == written
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_scale_part():
    """Scale part: every coordinate of the project's own part times the factor, exactly, written over the same
    file (one STL in part/ after three scales; by 2 then by 3 is by 6); the STL it came from is untouched;
    a project not in a folder of its own, and a factor that is not a number above 0, are refused."""
    import numpy as np
    from stl import mesh as stl_mesh
    d = _scratch()
    try:
        source = shutil.copy(STL, os.path.join(d, "export.STL"))           # the user's CAD export
        before = open(source, "rb").read()
        original = stl_mesh.Mesh.from_file(source).vectors.copy()
        p, _ = F.create(os.path.join(d, "P"), source)
        assert F.scale_part(p, 2.0) == os.path.abspath(p.stl_path())
        assert np.array_equal(stl_mesh.Mesh.from_file(p.stl_path()).vectors, original * 2.0)
        F.scale_part(p, 3)
        assert np.array_equal(stl_mesh.Mesh.from_file(p.stl_path()).vectors, original * 6.0)
        F.scale_part(p, 1 / 1000)
        assert np.array_equal(stl_mesh.Mesh.from_file(p.stl_path()).vectors, original * 6.0 * (1 / 1000))
        assert os.listdir(os.path.join(d, "P", "part")) == ["export.STL"] and open(source, "rb").read() == before
        for bad in (0, -2.0, float("inf"), True, "2"):
            try:
                F.scale_part(p, bad)
                raise AssertionError(bad)
            except ValueError as e:
                assert "above 0" in str(e), e
        example = Project.load(os.path.join(PROJECTS, "L-bracket.mmto"))   # its STL is the shipped one
        shipped = open(example.stl_path(), "rb").read()
        try:
            F.scale_part(example, 2.0)
            raise AssertionError("an example's STL was scaled")
        except ValueError as e:
            assert "folder of its own" in str(e), e
        assert open(example.stl_path(), "rb").read() == shipped
    finally:
        shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    failed = 0
    for name, fn in [(k, v) for k, v in list(globals().items()) if k.startswith("test_")]:
        try:
            fn()
            print(f"ok   {name}")
        except Exception as e:                                     # noqa: BLE001
            failed += 1
            print(f"FAIL {name}: {type(e).__name__}: {e}")
    print("all passed" if not failed else f"{failed} failed")
    sys.exit(1 if failed else 0)


def test_new_project_has_the_physics_chosen():
    """New project lists every physics, the first chosen; the project is made with the one chosen."""
    import shutil
    import tempfile
    from lsr_mmto.physics.available_physics import PHYSICS
    from lsr_mmto.problem import project_folder
    from lsr_mmto.problem.project import Project
    stl = os.path.join(ROOT, "data", "models", "UBracket", "UBracket.STL")
    with tempfile.TemporaryDirectory() as d:
        p, path = project_folder.create(os.path.join(d, "P"), stl, physics=next(iter(PHYSICS)))
        assert Project.load(path).physics == next(iter(PHYSICS))
        q, _ = project_folder.create(os.path.join(d, "Q"), stl)
        assert q.physics == next(iter(PHYSICS))


def test_runs_survive_moving_the_project_folder():
    """A run's record holds its paths relative to the project folder, so after the folder is moved (or renamed,
    or zipped to someone else) the run still rebuilds its mesh, redraws and exports. An older record, with
    absolute paths (written on Windows, as C:\\...), finds them at the same place in the folder where it is now."""
    import json
    from types import SimpleNamespace
    from gui import draw_design
    from lsr_mmto.optimize.solid_part_analysis import analyse_solid_part
    from lsr_mmto.results import design_3d, load_results, stl_export
    d = _scratch()
    try:
        p = F.pack(Project.load(os.path.join(PROJECTS, "U-bracket.mmto")), os.path.join(d, "A"))
        p.n_dof = 3000
        p.save(os.path.join(d, "A", "A.mmto"))
        run_folder = F.new_run_folder(p)
        analyse_solid_part(p, 0, save_dir=run_folder)                    # saved as a run is saved
        stamp = os.path.basename(run_folder)
        manifest = os.path.join("results", stamp, "run_data.config.json")
        record = json.load(open(os.path.join(d, "A", manifest)))
        spec = record["arguments_resolved"]["to_problem"]
        assert spec["base_dir"] == "../.." and spec["stl_file_path"] == "part/UBracket.STL"
        assert record["topopt_params"]["MaterialsExcelFile"] == "materials/6Materials_AF_3Attr.xlsx"
        assert "PROJECTDEMO" not in open(os.path.join(d, "A", "results", stamp, "run_config.yaml")).read()

        shutil.move(os.path.join(d, "A"), os.path.join(d, "B"))           # the whole folder moved
        moved = os.path.join(d, "B", "results", stamp)

        def works():
            run = load_results.load(moved)
            spec = run.config["arguments_resolved"]["to_problem"]
            assert spec["base_dir"] == os.path.join(d, "B")
            assert spec["stl_file_path"] == os.path.join(d, "B", "part", "UBracket.STL")
            design_3d._MESH_CACHE.clear()
            mesh = design_3d.mesh_for(run)                                  # rebuilt from the moved part
            assert draw_design.run_mesh(SimpleNamespace(), run).num_elems == mesh.num_elems == run.n_elems
            report = stl_export.export(run, mesh, moved)
            assert report.whole_closed and all(piece.closed for piece in report.pieces)
        works()

        old = "C:\\PROJECTDEMO\\A"                                       # an older record: absolute, elsewhere
        spec = dict(record["arguments_resolved"]["to_problem"], base_dir=old,
                    stl_file_path=old + "\\part/UBracket.STL")
        record["arguments_resolved"].update(to_problem=spec, save_dir=old + "\\results\\" + stamp)
        record["topopt_params"]["MaterialsExcelFile"] = old + "\\materials/6Materials_AF_3Attr.xlsx"
        json.dump(record, open(os.path.join(d, "B", manifest), "w"))
        works()
        config = load_results.load(moved).config
        assert config["topopt_params"]["MaterialsExcelFile"] == os.path.join(d, "B", "materials",
                                                                              "6Materials_AF_3Attr.xlsx")
        assert config["arguments_resolved"]["save_dir"] == moved
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_new_project_dialog_offers_every_physics():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt5 import QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from gui import main_window as G
    from lsr_mmto.physics.available_physics import PHYSICS
    w = G.MMTOWindow()
    d = G.ProjectFolderDialog(w, "New project", ROOT, stl="")
    assert [d.physics.itemData(i) for i in range(d.physics.count())] == list(PHYSICS)
    assert d.physics_name() == next(iter(PHYSICS)) and d.physics.currentText().startswith("Structural: ")
    assert G.ProjectFolderDialog(w, "Save As", ROOT).physics_name() is None      # only a new project asks
