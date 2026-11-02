"""
test_project_folder.py
======================
A project folder holds everything its project uses: create makes one with the STL copied in;
pack copies an example's STL, database and network into a new folder, leaving the example as
it was, and the copy checks and builds exactly as the example does; copy_in keeps a different
file of the same name and never copies the same file twice; runs are listed newest first and
only a run's folder can be deleted. A few seconds.

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
        F.delete_run(made[0])
        assert F.runs(p) == [made[1]] and os.path.isdir(os.path.join(d, "P", "part"))
        old = Project.load(os.path.join(PROJECTS, "beam.mmto"))              # not in a folder of its own:
        assert F.results_root(old) == os.path.join(PROJECTS, "results", "beam")   # results as before
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
