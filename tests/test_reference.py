"""
test_reference.py
=================
docs/reference.md must be what the code generates: after adding a quantity,
a selection type, a boundary-condition kind or a setting, or changing a docstring the
Python section is read from, run

    python docs/make_reference.py

and commit the page with the change. Every example of the Python section must run, one after
another as the page says, on the shipped U-bracket.
"""
import os
import shutil
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "docs"))
import make_reference as reference  # noqa: E402


def test_reference_page_is_current():
    with open(reference.PAGE, encoding="utf-8") as fh:
        page = fh.read()
    assert page == reference.generate(), "docs/reference.md is out of date: run python docs/make_reference.py"


def test_every_python_example_runs():
    """The examples, run in their order from a copy of the LSR MMTO folder's U-bracket project and part, make,
    save and write as a script the project they build, then run the shipped U-bracket and read it back.
    run(p) is stood in for by Analyse (the solid part solved once and saved as a run is saved), so nothing
    is optimized; analyse_design, which takes only a run, is stood in for the same way, once its run folder
    is checked to be one."""
    import matplotlib
    matplotlib.use("Agg")
    from lsr_mmto.optimize.solid_part_analysis import analyse_solid_part
    from lsr_mmto.problem.project_folder import new_run_folder

    def analysed(project, **options):
        folder = new_run_folder(project)
        analyse_solid_part(project, 0, save_dir=folder)
        return folder

    def design_analysed(project, run_folder, save_dir=None, should_stop=None):
        from lsr_mmto.problem.project_folder import is_run
        assert is_run(run_folder)
        return analyse_solid_part(project, 0, save_dir=save_dir)

    here = os.getcwd()
    with tempfile.TemporaryDirectory() as d:
        os.makedirs(os.path.join(d, "projects"))
        shutil.copy(os.path.join(ROOT, "projects", "U-bracket.mmto"), os.path.join(d, "projects"))
        shutil.copytree(os.path.join(ROOT, "data", "models", "UBracket"), os.path.join(d, "data", "models", "UBracket"))
        names = {}
        os.chdir(d)
        try:
            for imports, example in reference.examples():
                if imports:
                    exec(imports, names)
                    if "run" in names:
                        names["run"] = analysed
                    if "analyse_design" in names:
                        names["analyse_design"] = design_analysed
                exec(example, names)
        finally:
            os.chdir(here)
        built = os.path.join(d, "my bracket")
        assert sorted(os.listdir(built)) == ["materials", "my bracket.mmto", "my bracket_setup.py", "part"]
        assert names["r"].analysed_material is not None and names["report"].pieces
        assert all(os.path.isfile(piece.path) for piece in names["report"].pieces)


if __name__ == "__main__":
    test_reference_page_is_current()
    test_every_python_example_runs()
    print("all passed")
