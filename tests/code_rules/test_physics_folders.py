"""
test_physics_folders.py
=======================
Every physics folder keeps the shape of the form (lsr_mmto/physics/template/):

    the five files       <name>_module.py, _physics.py, _supports_and_loads.py, _quantities.py, _results.py
    the same places      each entry of the front page (MODULE_ENTRIES) a physics gives sits in the file the
                         form puts it in, and the form gives or lists every one
    the markers          each top-level function and class has "# ESSENTIAL:", "# OPTIONAL:" or "# HELPER:"
                         on the line above it
    the front page only  no file outside a physics' folder (the core, the GUI) imports any of its files but
                         its front page
    the final check      final_summary reads its designs only with "for ... in designs" or by an index
                         (designs[-1]), so the run knows which design each solve is for and labels it
                         (optimize/final_design_analysis.py): never list(designs), sorted(designs), or designs
                         handed to another function; and it solves each by calling the physics' own analyse
                         (on design_for_analyse), so no setting is read twice and the final check solves as
                         the iterations did

A test for each check shows it catches what it is for.

    python tests/code_rules/test_physics_folders.py        (or: python -m pytest tests)
"""
import ast
import glob
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from lsr_mmto.physics.available_physics import MODULE_ENTRIES  # noqa: E402

PHYSICS_DIR = os.path.join(ROOT, "lsr_mmto", "physics")
FOLDERS = sorted(d for d in os.listdir(PHYSICS_DIR) if os.path.isfile(os.path.join(PHYSICS_DIR, d, f"{d}_module.py")))
JOBS = ("module", "physics", "supports_and_loads", "quantities", "results")
MARKER = re.compile(r"^\s*#\s*(ESSENTIAL|OPTIONAL|HELPER):\s*(\w+)?")


def unmarked(source, name):
    """'name:line function' for each top-level function or class of source without a marker on the line
    above it (above its decorators, when it has some)."""
    lines = source.split("\n")
    out = []
    for node in ast.parse(source).body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            first = min([node.lineno] + [d.lineno for d in node.decorator_list])
            if first < 2 or not MARKER.match(lines[first - 2]):
                out.append(f"{name}:{node.lineno} {node.name}")
    return out


def entries_by_file(folder):
    """{entry: job} for each front-page entry the folder gives: defined at the top level of one of its five
    files, or named first on a marker line there (the form's commented-out OPTIONAL entries)."""
    out = {}
    for job in JOBS:
        source = open(os.path.join(PHYSICS_DIR, folder, f"{folder}_{job}.py"), encoding="utf-8").read()
        names = set()
        for node in ast.parse(source).body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                names.add(node.name)
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                names |= {t.id for t in targets if isinstance(t, ast.Name)}
        names |= {m.group(2) for m in map(MARKER.match, source.split("\n")) if m and m.group(2)}
        for entry in names & set(MODULE_ENTRIES):
            out.setdefault(entry, job)
    return out


def misplaced(given, form):
    """'entry: in <job>, the form has it in <job>' for each entry of given ({entry: job}) the form ({entry:
    job}) puts elsewhere or not at all."""
    return [f"{e}: in its {given[e]} file, the form's is in {form.get(e, 'none')}"
            for e in sorted(given) if given[e] != form.get(e)]


def private_imports(source, name):
    """'name:line module' for each import in source of a physics folder's file other than its front page."""
    out = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom) and node.module:
            parts = node.module.split(".")
            targets = [parts + [a.name] for a in node.names] if len(parts) == 3 else [parts]
        elif isinstance(node, ast.Import):
            targets = [a.name.split(".") for a in node.names]
        else:
            continue
        for t in targets:
            if t[:2] == ["lsr_mmto", "physics"] and len(t) >= 4 and t[2] in FOLDERS and t[3] != f"{t[2]}_module":
                out.append(f"{name}:{node.lineno} {'.'.join(t[:4])}")
    return out


def designs_misread(source, name):
    """'name:line' for each place final_summary reads its designs (its first argument, whatever its name) other
    than "for ... in designs" or designs[<an index>]."""
    out = []
    for fn in ast.walk(ast.parse(source)):
        if not (isinstance(fn, ast.FunctionDef) and fn.name == "final_summary" and fn.args.args):
            continue
        designs = fn.args.args[0].arg
        allowed = set()
        for node in ast.walk(fn):
            if isinstance(node, (ast.For, ast.AsyncFor)) and isinstance(node.iter, ast.Name):
                allowed.add(id(node.iter))
            if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) and \
                    not isinstance(node.slice, ast.Slice):
                allowed.add(id(node.value))
        out += [f"{name}:{node.lineno}" for node in ast.walk(fn) if isinstance(node, ast.Name)
                and node.id == designs and isinstance(node.ctx, ast.Load) and id(node) not in allowed]
    return out


def solves_without_analyse(source, name):
    """[name] when source has a final_summary that never calls analyse (as analyse(...) or something.analyse(...)),
    so it repeats the solve itself; [] otherwise."""
    for fn in ast.walk(ast.parse(source)):
        if isinstance(fn, ast.FunctionDef) and fn.name == "final_summary":
            calls = [node.func for node in ast.walk(fn) if isinstance(node, ast.Call)]
            if not any((isinstance(f, ast.Name) and f.id == "analyse") or
                       (isinstance(f, ast.Attribute) and f.attr == "analyse") for f in calls):
                return [name]
    return []


def test_every_physics_has_the_five_files():
    missing = [f"{d}/{d}_{job}.py" for d in FOLDERS for job in JOBS
               if not os.path.isfile(os.path.join(PHYSICS_DIR, d, f"{d}_{job}.py"))]
    assert {"structural", "template", "thermal"} <= set(FOLDERS), FOLDERS
    assert not missing, missing


def test_the_form_lists_every_entry_and_each_physics_puts_them_where_it_does():
    form = entries_by_file("template")
    assert sorted(form) == sorted(MODULE_ENTRIES), sorted(set(MODULE_ENTRIES) - set(form))
    wrong = [f"{d}: {m}" for d in FOLDERS if d != "template" for m in misplaced(entries_by_file(d), form)]
    assert not wrong, "\n".join(wrong)


def test_every_function_of_a_physics_is_marked():
    wrong = [u for path in sorted(glob.glob(os.path.join(PHYSICS_DIR, "*", "*.py")))
             for u in unmarked(open(path, encoding="utf-8").read(), os.path.relpath(path, ROOT))]
    assert not wrong, "\n".join(wrong)


def test_outside_a_physics_only_its_front_page_is_imported():
    inside = {os.path.join(PHYSICS_DIR, d) for d in FOLDERS}
    paths = glob.glob(os.path.join(ROOT, "lsr_mmto", "**", "*.py"), recursive=True) + \
        glob.glob(os.path.join(ROOT, "gui", "**", "*.py"), recursive=True) + glob.glob(os.path.join(ROOT, "*.py"))
    wrong = [p for path in sorted(paths) if os.path.dirname(path) not in inside
             for p in private_imports(open(path, encoding="utf-8").read(), os.path.relpath(path, ROOT))]
    assert not wrong, "\n".join(wrong)


def test_the_final_check_reads_its_designs_one_by_one():
    wrong = [w for d in FOLDERS for w in designs_misread(
        open(os.path.join(PHYSICS_DIR, d, f"{d}_results.py"), encoding="utf-8").read(), f"{d}_results.py")]
    assert not wrong, "final_summary reads designs other than one by one:\n" + "\n".join(wrong)


def test_the_final_check_solves_with_the_physics_own_analyse():
    """final_summary solves each design with the physics' own analyse, never a solve of its own."""
    wrong = [w for d in FOLDERS if d != "template" for w in solves_without_analyse(
        open(os.path.join(PHYSICS_DIR, d, f"{d}_results.py"), encoding="utf-8").read(), f"{d}_results.py")]
    assert not wrong, ("final_summary solves without calling the physics' analyse (build each design with "
                       "design_for_analyse and call analyse on it):\n" + "\n".join(wrong))


def test_the_checks_catch_what_they_are_for():
    assert unmarked("def f():\n    pass\n", "a.py") == ["a.py:1 f"]
    assert unmarked("# HELPER: of g\n@staticmethod\ndef f():\n    pass\n", "a.py") == []
    assert unmarked("x = 1\n# a comment, no marker\nclass C:\n    pass\n", "a.py") == ["a.py:3 C"]
    assert misplaced({"analyse": "results", "NAME": "module"}, {"analyse": "physics", "NAME": "module"}) == \
        ["analyse: in its results file, the form's is in physics"]
    assert private_imports("from lsr_mmto.physics.thermal.thermal_physics import analyse\n", "c.py") == \
        ["c.py:1 lsr_mmto.physics.thermal.thermal_physics"]
    assert private_imports("def f():\n    from lsr_mmto.physics.structural import structural_results\n", "c.py") \
        == ["c.py:2 lsr_mmto.physics.structural.structural_results"]
    assert private_imports("import lsr_mmto.physics.thermal.thermal_results as R\n", "c.py") == \
        ["c.py:1 lsr_mmto.physics.thermal.thermal_results"]
    assert private_imports("from lsr_mmto.physics.thermal import thermal_module\n"
                           "from lsr_mmto.physics.available_physics import PHYSICS\n", "c.py") == []
    one_by_one = "def final_summary(designs):\n    for l, x, z in designs:\n        pass\n    _, x, z = designs[-1]\n"
    assert designs_misread(one_by_one, "r.py") == []
    assert designs_misread("def final_summary(designs):\n    all = list(designs)\n", "r.py") == ["r.py:2"]
    assert designs_misread("def final_summary(designs):\n    return [d for d in designs]\n", "r.py") == ["r.py:2"]
    assert designs_misread("def final_summary(cases):\n    return sorted(cases)\n", "r.py") == ["r.py:2"]
    assert designs_misread("def final_summary(d):\n    for i in range(3):\n        d[i]\n    d[1:]\n", "r.py") \
        == ["r.py:4"]
    assert solves_without_analyse("def final_summary(designs):\n    analyse(d)\n", "r.py") == []
    assert solves_without_analyse("def final_summary(designs):\n    physics.analyse(d)\n", "r.py") == []
    assert solves_without_analyse("def final_summary(designs):\n    solver.solve(x)\n", "r.py") == ["r.py"]


if __name__ == "__main__":
    test_every_physics_has_the_five_files()
    test_the_form_lists_every_entry_and_each_physics_puts_them_where_it_does()
    test_every_function_of_a_physics_is_marked()
    test_outside_a_physics_only_its_front_page_is_imported()
    test_the_final_check_reads_its_designs_one_by_one()
    test_the_final_check_solves_with_the_physics_own_analyse()
    test_the_checks_catch_what_they_are_for()
    print("ok")
