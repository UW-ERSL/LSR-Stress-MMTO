"""
make_testing_doc.py
===================
Writes docs/testing_the_codebase.md from the tests themselves: one section for each folder of tests/,
what the folder checks (FOLDERS below), then each file in it with what it checks (the first paragraph
of its docstring) and each of its tests (the first paragraph of the test's docstring, or its name as a
sentence when it has none). A file marked slow (pytestmark = pytest.mark.slow) says so.

    python docs/make_testing_doc.py        (from the repository folder)

The page is generated, not edited by hand; tests/docs/test_testing_doc.py fails when the page in the
repository differs from what the tests generate, so a new test is listed as soon as it is written. A
folder of tests with no line in FOLDERS, or a test file outside the folders, stops the page being written.
"""
import ast
import contextlib
import glob
import importlib.util
import inspect
import io
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TESTS = os.path.join(ROOT, "tests")
PAGE = os.path.join(ROOT, "docs", "testing_the_codebase.md")

# each folder of tests/, in the page's order, and what its tests check
FOLDERS = {
    "physics": "Each physics gives the right answers: its solver on cases with a known answer, its gradients, "
               "its runs.",
    "gradients": "The autodiff gradients of the objectives and constraints agree with central finite differences.",
    "problem": "Setting up a project: saving and loading, the checks before a run, the selections, the mesh, "
               "the project folder.",
    "materials": "Material databases: their attributes and units, the check of a database file, the latent "
                 "dimension recommended for one.",
    "runs": "A run or an analysis goes through and saves what it should, and its numbers do not drift.",
    "gui": "The window and the desktop start, off-screen: the panels, the buttons, a project taken through the "
           "window the way a user would.",
    "docs": "The documents match the code: the generated pages are current, and the notebook and the examples run.",
    "code_rules": "The code keeps its own rules: one entry adds a feature, every physics folder keeps the "
                  "template's shape, every module imports.",
}


def _paragraph(doc: str | None, skip_title: bool = False) -> str:
    """The first paragraph of a docstring as one line; skip_title leaves out a "name.py" line underlined
    with "=" at its top."""
    lines = inspect.cleandoc(doc or "").splitlines()
    if skip_title and len(lines) > 1 and set(lines[1].strip()) == {"="}:
        lines = lines[2:]
    while lines and not lines[0].strip():
        lines = lines[1:]
    first = []
    for line in lines:
        if not line.strip():
            break
        first.append(line.strip())
    return " ".join(first)


def _sentence(name: str) -> str:
    """A test's name as a sentence: test_a_bar_grows_freely is "A bar grows freely."."""
    words = name[len("test_"):].replace("_", " ")
    return words[:1].upper() + words[1:] + "."


def _module(path: str) -> "object":
    """The test file, imported under a name of its own (so pytest's own import of it is not touched), from
    its folder, as pytest imports it; what it prints is not shown."""
    folder = os.path.dirname(path)
    for p in (ROOT, folder):
        if p not in sys.path:
            sys.path.insert(0, p)
    name = "_testing_doc_" + os.path.splitext(os.path.basename(path))[0]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    with contextlib.redirect_stdout(io.StringIO()):
        spec.loader.exec_module(module)
    return module


def _tests(module: "object") -> list[tuple[str, str]]:
    """(name, what it checks) of each test of the module, in the order the file makes them."""
    out = []
    for name, value in vars(module).items():
        if name.startswith("test_") and callable(value):
            out.append((name, _paragraph(value.__doc__) or _sentence(name)))
    return out


def _is_slow(module: "object") -> bool:
    marks = getattr(module, "pytestmark", [])
    marks = marks if isinstance(marks, list) else [marks]
    return any(getattr(m, "name", None) == "slow" for m in marks)


def generate() -> str:
    """The page, from the test files as they are."""
    loose = sorted(os.path.basename(p) for p in glob.glob(os.path.join(TESTS, "test_*.py")))
    if loose:
        raise ValueError(f"test files outside the folders of tests/: {', '.join(loose)}; move each into one")
    found = sorted(os.path.basename(os.path.dirname(p)) for p in glob.glob(os.path.join(TESTS, "*", "test_*.py")))
    missing = sorted(set(found) - set(FOLDERS))
    if missing:
        raise ValueError(f"folders of tests with no line in FOLDERS (docs/make_testing_doc.py): {', '.join(missing)}")
    gone = sorted(set(FOLDERS) - set(found))
    if gone:
        raise ValueError(f"FOLDERS names folders with no tests: {', '.join(gone)}")

    count = 0
    body = []
    for folder, about in FOLDERS.items():
        body += [f"## {folder}", "", about, ""]
        for path in sorted(glob.glob(os.path.join(TESTS, folder, "test_*.py"))):
            module = _module(path)
            tests = _tests(module)
            count += len(tests)
            name = os.path.basename(path)
            slow = " (slow: left out by `-m \"not slow\"`)" if _is_slow(module) else ""
            body += [f"### `tests/{folder}/{name}`{slow}", "", _paragraph(module.__doc__, skip_title=True), ""]
            body += [f"- `{test}`: {what}" for test, what in tests] + [""]
        others = sorted(os.path.basename(p) for p in glob.glob(os.path.join(TESTS, folder, "*.py"))
                        if not os.path.basename(p).startswith(("test_", "conftest")))
        for other in others:                            # a script run by hand: its docstring only, not imported
            with open(os.path.join(TESTS, folder, other), encoding="utf-8") as fh:
                doc = ast.get_docstring(ast.parse(fh.read()))
            body += [f"### `tests/{folder}/{other}` (a script, not collected by pytest)", "",
                     _paragraph(doc, skip_title=True), ""]

    head = ["# Testing the codebase", "",
            "Every test of LSR MMTO, folder by folder: what each folder checks, what each file in it checks, and "
            f"each test in it, {count} in all. Generated from the tests' own docstrings by "
            "`python docs/make_generated_docs.py`; do not edit it by hand.", "",
            "Run them all from the repository folder with `python -m pytest tests -q -n 4` (`-n 2` on a computer "
            "with less than 16 GB of memory), one folder with `python -m pytest tests/<folder> -q`, or one file "
            "with `python -m pytest tests/<folder>/<file> -q`.", ""]
    return "\n".join(head + body).rstrip() + "\n"


if __name__ == "__main__":
    text = generate()
    with open(PAGE, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    print(f"wrote {PAGE}")
