"""
test_testing_doc.py
===================
docs/testing_the_codebase.md must be what the tests generate: after adding, removing or renaming a test,
changing a test's docstring, or moving a test file, run

    python docs/make_generated_docs.py

and commit the page with the change. A folder of tests must have its line in FOLDERS
(docs/make_testing_doc.py), and no test file may sit outside the folders.
"""
import os
import shutil
import sys
import tempfile

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "docs"))
import make_testing_doc as testing_doc  # noqa: E402


def test_testing_page_is_current():
    """docs/testing_the_codebase.md is what docs/make_testing_doc.py writes from the tests as they are."""
    with open(testing_doc.PAGE, encoding="utf-8") as fh:
        page = fh.read()
    assert page == testing_doc.generate(), \
        "docs/testing_the_codebase.md is out of date: run python docs/make_generated_docs.py"


def test_a_folder_without_its_line_or_a_loose_test_file_stops_the_page(monkeypatch):
    """A folder of tests with no line in FOLDERS, a line for a folder with no tests, and a test file directly in
    tests/ each stop the page with a message naming them, so a new folder or file cannot go undescribed."""
    monkeypatch.setattr(testing_doc, "FOLDERS", {k: v for k, v in testing_doc.FOLDERS.items() if k != "gui"})
    with pytest.raises(ValueError, match="no line in FOLDERS.*gui"):
        testing_doc.generate()
    monkeypatch.setattr(testing_doc, "FOLDERS", dict(testing_doc.FOLDERS, gui="", test_nowhere="nothing"))
    with pytest.raises(ValueError, match="no tests: test_nowhere"):
        testing_doc.generate()
    with tempfile.TemporaryDirectory() as d:
        shutil.copytree(os.path.join(ROOT, "tests", "docs"), os.path.join(d, "docs"))
        open(os.path.join(d, "test_loose.py"), "w").close()
        monkeypatch.setattr(testing_doc, "TESTS", d)
        with pytest.raises(ValueError, match="outside the folders of tests/: test_loose.py"):
            testing_doc.generate()
