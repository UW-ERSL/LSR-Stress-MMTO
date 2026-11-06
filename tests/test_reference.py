"""
test_reference.py
=================
docs/reference.md must be what the code generates: after adding a quantity,
a selection type, a boundary-condition kind or a setting, run

    python docs/make_reference.py

and commit the page with the change.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "docs"))
import make_reference as reference  # noqa: E402


def test_reference_page_is_current():
    with open(reference.PAGE, encoding="utf-8") as fh:
        page = fh.read()
    assert page == reference.generate(), "docs/reference.md is out of date: run python docs/make_reference.py"


if __name__ == "__main__":
    test_reference_page_is_current()
    print("all passed")
