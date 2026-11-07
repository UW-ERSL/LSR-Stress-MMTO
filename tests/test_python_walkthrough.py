"""
test_python_walkthrough.py
==========================
Python_walkthrough.ipynb runs from top to bottom with the settings it uses, in a copy of the LSR MMTO
folder that holds the notebook and the U-bracket's part, and leaves the project folder its notes
describe; every link in its notes leads to an entry of docs/reference.md.

    python -m pytest tests/test_python_walkthrough.py
"""
import asyncio
import json
import os
import re
import shutil
import sys
import tempfile

import nbformat

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NOTEBOOK = os.path.join(ROOT, "Python_walkthrough.ipynb")
KERNEL = "lsr-stress-mmto-ad"                       # the kernel the setup registers, which the notebook names


def _anchor(heading):
    """The link target of a heading, as GitHub and VS Code make it."""
    return re.sub(r"[^\w\- ]", "", heading.strip().lower()).replace(" ", "-")


def test_every_link_leads_to_a_reference_entry():
    with open(os.path.join(ROOT, "docs", "reference.md"), encoding="utf-8") as fh:
        anchors = {_anchor(h) for h in re.findall(r"^#+ (.+)$", fh.read(), re.M)}
    nb = nbformat.read(NOTEBOOK, as_version=4)
    notes = [c.source for c in nb.cells if c.cell_type == "markdown"]
    linked = [a for text in notes for a in re.findall(r"\]\(docs/reference\.md#([^)]+)\)", text)]
    assert linked and set(linked) <= anchors, sorted(set(linked) - anchors)
    codes = [c for c in nb.cells if c.cell_type == "code"]
    assert all(nb.cells[nb.cells.index(c) - 1].cell_type == "markdown" for c in codes)    # a note beside each cell
    assert nb.metadata["kernelspec"]["name"] == KERNEL


def test_the_notebook_runs_from_top_to_bottom(monkeypatch):
    from nbclient import NotebookClient
    with tempfile.TemporaryDirectory() as d:
        kernels = os.path.join(d, "jupyter", "kernels", KERNEL)
        os.makedirs(kernels)
        with open(os.path.join(kernels, "kernel.json"), "w") as fh:     # this Python, as the setup's kernel is
            json.dump({"argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
                       "display_name": KERNEL, "language": "python",
                       "env": {"PYTHONPATH": os.pathsep.join([ROOT] + [p for p in [os.environ.get("PYTHONPATH")] if p]),
                               "MPLBACKEND": "agg"}}, fh)
        monkeypatch.setenv("JUPYTER_PATH", os.path.join(d, "jupyter"))
        folder = os.path.join(d, "LSR MMTO")
        shutil.copytree(os.path.join(ROOT, "data", "models", "UBracket"), os.path.join(folder, "data", "models", "UBracket"))
        nb = nbformat.read(NOTEBOOK, as_version=4)
        before = asyncio.get_event_loop_policy()
        if sys.platform == "win32":                # the kernel's messages need the selector event loop on Windows
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        try:
            NotebookClient(nb, timeout=600, resources={"metadata": {"path": folder}}).execute()
        finally:
            asyncio.set_event_loop_policy(before)
        errors = [o for c in nb.cells if c.cell_type == "code" for o in c.outputs if o.output_type == "error"]
        assert not errors, errors
        project = os.path.join(folder, "python_walkthrough_project")
        assert sorted(os.listdir(project)) == ["U-bracket.mmto", "materials", "part", "results"]
        (run,) = os.listdir(os.path.join(project, "results"))
        assert os.path.isfile(os.path.join(project, "results", run, "design.stl"))
        printed = "".join(o.get("text", "") for c in nb.cells if c.cell_type == "code" for o in c.outputs)
        assert "The trained network gives back every material within" in printed


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-q"]))
