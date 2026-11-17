"""
test_every_physics_runs.py
==========================
Every physics in available_physics.MODULES runs: its tiny example passes the project's check, goes through a
short run of 2 iterations, is saved, reads back with its final check's results, and its objective is a number at
every iteration. This file names no physics:
each gives its own example, as tiny_example in its own test file, tests/physics/test_<physics>_physics.py, where
everything about testing that physics lives. A physics added to MODULES is run here at once; until its test file
gives a tiny_example, this test fails saying what to add.

    python -m pytest tests/runs/test_every_physics_runs.py
"""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from lsr_mmto.physics.available_physics import MODULES, PHYSICS  # noqa: E402


@pytest.mark.parametrize("physics", [m.NAME for m in MODULES])
def test_every_physics_runs_its_tiny_example(physics, tiny_run, tmp_path):
    """The physics' tiny example passes the project's check, so a failure of its run is the run's, not the
    setup's. Its run (tiny_run) was saved; read back, it is a run of that physics with every result its
    RESULT_FIELDS lists, each a finite number, and a finite objective at the start and at each of 2 iterations."""
    from conftest import tiny_example_of
    fresh = tiny_example_of(physics)(str(tmp_path))
    fresh.complete_settings()
    assert fresh.validate() == [], f"{physics}'s tiny_example does not pass the project's check"
    assert fresh.settings["common"]["maxIterations"] == 2, f"{physics}'s tiny_example must run 2 iterations"
    _, run = tiny_run(physics)
    assert run.physics == physics
    assert set(run.results) == set(PHYSICS[physics].RESULT_FIELDS)
    for name, values in run.results.items():
        assert np.all(np.isfinite(values)), f"{name} has a value that is not a number"
    objective = np.asarray(run.history["objective"], dtype=float)
    assert len(objective) >= 2 and np.all(np.isfinite(objective)), f"{physics}: an objective that is not a number"


def test_a_physics_with_no_test_file_or_no_tiny_example_is_told_what_to_add(monkeypatch, tmp_path):
    """A physics whose test file is missing fails here with the file to add and what to put in it; one whose test
    file has no tiny_example fails saying to add one."""
    import conftest
    with pytest.raises(pytest.fail.Exception, match="add tests/physics/test_made_up_physics.py, with a tiny_example"):
        conftest.tiny_example_of("made_up")
    (tmp_path / "tests" / "physics").mkdir(parents=True)
    (tmp_path / "tests" / "physics" / "test_exampleless_physics.py").write_text("NAME = 'exampleless'\n")
    monkeypatch.setattr(conftest, "ROOT", str(tmp_path))
    with pytest.raises(pytest.fail.Exception, match="test_exampleless_physics.py has no tiny_example: add one"):
        conftest.tiny_example_of("exampleless")
