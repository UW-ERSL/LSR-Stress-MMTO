"""
settings_for_a_test.py
======================
A run's settings for a test that builds its own model: default_run_settings(**changes) gives every setting at
its default, as a new project has it, with the test's changes on top. A test that wrote out only the settings it
needs would break whenever a setting is added that the code it calls reads; one made here cannot. Not a test
file: a test file imports it from here.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from lsr_mmto.problem import settings  # noqa: E402


def default_run_settings(**changes: object) -> "settings.RunSettings":
    """The RunSettings of a new project (each setting's default, a setting with fixed choices as its Enum member,
    as run_settings gives it), with changes ({key: value}) on top. A key that is no setting is an error."""
    unknown = sorted(set(changes) - set(settings.RUN_KEYS))
    if unknown:
        raise ValueError(f"no such settings: {', '.join(unknown)}")
    values = {}
    for key in settings.RUN_KEYS:
        s = settings.BY_KEY[key]
        values[key] = s.choices[s.default] if s.choices is not None and isinstance(s.default, str) else s.default
    values.update(changes)
    return settings.RunSettings(values)
