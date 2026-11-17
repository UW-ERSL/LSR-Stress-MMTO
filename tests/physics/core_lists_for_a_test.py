"""
core_lists_for_a_test.py
========================
The core's lists changed for one test, and put back after it: registered(module) adds a physics as its line
in available_physics.MODULES would, making every list with the functions start-up makes them with
(remake_lists), so whatever start-up would refuse is refused here too; added(entries, ...) adds entries to one
list. Not a test file: a physics' own test file imports what it needs from here, and copying a test file
never copies these.
"""
import contextlib
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from lsr_mmto.physics import available_physics  # noqa: E402


@contextlib.contextmanager
def registered(module):
    """The core's lists made again with module added, by the functions start-up makes them with from
    available_physics.MODULES (remake_lists), so whatever start-up would refuse (a missing entry, two things of
    one name) is refused here too; made again without it afterwards. A module already in MODULES is left as it
    is: nothing added, nothing taken out."""
    if module in available_physics.MODULES:
        yield
        return
    from lsr_mmto.materials import databases
    available_physics._checked(module)
    added_u = [k for k in module.UNIT_KINDS if k not in databases.UNITS]
    databases._add_unit_kinds(module)                     # first: an attribute's kind must be a unit kind
    available_physics.MODULES.append(module)
    try:
        remake_lists()
    except Exception:
        _taken_out(module, added_u)
        raise
    try:
        yield
    finally:
        _taken_out(module, added_u)


def _taken_out(module, added_u):
    """module taken out of available_physics.MODULES, the core's lists made again without it, and the unit kinds
    only it gave removed."""
    from lsr_mmto.materials import databases
    available_physics.MODULES.remove(module)
    remake_lists()
    for k in added_u:
        del databases.UNITS[k]
        for table in databases.UNIT_SYSTEMS.values():
            del table[k]


def remake_lists():
    """Every core list made again from available_physics.MODULES, in place (the core reads the same objects),
    by the functions start-up makes them with: the physics, the material properties and their aliases, the
    quantities, the penalties, the kinds of support and load and the settings. All are made before any is
    replaced, so a refused one leaves every list as it was."""
    from lsr_mmto.design import latent_penalties
    from lsr_mmto.formulation import quantities
    from lsr_mmto.materials import databases
    from lsr_mmto.problem import boundary_conditions, settings
    modules = available_physics.MODULES
    physics = {m.NAME: available_physics._checked(m) for m in modules}
    attributes = databases.attributes_of(modules)
    aliases = databases.aliases_of(attributes)
    made_quantities = quantities.quantities_of(modules)
    penalties = latent_penalties.penalties_of(modules)
    kinds = boundary_conditions.bc_kinds_of(modules)
    made_settings = settings.settings_of(modules)
    for table, made in ((available_physics.PHYSICS, physics), (databases.ATTRIBUTES, attributes),
                        (databases.SHOWN, {a.name: a.shown for a in attributes.values()}),
                        (databases._BY_ALIAS, aliases), (boundary_conditions.BC_KINDS, kinds)):
        table.clear()
        table.update(made)
    quantities.QUANTITIES[:] = made_quantities
    latent_penalties.PENALTIES[:] = penalties
    settings.SETTINGS[:] = made_settings
    settings._make_views()


@contextlib.contextmanager
def added(entries, *new, then=None):
    """The core's list entries with new added at its end, in place (the core reads the same list), and taken
    out again afterwards; then(), when given, is called after each change (settings._make_views)."""
    entries.extend(new)
    try:
        if then is not None:
            then()
        yield
    finally:
        for e in new:
            entries.remove(e)
        if then is not None:
            then()
