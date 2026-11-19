"""
quantity_states.py
==================
The state a quantity keeps from one iteration of a run to the next (its between_iterations,
formulation/quantities.py). The run's model holds one dict for each entry of the formulation, empty at the
start of the run; a design analysed on its own (Analyse the solid part) has none. Nothing here imports the
rest of the package, so a physics' quantities can read their state too.
"""


def state_key(name: str, opt: dict[str, int | float | str] | None) -> tuple[str, str]:
    """The key of a quantity's state: its name and options. Two entries of the formulation with both the same
    (at most and at least one value) have the same value, so they keep one state."""
    return name, repr(sorted((opt or {}).items()))


def run_state(m: object, name: str, opt: dict[str, int | float | str] | None) -> dict | None:
    """The state the run keeps for the formulation's entry of quantity name with options opt, or None outside a
    run: a design analysed on its own has no run."""
    states = getattr(m, "quantity_states", None)
    if states is None:
        return None
    return states.setdefault(state_key(name, opt), {})
