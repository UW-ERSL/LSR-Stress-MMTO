"""
test_docstring_reads_and_sets.py
================================
A function that changes an object it is given says so in its docstring, in one line of code names:

    Reads it.zeta, state.beta. Sets it.z, state.iteration.

For each function in lsr_mmto/ with a parameter named state (the run state), it (an iteration) or m or
model (the model), the fields it assigns on them, item by item or with a method such as append, are the
ones its docstring names after "Sets"; the fields of an iteration it reads, and those of the run state
that change during the run (CHANGING_STATE), are named after "Reads" or "Sets". A name the docstring gives
that the code does not touch fails too. Only that line is read: the rest of the docstring is plain words.

    python tests/test_docstring_reads_and_sets.py        (or: python -m pytest tests)
"""
import ast
import glob
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

WATCHED = {"state", "it", "m", "model"}          # the parameter names of the shared objects
READS_LISTED = {"it"}                           # every field read of these is listed
CHANGING_STATE = {"iteration", "beta", "objective_scale", "penalties", "penalty_scales", "gamma",
                  "objective_history", "gamma_ramp_started", "use_z_smoothing", "use_distance_penalty",
                  "during_run", "last_zeta"}     # the run state's fields that change during the run
MUTATORS = {"append", "extend", "update", "pop", "insert", "clear", "setdefault", "remove", "add", "fill"}
NAME = re.compile(r"\b(state|it|m|model)\.([A-Za-z_]\w*)")


def touched(function, params):
    """({(param, field)} the function sets, {(param, field)} it reads that must be listed)."""
    sets, reads = set(), set()

    def field(node):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id in params:
            return node.value.id, node.attr
        return None

    for node in ast.walk(function):
        targets = node.targets if isinstance(node, ast.Assign) else \
            [node.target] if isinstance(node, (ast.AugAssign, ast.AnnAssign)) else []
        for target in targets:
            for t in (target.elts if isinstance(target, ast.Tuple) else [target]):
                f = field(t) or (field(t.value) if isinstance(t, ast.Subscript) else None)
                if f:
                    sets.add(f)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in MUTATORS:
            f = field(node.func.value)
            if f:
                sets.add(f)
        if isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load):
            f = field(node)
            if f and (f[0] in READS_LISTED or (f[0] == "state" and f[1] in CHANGING_STATE)):
                reads.add(f)
    return sets, reads


def problems(source, name):
    """'file:line function: what differs' for each watched function of the source whose docstring line
    does not match its code."""
    out = []
    for function in ast.walk(ast.parse(source)):
        if not isinstance(function, ast.FunctionDef):
            continue
        params = {a.arg for a in function.args.args + function.args.kwonlyargs} & WATCHED
        if not params:
            continue
        sets, reads = touched(function, params)
        doc = ast.get_docstring(function) or ""
        at_sets, at_reads = doc.find("Sets "), doc.find("Reads ")
        said_sets = set(NAME.findall(doc[at_sets:])) if at_sets >= 0 else set()
        said_reads = set(NAME.findall(doc[at_reads:at_sets if at_sets > at_reads else None])) \
            if at_reads >= 0 else set()
        differ = [("sets, not in its Sets", sets - said_sets),
                  ("in its Sets, never set", said_sets - sets),
                  ("reads, not in its Reads", reads - said_reads - said_sets - sets),   # one set is reported once
                  ("in its Reads, never read", said_reads - reads)]
        for what, names in differ:
            if names:
                out.append(f"{name}:{function.lineno} {function.name}: {what}: "
                           + ", ".join(sorted(".".join(n) for n in names)))
    return out


def test_every_docstring_names_what_its_function_reads_and_sets():
    wrong = [p for path in sorted(glob.glob(os.path.join(ROOT, "lsr_mmto", "**", "*.py"), recursive=True))
             for p in problems(open(path, encoding="utf-8").read(), os.path.relpath(path, ROOT))]
    assert not wrong, "\n".join(wrong)


def test_the_check_catches_a_field_left_out():
    old = ('def step(state, it):\n    """Reads it.a. Sets it.b."""\n    it.b = it.a + 1\n    it.c = 0\n'
           '    state.during_run["x"] = 1\n')
    assert problems(old, "old.py") == ["old.py:1 step: sets, not in its Sets: it.c, state.during_run"]
    right = old.replace("Sets it.b.", "Sets it.b, it.c, state.during_run.")
    assert problems(right, "right.py") == []


if __name__ == "__main__":
    test_every_docstring_names_what_its_function_reads_and_sets()
    test_the_check_catches_a_field_left_out()
    print("ok")
