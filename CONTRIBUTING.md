# Contributing

How changes to LSR MMTO are made. `docs/how_the_code_works.md` describes the code and
`docs/how_to_add_features.md` where each kind of addition goes.

## Where a change goes

Every change is one of three kinds.

| Kind | What it is | Where |
|---|---|---|
| A module | a new physics, or more of one: its solver, supports and loads, quantities, settings, results | its folder in `lsr_mmto/physics/`, plus one line in `MODULES`; no core file changes |
| A widening of the core | a new option the core does not have yet, e.g. an augmented Lagrangian next to the p-norm, or a field solve that is nonlinear in temperature | the one layer it belongs to |
| A new repository | anything every layer would have to check for | not here |

A new physics starts as a copy of `lsr_mmto/physics/template/`.

A widening of the core follows four rules:

1. It is a new option, kept in one layer; what runs today runs the same with it.
2. It is made on a branch of its own and merged when it is tested.
3. It rests on a published method, cited in the code where it is used.
4. One at a time: the next starts when the last is merged.

## Before writing anything

- **Check PyTO-AD first.** If PyTO-AD has it, import it through `lsr_mmto/pyto_ad_imports.py`, or
  subclass it where it needs a different material in each element. PyTO-AD is pinned to one commit
  and never edited.
- **One way to do each thing.** A project runs only through `run(p)`. A second way of doing
  something that already has one is removed, not kept beside it.
- **Generic code.** Nothing is special-cased by name (`if name == "Mass"`): what differs between
  entries is declared by the entry, in its list.

## Names

A file's name says what it is for. The test: from the name alone, someone new to the code can say
in one line what is in the file. `structural_supports_and_loads.py` passes; `loop.py` and
`utils.py` do not.

## Checking a change

- `python -m pytest tests -q` passes.
- A change that moves code or renames things changes no number: a short run of every shipped project
  gives the same history, to the last digit, before and after.
- Text the GUI shows is written for someone who has never seen the code: plain words, the labels
  the form itself shows, and the details behind an (i).
- After a change to a list (settings, quantities, supports and loads, selection types, material
  attributes), run `python docs/make_project_file_reference.py` and
  `python -m gui.user_guide_window --write`; the reference page and the user guide's lists are made
  from the code.

## Commit messages

Plain sentences saying what changed and why, in the present tense, ending with "No number
changes." when that is so.
