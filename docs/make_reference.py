"""
make_reference.py
=================
Writes docs/reference.md from the code: the project file's sections, the
selection types, the boundary-condition kinds, the quantities, the settings, and the
Python calls a setup script uses.

    python docs/make_reference.py        (from the repository folder)

The page is generated, not edited by hand; tests/test_reference.py fails when the
page in the repository differs from what the code generates.

The Python section is read from the docstrings of the calls (python_entries lists them):
first what the call does, then one line per argument, "name: what it is and its format",
continued on lines indented further, and an "Example: ..." line with one line of code. An
indented "Default: ..." under an argument stands for its default where the signature has
none to give. A class's docstring may also describe what is set on it once it is made,
each such line with its own indented "Example: ..." (Project: database, settings, ...).
Every argument needs its line; a missing one stops the page being written.
"""
import inspect
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAGE = os.path.join(ROOT, "docs", "reference.md")

SECTIONS = [
    ("schema_version", "version of the project format"),
    ("name, comment", "the project's name (also the run's name) and a description"),
    ("physics", "\"structural\" for now"),
    ("mesh", "n_dof (degrees of freedom asked for) and single_layer (one element through the thickness)"),
    ("selections", "named regions of the mesh, see Selection types"),
    ("bcs", "boundary conditions, each on a named selection, see Boundary conditions"),
    ("keep_solid", "names of element selections kept solid"),
    ("keep_support_elements", "also keep solid the elements touching fixed dofs"),
    ("database", "a database name in data/databases, or a path to an .xlsx"),
    ("formulation", "objective and constraints, see Quantities"),
    ("settings", "common, advanced and vae, see Settings"),
]


PROJECT_CALLS = ("Project", "units", "comment", "physics", "add_selection", "mesh", "evaluate", "add_bc",
                 "keep_solid", "keep_support_elements", "database", "network_error", "vae_check", "objective",
                 "constraint", "settings",
                 "save", "to_script", "load")                  # Project's calls, in the order a script uses them
ARGUMENT = re.compile(r"^(\w+)((?:\[\w+\]|\(\w+\))*): (.*)$")       # name: text, or settings[section][name]: text


def _value(v):
    return "None" if v is None else repr(v) if isinstance(v, str) else str(v)


def generate():
    sys.path.insert(0, ROOT)
    from lsr_mmto.problem import selections as S
    from lsr_mmto.formulation.quantities import QUANTITIES
    from lsr_mmto.physics.available_physics import PHYSICS
    from lsr_mmto.problem.boundary_conditions import BC_KINDS
    from lsr_mmto.problem.settings import FROM_DATABASE, SETTINGS
    from lsr_mmto.materials.databases import ATTRIBUTES, UNIT_SYSTEMS

    out = ["# Reference", "",
           "Generated from the code by `python docs/make_reference.py`; do not edit by hand.", "",
           "A project is an LSR MMTO project file (.mmto): the layout of a PyTO-AD project file, with an "
           "`mmto` section, which holds the whole "
           "problem. Supports and loads picked on the part in the window become selections (`gui_support1`, "
           "`gui_load1`, ...) and boundary conditions of the mmto section as soon as they are applied; "
           "the rest of what the window saves in the file (its settings) is kept.", "",
           "## The mmto section", "", "| key | holds |", "|---|---|"]
    out += [f"| `{k}` | {v} |" for k, v in SECTIONS]

    out += ["", "## Selection types", "",
            "Lengths are numbers in model units, or text in numbers, pi and h, the largest element size: "
            "`\"<k>h\"` for k times it, or arithmetic such as `\"5.64 + 1h\"`, worked out on each mesh. "
            "`pad` widens every bound; `\"auto\"` is half an element for nodes and 0 for element centres "
            "(faces and planes always half an element).", "",
            "| type | fields | selects |", "|---|---|---|"]
    for name, cls in sorted(S._TYPES.items()):
        fields = "parts" if cls in (S.Union, S.Intersect, S.Difference) else ", ".join(cls.fields)
        doc = _docstring(cls.__doc__, cls.fields)[0].replace("|", "\\|")      # what it selects
        out.append(f"| `{name}` | {fields} | {doc} |")

    out += ["", "## Boundary conditions", "",
            "A project file keeps each as {\"kind\": kind, \"on\": a selection's name, ...its values}; a setup "
            "script adds one with p.add_bc(kind, on, **values).", "",
            "| kind | values | means |", "|---|---|---|"]
    out += [f"| `{k}` | " + ", ".join(f"`{v.name}` ({v.doc})" for v in kind.values) + f" | {kind.help} |"
            for k, kind in BC_KINDS.items()]

    out += ["", "## Material database attributes", "",
            "A database is an .xlsx file: attribute names in row 1, units in row 2, then one row per "
            "material with its name in column A. Every column must be one of these attributes (or one "
            "of its other names); the Materials panel of the app shows the errors and warnings of a file.", "",
            "| attribute | other names | meaning | allowed range | units |", "|---|---|---|---|---|"]
    out += [f"| `{a.name}` | {', '.join(a.aliases)} | {a.doc} | {a.range_text()} | "
            f"{'none (may be blank)' if a.measures == 'ratio' else ', '.join(a.units)} |"
            for a in ATTRIBUTES.values()]
    out += ["", "An optional sheet `Display` gives each material the colour and the label it is shown with: "
            "columns Material, Color (#rrggbb) and Label, one row per material, matched by name. An empty "
            "Color is the default colour, an empty Label the material's name. Two materials with the same "
            "colour or the same label are an error. The run record stores both."]

    kinds = ("length", "force", "stress", "density", "mass", "cost per mass")
    out += ["", "## Unit systems", "",
            "A project declares the units its numbers are in, as the PyTO-AD GUI does (`settings.unit_system` "
            "in the project file; `p.units` in code). Nothing is converted: forces, lengths, bounds and the "
            "database are used as written. A database column in other units is reported before a run, with "
            "what it does to the results. `none` (no unit system) checks nothing.", "",
            "| system | " + " | ".join(kinds) + " |", "|---|" + "---|" * len(kinds)]
    out += [f"| `{k}` | " + " | ".join(v[q] for q in kinds) + " |" for k, v in UNIT_SYSTEMS.items()]

    out += ["", "## Physics", "",
            "The `physics` of a project names the model its optimization uses.", "",
            "| physics | model | needs columns |", "|---|---|---|"]
    out += [f"| `{k}` | {v.DOC} | {', '.join(v.PROPERTIES)} |" for k, v in PHYSICS.items()]

    worded = "".join(f" `{q.name}` also takes {' or '.join(f'`{w}`' for w in q.words('material'))} in its "
                     f"place, e.g. `{q.name}({q.words('material')[-1]})`."
                     for q in QUANTITIES if q.words("material"))
    out += ["", "## Quantities", "",
            "An objective or a constraint is one quantity, e.g. `Mass()` or "
            "`MaterialVolumeFraction(2, tau=0.5)`. A material is its row in the database, counted from 0."
            + worded + " A constraint is value / bound - 1 <= 0.", "",
            "| name | shown as | objective | constraint | needs columns | value | meaning |",
            "|---|---|---|---|---|---|---|"]
    for q in QUANTITIES:
        out.append(f"| `{q.name}` | {q.shown} | {'yes' if q.objective else ''} | {'yes' if q.constraint else ''} | "
                   f"{', '.join(q.needs)} | {q.doc} | {q.help} |")

    out += ["", "## Settings", "",
            "Every setting a run takes, under its own name. A new project starts from these values "
            "(the settings of the paper's examples); a file missing some is completed from them, and "
            "the run says which. The VAE's own settings start from the database's known values. The "
            "mesh settings are the file's `mesh` section; the others are `settings.<section>`.", ""]
    for section in ("mesh", "common", "advanced", "vae"):
        out += [f"### {section}", "", "| setting | default | meaning |", "|---|---|---|"]
        for e in (e for e in SETTINGS if e.section == section):
            default = "from the database" if e.default is FROM_DATABASE else _value(e.default)
            if e.choices is not None:
                default += f" (one of {', '.join(e.choices.__members__)})"
            meaning = e.help + (f" Empty (None): {e.none_means}." if e.none_means else "")
            out.append(f"| `{e.key}` | {default} | {meaning} |")
        out.append("")
    out += python_section()
    return "\n".join(out)


# ---------------------------------------------------------------------- the Python section
def _docstring(doc, known):
    """(description, {name: {"written", "text", "default", "example"}}, example) read from a docstring as
    the module docstring says; known: the names whose lines are arguments."""
    description, arguments, example, current = [], {}, None, None
    for line in inspect.cleandoc(doc or "").splitlines():
        m = ARGUMENT.match(line)
        if line.startswith("Example: "):
            example, current = line[len("Example: "):].strip(), None
        elif m and m.group(1) in known:
            current = arguments[m.group(1)] = {"written": m.group(1) + m.group(2), "text": m.group(3).strip(),
                                               "default": None, "example": None}
        elif current is not None and line[:1].isspace():
            text = line.strip()
            for key in ("Default", "Example"):
                if text.startswith(key + ": "):
                    current[key.lower()] = text[len(key) + 2:].strip()
                    break
            else:
                current["text"] += " " + text
        elif not arguments and example is None:
            description.append(line)
        else:
            current = None
    return " ".join(" ".join(description).split()), arguments, example


def _cell(text):
    """Text for the page: | escaped for the tables, and < before a letter so it is not read as a tag."""
    return re.sub(r"<(?=[A-Za-z])", r"\\<", str(text).replace("|", "\\|"))


def python_entries():
    """[(heading, object, how)] of the Python section, in order; how is "call" (a function or a class),
    "method", "classmethod", "attribute" (a property or a value set on a Project) or "fields" (what a
    Run holds). Every selection type is written as the call of its name."""
    from lsr_mmto.optimize.solid_part_analysis import analyse_design
    from lsr_mmto.problem import selections as S
    from lsr_mmto.problem.project import Project, run
    from lsr_mmto.results import design_3d, load_results, run_plots, stl_export, tables
    out = [(name, getattr(S, name), "call") for name in S._TYPES]
    for name in PROJECT_CALLS:
        found = inspect.getattr_static(Project, name, None)
        how = ("call" if name == "Project" else "classmethod" if isinstance(found, classmethod) else
               "method" if inspect.isfunction(found) else "attribute")
        heading = name if how == "call" else f"Project.{name}" if how == "classmethod" else f"p.{name}"
        out.append((heading, Project if how == "call" else found, how))
    out += [("run", run, "call"), ("load", load_results.load, "call"), ("Run fields", load_results.Run, "fields"),
            ("summary", tables.summary, "call"), ("design", run_plots.design, "call"),
            ("mesh_for", design_3d.mesh_for, "call"), ("export", stl_export.export, "call"),
            ("analyse_design", analyse_design, "call")]
    return out


def _options(obj):
    """The keyword options a call passes on under its **name (run's run_options: those _run_arguments
    takes), as parameters; none for other calls."""
    from lsr_mmto.problem.project import _run_arguments, run
    if obj is not run:
        return []
    taken = inspect.signature(run).parameters
    return [q for n, q in inspect.signature(_run_arguments).parameters.items() if n not in taken and n != "project"]


def _entry(heading, obj, how):
    """(lines of the page, import line or None, example) of one entry."""
    import lsr_mmto
    from lsr_mmto.problem.project import Project
    new = Project("name", "part.stl")
    rows, imports = [], None
    if how == "attribute":
        doc = obj.__doc__ if isinstance(obj, property) else Project.__doc__
        name = heading[2:]
        description, arguments, example = _docstring(doc, {name})
        if name not in arguments:
            raise ValueError(f"{heading}: its docstring has no line for it")
        a = arguments[name]
        if isinstance(obj, property):
            a["example"], description = example, description
        else:
            description = ""
        call = f"p.{a['written']} = value"
        rows.append(("value", a["text"], a["default"] or f"`{getattr(new, name)!r}`"))
        example = a["example"]
    elif how == "fields":
        fields = set(getattr(obj, "__annotations__", {})) | set(dir(obj))
        description, arguments, example = _docstring(obj.__doc__, fields)
        call = None
        rows = [(a["written"], a["text"]) for a in arguments.values()]
    else:
        function = obj.__func__ if how == "classmethod" else obj
        parameters = list(inspect.signature(function).parameters.values())
        if how in ("method", "classmethod"):
            parameters = parameters[1:]                                             # self, cls
        options = _options(obj)
        description, arguments, example = _docstring(obj.__doc__, {q.name for q in parameters + options})
        signature = str(inspect.Signature([q.replace(annotation=inspect.Parameter.empty) for q in parameters]))
        call = (f"p.{heading[2:]}{signature}" if how == "method" else f"{heading}{signature}")
        for q in parameters + options:
            if q.name not in arguments:
                raise ValueError(f"{heading}: its docstring has no line for the argument {q.name}")
            a = arguments[q.name]
            default = a["default"] or ("required" if q.default is inspect.Parameter.empty else f"`{q.default!r}`")
            rows.append((a["written"], a["text"], default))
        if how == "call":
            module = "lsr_mmto" if getattr(lsr_mmto, obj.__name__, None) is obj else obj.__module__
            imports = f"from {module} import {obj.__name__}"
    if not example:
        raise ValueError(f"{heading}: its docstring has no Example: line")
    if how == "fields":
        imports = f"from {obj.__module__} import {obj.__name__}"
    lines = [f"### {heading}", ""]
    if call:
        lines += [f"`{call}`" + (f", after `{imports}`" if imports else ""), ""]
    if description:
        lines += [_cell(description), ""]
    if how == "fields":
        lines += ["| field | what it is |", "|---|---|"] + [f"| `{n}` | {_cell(t)} |" for n, t in rows] + [""]
    elif rows:
        lines += ["| argument | format | default |", "|---|---|---|"] + \
                 [f"| `{n}` | {_cell(t)} | {d if d.startswith('`') else _cell(d)} |" for n, t, d in rows] + [""]
    lines += ["```python", example, "```", ""]
    return lines, imports if how == "call" else None, example


def examples():
    """[(import line or None, example)] of the Python section's entries, in their order: run one after
    another from the repository folder, they make, save, run and read back the projects they describe."""
    return [_entry(*e)[1:] for e in python_entries()]


def python_section():
    out = ["", "## Python", "",
           "In the examples, `p` is your project: `p = Project(...)` for a new one, or "
           "`p = Project.load('my project.mmto')` for a saved one.", "",
           "Every call a setup script uses, with its arguments, read from the code's own docstrings; "
           "`Python_walkthrough.ipynb` uses them on one whole project. Model units "
           "are the project's units (`p.units`). A length is a number in model units, or text in numbers, pi "
           "and h, the largest element size, such as `\"1.5h\"` or `\"5.64 + 1h\"`. A selection is made "
           "with one of the selection types first and then added to a project under a name. The examples "
           "follow on from one another, run from the LSR MMTO folder: the selections, then a new project "
           "that the next examples build up, save and write as a script, then the shipped U-bracket, run "
           "and read back.", ""]
    for e in python_entries():
        out += _entry(*e)[0]
    return out


if __name__ == "__main__":
    os.makedirs(os.path.dirname(PAGE), exist_ok=True)
    with open(PAGE, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(generate())
    print(f"wrote {PAGE}")
