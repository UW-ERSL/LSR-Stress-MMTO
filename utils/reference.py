"""
reference.py
============
Writes docs/project_reference.md from the code: the project file's sections, the
selection types, the boundary-condition kinds, the quantities and the settings.

    python utils/reference.py        (from the repository folder)

The page is generated, not edited by hand; tests/test_reference.py fails when the
page in the repository differs from what the code generates.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAGE = os.path.join(ROOT, "docs", "project_reference.md")

SECTIONS = [
    ("schema_version", "version of the project format"),
    ("name, comment", "the project's name (also the run's name) and a description"),
    ("physics", "\"structural\" for now"),
    ("mesh", "n_dof (degrees of freedom asked for) and single_layer (one element through the thickness)"),
    ("selections", "named regions of the mesh, see Selection types"),
    ("bcs", "boundary conditions, each on a named selection, see Boundary conditions"),
    ("keep_solid", "names of element selections kept solid"),
    ("keep_support_elements", "also keep solid the elements touching fixed dofs"),
    ("database", "a database name in 1-PureStructural/MaterialDataPureStructural, or a path to an .xlsx"),
    ("formulation", "objective and constraints, see Quantities"),
    ("settings", "common, advanced and vae, see Settings"),
]


def _value(v):
    return "None" if v is None else repr(v) if isinstance(v, str) else str(v)


def generate():
    sys.path.insert(0, ROOT)
    from utils import project_paths
    project_paths.bootstrap(chdir=False)
    import selections as S                                                  # type: ignore
    from AutodiffPureStructural import QUANTITIES                           # type: ignore
    from project import BC_KINDS, DEFAULT_SETTINGS, SETTING_DOCS, VAE_PARAMS  # type: ignore

    out = ["# Project reference", "",
           "Generated from the code by `python utils/reference.py`; do not edit by hand.", "",
           "A project is a PyTO-AD project file (.pyto) with an `mmto` section, which holds the whole "
           "problem. Supports and loads picked in the PyTO-AD GUI become selections (`gui_support1`, "
           "`gui_load1`, ...) and boundary conditions of the mmto section when the project is loaded, "
           "and are removed from the GUI's own section; the rest of what the GUI saved is kept.", "",
           "## The mmto section", "", "| key | holds |", "|---|---|"]
    out += [f"| `{k}` | {v} |" for k, v in SECTIONS]

    out += ["", "## Selection types", "",
            "Lengths are numbers in model units, or `\"<k>h\"` for k times the largest element size. "
            "`pad` widens every bound; `\"auto\"` is half an element for nodes and 0 for element centres "
            "(faces and planes always half an element).", "",
            "| type | fields | selects |", "|---|---|---|"]
    for name, cls in sorted(S._TYPES.items()):
        fields = "parts" if cls in (S.Union, S.Intersect, S.Difference) else ", ".join(cls.fields)
        doc = " ".join((cls.__doc__ or "").split()).replace("|", "\\|")
        out.append(f"| `{name}` | {fields} | {doc} |")

    out += ["", "## Boundary conditions", "", "| kind | means |", "|---|---|"]
    out += [f"| `{k}` | {v.doc} |" for k, v in BC_KINDS.items()]

    out += ["", "## Quantities", "",
            "An objective or a constraint is one quantity, e.g. `Mass()` or "
            "`MaterialVolumeFraction(2, tau=0.5)`. A constraint is value / bound - 1 <= 0.", "",
            "| name | objective | constraint | needs columns | value |", "|---|---|---|---|---|"]
    for q in QUANTITIES:
        out.append(f"| `{q.name}` | {'yes' if q.objective else ''} | {'yes' if q.constraint else ''} | "
                   f"{', '.join(q.needs)} | {q.doc} |")

    out += ["", "## Settings", "",
            "Every setting of `run_topopt`, under its own name. A new project starts from these values "
            "(the settings of the paper's examples); a file missing some is completed from them, and "
            "the run says which. The VAE's own settings start from the database's known values.", ""]
    for section in ("common", "advanced", "vae"):
        out += [f"### {section}", "", "| setting | default | meaning |", "|---|---|---|"]
        for k, v in DEFAULT_SETTINGS[section].items():
            out.append(f"| `{k}` | {_value(v)} | {SETTING_DOCS[k]} |")
        if section == "vae":
            out += [f"| `{k}` | from the database | {SETTING_DOCS[k]} |" for k in VAE_PARAMS]
        out.append("")
    return "\n".join(out)


if __name__ == "__main__":
    os.makedirs(os.path.dirname(PAGE), exist_ok=True)
    with open(PAGE, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(generate())
    print(f"wrote {PAGE}")
