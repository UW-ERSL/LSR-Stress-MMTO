"""
project_folder.py
=================
A project folder holds everything its project uses, so the folder can be zipped and sent, and
every result made from it can be shown and run again:

    Bracket study/
      Bracket study.mmto
      Bracket study_setup.py              the project as a Python script
      part/bracket.STL                    the part, copied in
      materials/steels.xlsx, steels.nt    the database and its trained network, copied in
      results/2026-11-01_16-04-41/        one folder per run: run_data.npz, run_data.config.json
                                          (with the project as it was run) and run_config.yaml

The window and a script keep a project the same way. File > New makes the folder; every save
(Project.save, from the window or from a script) copies into the project's folder what the
project uses: the part into part/, the database and its network into materials/. Every run,
from the window or from run(p), puts its results in results/<date and time>.

The window also writes the setup script, <name>_setup.py, at File > New and again at every save,
so it always matches the project: edit it in any editor and run it with File > Run script. Edits
made in it are replaced at the next save in the window. A save from a script leaves the setup
script alone, so a script never rewrites itself.

    create(folder, stl)          a new project in a new folder
    pack(project, folder)        everything the project uses copied into folder (every save)
    bring_in_database(project)   the project's database and network copied into its folder
    scale_part(project, factor)  the project's own part, in part/, scaled in place
    convert_database_units(project)  the project's own copy of its database in the project's units
    setup_script(project)        the path of the project's setup script
    write_setup_script(project)  the setup script written from the project (the window only)
    new_run_folder(project)      the folder for the next run's results
    runs(project)                the project's runs, newest first
    run_name(folder)             the name the user gave a run, or None
    rename_run(folder, new)      a run's folder renamed
"""
import datetime
import filecmp
import json
import math
import os
import re
import shutil

PART, MATERIALS, RESULTS = "part", "materials", "results"
BAD_CHARACTERS = '\\/:*?"<>|'
RUN_FILE = "run_data.npz"


def name_problem(name):
    """What is wrong with a name for a project folder, or None."""
    name = name.strip()
    if not name:
        return "a name is needed"
    if any(c in name for c in BAD_CHARACTERS) or name in (".", ".."):
        return f"a name cannot contain any of {' '.join(BAD_CHARACTERS)}"
    return None


def folder_problem(folder):
    """What stops a new project folder being made at folder, or None."""
    problem = name_problem(os.path.basename(os.path.normpath(folder)))
    if problem:
        return problem
    parent = os.path.dirname(os.path.abspath(folder))
    if not os.path.isdir(parent):
        return f"{parent} is not a folder"
    if os.path.exists(folder) and (not os.path.isdir(folder) or os.listdir(folder)):
        return f"{os.path.basename(os.path.normpath(folder))} already exists in {parent}; choose another name"
    return None


def inside(path, folder):
    """True when path is folder or anything in it."""
    path, folder = os.path.normcase(os.path.abspath(path)), os.path.normcase(os.path.abspath(folder))
    return path == folder or path.startswith(folder.rstrip(os.sep) + os.sep)


def copy_in(path, folder, sub):
    """path copied into folder/sub, unless it is in folder already. Returns its path relative to
    folder, with / between the parts. A different file of the same name there is kept, and the
    copy is named name_2, name_3, ...; the same file is not copied twice."""
    if inside(path, folder):
        return os.path.relpath(path, folder).replace("\\", "/")
    target = os.path.join(folder, sub)
    os.makedirs(target, exist_ok=True)
    stem, ext = os.path.splitext(os.path.basename(path))
    dest, k = os.path.join(target, stem + ext), 2
    while os.path.exists(dest) and not filecmp.cmp(path, dest, shallow=False):
        dest, k = os.path.join(target, f"{stem}_{k}{ext}"), k + 1
    if not os.path.exists(dest):
        shutil.copy2(path, dest)
    return os.path.relpath(dest, folder).replace("\\", "/")


def has_own_folder(project):
    """True when the project lives in a project folder: its STL is in the folder's part/."""
    return inside(project.stl_path(), os.path.join(project.base_dir, PART))


def _database_files(project):
    """(xlsx, nt) as absolute paths: the database and the network next to it (None, None with no
    database). nt is None when settings.vae.saveNet names a network elsewhere: that one stays."""
    if project.database is None:
        return None, None
    xlsx, _ = project.database_paths()
    nt = None if project.settings.get("vae", {}).get("saveNet") else os.path.splitext(xlsx)[0] + ".nt"
    return os.path.abspath(xlsx), None if nt is None else os.path.abspath(nt)


def _copy_database(project, xlsx, nt, folder):
    """The database file and its network into folder/materials; project.database names the copy.
    A network already beside a copy is kept: it may have been retrained there."""
    if not os.path.isfile(xlsx):                                # the same database moved: its settings stay
        project._database = xlsx                                # its error stays readable: the full path
        return None
    rel = copy_in(xlsx, folder, MATERIALS)
    project._database = rel
    if nt is not None and os.path.isfile(nt):
        dest = os.path.splitext(os.path.join(folder, rel))[0] + ".nt"
        if not os.path.exists(dest):
            shutil.copy2(nt, dest)
    return rel


def bring_in_database(project):
    """The project's database (a shipped one too) and its trained network copied into the
    project's materials/; project.database then names the copy. Returns the copy's path relative
    to the project folder, or None when there was nothing to copy."""
    xlsx, nt = _database_files(project)
    if xlsx is None or inside(xlsx, project.base_dir):
        return None
    project.complete_settings()             # a shipped database's network settings, while it is named
    return _copy_database(project, xlsx, nt, project.base_dir)


def pack(project, folder):
    """Everything the project uses copied into folder, which becomes the project's folder: the
    STL, and any file a region names, into part/, the database and its network into materials/.
    Results stay where they are."""
    folder = os.path.abspath(folder)
    os.makedirs(folder, exist_ok=True)
    project.complete_settings()             # a shipped database's network settings, while it is named
    stl = os.path.abspath(project.stl_path())
    xlsx, nt = _database_files(project)
    old_base = project.base_dir
    project.base_dir = folder
    project.stl = copy_in(stl, folder, PART) if os.path.isfile(stl) else stl
    for selection in project.selections.values():
        _bring_in_files(selection, old_base, folder)
    if xlsx is not None:
        _copy_database(project, xlsx, nt, folder)
    return project


def _bring_in_files(selection, old_base, folder):
    """The files a region names (an inside_stl surface, a function selection's .py) copied into part/
    of folder, and the region pointed at the copies; inside combinations too."""
    from lsr_mmto.problem import selections as S
    for part in getattr(selection, "parts", ()):
        _bring_in_files(part, old_base, folder)
    if isinstance(selection, S.inside_stl):
        path = S._resolve(selection.path, old_base)
        if os.path.isfile(path):
            selection.path = copy_in(os.path.abspath(path), folder, PART)
    elif isinstance(selection, S.function):
        file, name = selection.ref.rsplit(":", 1)
        path = S._resolve(file, old_base)
        if os.path.isfile(path):
            selection.ref = f"{copy_in(os.path.abspath(path), folder, PART)}:{name}"


def create(folder, stl, physics=None):
    """A new project in a new folder, named after the folder, with the STL copied into it, the physics
    given (the first of the list when None) and every setting at its default; saved as
    folder/<name>.mmto. Returns (project, its path)."""
    from lsr_mmto.problem.project import Project
    problem = folder_problem(folder)
    if problem:
        raise ValueError(problem)
    if not os.path.isfile(stl):
        raise ValueError(f"{stl} is not a file")
    folder = os.path.abspath(folder)
    name = os.path.basename(folder)
    os.makedirs(folder, exist_ok=True)
    p = Project(name, stl=copy_in(os.path.abspath(stl), folder, PART), base_dir=folder)
    if physics is not None:
        p.physics = physics
    p.complete_settings()
    path = os.path.join(folder, f"{name}.mmto")
    p.save(path)
    return p, path


def scale_part(project, factor):
    """Every coordinate of the project's part times factor, written over the same file in part/ as a
    binary STL: part/ holds no more files than before, and scaling again multiplies again. Only the
    part of a project in a folder of its own is written; an STL anywhere else (an example's, the one
    the part came from) never is. Nothing else changes: the units, the loads, the limits and the
    database stay as they are."""
    from stl import mesh as stl_mesh
    from stl import Mode
    if not has_own_folder(project):
        raise ValueError("only the part of a project in a folder of its own is scaled; save the project first")
    if isinstance(factor, bool) or not isinstance(factor, (int, float)) or not math.isfinite(factor) or factor <= 0:
        raise ValueError(f"the part is scaled by a number above 0, not {factor!r}")
    path = os.path.abspath(project.stl_path())
    part = stl_mesh.Mesh.from_file(path)
    part.vectors *= factor                                  # the triangles' normals keep their direction
    written = path + ".scaling"                             # written in full, then put in the part's place
    try:
        part.save(written, mode=Mode.BINARY, update_normals=False)
        os.replace(written, path)
    finally:
        if os.path.exists(written):
            os.remove(written)
    return path


def convert_database_units(project, limits=True):
    """The project's database in the project's units (Project.unit_conversion): each column in other units
    multiplied by its factor, written over the project's own copy in its folder. A database from elsewhere
    (a shipped one too) is first copied in with its network (bring_in_database), and only the copy is
    written. The trained network is kept as it is: it learns each column scaled to 0 to 1, which a change
    of unit leaves as it was. With limits, the limits that meant something for the old numbers are
    converted too. Returns (columns, limits) converted, as Project.unit_conversion gives them."""
    from lsr_mmto.materials.databases import database_table, material_display, write_database
    if not has_own_folder(project):
        raise ValueError("only a project in a folder of its own has a database of its own; save the project first")
    columns, bounds = project.unit_conversion()
    if not columns:
        return [], []
    bring_in_database(project)
    xlsx, _ = project.database_paths()
    if not inside(xlsx, project.base_dir):
        raise ValueError(f"{xlsx} is not in the project's folder")
    names, attributes, units, values = database_table(xlsx)
    colours, labels = material_display(xlsx)
    to = {name: (unit, factor) for name, _, unit, factor in columns}
    for j, name in enumerate(attributes):
        if name in to:
            units[j] = to[name][0]
            values[:, j] = values[:, j] * to[name][1]
    write_database(xlsx, attributes, units, names, [None if label == name else label
                                                     for label, name in zip(labels, names)], colours, values)
    if not limits:
        return columns, []
    for i, _, _, bound in bounds:
        project.formulation["constraints"][i]["bound"] = bound
    return columns, bounds


def setup_script(project):
    """<name>_setup.py beside the project file: the project as a Python script."""
    return os.path.join(project.base_dir, f"{project.name}_setup.py")


def write_setup_script(project):
    """The setup script written from the project as it is (Project.to_script). The window calls this
    at File > New and at every save; a script's own save does not. Returns its path."""
    return project.to_script(setup_script(project))


# ---------------------------------------------------------------------- results
def results_root(project):
    """Where the project's runs go: results/ in its folder; for a project not in a folder of its
    own, results/<project name> beside the project file, as before."""
    if has_own_folder(project):
        return os.path.join(project.base_dir, RESULTS)
    return os.path.join(project.base_dir, RESULTS, project.name)


def new_run_folder(project):
    """results/<date and time> for the next run (not made yet)."""
    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    root = results_root(project)
    path, k = os.path.join(root, stamp), 2
    while os.path.exists(path):
        path, k = os.path.join(root, f"{stamp}_{k}"), k + 1
    return path


STAMP = re.compile(r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}(_\d+)?")   # a name new_run_folder gives


def run_name(folder):
    """The name the user gave a run (or an analysis) by renaming its folder, or None while its folder still
    has a name new_run_folder gives."""
    name = os.path.basename(os.path.normpath(folder))
    return None if STAMP.fullmatch(name) else name


def rename_run(folder, new):
    """Renames a run's folder, beside the others; returns its new path. ValueError, with the reason in plain
    words, when the name cannot be used; OSError when the system refuses (a file of the run is open)."""
    new = new.strip()
    problem = name_problem(new)
    if problem:
        raise ValueError(problem)
    target = os.path.join(os.path.dirname(os.path.normpath(folder)), new)
    if os.path.normcase(os.path.abspath(target)) == os.path.normcase(os.path.abspath(folder)):
        return folder
    if os.path.exists(target) and os.path.normcase(new) != os.path.normcase(os.path.basename(folder)):
        raise ValueError(f"{new} is already used in this project's results; choose another name")
    os.rename(folder, target)
    return target


def is_run(folder):
    return os.path.isfile(os.path.join(folder, RUN_FILE))


def runs(project):
    """The project's run folders, newest first."""
    root = results_root(project)
    if not os.path.isdir(root):
        return []
    found = [os.path.join(root, d) for d in os.listdir(root) if is_run(os.path.join(root, d))]
    return sorted(found, key=lambda d: (os.path.getmtime(os.path.join(d, RUN_FILE)), d), reverse=True)


def run_facts(folder):
    """(when, iterations, stopped) of a run, from its record's manifest; what is missing is None.
    iterations counts as the Run strip does: the starting design is iteration 0, so a run of 40
    iterations (41 designs in its record) gives 40."""
    when, iterations, stopped = None, None, False
    try:
        with open(os.path.join(folder, "run_data.config.json")) as fh:
            config = json.load(fh)
        when = (config.get("run") or {}).get("timestamp")
        iterations = (config.get("run") or {}).get("iterations")
        iterations = None if iterations is None else max(int(iterations) - 1, 0)
        stopped = "stopped_at_iter" in (config.get("during_run") or {})
    except (OSError, ValueError):
        pass
    if when is None:
        when = datetime.datetime.fromtimestamp(os.path.getmtime(os.path.join(folder, RUN_FILE))).isoformat(
            timespec="seconds")
    return when.replace("T", " "), iterations, stopped


def analysed_material(folder):
    """The material of an analysis (Analyse: the solid part in one material, not optimized), as its label, from
    its record's manifest; None for a run, or when the manifest cannot be read."""
    try:
        with open(os.path.join(folder, "run_data.config.json")) as fh:
            return ((json.load(fh) or {}).get("analysis") or {}).get("material")
    except (OSError, ValueError):
        return None


def delete_run(folder):
    """A run's folder and everything in it removed; only a folder that holds a run."""
    if not is_run(folder):
        raise ValueError(f"{folder} holds no run")
    shutil.rmtree(folder)
