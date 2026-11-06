"""
load_results.py
===============
An optimization run, read back for plotting.

load(path)        a run saved to disk, as a Run.
from_arrays(...)  the same from the record's arrays and manifest, written or not (an analysis
                  shown without being saved).

The paths a record holds (RECORD_PATHS) are written relative to the project folder, and the project
folder relative to the run's folder (relative_paths), so a project folder can be moved, renamed or
zipped to someone else with its runs; load makes them absolute again (absolute_paths), also for an
older record that holds them absolute.

Nothing here imports torch, pyvista or PyTO.
"""
from __future__ import annotations

import copy
import json
import ntpath
import os
import posixpath
from dataclasses import dataclass, field

import numpy as np

# the project folder in a record (relative to the run's folder), and the paths a record holds relative to it
PROJECT_FOLDER = ("arguments_resolved", "to_problem", "base_dir")
RECORD_PATHS = (("arguments_resolved", "to_problem", "stl_file_path"), ("arguments_resolved", "saveNet"),
                ("arguments_resolved", "save_dir"), ("topopt_params", "MaterialsExcelFile"))


def _get(config, keys):
    for key in keys:
        config = config.get(key) if isinstance(config, dict) else None
    return config if isinstance(config, str) and config else None


def _set(config, keys, value):
    for key in keys[:-1]:
        config = config[key]
    config[keys[-1]] = value


def _absolute(path):
    """True for an absolute path of either kind (a record written on Windows read elsewhere, and back)."""
    return ntpath.isabs(path) or posixpath.isabs(path)


def _below(path, folder):
    """path relative to folder, its parts joined by /, when it is inside folder; else None. Either kind of
    separator, and case, are ignored, so a record written on Windows is read anywhere."""
    p, f = path.replace("\\", "/"), folder.replace("\\", "/").rstrip("/")
    return p[len(f) + 1:] if p.lower().startswith(f.lower() + "/") else None


def relative_paths(config, run_folder):
    """A copy of the record's manifest with RECORD_PATHS inside the project folder relative to it, and the
    project folder relative to the run's folder (run_folder), joined by /. A path elsewhere stays as it is."""
    config = copy.deepcopy(config)
    base = _get(config, PROJECT_FOLDER)
    if base is None or not _absolute(base):
        return config
    for keys in RECORD_PATHS:
        path = _get(config, keys)
        if path is not None and _absolute(path) and _below(path, base) is not None:
            _set(config, keys, _below(path, base))
    try:
        _set(config, PROJECT_FOLDER, os.path.relpath(base, os.path.abspath(run_folder)).replace("\\", "/"))
    except ValueError:                                              # another drive on Windows: kept absolute
        pass
    return config


def absolute_paths(config, run_folder):
    """A copy of a record's manifest with its paths absolute again, for the run in run_folder: relative ones
    against the project folder, found from the run's folder. An older record holds them absolute: when one
    no longer exists, the same place inside the run's project folder now is taken, if it is there (the
    project folder was moved or renamed)."""
    config = copy.deepcopy(config)
    base = _get(config, PROJECT_FOLDER)
    if base is None:
        return config
    run_folder = os.path.abspath(run_folder)
    old = base if _absolute(base) else None
    folder = base if old else os.path.normpath(os.path.join(run_folder, base))
    if old and not os.path.isdir(old):                              # moved: the run sits where it sat in it
        where = _below(_get(config, ("arguments_resolved", "save_dir")) or "", old)
        folder = run_folder
        for _ in range(len(where.split("/")) if where else 2):     # an analysis records no save_dir:
            folder = os.path.dirname(folder)                        # <project>/results/<run>
    _set(config, PROJECT_FOLDER, folder)
    for keys in RECORD_PATHS:
        path = _get(config, keys)
        if path is None:
            continue
        if not _absolute(path):
            _set(config, keys, os.path.normpath(os.path.join(folder, path)))
        elif old and not os.path.exists(path) and _below(path, old) is not None:
            moved = os.path.normpath(os.path.join(folder, _below(path, old)))
            if os.path.exists(moved):
                _set(config, keys, moved)
    return config



class RecordError(RuntimeError):
    pass


@dataclass
class Run:
    """One optimization run. Per-element arrays are in solver element order.

    path: the run_data.npz file it was read from
    n_elems: how many elements the mesh has
    solid: one True or False per element: material or empty, once the design is made all or nothing
    x_cont: each element's density, 0 to 1, before that
    material_indices: each element's material, its row in the database, counted from 0
    material_labels: the materials' names, as tables and legends show them
    results: {name: one value per element}, the physics' results, e.g. results["von_mises"]
    history: {name: one value per iteration}, e.g. history["J_phys"], the objective
    iterations: how many iterations the optimizer made
    final_objective: the objective at the last iteration
    objective_label: the objective's name as the GUI shows it
    constraint(i): constraint i (counted from 0) at the last iteration, as value / bound - 1: 0 or less is met
    Example: print(r.iterations, r.final_objective)
    """
    path: str
    meta: dict
    config: dict

    n_elems: int
    material_indices: np.ndarray   # (n_elems,)      int
    solid: np.ndarray              # (n_elems,)      bool, binarized topology
    x_cont: np.ndarray             # (n_elems,)      float32
    results: dict                  # name -> (n_elems,) float32, the physics' results (von_mises)
    result_info: dict              # name -> {view, title, measures}
    z_elem: np.ndarray             # (n_elems, n_z)  float32, NaN in voids
    z_real: np.ndarray             # (n_mats, n_z)   latent anchors
    material_names: list
    material_colors: list          # (n_mats,)       #rrggbb, as drawn
    material_labels: list          # (n_mats,)       as shown in legends and tables
    history: dict = field(repr=False)

    @property
    def example(self) -> str:
        return self.meta.get("example", "?")

    @property
    def descriptor(self) -> str:
        return self.meta.get("descriptor", "?")

    @property
    def analysed_material(self):
        """The material of an analysis (the solid part in one material, analysed once, not optimized:
        optimize/solid_part_analysis.py), as its label; None for a run."""
        return ((self.config or {}).get("analysis") or {}).get("material")

    @property
    def n_latent(self) -> int:
        return self.z_elem.shape[1]

    def last(self, key: str, col: int = 0) -> float:
        """Final value of a history series, NaN if absent or empty.

        mass_ratio and max_failure_factor are all-NaN on the L-bracket runs,
        which are constrained by volume fraction and minimise the p-norm.
        """
        a = self.history.get(key)
        if a is None or np.size(a) == 0:
            return float("nan")
        a = np.asarray(a)
        return float(a.reshape(a.shape[0], -1)[-1, col]) if a.ndim > 1 else float(a[-1])

    @property
    def iterations(self) -> int:
        """Optimizer iterations. The record holds the starting design as well, so a run of 300
        iterations has 301 designs in its history (main and autodiff alike)."""
        n = self.meta.get("iterations")
        if n is None:
            n = np.asarray(self.history.get("J_phys", [])).size
        return max(int(n) - 1, 0)

    @property
    def final_objective(self) -> float:
        """The objective at the last iteration, without the penalties."""
        return self.last("J_phys")

    @property
    def objective_label(self) -> str:
        """The objective's name as the app shows it, e.g. 'Failure factor (p-norm)'."""
        from lsr_mmto.formulation.quantities import quantity_label
        obj = self.config.get("topopt_params", {}).get("Objective") or ["objective", None]
        return quantity_label(obj[0], obj[1] if len(obj) > 1 else None, self.material_labels)

    def constraint_label(self, i: int) -> str:
        """Constraint i's name as the app shows it, e.g. 'Material volume fraction of Ti-6Al-4V'."""
        from lsr_mmto.formulation.quantities import quantity_label
        spec = self.constraint_spec[i]
        return quantity_label(spec[0], spec[1] if len(spec) > 1 else None, self.material_labels)

    @property
    def max_ff(self) -> float:
        """Max failure factor as tracked by the solver, on the relaxed stress.

        Not max(von_mises / Y), which uses the raw stress and gives a different
        number.
        """
        return self.last("max_failure_factor")

    @property
    def grey(self) -> float:
        return self.last("grey")

    def constraint(self, i: int) -> float:
        """g_i at the final iteration, stored as value/budget - 1."""
        return self.last("constraints", col=i)

    def utilisation(self, i: int) -> float:
        """value/budget at the final iteration."""
        return self.constraint(i) + 1.0

    @property
    def constraint_spec(self) -> list:
        return self.config.get("topopt_params", {}).get("Constraints", [])

    def material_property(self, key: str) -> np.ndarray:
        """Per-candidate-material property, e.g. Youngs_Modulus, Density."""
        try:
            return np.asarray(
                self.config["materials"]["real_material_properties"][key], float)
        except KeyError as e:
            avail = list(self.config.get("materials", {})
                         .get("real_material_properties", {}))
            raise RecordError(
                f"{self.descriptor}: no material property {key!r}. "
                f"Available: {avail}") from e

    @property
    def elem_size(self) -> np.ndarray:
        try:
            return np.asarray(self.config["fea"]["mesh"]["elem_size"], float)
        except KeyError as e:
            raise RecordError(
                f"{self.descriptor}: config has no fea.mesh.elem_size") from e

    @property
    def elem_volume(self) -> float:
        return float(np.prod(self.elem_size))

    def materials_used(self) -> np.ndarray:
        return np.unique(self.material_indices[self.solid])

    def mass_by_material(self) -> np.ndarray:
        """Mass per candidate material, length n_mats, zero where unselected, from the column that
        measures density."""
        from lsr_mmto.materials.databases import density_column
        names = list(self.config.get("materials", {}).get("real_material_properties", {}))
        rho = self.material_property(density_column(names) or "Density")
        w = self.x_cont * self.elem_volume * rho[self.material_indices]
        out = np.zeros(rho.size)
        np.add.at(out, self.material_indices, w)
        return out

    def z_solid(self) -> np.ndarray:
        """Latent field on solid elements only, no NaNs."""
        return self.z_elem[self.solid]


def _build(path, meta, config, n, mat, solid, x_cont, results, z, z_real,
           names, history, colours, labels) -> Run:
    tag = meta.get("descriptor", "?")
    mat = np.asarray(mat).ravel()
    if mat.size != n:
        raise RecordError(f"{tag}: material_indices has {mat.size} entries, "
                          f"expected n_elems={n}")
    if solid.size != n:
        raise RecordError(f"{tag}: x_bin has {solid.size} entries, "
                          f"expected n_elems={n}")
    if z.ndim != 2 or z.shape[0] != n:
        raise RecordError(f"{tag}: z_elem has shape {z.shape}, "
                          f"expected ({n}, n_latent)")
    names = [str(s) for s in names]
    return Run(path=path, meta=meta, config=config, n_elems=n,
               material_indices=mat.astype(np.int64), solid=solid,
               x_cont=np.asarray(x_cont).ravel().astype(np.float32),
               results={k: np.asarray(v).ravel().astype(np.float32) for k, v in results.items()},
               result_info=dict(meta.get("results") or {}),
               z_elem=z, z_real=np.asarray(z_real).astype(np.float32),
               material_names=names, material_colors=[str(c) for c in colours],
               material_labels=[str(s) for s in labels], history=history)


def load(path: str) -> Run:
    """A saved run read back, as a Run.

    path: the run's folder (one of those in results in the project's folder), or the run_data.npz in it;
        the .npz may be left off
    Example: r = load(folder)
    """
    if os.path.isdir(path):
        path = os.path.join(path, "run_data.npz")
    if not path.endswith(".npz"):
        path = path + ".npz"
    if not os.path.isfile(path):
        raise RecordError(f"no record at {path}")

    try:
        r = np.load(path, allow_pickle=False)
    except Exception as e:
        raise RecordError(f"{path} is not a readable .npz: {e}") from e

    config = None
    if os.path.isfile(path[:-4] + ".config.json"):
        with open(path[:-4] + ".config.json") as fh:
            config = absolute_paths(json.load(fh), os.path.dirname(path))
    return from_arrays(path, r, config)


def from_arrays(path: str, r, config) -> Run:
    """A record from its arrays (r: run_data.npz's, or the same arrays not written, as
    save_results.record_arrays makes them) and its manifest (run_data.config.json's; None when the
    record has none)."""
    keys = list(r.keys())
    files = set(keys)
    if "schema_version" not in files or int(r["schema_version"]) != 2:
        raise RecordError(f"{os.path.basename(path)} was saved by an earlier version of LSR MMTO; run the "
                          f"project again to see it here")
    need = {"material_indices", "x_bin_packed", "n_elems", "x_cont", "z_elem_solid",
            "zRealPoints", "meta_json", "material_names", "material_colors", "material_labels"}
    missing = need - files
    if missing:
        raise RecordError(f"{os.path.basename(path)} is missing {sorted(missing)}. "
                          f"Present: {sorted(files)}")

    n = int(r["n_elems"])
    solid = np.unpackbits(r["x_bin_packed"])[:n].astype(bool)   # unpackbits pads to 8

    # z_elem is stored for solid rows only. Scatter back so it is element-indexed
    # like every other field; voids are NaN so a read of them is visible.
    z_solid = r["z_elem_solid"]
    n_solid = int(solid.sum())
    if z_solid.shape[0] != n_solid:
        raise RecordError(
            f"{os.path.basename(path)}: z_elem_solid has {z_solid.shape[0]} rows "
            f"but x_bin marks {n_solid} solid elements")
    z = np.full((n, z_solid.shape[1]), np.nan, dtype=np.float32)
    z[solid] = z_solid

    if config is None:
        raise RecordError(f"no {os.path.basename(path)[:-4]}.config.json beside {os.path.basename(path)}")
    return _build(path, json.loads(str(r["meta_json"])), config, n,
                  r["material_indices"], solid, r["x_cont"],
                  {k[2:]: r[k] for k in keys if k.startswith("r_")},
                  z, r["zRealPoints"], [str(s) for s in r["material_names"]],
                  {k[2:]: r[k] for k in keys if k.startswith("h_")},
                  [str(s) for s in r["material_colors"]],
                  [str(s) for s in r["material_labels"]])


def label(run: "Run") -> str:
    """Readable name for a run: its project and its folder."""
    return f"{run.example}  {run.descriptor}"
