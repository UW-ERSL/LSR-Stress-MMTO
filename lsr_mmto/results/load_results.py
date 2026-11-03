"""
load_results.py
===============
An optimization run, read back for plotting.

load(path)        a run saved to disk, as a Run.

Nothing here imports torch, pyvista or PyTO.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import numpy as np



class RecordError(RuntimeError):
    pass


@dataclass
class Run:
    """One optimization run. Per-element arrays are in solver element order."""
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
    """Load one record. The .npz extension may be omitted."""
    if not path.endswith(".npz"):
        path = path + ".npz"
    if not os.path.isfile(path):
        raise RecordError(f"no record at {path}")

    try:
        r = np.load(path, allow_pickle=False)
    except Exception as e:
        raise RecordError(f"{path} is not a readable .npz: {e}") from e

    if "schema_version" not in r.files or int(r["schema_version"]) != 2:
        raise RecordError(f"{os.path.basename(path)} was saved by an earlier version of LSR MMTO; run the "
                          f"project again to see it here")
    need = {"material_indices", "x_bin_packed", "n_elems", "x_cont", "z_elem_solid",
            "zRealPoints", "meta_json", "material_names", "material_colors", "material_labels"}
    missing = need - set(r.files)
    if missing:
        raise RecordError(f"{os.path.basename(path)} is missing {sorted(missing)}. "
                          f"Present: {sorted(r.files)}")

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

    cfg_path = path[:-4] + ".config.json"
    if not os.path.isfile(cfg_path):
        raise RecordError(f"no {os.path.basename(cfg_path)} beside {os.path.basename(path)}")
    with open(cfg_path) as fh:
        config = json.load(fh)

    return _build(path, json.loads(str(r["meta_json"])), config, n,
                  r["material_indices"], solid, r["x_cont"],
                  {k[2:]: r[k] for k in r.files if k.startswith("r_")},
                  z, r["zRealPoints"], [str(s) for s in r["material_names"]],
                  {k[2:]: r[k] for k in r.files if k.startswith("h_")},
                  [str(s) for s in r["material_colors"]],
                  [str(s) for s in r["material_labels"]])


def label(run: "Run") -> str:
    """Readable name for a run: its project and its folder."""
    return f"{run.example}  {run.descriptor}"
