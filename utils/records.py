"""
records.py
==========
An optimization run, read back for plotting.

from_payload(d)   a run still in memory, i.e. run_io.LAST_RUN
load(path)        a run saved to disk with run_io.WRITE_FILES = True

Both return a Run. Nothing here imports torch, pyvista or PyTO.
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
    von_mises: np.ndarray          # (n_elems,)      float32
    z_elem: np.ndarray             # (n_elems, n_z)  float32, NaN in voids
    z_real: np.ndarray             # (n_mats, n_z)   latent anchors
    material_names: list
    history: dict = field(repr=False)

    @property
    def example(self) -> str:
        return self.meta.get("example", "?")

    @property
    def descriptor(self) -> str:
        return self.meta.get("descriptor", "?")

    @property
    def params(self) -> dict:
        return self.meta.get("params", {})

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
    def phi_p(self) -> float:
        """p-norm failure factor, penalty-free."""
        return self.last("J_phys")

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
        """Mass per candidate material, length n_mats, zero where unselected."""
        rho = self.material_property("Density")
        w = self.x_cont * self.elem_volume * rho[self.material_indices]
        out = np.zeros(rho.size)
        np.add.at(out, self.material_indices, w)
        return out

    def z_solid(self) -> np.ndarray:
        """Latent field on solid elements only, no NaNs."""
        return self.z_elem[self.solid]


def _build(path, meta, config, n, mat, solid, x_cont, von_mises, z, z_real,
           names, history) -> Run:
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
    return Run(path=path, meta=meta, config=config, n_elems=n,
               material_indices=mat.astype(np.int64), solid=solid,
               x_cont=np.asarray(x_cont).ravel().astype(np.float32),
               von_mises=np.asarray(von_mises).ravel().astype(np.float32),
               z_elem=z, z_real=np.asarray(z_real).astype(np.float32),
               material_names=[str(s) for s in names], history=history)


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

    need = {"material_indices", "x_bin_packed", "n_elems", "x_cont",
            "von_mises", "z_elem_solid", "zRealPoints", "meta_json"}
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
    config = {}
    if os.path.isfile(cfg_path):
        with open(cfg_path) as fh:
            config = json.load(fh)

    names = [str(s) for s in r["material_names"]] if "material_names" in r.files else []

    return _build(path, json.loads(str(r["meta_json"])), config, n,
                  r["material_indices"], solid, r["x_cont"], r["von_mises"],
                  z, r["zRealPoints"], names,
                  {k[2:]: r[k] for k in r.files if k.startswith("h_")})


def from_payload(payload: dict, descriptor: str = "in-memory") -> Run:
    """Build a Run from a run still in memory, i.e. run_io.LAST_RUN.

    The payload holds z_elem for every element. Voids are masked to NaN here so
    an in-memory Run behaves the same as a loaded one.
    """
    if not isinstance(payload, dict) or "fields" not in payload:
        raise RecordError(
            "payload is not a run dict. Expected run_io.LAST_RUN, which is None "
            "until a run finishes.")

    f = payload["fields"]
    need = {"material_indices", "x_bin", "x_cont", "von_mises", "z_elem"}
    missing = need - set(f)
    if missing:
        raise RecordError(f"payload fields missing {sorted(missing)}. "
                          f"Present: {sorted(f)}")

    n = int(payload["num_elems"])
    solid = np.asarray(f["x_bin"]).ravel() > 0.5
    z = np.asarray(f["z_elem"], dtype=np.float32).copy()
    if z.ndim == 2 and z.shape[0] == n:
        z[~solid] = np.nan

    cfg = payload.get("config") or {}
    tp = cfg.get("topopt_params") or {}
    meta = {
        "example": str(payload.get("example_name", "?")),
        "descriptor": descriptor,
        "iterations": (cfg.get("run") or {}).get("iterations"),
        "constraints": tp.get("Constraints"),
        "vae_net": os.path.basename(
            str(tp.get("MaterialsExcelFile", ""))).replace(".xlsx", ".nt"),
        "params": {},
    }

    # history is cast to float32, as the record writer does, so an in-memory Run
    # and the same run saved and reloaded report identical digits
    hist = {}
    for k, v in (payload.get("history") or {}).items():
        a = np.asarray(v)
        hist[k] = a.astype(np.float32) if a.dtype.kind in "fiu" else a

    return _build("<memory>", meta, cfg, n, f["material_indices"], solid,
                  f["x_cont"], f["von_mises"], z, payload["zRealPoints"],
                  payload.get("material_names", []), hist)


def label(run: "Run") -> str:
    """Readable name for a run, e.g. 'U-bracket  P=16, r_rho=3.5'."""
    g, p = run.meta.get("group") or _group_of(run), run.params
    if g == "lbracket_bm1":
        return f"L-bracket  r_rho={p['r_rho']:g}, r_z={p['r_z']:g}"
    if g == "ubracket_sweep":
        return f"U-bracket  P={p['P']:g}, r_rho={p['r_rho']:g}"
    if g == "ubracket_mesh":
        kind = "mesh control" if p.get("fixed_radius") else "mesh"
        return f"U-bracket  {kind} {p['nx']} x {p['nx']}"
    if g == "ubracket_single":
        return f"U-bracket  single material {p['material']}"
    if g == "beam_pareto":
        return f"Beam  mass cap {p['mass_cap']:.1e}".replace("e+0", "e")
    if g == "bracket":
        return f"Bracket  mass cap {p['mass_cap']:.1e}".replace("e+0", "e")
    return f"{run.example}  {run.descriptor}"


def _group_of(run: "Run") -> str:
    return os.path.basename(os.path.dirname(run.path))
