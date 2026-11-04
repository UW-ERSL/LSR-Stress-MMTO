"""
save_results.py
===============
Persistence layer for MMTO runs.

Writes into save_dir:
    run_data.npz         slim record: final fields + history
    run_data.config.json the manifest as JSON, beside the record
    run_config.yaml      the same manifest, sectioned
    run_data.pkl         only when SAVE_PICKLE is True

Nothing is written unless a save_dir is passed. Reader: results/load_results.py.
"""
import os
import sys
import json
import pickle
import datetime
import platform
import importlib
import subprocess
import numpy as np


# Full-precision pickle alongside the record: zeta, void latents, float64,
# VAE weights. Off by default.
SAVE_PICKLE = False

# For the tests only: False makes save_run_data (and run(p)) write nothing.
WRITE_FILES = True


# --------------------------------------------------------------------------- #
# sanitisation: turn enums / numpy / torch / dataclasses / params into plain   #
# python so the config is both YAML-readable and pickle-safe.                  #
# --------------------------------------------------------------------------- #
def _to_plain(obj, _depth=0):
    if _depth > 12:
        return str(obj)
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    try:
        import enum
        if isinstance(obj, enum.Enum):
            return obj.name
    except Exception:
        pass
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        if obj.size <= 64:
            return obj.astype(float).tolist() if obj.dtype.kind in "fiu" else obj.tolist()
        return {"_ndarray_shape": list(obj.shape), "_dtype": str(obj.dtype),
                "_preview": obj.flatten()[:8].tolist()}
    try:
        import torch
        if isinstance(obj, torch.Tensor):
            return _to_plain(obj.detach().cpu().numpy(), _depth + 1)
    except Exception:
        pass
    if hasattr(obj, "to_record"):                     # a Project: stored whole, as saved
        return _to_plain(obj.to_record(), _depth + 1)
    try:
        import dataclasses
        if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
            return {k: _to_plain(v, _depth + 1)
                    for k, v in dataclasses.asdict(obj).items()}
    except Exception:
        pass
    if isinstance(obj, dict):
        return {str(k): _to_plain(v, _depth + 1) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_to_plain(v, _depth + 1) for v in obj]
    if hasattr(obj, "__dict__"):
        return {k: _to_plain(v, _depth + 1) for k, v in vars(obj).items()
                if not k.startswith("_")}
    return str(obj)


def _params_dict(p):
    """Every public field of a params object (dataclass or plain class)."""
    try:
        import dataclasses
        if dataclasses.is_dataclass(p) and not isinstance(p, type):
            return {k: _to_plain(v) for k, v in dataclasses.asdict(p).items()}
    except Exception:
        pass
    out = {}
    for k in dir(p):
        if k.startswith("_"):
            continue
        try:
            v = getattr(p, k)
        except Exception:
            continue
        if callable(v):
            continue
        out[k] = _to_plain(v)
    return out


def _read_pnorm_exponent():
    """The p-norm exponent is a module global on topopt_material_model."""
    for getter in (
        lambda: importlib.import_module("topopt_material_model"),
        lambda: __import__("lsr_mmto.pyto_ad_imports", fromlist=["topopt_material_model"]).topopt_material_model,
    ):
        try:
            m = getter()
            if hasattr(m, "_PNORM_EXPONENT"):
                return float(m._PNORM_EXPONENT)
        except Exception:
            pass
    return None


def _loading_bc_summary(mesh, bc, ndof_per_node=3):
    out = {}
    try:
        force = np.asarray(bc.force).ravel()
        nz = np.where(force != 0.0)[0]
        load_nodes = np.unique(nz // ndof_per_node)
        net = [float(force[ndof_per_node * load_nodes + d].sum())
               for d in range(ndof_per_node)]
        out["load_node_count"] = int(load_nodes.size)
        out["net_force_xyz"] = net
        out["total_force_magnitude"] = float(np.linalg.norm(net))
        if load_nodes.size:                         # no loads on this field: no centroid to give
            try:
                c = np.asarray(mesh.node_xyz)[load_nodes].mean(axis=0)
                out["load_centroid_xyz"] = [float(v) for v in c]
            except Exception:
                pass
    except Exception as e:
        out["load_summary_error"] = str(e)
    try:
        fd = np.asarray(bc.fixed_dofs).ravel()
        out["fixed_dof_count"] = int(fd.size)
        out["fixed_node_count"] = int(np.unique(fd // ndof_per_node).size)
    except Exception as e:
        out["bc_summary_error"] = str(e)
    return out


def _mesh_summary(mesh):
    out = {}
    for attr in ("num_elems", "num_nodes"):
        try:
            out[attr] = int(getattr(mesh, attr))
        except Exception:
            pass
    try:
        out["elem_size"] = [float(v) for v in np.asarray(mesh.elem_size).ravel()]
    except Exception:
        pass
    try:
        xyz = np.asarray(mesh.node_xyz)
        out["domain_min_xyz"] = [float(v) for v in xyz.min(axis=0)]
        out["domain_max_xyz"] = [float(v) for v in xyz.max(axis=0)]
    except Exception:
        pass
    return out


def _material_summary(mat_encoder, z_real_points, material_colors, material_labels=None):
    out = {}
    try:
        out["material_names"] = list(mat_encoder.materialNames)
        out["material_count"] = len(out["material_names"])
    except Exception:
        pass
    try:
        out["zRealPoints"] = np.asarray(z_real_points).astype(float).tolist()
    except Exception:
        pass
    try:
        out["material_colors"] = list(material_colors)
        if material_labels is not None:
            out["material_labels"] = list(material_labels)
    except Exception:
        pass
    try:
        import torch
        zr = torch.as_tensor(np.asarray(z_real_points), dtype=torch.get_default_dtype())
        props = mat_encoder.getMaterialProperties(mat_encoder.vaeNet.decoder(zr))
        out["real_material_properties"] = {
            k: np.asarray(v.detach().cpu()).astype(float).ravel().tolist()
            for k, v in props.items()
        }
    except Exception as e:
        out["real_material_properties_error"] = str(e)
    return out


# --------------------------------------------------------------------------- #
# provenance: the code and the packages a run was made with                   #
# --------------------------------------------------------------------------- #
_PACKAGES = ("numpy", "scipy", "torch", "numba", "pyvista", "pandas", "openpyxl", "mmapy", "pypardiso")


NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)   # Windows: no console window flashing up for git,
                                                         # when the app runs without a console of its own


def _git(folder, *args):
    try:
        out = subprocess.run(["git", "-C", folder, *args], capture_output=True, text=True, timeout=10,
                             creationflags=NO_WINDOW)
        return out.stdout if out.returncode == 0 else None
    except Exception:
        return None


def git_state(folder):
    """Commit, branch and the tracked files changed since that commit; "unknown" when the
    folder is not a git checkout (a zip download) or git is not installed."""
    commit = _git(folder, "rev-parse", "HEAD")
    if commit is None:
        return {"commit": "unknown", "branch": "unknown", "changed_files": "unknown"}
    status = _git(folder, "status", "--porcelain", "--untracked-files=no") or ""
    return {"commit": commit.strip(),
            "branch": (_git(folder, "rev-parse", "--abbrev-ref", "HEAD") or "unknown").strip(),
            "changed_files": [line[3:] for line in status.splitlines() if line.strip()]}


def provenance():
    """The code this run is made with: this repository and PyTO-AD as git commits with
    their uncommitted changes, the Python version, the platform and the package versions.
    Never raises."""
    out = {}
    try:
        out["repository"] = git_state(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        try:
            from lsr_mmto.pyto_ad_imports import PYTO_AD_DIR
            out["pyto_ad"] = git_state(PYTO_AD_DIR)
        except Exception:
            out["pyto_ad"] = {"commit": "unknown", "branch": "unknown", "changed_files": "unknown"}
        out["python"] = sys.version.split()[0]
        out["platform"] = platform.platform()
        from importlib import metadata
        packages = {}
        for name in _PACKAGES:
            try:
                packages[name] = metadata.version(name)
            except Exception:
                packages[name] = "not installed"
        out["packages"] = packages
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"
    return out


# --------------------------------------------------------------------------- #
# public API                                                                  #
# --------------------------------------------------------------------------- #
def build_run_config(example_name, run_args, to_params, vae_params, beta_schedule,
                     mesh, bcs, fields, mat_encoder, z_real_points, material_colors,
                     mma_settings=None, runtime_sec=None, iterations=None,
                     comment="", pnorm_exponent=None, material_labels=None):
    """Assemble the complete sectioned parameter manifest. Never raises."""
    if pnorm_exponent is None:
        pnorm_exponent = _read_pnorm_exponent()
    ra = run_args if isinstance(run_args, dict) else {}
    cfg = {}
    cfg["run"] = {
        "example": str(example_name),
        "comment": comment or getattr(to_params, "Comment", ""),
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "runtime_seconds": None if runtime_sec is None else float(runtime_sec),
        "iterations": None if iterations is None else int(iterations),
    }
    cfg["arguments_resolved"] = _to_plain(ra)
    cfg["penalization_distance"] = {
        "use_penalization": ra.get("use_penalization"),
        "gamma_init": ra.get("gamma_init"),
        "gamma_max": ra.get("gamma_max"),
        "gamma_factor": ra.get("gamma_factor"),
        "penalty_start_iter": ra.get("penalty_start_iter"),
        "distance_penalty_disable_after_iter": ra.get("distance_penalty_disable_after_iter"),
    }
    cfg["penalization_zsmoothing"] = {
        "use_z_smoothing": ra.get("use_z_smoothing"),
        "z_smoothing_weight": ra.get("z_smoothing_weight"),
        "z_smoothing_disable_after_iter": ra.get("z_smoothing_disable_after_iter"),
    }
    cfg["heaviside_beta"] = {
        "use_heaviside_projection": ra.get("use_heaviside_projection"),
        "eta_proj": ra.get("eta_proj"),
        "beta_schedule_scheme": _to_plain(ra.get("beta_schedule_scheme")),
        "beta_schedule_values": _to_plain(beta_schedule),
    }
    topopt = _params_dict(to_params)
    topopt["pnorm_exponent"] = pnorm_exponent
    cfg["topopt_params"] = topopt
    cfg["vae_params"] = _params_dict(vae_params)
    cfg["materials"] = _material_summary(mat_encoder, z_real_points, material_colors, material_labels)
    cfg["fea"] = {
        "mesh": _mesh_summary(mesh),
        "loading_and_bc": {field: _loading_bc_summary(mesh, bc, fields[field]) for field, bc in bcs.items()},
        "use_constant_poissons_ratio": ra.get("use_constant_poissons_ratio"),
        "constant_poissons_ratio": ra.get("constant_poissons_ratio"),
    }
    cfg["optimizer_mma"] = _to_plain(mma_settings or {})
    return cfg


def write_run_config_yaml(config, path):
    """Sectioned, human-readable manifest. Prefers PyYAML, falls back cleanly."""
    sections = list(config.items())
    try:
        import yaml
        with open(path, "w", encoding="utf-8") as f:
            f.write("# MMTO run configuration  (auto-generated; do not edit)\n")
            for name, body in sections:
                f.write("\n# " + "=" * 64 + f"\n# {name.upper()}\n# " + "=" * 64 + "\n")
                yaml.safe_dump({name: body}, f, default_flow_style=False, sort_keys=False)
        return
    except Exception:
        pass

    def _emit(f, val, indent):
        pad = "  " * indent
        if isinstance(val, dict):
            for k, v in val.items():
                if isinstance(v, (dict, list)):
                    f.write(f"{pad}{k}:\n"); _emit(f, v, indent + 1)
                else:
                    f.write(f"{pad}{k}: {v}\n")
        elif isinstance(val, list):
            for v in val:
                if isinstance(v, (dict, list)):
                    f.write(f"{pad}-\n"); _emit(f, v, indent + 1)
                else:
                    f.write(f"{pad}- {v}\n")
        else:
            f.write(f"{pad}{val}\n")

    with open(path, "w", encoding="utf-8") as f:
        f.write("# MMTO run configuration  (auto-generated; do not edit)\n")
        for name, body in sections:
            f.write("\n# " + "=" * 64 + f"\n# {name.upper()}\n# " + "=" * 64 + f"\n{name}:\n")
            _emit(f, body, 1)


SCHEMA_VERSION = 2          # 2: the physics' per-element results by name (results), not a von_mises field


def files_written():
    """The names of the files save_run_data writes into a run's folder."""
    return ["run_data.npz", "run_data.config.json", "run_config.yaml"] + (["run_data.pkl"] if SAVE_PICKLE else [])


def save_run_data(save_dir, config, fields, vae_state_dict, history,
                  example_name, z_real_points, material_colors, material_names,
                  latent_dim, num_elems, num_nodes, material_labels=None, results=None):
    """Write the record, run_config.yaml, and the pickle if SAVE_PICKLE. results: the physics'
    per-element results, {name: dict(view, title, measures, value)} with value the array.

    Returns the path of the primary artefact: the .pkl when SAVE_PICKLE is on,
    otherwise the .npz. Returns None when WRITE_FILES is False (the tests only).
    """
    if WRITE_FILES:
        os.makedirs(save_dir, exist_ok=True)
    data = run_data(config, fields, vae_state_dict, history, example_name, z_real_points, material_colors,
                    material_names, latent_dim, num_elems, num_nodes, material_labels, results)
    if not WRITE_FILES:
        return None
    return write_run_data(save_dir, data)


def run_data(config, fields, vae_state_dict, history, example_name, z_real_points, material_colors,
             material_names, latent_dim, num_elems, num_nodes, material_labels=None, results=None):
    """Everything save_run_data writes, as one dict (its arguments as save_run_data takes them)."""
    sd = {}
    try:
        for k, v in (vae_state_dict or {}).items():
            sd[k] = v.detach().cpu() if hasattr(v, "detach") else v
    except Exception:
        sd = vae_state_dict
    data = {
        "schema_version": SCHEMA_VERSION,
        "config": config,
        "example_name": str(example_name),
        "latentDim": int(latent_dim),
        "num_elems": int(num_elems),
        "num_nodes": int(num_nodes),
        "fields": {k: np.asarray(v) for k, v in fields.items()},
        "results": {name: np.asarray(r["value"]) for name, r in (results or {}).items()},
        "result_info": {name: {k: r[k] for k in ("view", "title", "measures")} for name, r in (results or {}).items()},
        "vae_state_dict": sd,
        "history": history,
        "zRealPoints": np.asarray(z_real_points).astype(float),
        "material_colors": list(material_colors),
        "material_names": list(material_names),
        "material_labels": list(material_names if material_labels is None else material_labels),
    }
    return data


def write_run_data(save_dir, data):
    """run_data's dict written into save_dir: the record, run_config.yaml, and the pickle if SAVE_PICKLE.
    Returns the path of the primary artefact, as save_run_data."""
    rec_path = _save_record(save_dir, data)

    pkl_path = None
    if SAVE_PICKLE:
        pkl_path = os.path.join(save_dir, "run_data.pkl")
        with open(pkl_path, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    try:
        write_run_config_yaml(data["config"], os.path.join(save_dir, "run_config.yaml"))
    except Exception as e:
        print(f"[save] WARNING: run_config.yaml not written: {e}")
    return pkl_path or rec_path


def _save_record(save_dir, data):
    """Write run_data.npz and run_data.config.json into save_dir.

    x_bin is stored bitpacked; z_elem is stored for solid elements only.
    results/load_results.py undoes both. zeta, the property_<name> fields and vae_state_dict
    are not stored; they are recoverable from z_elem and the .nt named in meta.
    Each of the physics' results is stored as r_<name>, described in meta's results.
    """
    rec, cfg = record_arrays(save_dir, data)
    base = os.path.join(save_dir, "run_data")
    np.savez_compressed(base + ".npz", **rec)
    try:
        with open(base + ".config.json", "w") as fh:
            json.dump(cfg, fh, indent=1, default=str)
    except Exception as e:
        print(f"[save] WARNING: run_data.config.json not written: {e}")
    return base + ".npz"


def record_arrays(save_dir, data):
    """(the arrays of run_data.npz, the manifest of run_data.config.json) of run_data's dict, as
    _save_record writes them for a record in save_dir (its folder name is the record's descriptor)."""
    f = data["fields"]
    n = int(data["num_elems"])

    missing = {"material_indices", "x_bin", "x_cont", "z_elem"} - set(f)
    if missing:
        raise KeyError(
            f"[save] cannot write record: fields is missing {sorted(missing)}. "
            f"Present: {sorted(f)}")

    x_bin = np.asarray(f["x_bin"]).ravel() > 0.5
    z = np.asarray(f["z_elem"])
    if z.ndim != 2 or z.shape[0] != n:
        raise ValueError(
            f"[save] z_elem has shape {z.shape}; expected ({n}, latent_dim)")

    rec = {
        "material_indices": np.asarray(f["material_indices"]).ravel().astype(np.uint8),
        "x_bin_packed":     np.packbits(x_bin),
        "n_elems":          np.int64(n),
        "x_cont":           np.asarray(f["x_cont"]).ravel().astype(np.float32),
        "z_elem_solid":     z[x_bin].astype(np.float32),
        "z_latent_dim":     np.int64(z.shape[1]),
        "zRealPoints":      np.asarray(data["zRealPoints"]).astype(np.float32),
        "material_names":   np.array([str(s) for s in data["material_names"]]),
        "material_colors":  np.array([str(s) for s in data["material_colors"]]),
        "material_labels":  np.array([str(s) for s in data["material_labels"]]),
        "schema_version":   np.int64(data["schema_version"]),
    }
    for name, a in data["results"].items():
        rec["r_" + name] = np.asarray(a).ravel().astype(np.float32)
    for k, v in (data.get("history") or {}).items():
        a = np.asarray(v)
        if a.dtype.kind in "fiu":
            rec["h_" + k] = a.astype(np.float32)

    cfg = data.get("config") or {}
    tp = cfg.get("topopt_params") or {}
    rec["meta_json"] = np.array(json.dumps({
        "example":     str(data["example_name"]),
        "descriptor":  os.path.basename(os.path.normpath(save_dir)),
        "iterations":  (cfg.get("run") or {}).get("iterations"),
        "constraints": tp.get("Constraints"),
        "vae_net":     os.path.basename(
                           str(tp.get("MaterialsExcelFile", ""))).replace(".xlsx", ".nt"),
        "results":     data["result_info"],
    }, default=str))
    return rec, cfg
