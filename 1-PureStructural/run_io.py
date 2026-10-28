"""
run_io.py
=========
Persistence layer for MMTO runs.

Writes into save_dir:
    run_data.npz         slim record: final fields + history
    run_data.config.json the manifest as JSON, beside the record
    run_config.yaml      the same manifest, sectioned
    run_data.pkl         only when SAVE_PICKLE is True

Nothing is written unless a save_dir is passed. Reader: utils/records.py.
"""
import os
import json
import pickle
import datetime
import importlib
import numpy as np


# Full-precision pickle alongside the record: zeta, void latents, float64,
# VAE weights. Off by default.
SAVE_PICKLE = False

# When False, save_run_data captures the payload in LAST_RUN and writes nothing.
WRITE_FILES = True
LAST_RUN = None


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
        lambda: __import__("PyTOImports", fromlist=["topopt_material_model"]).topopt_material_model,
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


def _material_summary(mat_encoder, z_real_points, material_colors):
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
# public API                                                                  #
# --------------------------------------------------------------------------- #
def build_run_config(example_name, run_args, to_params, vae_params, beta_schedule,
                     mesh, bc, mat_encoder, z_real_points, material_colors,
                     mma_settings=None, runtime_sec=None, iterations=None,
                     comment="", pnorm_exponent=None):
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
    cfg["materials"] = _material_summary(mat_encoder, z_real_points, material_colors)
    cfg["fea"] = {
        "mesh": _mesh_summary(mesh),
        "loading_and_bc": _loading_bc_summary(mesh, bc),
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


def save_run_data(save_dir, config, fields, vae_state_dict, history,
                  example_name, z_real_points, material_colors, material_names,
                  latent_dim, num_elems, num_nodes):
    """Write the record, run_config.yaml, and the pickle if SAVE_PICKLE.

    Returns the path of the primary artefact: the .pkl when SAVE_PICKLE is on,
    otherwise the .npz. Returns None when WRITE_FILES is False, in which case
    the payload is left in LAST_RUN.
    """
    if WRITE_FILES:
        os.makedirs(save_dir, exist_ok=True)
    sd = {}
    try:
        for k, v in (vae_state_dict or {}).items():
            sd[k] = v.detach().cpu() if hasattr(v, "detach") else v
    except Exception:
        sd = vae_state_dict
    data = {
        "schema_version": 1,
        "config": config,
        "example_name": str(example_name),
        "latentDim": int(latent_dim),
        "num_elems": int(num_elems),
        "num_nodes": int(num_nodes),
        "fields": {k: np.asarray(v) for k, v in fields.items()},
        "vae_state_dict": sd,
        "history": history,
        "zRealPoints": np.asarray(z_real_points).astype(float),
        "material_colors": list(material_colors),
        "material_names": list(material_names),
    }

    global LAST_RUN
    LAST_RUN = data
    if not WRITE_FILES:
        return None

    rec_path = _save_record(save_dir, data)

    pkl_path = None
    if SAVE_PICKLE:
        pkl_path = os.path.join(save_dir, "run_data.pkl")
        with open(pkl_path, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    try:
        write_run_config_yaml(config, os.path.join(save_dir, "run_config.yaml"))
    except Exception as e:
        print(f"[run_io] WARNING: run_config.yaml not written: {e}")
    return pkl_path or rec_path


def _save_record(save_dir, data):
    """Write run_data.npz and run_data.config.json into save_dir.

    x_bin is stored bitpacked; z_elem is stored for solid elements only.
    utils/records.py undoes both. zeta, youngs_modulus and vae_state_dict
    are not stored; they are recoverable from z_elem and the .nt named in meta.
    """
    f = data["fields"]
    n = int(data["num_elems"])

    missing = {"material_indices", "x_bin", "x_cont", "von_mises", "z_elem"} - set(f)
    if missing:
        raise KeyError(
            f"[run_io] cannot write record: fields is missing {sorted(missing)}. "
            f"Present: {sorted(f)}")

    x_bin = np.asarray(f["x_bin"]).ravel() > 0.5
    z = np.asarray(f["z_elem"])
    if z.ndim != 2 or z.shape[0] != n:
        raise ValueError(
            f"[run_io] z_elem has shape {z.shape}; expected ({n}, latent_dim)")

    rec = {
        "material_indices": np.asarray(f["material_indices"]).ravel().astype(np.uint8),
        "x_bin_packed":     np.packbits(x_bin),
        "n_elems":          np.int64(n),
        "x_cont":           np.asarray(f["x_cont"]).ravel().astype(np.float32),
        "von_mises":        np.asarray(f["von_mises"]).ravel().astype(np.float32),
        "z_elem_solid":     z[x_bin].astype(np.float32),
        "z_latent_dim":     np.int64(z.shape[1]),
        "zRealPoints":      np.asarray(data["zRealPoints"]).astype(np.float32),
        "material_names":   np.array([str(s) for s in data["material_names"]]),
    }
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
        "params":      {},
    }, default=str))

    base = os.path.join(save_dir, "run_data")
    np.savez_compressed(base + ".npz", **rec)
    try:
        with open(base + ".config.json", "w") as fh:
            json.dump(cfg, fh, indent=1, default=str)
    except Exception as e:
        print(f"[run_io] WARNING: run_data.config.json not written: {e}")
    return base + ".npz"
