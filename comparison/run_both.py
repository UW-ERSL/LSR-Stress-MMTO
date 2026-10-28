"""
run_both.py
===========
Makes the reference runs in comparison/reference: each example on main, then on
autodiff, one after the other on this machine, with identical settings.

    python comparison/run_both.py
    python comparison/run_both.py --examples L-bracket

Run it from an Anaconda Prompt, from the base environment, inside this repository.
It needs:
  - a checkout of main beside this one. Make it once, from this repository:
        git worktree add ../LSR-Stress-MMTO-main main
    (or pass --main-repo with the folder of another checkout of main)
  - the environment of each branch: lsr-stress-mmto for main, lsr-stress-mmto-ad
    for this branch (each made by its branch's setup_environment.py)

Each run goes to comparison/reference/<branch>/<example>/, with its log in run.log.
The settings are the ones in Tutorial.ipynb, section 1 and section 7.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
REFERENCE = os.path.join(HERE, "reference")

# Tutorial.ipynb, section 1
SETUP = {
    "L-bracket": dict(
        problem="LBracketTopLoad_Stress_BM1",
        database="BM1_3Mats_2Attr",
        relative_filter_radius=3.5, material_filter_radius=3.5,
        maxIterations=300, beta="PAPER_BM1"),
    "U-bracket": dict(
        problem="UBracket_Stress_FF",
        database="6Materials_AF_3Attr",
        relative_filter_radius=3.5, material_filter_radius=2.5,
        pnorm_exponent=16.0, maxIterations=350, beta="PAPER"),
    "beam": dict(
        problem="Beam_Liao_Pareto_FF",
        database="Beam_8Mats_3Attr_v4_spectrum2D",
        relative_filter_radius=None, material_filter_radius=None,
        maxIterations=350, beta="PAPER",
        use_constant_poissons_ratio=True,
        constant_poissons_ratio=0.37),
    "Airplane bracket": dict(
        problem="AirplaneBracket_Stress",
        database="CustomAlloys20_3D",
        relative_filter_radius=None, material_filter_radius=None,
        maxIterations=300, beta="PAPER"),
}


def run_args(example):
    """Tutorial.ipynb, section 7, with enums as names so they can cross processes."""
    cfg = SETUP[example]
    return dict(
        to_problem=cfg["problem"],
        saveNet=f"./1-PureStructural/MaterialDataPureStructural/{cfg['database']}.nt",
        use_pretrained_vae=True,

        relative_filter_radius=cfg["relative_filter_radius"],
        material_filter_radius=cfg["material_filter_radius"],
        pnorm_exponent=cfg.get("pnorm_exponent"),
        maxIterations=cfg["maxIterations"],
        rel_conv_tol=1e-7,

        # distance-to-material penalty
        use_penalization=True,
        gamma_init=1e-3,
        gamma_factor=1.1,
        gamma_max=10,
        penalty_start_iter=0,
        distance_penalty_disable_after_iter=250,
        use_dynamic_gamma_scheduling=True,

        # latent smoothing
        use_z_smoothing=True,
        z_smoothing_weight=1e-4,
        z_smoothing_disable_after_iter=250,

        use_heaviside_projection=True,
        beta_schedule_scheme=cfg["beta"],
        eta_proj=0.5,
        z0_init_method="ORIGIN",
        snap_to_real_material=False,
        use_constant_poissons_ratio=cfg.get("use_constant_poissons_ratio", False),
        constant_poissons_ratio=cfg.get("constant_poissons_ratio", 0.3),
        use_continuation=False,
        binarize_topology=True,
        plot_progress=False,
        verbose=True,
    )


# runs inside the branch's own checkout and environment
RUNNER = r'''
import json, sys
args, save_dir = json.loads(sys.argv[1]), sys.argv[2]
sys.path.insert(0, ".")
from utils import project_paths
project_paths.bootstrap()
import run_io
from ExamplesPureStructural import MMTOExamplesPureStructural
from MainPureStructural import run_topopt, Z0InitMethod, BetaScheduleScheme
args["to_problem"] = MMTOExamplesPureStructural[args["to_problem"]]
args["beta_schedule_scheme"] = BetaScheduleScheme[args["beta_schedule_scheme"]]
args["z0_init_method"] = Z0InitMethod[args["z0_init_method"]]
run_io.WRITE_FILES = True
run_topopt(save_dir=save_dir, **args)
'''


def fail(msg):
    print(f"\nFAILED: {msg}")
    sys.exit(1)


def env_python(env):
    """Path to an environment's python, from `conda env list`."""
    conda = os.environ.get("CONDA_EXE") or shutil.which("conda")
    if not conda:
        fail("conda was not found; run this from an Anaconda Prompt")
    out = subprocess.run([conda, "env", "list"], capture_output=True, text=True).stdout
    for line in out.splitlines():
        parts = line.split()
        if parts and parts[0] == env:
            exe = (os.path.join(parts[-1], "python.exe") if os.name == "nt"
                   else os.path.join(parts[-1], "bin", "python"))
            if os.path.isfile(exe):
                return exe
    fail(f"environment {env} not found; make it with that branch's setup_environment.py")


def branch_of(repo):
    out = subprocess.run(["git", "-C", repo, "rev-parse", "--abbrev-ref", "HEAD"],
                         capture_output=True, text=True)
    return out.stdout.strip()


def one_run(branch, repo, python, example):
    save_dir = os.path.join(REFERENCE, branch, example)
    os.makedirs(save_dir, exist_ok=True)
    log = os.path.join(save_dir, "run.log")
    print(f"[{time.strftime('%H:%M:%S')}] {example} on {branch} ... log: {log}", flush=True)
    t0 = time.time()
    with open(log, "w", encoding="utf-8") as fh:
        code = subprocess.call([python, "-c", RUNNER, json.dumps(run_args(example)), save_dir],
                               cwd=repo, stdout=fh, stderr=subprocess.STDOUT)
    if code != 0:
        fail(f"{example} on {branch} stopped with exit code {code}; see {log}")
    print(f"           done in {(time.time() - t0) / 60:.1f} min", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--examples", default="L-bracket,U-bracket",
                   help="comma-separated, from: " + ", ".join(SETUP))
    p.add_argument("--main-repo", default=os.path.join(os.path.dirname(REPO), "LSR-Stress-MMTO-main"))
    p.add_argument("--main-env", default="lsr-stress-mmto")
    p.add_argument("--ad-env", default="lsr-stress-mmto-ad")
    a = p.parse_args()

    examples = [e.strip() for e in a.examples.split(",") if e.strip()]
    for e in examples:
        if e not in SETUP:
            fail(f"unknown example {e!r}; choose from {', '.join(SETUP)}")
    if not os.path.isdir(a.main_repo):
        fail(f"no checkout of main at {a.main_repo}. Make one, from this repository:\n"
             f"    git worktree add {a.main_repo} main")
    if branch_of(a.main_repo) != "main":
        fail(f"{a.main_repo} is on branch {branch_of(a.main_repo)!r}, not main")
    if branch_of(REPO) != "autodiff":
        fail(f"this repository is on branch {branch_of(REPO)!r}, not autodiff")

    py_main, py_ad = env_python(a.main_env), env_python(a.ad_env)
    print(f"main     : {a.main_repo}  ({py_main})")
    print(f"autodiff : {REPO}  ({py_ad})")
    print(f"examples : {', '.join(examples)}\n")

    for e in examples:
        one_run("main", a.main_repo, py_main, e)
        one_run("autodiff", REPO, py_ad, e)

    print("\nDone. The runs are in comparison/reference; Tutorial.ipynb section 11 compares them.")


if __name__ == "__main__":
    main()
