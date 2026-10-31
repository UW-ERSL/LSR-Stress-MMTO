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
The problem and every setting come from projects/<example>.mmto. main has no project
files, so it gets its own coded version of the example, under the name below, and the
same settings.
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

# the name of each example on main
MAIN_NAMES = {
    "L-bracket":        "LBracketTopLoad_Stress_BM1",
    "U-bracket":        "UBracket_Stress_FF",
    "beam":             "Beam_Liao_Pareto_FF",
    "Airplane bracket": "AirplaneBracket_Stress",
}

# runs in the autodiff environment: the project's run_topopt arguments, enums as names
SETTINGS = r'''
import json, os, sys
sys.path.insert(0, ".")
from utils import project_paths
project_paths.bootstrap()
from project import Project, run_arguments, DEFAULT_SETTINGS
p = Project.load(sys.argv[1])
args = run_arguments(p)
if args["saveNet"] is None:                                  # the network shipped with the database
    args["saveNet"] = os.path.splitext(p.database_paths()[0])[0] + ".nt"
args = {k: getattr(v, "name", v) for k, v in args.items() if k not in ("to_problem", "save_dir", "plotter")}
defaults = {k: v for section in DEFAULT_SETTINGS.values() for k, v in section.items()}
print("SETTINGS " + json.dumps({"args": args, "defaults": defaults}))
'''

# runs inside the branch's own checkout and environment
RUNNER = r'''
import inspect, json, os, sys
spec, save_dir, branch = json.loads(sys.argv[1]), sys.argv[2], sys.argv[3]
sys.path.insert(0, ".")
from utils import project_paths
project_paths.bootstrap()
import run_io
from MainPureStructural import run_topopt, Z0InitMethod, BetaScheduleScheme
args = dict(spec["args"])
if branch == "autodiff":
    from project import Project
    args["to_problem"] = Project.load(os.path.join("projects", spec["example"] + ".mmto"))
else:
    accepted = set(inspect.signature(run_topopt).parameters)
    for k in sorted(set(args) - accepted):
        if spec["defaults"].get(k, object()) != args[k]:
            sys.exit(f"main's run_topopt has no setting {k!r}, and the project sets it to {args[k]!r}")
        print(f"main's run_topopt has no setting {k!r}; it is at its default, {args.pop(k)!r}")
    from ExamplesPureStructural import MMTOExamplesPureStructural
    args["to_problem"] = MMTOExamplesPureStructural[spec["main_name"]]
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


def settings_of(example, python):
    """The project's settings, read in the autodiff environment."""
    path = os.path.join(REPO, "projects", f"{example}.mmto")
    out = subprocess.run([python, "-c", SETTINGS, path], cwd=REPO, capture_output=True, text=True)
    lines = [l for l in out.stdout.splitlines() if l.startswith("SETTINGS ")]
    if out.returncode != 0 or not lines:
        fail(f"could not read {path}:\n{out.stdout}\n{out.stderr}")
    spec = json.loads(lines[-1][len("SETTINGS "):])
    spec.update(example=example, main_name=MAIN_NAMES[example])
    return spec


def one_run(branch, repo, python, example, spec):
    save_dir = os.path.join(REFERENCE, branch, example)
    os.makedirs(save_dir, exist_ok=True)
    log = os.path.join(save_dir, "run.log")
    print(f"[{time.strftime('%H:%M:%S')}] {example} on {branch} ... log: {log}", flush=True)
    t0 = time.time()
    with open(log, "w", encoding="utf-8") as fh:
        code = subprocess.call([python, "-c", RUNNER, json.dumps(spec), save_dir, branch],
                               cwd=repo, stdout=fh, stderr=subprocess.STDOUT)
    if code != 0:
        fail(f"{example} on {branch} stopped with exit code {code}; see {log}")
    print(f"           done in {(time.time() - t0) / 60:.1f} min", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--examples", default="L-bracket,U-bracket",
                   help="comma-separated, from: " + ", ".join(MAIN_NAMES))
    p.add_argument("--main-repo", default=os.path.join(os.path.dirname(REPO), "LSR-Stress-MMTO-main"))
    p.add_argument("--main-env", default="lsr-stress-mmto")
    p.add_argument("--ad-env", default="lsr-stress-mmto-ad")
    a = p.parse_args()

    examples = [e.strip() for e in a.examples.split(",") if e.strip()]
    for e in examples:
        if e not in MAIN_NAMES:
            fail(f"unknown example {e!r}; choose from {', '.join(MAIN_NAMES)}")
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
        spec = settings_of(e, py_ad)
        one_run("main", a.main_repo, py_main, e, spec)
        one_run("autodiff", REPO, py_ad, e, spec)

    print("\nDone. The runs are in comparison/reference; Tutorial.ipynb section 11 compares them.")


if __name__ == "__main__":
    main()
