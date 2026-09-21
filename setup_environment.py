"""
setup_environment.py
====================
Builds everything needed to run this repository, in one command:

    python setup_environment.py

1. creates a conda environment, lsr-stress-mmto, with Python 3.11
2. fetches the solver library beside this repository, at a fixed commit,
   with git if it is installed and as a download otherwise
3. installs requirements.txt into that environment
4. registers the environment as a Jupyter kernel
5. checks that the solver builds a mesh and a material decoder loads

Safe to run again: each step is skipped if already done.

Afterwards, open Tutorial.ipynb in a code editor and select the kernel
"Python (lsr-stress-mmto)". The README covers running it in a browser instead.
"""
import os
import shutil
import subprocess
import sys
import time

try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass
os.environ["PYTHONUNBUFFERED"] = "1"

ENV = "lsr-stress-mmto"
PYTHON = "3.11"

SOLVER_URL = "https://github.com/UW-ERSL/PyTO.git"
SOLVER_COMMIT = "b109a1691c5526001c766221d00e84e26470dfdb"
SOLVER_DIR = "PyTO"

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
SOLVER_PATH = os.path.join(PARENT, SOLVER_DIR)


_T0 = [time.time()]


def step(n, text, estimate=None):
    if n > 1:
        print(f"      done in {_fmt(time.time() - _T0[0])}", flush=True)
    _T0[0] = time.time()
    tail = f"   (usually {estimate})" if estimate else ""
    print(f"\n[{n}/5] {text}{tail}", flush=True)


def _fmt(sec):
    return f"{sec:.0f}s" if sec < 90 else f"{sec / 60:.1f} min"


def fail(msg):
    print(f"\nFAILED: {msg}")
    sys.exit(1)


def run(cmd, **kw):
    """Run a command, streaming its output. Returns the exit code."""
    print("      $ " + " ".join(cmd), flush=True)
    return subprocess.call(cmd, **kw)


def conda_exe():
    exe = os.environ.get("CONDA_EXE") or shutil.which("conda")
    if not exe:
        fail("conda was not found. Install Anaconda or Miniconda, then run this "
             "from an Anaconda Prompt, or a terminal where `conda` works.")
    return exe


def env_exists(conda):
    out = subprocess.run([conda, "env", "list"], capture_output=True, text=True).stdout
    return any(line.split() and line.split()[0] == ENV for line in out.splitlines())


def env_python(conda):
    """Path to the environment's own python, from `conda env list`."""
    out = subprocess.run([conda, "env", "list"], capture_output=True, text=True).stdout
    for line in out.splitlines():
        parts = line.split()
        if parts and parts[0] == ENV:
            prefix = parts[-1]
            exe = (os.path.join(prefix, "python.exe") if os.name == "nt"
                   else os.path.join(prefix, "bin", "python"))
            if os.path.isfile(exe):
                return exe
    fail(f"{ENV} exists but has no python inside, probably from an interrupted "
         f"run. Remove it with `conda env remove -n {ENV}` and run this again.")


def in_env(py, *args):
    """Run with the environment's interpreter directly. Unlike `conda run`, this
    keeps the real console attached, so pip shows its progress bars."""
    return run([py, *args])


def report_existing_solver():
    head = ""
    if shutil.which("git") and os.path.isdir(os.path.join(SOLVER_PATH, ".git")):
        head = subprocess.run(["git", "-C", SOLVER_PATH, "rev-parse", "HEAD"],
                              capture_output=True, text=True).stdout.strip()
    if not head:
        marker = os.path.join(SOLVER_PATH, ".source_commit")
        if os.path.isfile(marker):
            head = open(marker).read().strip()
    if not head:
        print("      already present, but its commit is unknown; leaving it as is")
        print(f"      results were produced at {SOLVER_COMMIT[:8]}")
    elif head.startswith(SOLVER_COMMIT[:8]) or SOLVER_COMMIT.startswith(head):
        print(f"      already present at {head[:8]}, the expected commit")
    else:
        print(f"      already present at {head[:8]}, leaving it as is")
        print(f"      note: results were produced at {SOLVER_COMMIT[:8]}")


def download_solver():
    """Fetch the pinned commit as a zip, for machines without git."""
    import tempfile
    import urllib.request
    import zipfile

    url = f"https://github.com/UW-ERSL/PyTO/archive/{SOLVER_COMMIT}.zip"
    tmp = tempfile.mkdtemp(prefix="solver_")
    zpath = os.path.join(tmp, "solver.zip")
    print(f"      $ download {url}", flush=True)

    last = [-1]

    def progress(blocks, block_size, total):
        done = blocks * block_size
        if total > 0:
            pct = min(100, int(100 * done / total))
            if pct // 10 != last[0]:
                last[0] = pct // 10
                print(f"      {pct:3d}%  {done / 2**20:6.1f} MB", flush=True)
        elif blocks % 1200 == 0:
            print(f"      {done / 2**20:6.1f} MB", flush=True)

    try:
        urllib.request.urlretrieve(url, zpath, reporthook=progress)
    except Exception as e:
        fail(f"could not download {url}: {e}")

    with zipfile.ZipFile(zpath) as z:
        top = z.namelist()[0].split("/")[0]
        z.extractall(tmp)
    shutil.move(os.path.join(tmp, top), SOLVER_PATH)
    with open(os.path.join(SOLVER_PATH, ".source_commit"), "w") as fh:
        fh.write(SOLVER_COMMIT + "\n")
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"      downloaded at {SOLVER_COMMIT[:8]}", flush=True)


def main():
    print(f"repository : {HERE}", flush=True)
    print(f"solver     : {SOLVER_PATH}", flush=True)
    print(f"environment: {ENV}", flush=True)
    print("\nFive steps. Step 3 downloads a few GB and is the slow one.", flush=True)

    conda = conda_exe()

    if os.environ.get("CONDA_DEFAULT_ENV") == ENV:
        fail(f"you are inside {ENV}. Run this from the base environment "
             f"(`conda deactivate` first), so the environment can be rebuilt.")

    # ------------------------------------------------------------------ 1
    step(1, "conda environment", "1 to 3 min")
    if env_exists(conda):
        print(f"      {ENV} exists, skipping")
    # Python from conda-forge rather than Anaconda's default channel, which
    # refuses to install anything until its terms of service are accepted
    elif run([conda, "create", "-y", "-n", ENV, "--override-channels",
              "-c", "conda-forge", f"python={PYTHON}"]) != 0:
        fail(f"could not create {ENV}")

    # ------------------------------------------------------------------ 2
    step(2, "solver library", "under a minute")
    if os.path.isdir(os.path.join(SOLVER_PATH, "src")):
        report_existing_solver()
    elif shutil.which("git"):
        if run(["git", "clone", SOLVER_URL, SOLVER_PATH]) != 0:
            fail(f"could not clone {SOLVER_URL}")
        if run(["git", "-C", SOLVER_PATH, "checkout", "-q", SOLVER_COMMIT]) != 0:
            fail(f"could not check out commit {SOLVER_COMMIT}")
        print(f"      cloned at {SOLVER_COMMIT[:8]}", flush=True)
    else:
        print("      git not found; downloading the same commit instead", flush=True)
        download_solver()

    # ------------------------------------------------------------------ 3
    step(3, "dependencies", "10 to 30 min; pip shows progress below")
    py = env_python(conda)
    print(f"      using {py}", flush=True)
    req = os.path.join(HERE, "requirements.txt")
    if in_env(py, "-m", "pip", "install", "--upgrade", "pip") != 0:
        fail("could not upgrade pip inside the environment")
    if in_env(py, "-m", "pip", "install", "--progress-bar", "on", "-r", req) != 0:
        fail("dependency install failed; the lines above say which package")

    # ------------------------------------------------------------------ 4
    step(4, "Jupyter kernel", "a few seconds")
    py = env_python(conda)
    if in_env(py, "-m", "ipykernel", "install", "--user",
              "--name", ENV, "--display-name", f"Python ({ENV})") != 0:
        fail("could not register the Jupyter kernel")

    # ------------------------------------------------------------------ 5
    step(5, "verification", "under a minute")
    check = os.path.join(HERE, "_verify_environment.py")
    with open(check, "w") as fh:
        fh.write(VERIFY)
    try:
        code = in_env(env_python(conda), check)
    finally:
        os.remove(check)
    if code != 0:
        fail("the environment is built but something does not import; see above")

    print(f"      done in {_fmt(time.time() - _T0[0])}", flush=True)
    print(f"""
Done.

Open Tutorial.ipynb in your code editor and select the kernel
"Python ({ENV})". The README covers running it in a browser instead.
""")


VERIFY = r'''
import contextlib, io, os, sys, importlib
here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here)

ok = True
def check(label, fn):
    global ok
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            fn()
        print(f"      ok    {label}", flush=True)
    except Exception as e:
        ok = False
        print(f"      FAIL  {label}: {type(e).__name__}: {e}", flush=True)

import numpy, numba
print(f"      numpy {numpy.__version__}, numba {numba.__version__}", flush=True)

from utils import project_paths
root = project_paths.bootstrap()
check("repository found", lambda: root or (_ for _ in ()).throw(RuntimeError("not found")))

for m in ("records", "render", "figures", "tables", "display"):
    check(f"utils.{m} imports", lambda m=m: importlib.import_module(f"utils.{m}"))
check("solver library imports", lambda: importlib.import_module("ExamplesPureStructural"))
check("optimizer driver imports", lambda: importlib.import_module("MainPureStructural"))

def lbracket_mesh():
    from ExamplesPureStructural import (MMTOExamplesPureStructural,
                                        getMMTOProblemPureStructural)
    mesh = getMMTOProblemPureStructural(
        MMTOExamplesPureStructural.LBracketTopLoad_Stress_BM1)[0]
    if int(mesh.num_elems) != 12825:
        raise RuntimeError(f"expected 12825 elements, got {mesh.num_elems}")
check("the L-bracket meshes to 12,825 elements", lbracket_mesh)

def decoder():
    from ExamplesPureStructural import (MMTOExamplesPureStructural,
                                        getMMTOProblemPureStructural)
    from materialEncoder import MaterialEncoder
    vae_params = getMMTOProblemPureStructural(
        MMTOExamplesPureStructural.LBracketTopLoad_Stress_BM1)[5]
    db = "./1-PureStructural/MaterialDataPureStructural/BM1_3Mats_2Attr"
    enc = MaterialEncoder(vae_params)
    enc.readExcel(db + ".xlsx")
    enc.loadAutoencoderFromFile(db + ".nt")
check("a material database reads and its decoder loads", decoder)

sys.exit(0 if ok else 1)
'''


if __name__ == "__main__":
    main()
