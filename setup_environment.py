"""
setup_environment.py
====================
Builds everything needed to run this repository, in one command:

    python setup_environment.py

autodiff branch: gradients come from automatic differentiation through PyTO-AD.

1. creates a conda environment, lsr-stress-mmto-ad, with Python 3.11
2. fetches PyTO-AD, the differentiable topology optimization code this branch
   builds on, beside this repository, at a fixed commit,
   with git if it is installed and as a download otherwise
3. installs requirements.txt into that environment, and this repository as the package
   lsr_mmto (pip install -e .), so scripts anywhere can import it
4. checks that the solver builds a mesh and a material decoder loads
5. (Windows) adds LSR MMTO to the Start menu, and to the desktop if you say yes; the one question
   is asked before anything is installed, so the rest runs on its own

Safe to run again: each step is skipped if already done.

Afterwards, open LSR MMTO from the Start menu (type LSR).
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

ENV = "lsr-stress-mmto-ad"
PYTHON = "3.11"

SOLVER_URL = "https://github.com/UW-ERSL/PyTO-AD.git"
SOLVER_COMMIT = "790d935bf210cb115944495d50abce4173e46a27"
SOLVER_DIR = "PyTO-AD"

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
SOLVER_PATH = os.path.join(PARENT, SOLVER_DIR)


_T0 = [time.time()]


def steps(windows, desktop):
    """The install's steps in order, as (what it does, how long it usually takes). The [n/N] lines count
    this list, so a step added here is counted without changing a number anywhere else."""
    out = [("conda environment", "1 to 3 min"),
           ("PyTO-AD, the differentiable topology optimization code", "under a minute"),
           ("dependencies", "roughly 10 min; pip shows progress below"),
           ("verification", "under a minute")]
    if windows:
        out.append(("Start menu" + (" and desktop" if desktop else ""), "a few seconds"))
    return out


def step(n, all_steps):
    if n > 1:
        print(f"      done in {_fmt(time.time() - _T0[0])}", flush=True)
    _T0[0] = time.time()
    text, estimate = all_steps[n - 1]
    print(f"\n[{n}/{len(all_steps)}] {text}   (usually {estimate})", flush=True)


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
        fail("conda was not found. Install Miniforge (or Miniconda or Anaconda), then run this "
             "from a Miniforge Prompt, or a terminal where `conda` works.")
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
    if shutil.which("git") and os.path.isdir(os.path.join(SOLVER_PATH, ".git")):
        dirty = subprocess.run(
            ["git", "-C", SOLVER_PATH, "status", "--porcelain", "--untracked-files=no"],
            capture_output=True, text=True).stdout.strip()
        if dirty:
            print("      note: it has uncommitted changes, which will be used as they are")


def pyto_like_siblings():
    """Folders beside this repository that hold PyTO-AD under another name.

    A folder counts if it contains src/pyto/topopt/common.py. A link to PyTO-AD
    and PyTO-AD itself are the same folder, so a link does not count.
    """
    try:
        names = sorted(os.listdir(PARENT))
    except OSError:
        return []
    ours = os.path.realpath(SOLVER_PATH) if os.path.exists(SOLVER_PATH) else None
    found = []
    for name in names:
        path = os.path.join(PARENT, name)
        if name == SOLVER_DIR or os.path.realpath(path) == os.path.realpath(HERE):
            continue
        if not os.path.isfile(os.path.join(path, "src", "pyto", "topopt", "common.py")):
            continue
        if ours is None or os.path.realpath(path) != ours:
            found.append(name)
    return found


def download_solver():
    """Fetch the pinned commit as a zip, for machines without git."""
    import tempfile
    import urllib.request
    import zipfile

    url = f"https://github.com/UW-ERSL/PyTO-AD/archive/{SOLVER_COMMIT}.zip"
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


def ask_desktop(read=input):
    """Whether to put the app's icon on the desktop: asked once, at the start, so that nobody has to watch
    the install for it. Enter means yes; no answer possible (no keyboard) also means yes."""
    print("\n" + "=" * 64)
    print(" LSR MMTO will be added to the Start menu.")
    try:
        answer = read(" Also put an LSR MMTO icon on the desktop? [Y/n] ")
    except EOFError:
        answer = ""
    print("=" * 64 + "\n", flush=True)
    return answer.strip().lower() not in ("n", "no")


def main():
    desktop = ask_desktop() if os.name == "nt" else False
    print(f"repository : {HERE}", flush=True)
    print(f"PyTO-AD    : {SOLVER_PATH}", flush=True)
    print(f"environment: {ENV}", flush=True)
    all_steps = steps(os.name == "nt", desktop)
    print(f"\n{len(all_steps)} steps. Step 3 downloads a few GB and is the slow one.", flush=True)

    conda = conda_exe()

    if os.environ.get("CONDA_DEFAULT_ENV") == ENV:
        fail(f"you are inside {ENV}. Run this from the base environment "
             f"(`conda deactivate` first), so the environment can be rebuilt.")

    # ------------------------------------------------------------------ 1
    step(1, all_steps)
    if env_exists(conda):
        print(f"      {ENV} exists, skipping")
    # Python from conda-forge rather than Anaconda's default channel, which
    # refuses to install anything until its terms of service are accepted
    elif run([conda, "create", "-y", "-n", ENV, "--override-channels",
              "-c", "conda-forge", f"python={PYTHON}"]) != 0:
        fail(f"could not create {ENV}")

    # ------------------------------------------------------------------ 2
    step(2, all_steps)
    others = pyto_like_siblings()
    if os.path.isdir(os.path.join(SOLVER_PATH, "src")):
        report_existing_solver()
        for name in others:
            print(f"      note: {name} also looks like PyTO-AD; this branch uses "
                  f"{SOLVER_DIR}", flush=True)
    elif others:
        fail(f"found {others[0]} beside this repository, but it must be called "
             f"{SOLVER_DIR}.\nRename it, or keep the name and link it, in {PARENT}:\n"
             f"    mklink /J {SOLVER_DIR} {others[0]}\n"
             f"then run this again. Nothing was downloaded.")
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
    step(3, all_steps)
    py = env_python(conda)
    print(f"      using {py}", flush=True)
    req = os.path.join(HERE, "requirements.txt")
    if in_env(py, "-m", "pip", "install", "--no-warn-script-location", "--upgrade", "pip") != 0:
        fail("could not upgrade pip inside the environment")
    if in_env(py, "-m", "pip", "install", "--no-warn-script-location",
              "--progress-bar", "on", "-r", req) != 0:
        fail("dependency install failed; the lines above say which package")
    if in_env(py, "-m", "pip", "install", "--no-warn-script-location", "--no-deps", "-e", HERE) != 0:
        fail("could not install this repository as the package lsr_mmto")

    # ------------------------------------------------------------------ 4
    step(4, all_steps)
    check = os.path.join(HERE, "_verify_environment.py")
    with open(check, "w") as fh:
        fh.write(VERIFY)
    try:
        code = in_env(env_python(conda), check)
    finally:
        os.remove(check)
    if code != 0:
        fail("the environment is built but something does not import; see above")

    # ------------------------------------------------------------------ 5
    opened = ""
    if os.name == "nt":
        step(5, all_steps)
        args = [os.path.join(HERE, "make_shortcut.py")] + ([] if desktop else ["--no-desktop"])
        if in_env(env_python(conda), *args) == 0:
            opened = ("\nOpen LSR MMTO from the Start menu (type LSR)"
                      + (" or the desktop icon." if desktop else "."))
        else:                                       # never a reason to fail the install
            opened = ("\nThe shortcut could not be made (the reason is above). To open LSR MMTO, run\n"
                      f"    conda activate {ENV}\n    python run_gui.py\nin this folder.")

    print(f"      done in {_fmt(time.time() - _T0[0])}", flush=True)
    print(f"""
Done.
{opened}
""")


VERIFY = r'''
import contextlib, io, os, sys, importlib
from importlib.metadata import version
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

check("LSR MMTO is installed in the environment", lambda: version("lsr_mmto"))
import pkgutil
import lsr_mmto.results
for m in sorted(i.name for i in pkgutil.iter_modules(lsr_mmto.results.__path__)):   # every file there, by itself
    check(f"lsr_mmto.results.{m} imports", lambda m=m: importlib.import_module(f"lsr_mmto.results.{m}"))
check("PyTO-AD and the projects import", lambda: importlib.import_module("lsr_mmto.problem.project"))
check("optimizer driver imports", lambda: importlib.import_module("lsr_mmto.optimize.optimization_loop"))

def pyto_ad_only():
    import pyto
    where = os.path.realpath(os.path.dirname(pyto.__file__))
    expected = os.path.realpath(os.path.join(os.path.dirname(here), "PyTO-AD", "src", "pyto"))
    if where != expected:
        raise RuntimeError(f"pyto imported from {where}, expected {expected}")
    old = [m for m in ("topopt_common", "hex_structural_fea", "topopt_material_model") if m in sys.modules]
    if old:
        raise RuntimeError(f"PyTO (main) modules were imported: {old}")
check("only PyTO-AD is used (no PyTO main modules loaded)", pyto_ad_only)

LBRACKET = os.path.join(here, "projects", "L-bracket.mmto")

def lbracket_mesh():
    from lsr_mmto.problem.project import Project
    mesh = Project.load(LBRACKET).build(verbose=False)[0]
    if int(mesh.num_elems) != 12825:
        raise RuntimeError(f"expected 12825 elements, got {mesh.num_elems}")
check("the L-bracket meshes to 12,825 elements", lbracket_mesh)

def decoder():
    from lsr_mmto.problem.project import Project
    from lsr_mmto.materials.encoder import MaterialEncoder
    vae_params = Project.load(LBRACKET).build(verbose=False)[5]
    db = os.path.join(here, "data", "databases", "BM1_3Mats_2Attr")
    enc = MaterialEncoder(vae_params)
    enc.readExcel(db + ".xlsx")
    enc.loadAutoencoderFromFile(db + ".nt")
check("a material database reads and its decoder loads", decoder)

sys.exit(0 if ok else 1)
'''


if __name__ == "__main__":
    main()
