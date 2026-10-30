"""
project_paths.py
================
Locate the repo root and make the project importable from anywhere.

    import project_paths
    ROOT = project_paths.bootstrap()

Two things it fixes. The project modules live in 0-Common and 1-PureStructural,
neither of which is on sys.path by default. And the example definitions carry
repo-root-relative material paths, so cwd has to be the repo root for them to
resolve.

0-Common/PyTOImports.py resolves ../../PyTO-AD/src against its own location, so
putting 0-Common on sys.path is enough for PyTO-AD to follow.

bootstrap() also checks the environment for what is known to cause trouble and
prints a warning for each problem it finds. It never stops anything.
"""
import os
import re
import sys
from importlib.metadata import PackageNotFoundError, version

_MARKER_DIRS = ("0-Common", "1-PureStructural")


def find_root(start=None, max_up=6):
    """Nearest ancestor holding both marker directories, or None."""
    d = os.path.abspath(start or os.path.dirname(os.path.abspath(__file__)))
    for _ in range(max_up):
        if all(os.path.isdir(os.path.join(d, m)) for m in _MARKER_DIRS):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    return None


def bootstrap(chdir=True, verbose=False):
    """Put the project folders on sys.path and chdir to the repo root.

    Returns the root, or None if it could not be located, in which case nothing
    is changed.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    root = find_root(here)

    if root is None:
        print(f"[project_paths] repo root not found: no ancestor of {here} "
              f"contains both {' and '.join(_MARKER_DIRS)}. sys.path unchanged.")
        return None

    wanted = [os.path.join(root, "1-PureStructural"),
              os.path.join(root, "0-Common"),
              root]
    added = []
    for p in wanted:
        if os.path.isdir(p) and p not in sys.path:
            sys.path.insert(0, p)
            added.append(p)

    if chdir and os.path.abspath(os.getcwd()) != root:
        os.chdir(root)

    if verbose:
        print(f"[project_paths] root = {root}")
        for p in added:
            print(f"[project_paths]   + {p}")
        missing = [os.path.basename(p) for p in wanted if not os.path.isdir(p)]
        if missing:
            print(f"[project_paths]   ! not found: {missing}")

    check_environment(root)
    return root


def check_environment(root):
    """Warn about what is known to cause trouble. Packages are not imported."""
    warnings = []

    pyto_ad = os.path.join(os.path.dirname(root), "PyTO-AD", "src")
    if not os.path.isdir(pyto_ad):
        warnings.append(f"PyTO-AD not found at {pyto_ad}. Run setup_environment.py, "
                        "or see Setting up by hand in the README.")

    try:
        numpy_version = version("numpy")
        major, minor = (int(v) for v in re.findall(r"\d+", numpy_version)[:2])
        if (major, minor) >= (2, 3):
            warnings.append(f"numpy {numpy_version} is installed; this code is tested "
                            "with numpy below 2.3 (see requirements.txt).")
    except PackageNotFoundError:
        warnings.append("numpy is not installed in this environment.")

    try:
        version("torch")
    except PackageNotFoundError:
        warnings.append("torch is not installed in this environment; this branch "
                        "needs it for the gradients.")

    for w in warnings:
        print(f"[project_paths] warning: {w}")
