"""
uninstall_lsr_mmto.py
=====================
Removes LSR MMTO from this computer: Windows Settings > Apps > LSR MMTO > Uninstall runs it, with the
conda installation's own Python, as the entry make_shortcut.py writes says (the environment cannot be
removed while it runs this). It shows what it removes and what it keeps, asks, and then removes:

    the Start menu and desktop shortcuts
    the Jupyter kernel "Python (lsr-stress-mmto-ad)"
    the GUI's saved settings, in %APPDATA%\\LSR MMTO
    the environment lsr-stress-mmto-ad (conda env remove)
    its own entry in Settings > Apps, last, so that a removal that stops part way can be run again

It keeps this folder, PyTO-AD beside it, and every project, and says where they are.
"""
import os
import shutil
import subprocess
import sys

import make_shortcut as M
from setup_environment import ENV, env_exists

ROOT = os.path.dirname(os.path.abspath(__file__))
KERNEL = f"Python ({ENV})"


def shortcuts():
    """The shortcuts make_shortcut.py makes, where they are: the Start menu's, and the desktop's wherever
    Windows keeps it (OneDrive too)."""
    folders = [M.folder_path("Programs"), M.folder_path("Desktop")]
    folders += [os.path.join(os.environ[v], "Desktop") for v in ("USERPROFILE", "OneDrive") if os.environ.get(v)]
    found = []
    for folder in filter(None, folders):
        path = os.path.join(folder, M.NAME + ".lnk")
        if os.path.isfile(path) and os.path.normcase(path) not in map(os.path.normcase, found):
            found.append(path)
    return found


def kernel_folder():
    """Where the setup registered the Jupyter kernel (ipykernel install --user)."""
    return os.path.join(os.environ.get("APPDATA", ""), "jupyter", "kernels", ENV)


def settings_folder():
    """Where the GUI keeps its settings between sessions."""
    return os.path.join(os.environ.get("APPDATA", ""), M.NAME)


def conda():
    """The conda of the installation this Python belongs to; None when there is none."""
    for path in (os.path.join(sys.prefix, "Scripts", "conda.exe"), os.path.join(sys.prefix, "condabin", "conda.bat")):
        if os.path.isfile(path):
            return path
    return shutil.which("conda")


def inside_the_environment():
    """True when this runs inside the environment it would remove, which conda refuses to remove."""
    return (os.path.basename(os.path.normpath(sys.prefix)).lower() == ENV or
            os.environ.get("CONDA_DEFAULT_ENV") == ENV)


def plan(conda_path):
    """(what is removed, what is kept), each a list of (what, where) lines to show."""
    removed = [("Shortcut", path) for path in shortcuts()]
    if os.path.isdir(kernel_folder()):
        removed.append(("Jupyter kernel", KERNEL))
    if os.path.isdir(settings_folder()):
        removed.append(("Saved settings", settings_folder()))
    if conda_path and env_exists(conda_path):
        removed.append(("Python environment", ENV))
    removed.append(("Entry in Settings > Apps", M.NAME))
    kept = [("LSR MMTO's folder", ROOT), ("PyTO-AD", os.path.join(os.path.dirname(ROOT), "PyTO-AD")),
            ("Your projects", "where you saved them")]
    return removed, kept


def remove(removed, conda_path):
    """Remove what plan listed, the Settings > Apps entry last and only when everything else went; True when
    everything went."""
    import winreg
    whole = True
    for what, where in removed:
        try:
            if what == "Shortcut":
                os.remove(where)
            elif what == "Jupyter kernel":
                shutil.rmtree(kernel_folder())
            elif what == "Saved settings":
                shutil.rmtree(settings_folder())
            elif what == "Python environment":
                print(f"Removing the Python environment {ENV}, which takes a minute...")
                if subprocess.run([conda_path, "env", "remove", "-n", ENV, "-y"]).returncode != 0:
                    raise OSError("conda could not remove it; the lines above say why")
            elif whole:
                winreg.DeleteKey(winreg.HKEY_CURRENT_USER, M.UNINSTALL_KEY)
            else:
                continue
            print(f"Removed: {what}, {where}")
        except OSError as e:
            whole = False
            print(f"Not removed: {what}, {where}: {e}")
    return whole


def main(answer=input):
    if inside_the_environment():
        print(f"This cannot run inside {ENV}, the environment it removes. Uninstall LSR MMTO from Windows "
              f"Settings > Apps > {M.NAME} > Uninstall.")
        return 1
    conda_path = conda()
    removed, kept = plan(conda_path)
    width = max(len(what) for what, _ in removed + kept)
    print(f"{M.NAME} will be removed from this computer.\n\nTo be removed:")
    print("\n".join(f"  {what:<{width}}  {where}" for what, where in removed))
    print("\nTo be kept:")
    print("\n".join(f"  {what:<{width}}  {where}" for what, where in kept))
    if conda_path is None:
        print(f"\nconda was not found, so the environment {ENV} is not removed from here.")
    try:
        sure = answer(f"\nRemove {M.NAME}? Type y and press Enter: ").strip().lower() in ("y", "yes")
    except EOFError:
        sure = False
    if not sure:
        print("Nothing was removed.")
        return 1
    whole = remove(removed, conda_path)
    print(f"\n{M.NAME} is removed." if whole else f"\nSome parts were not removed (see above). Run Uninstall "
                                                  f"again from Settings > Apps once the reason is put right.")
    print("These stay, to delete by hand if you no longer want them:")
    print("\n".join(f"  {what:<{width}}  {where}" for what, where in kept))
    return 0 if whole else 1


if __name__ == "__main__":
    code = main()
    try:
        input("\nPress Enter to close this window.")
    except EOFError:
        pass
    sys.exit(code)
