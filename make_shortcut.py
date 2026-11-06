"""
make_shortcut.py
================
Put "LSR MMTO", with the app's icon, in the Start menu and on the desktop (Windows). Opening it
starts the GUI with no project and no console window. setup_environment.py does this at its end;
to do it by hand, in the Miniforge Prompt, from this folder:

    conda activate lsr-stress-mmto-ad
    python make_shortcut.py                  (--no-desktop: the Start menu only)

The shortcuts start this environment's pythonw with run_gui.py in this folder, so run it again if
the folder moves or the environment is made again.

It also lists LSR MMTO in Windows Settings > Apps (a per-user uninstall entry, no administrator
rights needed), whose Uninstall runs uninstall_lsr_mmto.py with the conda installation's own Python:
that is the one way to remove LSR MMTO. Running this again writes the same entry over the old one.
"""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
NAME = "LSR MMTO"
ICON = os.path.join(ROOT, "gui", "assets", "lsr_mmto.ico")
UNINSTALL_KEY = rf"Software\Microsoft\Windows\CurrentVersion\Uninstall\{NAME}"      # under HKEY_CURRENT_USER
UNINSTALLER = os.path.join(ROOT, "uninstall_lsr_mmto.py")


def _ps(text):
    """A PowerShell string in single quotes (a quote inside is doubled)."""
    return "'" + str(text).replace("'", "''") + "'"


def _where(folder):
    """PowerShell for the folder: "Desktop" or "Programs" (the Start menu) wherever Windows keeps them for
    this user (OneDrive too), or a folder's path."""
    return f"[Environment]::GetFolderPath({_ps(folder)})" if folder in ("Desktop", "Programs") else _ps(folder)


def folder_path(folder):
    """Where Windows keeps "Desktop" or "Programs" for this user, as the shortcuts are made there; None
    when PowerShell does not say."""
    done = subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", _where(folder)],
                          capture_output=True, text=True)
    return (done.stdout.strip() or None) if done.returncode == 0 else None


def powershell(pythonw, root=ROOT, icon=ICON, name=NAME, folder="Desktop"):
    """The PowerShell lines that write the shortcut in folder and print its path. folder is "Desktop" or
    "Programs" (the Start menu), wherever Windows keeps them (OneDrive too), or a folder's path."""
    script = os.path.join(root, "run_gui.py")
    return "; ".join([
        f"$folder = {_where(folder)}",
        f"$link = Join-Path $folder {_ps(name + '.lnk')}",
        "$s = (New-Object -ComObject WScript.Shell).CreateShortcut($link)",
        f"$s.TargetPath = {_ps(pythonw)}",
        f"$s.Arguments = {_ps(chr(34) + script + chr(34))}",
        f"$s.WorkingDirectory = {_ps(root)}",
        f"$s.IconLocation = {_ps(icon + ',0')}",
        f"$s.Description = {_ps('LSR MMTO: multi-material topology optimization')}",
        "$s.Save()",
        "Write-Output $link",
    ])


def problems():
    """Why the shortcut cannot be made from here, or an empty list."""
    out = []
    if os.name != "nt":
        out.append("the shortcut is for Windows")
    for module in ("PyQt5", "pyvista", "torch"):
        try:
            __import__(module)
        except ImportError:
            out.append(f"{module} is missing: run this in the lsr-stress-mmto-ad environment "
                       f"(conda activate lsr-stress-mmto-ad)")
    if os.name == "nt" and not os.path.isfile(os.path.join(sys.prefix, "pythonw.exe")):
        out.append(f"no pythonw.exe in {sys.prefix}")
    if not os.path.isfile(ICON):
        out.append(f"the icon is missing: {ICON}")
    return out


def make(folder):
    """The shortcut in folder ("Desktop" or "Programs"): its path, or raises RuntimeError with the reason."""
    done = subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command",
                           powershell(os.path.join(sys.prefix, "pythonw.exe"), folder=folder)],
                          capture_output=True, text=True)
    if done.returncode != 0:
        raise RuntimeError(done.stderr.strip() or "PowerShell failed")
    return done.stdout.strip()


def base_python(environ=None, prefix=None):
    """The python.exe of the conda installation this environment belongs to, which runs the uninstaller
    (the environment cannot remove itself while it runs): the one beside CONDA_EXE's Scripts folder, or
    else the one in the folder that holds envs. RuntimeError when neither is there."""
    environ = os.environ if environ is None else environ
    prefix = sys.prefix if prefix is None else prefix
    bases = [os.path.dirname(os.path.dirname(environ["CONDA_EXE"]))] if environ.get("CONDA_EXE") else []
    if os.path.basename(os.path.dirname(os.path.normpath(prefix))).lower() == "envs":
        bases.append(os.path.dirname(os.path.dirname(os.path.normpath(prefix))))
    for base in bases:
        if os.path.isfile(os.path.join(base, "python.exe")):
            return os.path.join(base, "python.exe")
    raise RuntimeError("the conda installation's own python.exe was not found")


def uninstall_values(python, root=ROOT, icon=ICON):
    """The values of the uninstall entry Windows lists under Settings > Apps."""
    return {"DisplayName": NAME, "DisplayIcon": icon, "InstallLocation": root, "NoModify": 1, "NoRepair": 1,
            "UninstallString": f'"{python}" "{os.path.join(root, os.path.basename(UNINSTALLER))}"'}


def write_uninstall_entry(python):
    """The uninstall entry written under HKEY_CURRENT_USER, over the old one if there is one."""
    import winreg
    with winreg.CreateKey(winreg.HKEY_CURRENT_USER, UNINSTALL_KEY) as key:
        for name, value in uninstall_values(python).items():
            winreg.SetValueEx(key, name, 0, winreg.REG_DWORD if isinstance(value, int) else winreg.REG_SZ, value)


if __name__ == "__main__":
    found = problems()
    if found:
        print("No shortcut made:\n  " + "\n  ".join(found))
        sys.exit(1)
    folders = ["Programs"] + ([] if "--no-desktop" in sys.argv[1:] else ["Desktop"])
    try:
        for folder in folders:
            print(f"Made {make(folder)}")
    except RuntimeError as e:
        print(f"No shortcut made:\n{e}")
        sys.exit(1)
    try:
        write_uninstall_entry(base_python())
        print(f"{NAME} is listed in Settings > Apps, where Uninstall removes it.")
    except (RuntimeError, OSError) as e:
        print(f"{NAME} could not be listed in Settings > Apps: {e}")
    print(f"Open {NAME} from the Start menu (type LSR)" + (" or the desktop icon." if "Desktop" in folders else "."))
