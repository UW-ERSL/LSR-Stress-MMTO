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
"""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
NAME = "LSR MMTO"
ICON = os.path.join(ROOT, "gui", "assets", "lsr_mmto.ico")


def _ps(text):
    """A PowerShell string in single quotes (a quote inside is doubled)."""
    return "'" + str(text).replace("'", "''") + "'"


def powershell(pythonw, root=ROOT, icon=ICON, name=NAME, folder="Desktop"):
    """The PowerShell lines that write the shortcut in folder and print its path. folder is "Desktop" or
    "Programs" (the Start menu), wherever Windows keeps them (OneDrive too), or a folder's path."""
    script = os.path.join(root, "run_gui.py")
    where = (f"[Environment]::GetFolderPath({_ps(folder)})" if folder in ("Desktop", "Programs") else _ps(folder))
    return "; ".join([
        f"$folder = {where}",
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
    print(f"Open {NAME} from the Start menu (type LSR)" + (" or the desktop icon." if "Desktop" in folders else "."))
