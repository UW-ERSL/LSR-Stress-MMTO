"""
test_launcher.py
================
The desktop start: the icon loads at every size Windows asks for, the start-up picture shows, a start
with no console writes what the app prints to the log file, the shortcut is written with every path
quoted for the desktop or the Start menu (on Windows, a real shortcut in a scratch folder), the
setup's one question takes Enter as yes, and Settings > Apps lists LSR MMTO, whose Uninstall removes
exactly what it lists (with stand-ins for the registry, the commands and the folders).

    python -m pytest tests/test_launcher.py
"""
import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import make_shortcut as M  # noqa: E402


def test_icon_sizes():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt5 import QtGui, QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])  # noqa: F841
    icon = QtGui.QIcon(M.ICON)
    assert not icon.isNull()
    sizes = {s.width() for s in icon.availableSizes()}
    assert {16, 24, 32, 48, 256} <= sizes, sizes
    assert not QtGui.QPixmap(os.path.join(ROOT, "gui", "assets", "lsr_mmto.png")).isNull()


def test_splash_screen():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    import run_gui
    splash = run_gui.splash_screen()
    assert splash.isVisible() and not splash.pixmap().isNull()
    splash.close()


def test_start_without_console_writes_the_log():
    with tempfile.TemporaryDirectory() as d:
        code = ("import sys; sys.stdout = sys.stderr = None; sys.path.insert(0, %r); import run_gui; "
                "assert run_gui.without_console(); print('hello from the app')" % ROOT)
        env = dict(os.environ, LOCALAPPDATA=d)
        done = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=60)
        assert done.returncode == 0, done.stderr
        log = os.path.join(d, "LSR MMTO", "gui.log")
        assert open(log, encoding="utf-8").read().strip() == "hello from the app"


def test_torch_loads_before_qt():
    """run_gui.py imports torch before anything from PyQt5: the other way round, torch's c10.dll fails to
    load on Windows (WinError 1114). The start is stopped as soon as PyQt5 is imported."""
    code = ("import builtins, runpy, sys\n"
            "seen, real = [], builtins.__import__\n"
            "def imp(name, *a, **k):\n"
            "    top = name.split('.')[0]\n"
            "    if top in ('torch', 'PyQt5') and top not in seen:\n"
            "        seen.append(top)\n"
            "        if top == 'PyQt5':\n"
            "            print(','.join(seen)); sys.stdout.flush(); raise SystemExit(0)\n"
            "    return real(name, *a, **k)\n"
            "builtins.__import__ = imp\n"
            "runpy.run_path(%r, run_name='__main__')\n" % os.path.join(ROOT, "run_gui.py"))
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen")
    done = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "torch,PyQt5", done.stdout


def test_shortcut_lines_quote_every_path():
    lines = M.powershell(r"C:\Users\o'brien\miniforge3\envs\lsr\pythonw.exe", root=r"C:\LSR\my repo",
                         icon=r"C:\LSR\my repo\gui\assets\lsr_mmto.ico")
    assert r"$s.TargetPath = 'C:\Users\o''brien\miniforge3\envs\lsr\pythonw.exe'" in lines
    script = os.path.join(r"C:\LSR\my repo", "run_gui.py")             # a backslash on Windows
    assert f"""$s.Arguments = '"{script}"'""" in lines
    assert r"$s.IconLocation = 'C:\LSR\my repo\gui\assets\lsr_mmto.ico,0'" in lines
    assert "GetFolderPath('Desktop')" in lines
    if os.name != "nt":
        assert M.problems() == ["the shortcut is for Windows"]


def test_start_menu_and_the_question():
    assert "GetFolderPath('Programs')" in M.powershell("pythonw.exe", folder="Programs")
    import setup_environment as S

    def answered(text):
        def read(prompt):
            if text is None:
                raise EOFError
            return text
        return read
    assert S.ask_desktop(answered("")) and S.ask_desktop(answered("Y")) and S.ask_desktop(answered(None))
    assert not S.ask_desktop(answered("n")) and not S.ask_desktop(answered(" No "))


def test_setup_step_lines_count_the_real_steps(capsys):
    """The install prints [n/N] with N the number of steps it really goes through: 6 on Windows (the
    Start menu step), 5 elsewhere, and main() calls step() once for each of them."""
    import inspect
    import re
    import setup_environment as S
    for windows, total in ((True, 6), (False, 5)):
        all_steps = S.steps(windows, desktop=True)
        assert len(all_steps) == total
        for n in range(1, total + 1):
            S.step(n, all_steps)
        lines = re.findall(r"^\[(\d+)/(\d+)\]", capsys.readouterr().out, re.M)
        assert lines == [(str(n), str(total)) for n in range(1, total + 1)]
    called = sorted(int(n) for n in re.findall(r"\bstep\((\d+), all_steps\)", inspect.getsource(S.main)))
    assert called == list(range(1, len(S.steps(True, True)) + 1))


def test_run_record_starts_git_without_a_window():
    """A run looks up the code's git commit for its record: with no console window on Windows (started from
    the shortcut, each git call otherwise flashes a black window up)."""
    import subprocess
    sys.path.insert(0, ROOT)
    from lsr_mmto.optimize import save_results as run_save
    seen = []
    real = subprocess.run

    def run(cmd, **kw):
        seen.append(kw.get("creationflags"))
        return real(cmd, **kw)
    subprocess.run = run
    try:
        run_save.git_state(ROOT)
    finally:
        subprocess.run = real
    assert seen and all(f == getattr(subprocess, "CREATE_NO_WINDOW", 0) for f in seen)
    if os.name == "nt":
        assert seen[0] == 0x08000000


def test_shortcut_made_on_windows():
    if os.name != "nt":
        return
    assert M.problems() == []
    with tempfile.TemporaryDirectory() as d:
        lines = M.powershell(os.path.join(sys.prefix, "pythonw.exe"), folder=d)
        done = subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", lines],
                              capture_output=True, text=True, timeout=120)
        assert done.returncode == 0, done.stderr
        assert os.path.isfile(os.path.join(d, "LSR MMTO.lnk"))


# ---------------------------------------------------------------------- Settings > Apps, and Uninstall
class FakeRegistry:
    """A stand-in for winreg, holding keys under HKEY_CURRENT_USER as {key: {name: (value, type)}}."""
    HKEY_CURRENT_USER, REG_SZ, REG_DWORD = "HKCU", 1, 4

    def __init__(self):
        self.keys = {}

    def CreateKey(self, root, key):
        values = self.keys.setdefault((root, key), {})

        class Key:
            def __enter__(self):
                return values

            def __exit__(self, *exc):
                return False
        return Key()

    def SetValueEx(self, values, name, reserved, kind, value):
        values[name] = (value, kind)

    def DeleteKey(self, root, key):
        del self.keys[(root, key)]


def test_uninstall_entry(monkeypatch):
    """The entry Settings > Apps lists: its values, the uninstaller run by the conda installation's own
    python (found from CONDA_EXE, or as the folder holding envs); written twice, still one entry."""
    reg = FakeRegistry()
    monkeypatch.setitem(sys.modules, "winreg", reg)
    with tempfile.TemporaryDirectory() as d:
        base = os.path.join(d, "miniforge3")
        os.makedirs(os.path.join(base, "Scripts"))
        open(os.path.join(base, "python.exe"), "w").close()
        env = os.path.join(base, "envs", "lsr-stress-mmto-ad")
        python = os.path.join(base, "python.exe")
        assert M.base_python({"CONDA_EXE": os.path.join(base, "Scripts", "conda.exe")}, "elsewhere") == python
        assert M.base_python({}, env) == python
        try:
            M.base_python({}, os.path.join(d, "venv"))
        except RuntimeError:
            pass
        else:
            raise AssertionError("a python.exe was found where there is none")
        M.write_uninstall_entry(python)
        M.write_uninstall_entry(python)
    assert list(reg.keys) == [("HKCU", r"Software\Microsoft\Windows\CurrentVersion\Uninstall\LSR MMTO")]
    values = reg.keys[("HKCU", M.UNINSTALL_KEY)]
    assert values == {"DisplayName": ("LSR MMTO", 1), "DisplayIcon": (M.ICON, 1), "InstallLocation": (ROOT, 1),
                      "NoModify": (1, 4), "NoRepair": (1, 4),
                      "UninstallString": (f'"{python}" "{os.path.join(ROOT, "uninstall_lsr_mmto.py")}"', 1)}


def _computer(d, monkeypatch):
    """A stand-in computer in folder d: shortcuts, kernels and settings among others that must stay, a code
    folder, a conda installation, the uninstall entry, and every command recorded instead of run."""
    import types
    import uninstall_lsr_mmto as U
    reg = FakeRegistry()
    monkeypatch.setitem(sys.modules, "winreg", reg)
    reg.keys[("HKCU", M.UNINSTALL_KEY)] = {"DisplayName": ("LSR MMTO", 1)}
    folders = {name: os.path.join(d, name) for name in ("Desktop", "Programs", "home", "appdata", "code", "base")}
    for f in folders.values():
        os.makedirs(f)
    files = [os.path.join(folders["Desktop"], "LSR MMTO.lnk"), os.path.join(folders["Desktop"], "Other.lnk"),
             os.path.join(folders["Programs"], "LSR MMTO.lnk"), os.path.join(folders["code"], "run_gui.py"),
             os.path.join(folders["appdata"], "jupyter", "kernels", "lsr-stress-mmto-ad", "kernel.json"),
             os.path.join(folders["appdata"], "jupyter", "kernels", "python3", "kernel.json"),
             os.path.join(folders["appdata"], "LSR MMTO", "LSR MMTO.ini"),
             os.path.join(folders["appdata"], "Other app", "settings.ini"),
             os.path.join(folders["base"], "Scripts", "conda.exe")]
    for f in files:
        os.makedirs(os.path.dirname(f), exist_ok=True)
        open(f, "w").close()
    ran = []

    def run(cmd, **kw):
        ran.append(list(cmd))
        out = ""
        if cmd[0] == "powershell":
            out = folders["Desktop"] if "'Desktop'" in cmd[-1] else folders["Programs"]
        elif cmd[1:3] == ["env", "list"]:
            out = f"base  {folders['base']}\nlsr-stress-mmto-ad  {folders['base']}/envs/lsr-stress-mmto-ad\n"
        return types.SimpleNamespace(returncode=0, stdout=out, stderr="")
    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(sys, "prefix", folders["base"])
    monkeypatch.setattr(U, "ROOT", folders["code"])
    monkeypatch.setenv("APPDATA", folders["appdata"])
    monkeypatch.setenv("USERPROFILE", folders["home"])
    monkeypatch.delenv("OneDrive", raising=False)
    monkeypatch.delenv("CONDA_DEFAULT_ENV", raising=False)
    return U, reg, files, ran


def test_uninstall_removes_exactly_what_it_lists(monkeypatch, capsys):
    """Answered no, nothing goes. Answered yes: the two shortcuts, the kernel, the GUI's settings, the
    environment (conda env remove) and, last, the entry go; everything else stays, the code folder
    untouched, and the folders kept are named."""
    with tempfile.TemporaryDirectory() as d:
        U, reg, files, ran = _computer(d, monkeypatch)
        assert U.main(answer=lambda prompt: "n") == 1
        assert all(os.path.isfile(f) for f in files) and reg.keys
        assert U.main(answer=lambda prompt: "y") == 0
        gone = {files[0], files[2], files[4], files[6]}
        assert [f for f in files if not os.path.exists(f)] == [f for f in files if f in gone]
        assert not os.path.exists(os.path.dirname(files[4])) and not os.path.exists(os.path.dirname(files[6]))
        assert os.listdir(os.path.join(d, "code")) == ["run_gui.py"]
        assert [c for c in ran if "remove" in c] == [[os.path.join(d, "base", "Scripts", "conda.exe"), "env", "remove",
                                                      "-n", "lsr-stress-mmto-ad", "-y"]]
        assert reg.keys == {}
    shown = capsys.readouterr().out
    assert "To be kept:" in shown and os.path.join(d, "code") in shown and "PyTO-AD" in shown


def test_uninstall_refuses_to_run_inside_the_environment(monkeypatch, capsys):
    with tempfile.TemporaryDirectory() as d:
        U, reg, files, ran = _computer(d, monkeypatch)
        monkeypatch.setattr(sys, "prefix", os.path.join(d, "base", "envs", "lsr-stress-mmto-ad"))
        assert U.main(answer=lambda prompt: "y") == 1
        assert all(os.path.isfile(f) for f in files) and reg.keys and ran == []
    assert "Settings > Apps" in capsys.readouterr().out


def test_no_other_way_to_remove():
    """Uninstall in Settings > Apps is the one way: no remove option in the setup or the shortcut maker."""
    for name in ("setup_environment.py", "make_shortcut.py", "run_gui.py"):
        with open(os.path.join(ROOT, name), encoding="utf-8") as fh:
            text = fh.read()
        assert "--remove" not in text and "--uninstall" not in text, name
