"""
test_launcher.py
================
The desktop start: the icon loads at every size Windows asks for, the start-up picture shows, a start
with no console writes what the app prints to the log file, the shortcut is written with every path
quoted for the desktop or the Start menu (on Windows, a real shortcut in a scratch folder), and the
setup's one question takes Enter as yes.

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


def test_run_record_starts_git_without_a_window():
    """A run looks up the code's git commit for its record: with no console window on Windows (started from
    the shortcut, each git call otherwise flashes a black window up)."""
    import subprocess
    sys.path.insert(0, ROOT)
    from lsr_mmto.optimize import save as run_save
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
