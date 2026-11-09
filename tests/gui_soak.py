"""Soak test: drives the real GUI in a loop for hours, to find what piles up or slows down with use.

Not part of the test suite (pytest never collects this file). Run it from the repository folder:

    python tests/gui_soak.py --hours 6        until 6 hours are up (the cycle under way is finished)
    python tests/gui_soak.py --cycles 3       3 cycles, a few minutes; a check that it works
    python tests/gui_soak.py --small ...      every project on a small mesh: more cycles per hour

One cycle, everything taken from the app's own lists, so new projects, side buttons and Results buttons are
covered without editing this file:
    the next project of projects/ (in turn), copied into soak_results and saved there as a project of its own;
    every side button, its window then closed with Cancel (looking changes nothing);
    a 3-iteration run with the Run button; Results on the new run, the 3D view turned a full circle, and every
    Results button pressed but those in SKIP; Analyse; Export STL of the run and of the newest analysis; the
    windows closed, the copy deleted.
Any question or dialog that appears is answered within a second: Cancel while looking, its default button while
acting (with any "Save..." box ticked: acting keeps what it makes); a file dialog is always cancelled. The crash report (crash_report.log) is on throughout.

Reading soak_results/soak_<date>_<time>.csv (one row per cycle; open it in Excel and plot a column):
    memory_mb, handles, gdi, user, os_threads   should level off after the first few cycles; a steady climb over
                                                many cycles is a leak (gdi and user: Windows' drawing objects,
                                                at most 10,000 for a program; a climb towards that ends in
                                                windows that stop drawing or responding)
    windows, objects_3d                         should be the same every cycle; a climb means something is never
                                                closed or never removed from the 3D view
    s_<step>                                    each step's seconds; compare rows of the same project (it cycles
                                                through the projects): a step that keeps getting slower means
                                                something piles up
    exports                                     what each Export STL gave: files, or refused (a design with no
                                                material yet, as after only a few iterations)
    pop_ups                                     the questions and dialogs answered in the cycle
    status                                      ok; or why the soak stopped: hung in a step (longer than
                                                --step-limit), or memory over half the computer's RAM
"""
import argparse
import csv
import datetime
import faulthandler
import glob
import os
import platform
import shutil
import sys
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

OUT = os.path.join(ROOT, "soak_results")            # git-ignored; one level below the repository, like projects/,
WORK = "work_"                                      # so the copied projects' paths are read the same
ITERATIONS = 3
SMALL_N_DOF = 3000
TURN_STEP = 10                                      # degrees per frame of the 3D view's full turn
SKIP = {"Open folder": "opens File Explorer",       # Results buttons not pressed, with the reason
        "Open results folder": "opens File Explorer",
        "Delete this run": "removes the run the cycle goes on looking at"}
STEPS = ("open", "side_buttons", "run", "results", "analyse", "export_stl", "close")
COLUMNS = ("cycle", "time", "project", "memory_mb", "handles", "gdi", "user", "os_threads", "python_threads",
           "windows", "objects_3d") + tuple(f"s_{s}" for s in STEPS) + ("exports", "pop_ups", "status")


# ---------------------------------------------------------------------------------------- what is measured
def measures():
    """Memory in use (MB), open handles, Windows' GDI and USER objects, operating-system threads, Python threads.
    Windows' own calls on Windows; /proc on Linux, where gdi and user stay blank."""
    m = dict(memory_mb="", handles="", gdi="", user="", os_threads="", python_threads=threading.active_count())
    if platform.system() == "Windows":
        import ctypes
        from ctypes import wintypes
        k32, psapi, user32 = ctypes.windll.kernel32, ctypes.windll.psapi, ctypes.windll.user32
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        me = k32.GetCurrentProcess()

        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + \
                       [(n, ctypes.c_size_t) for n in ("PeakWorkingSetSize", "WorkingSetSize",
                        "QuotaPeakPagedPoolUsage", "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage",
                        "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage", "PrivateUsage")]
        c = Counters()
        c.cb = ctypes.sizeof(c)
        if psapi.GetProcessMemoryInfo(me, ctypes.byref(c), c.cb):
            m["memory_mb"] = round(c.PrivateUsage / 2 ** 20, 1)
        n = wintypes.DWORD()
        if k32.GetProcessHandleCount(me, ctypes.byref(n)):
            m["handles"] = n.value
        m["gdi"], m["user"] = user32.GetGuiResources(me, 0), user32.GetGuiResources(me, 1)

        class Entry(ctypes.Structure):
            _fields_ = [("dwSize", wintypes.DWORD), ("cntUsage", wintypes.DWORD), ("th32ThreadID", wintypes.DWORD),
                        ("th32OwnerProcessID", wintypes.DWORD), ("tpBasePri", ctypes.c_long),
                        ("tpDeltaPri", ctypes.c_long), ("dwFlags", wintypes.DWORD)]
        k32.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
        snap = k32.CreateToolhelp32Snapshot(4, 0)                   # TH32CS_SNAPTHREAD
        e, pid, count = Entry(), os.getpid(), 0
        e.dwSize = ctypes.sizeof(e)
        ok = k32.Thread32First(snap, ctypes.byref(e))
        while ok:
            count += e.th32OwnerProcessID == pid
            ok = k32.Thread32Next(snap, ctypes.byref(e))
        k32.CloseHandle(snap)
        m["os_threads"] = count
    elif os.path.exists("/proc/self/status"):
        with open("/proc/self/status") as f:
            status = dict(line.split(":", 1) for line in f if ":" in line)
        m["memory_mb"] = round(int(status["VmRSS"].split()[0]) / 1024, 1)
        m["os_threads"] = int(status["Threads"])
        m["handles"] = len(os.listdir("/proc/self/fd"))
    return m


def ram_mb():
    """The computer's memory, in MB."""
    if platform.system() == "Windows":
        import ctypes

        class Status(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong)] + \
                       [(n, ctypes.c_ulonglong) for n in ("ullAvailPhys", "ullTotalPageFile", "ullAvailPageFile",
                                                          "ullTotalVirtual", "ullAvailVirtual",
                                                          "ullAvailExtendedVirtual")]
        s = Status()
        s.dwLength = ctypes.sizeof(s)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s))
        return s.ullTotalPhys / 2 ** 20
    return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2 ** 20


def projects():
    return sorted(glob.glob(os.path.join(ROOT, "projects", "*.mmto")))


# ---------------------------------------------------------------------------------------- the soak
class Soak:
    def __init__(self, app, window, args, csv_path):
        from PyQt5 import QtCore
        self.app, self.w, self.args = app, window, args
        self.mode = "act"                                       # "look": pop-ups get Cancel
        self.pop_ups = []
        self.step, self.since = None, time.monotonic()
        self.lock = threading.Lock()
        self.csv_path = csv_path
        self.row = {}
        self.stopped = None
        self.answer = QtCore.QTimer()                           # pop-ups answered within a second
        self.answer.timeout.connect(self.answer_pop_up)
        self.answer.start(300)
        self._seen = {}
        threading.Thread(target=self.watch, name="soak watchdog", daemon=True).start()

    # pop-ups ------------------------------------------------------------------------------------------
    def answer_pop_up(self):
        from PyQt5 import QtWidgets
        d = self.app.activeModalWidget()
        if d is None or not d.isVisible():
            return
        first = self._seen.setdefault(id(d), time.monotonic())
        if time.monotonic() - first < 0.5:                      # drawn, as a user would see it
            return
        self._seen.pop(id(d), None)
        title = d.windowTitle() or type(d).__name__
        cancel = self.mode == "look" or isinstance(d, QtWidgets.QFileDialog)
        self.pop_ups.append(("cancelled " if cancel else "answered ") + title)
        if isinstance(d, QtWidgets.QMessageBox):
            button = d.escapeButton() if cancel else (d.defaultButton() or None)
            if button is not None:
                button.click()
                return
            if not cancel and d.buttons():
                d.buttons()[0].click()
                return
        if cancel:
            d.reject()
        else:
            for box in d.findChildren(QtWidgets.QCheckBox):         # acting keeps what it makes: Save... ticked
                if box.text().startswith("Save") and box.isEnabled():
                    box.setChecked(True)
            default = next((b for b in d.findChildren(QtWidgets.QPushButton) if b.isDefault() and b.isEnabled()),
                           None)
            default.click() if default is not None else d.accept()

    # waiting --------------------------------------------------------------------------------------------
    def wait(self, seconds):
        from PyQt5 import QtTest
        QtTest.QTest.qWait(int(seconds * 1000))

    def wait_until(self, done):
        while not done():
            self.wait(0.2)

    def busy(self):
        return self.w.run_going_now() or getattr(self.w, "exporting", False)

    # steps ------------------------------------------------------------------------------------------------
    def timed(self, name, work):
        with self.lock:
            self.step, self.since = name, time.monotonic()
        work()
        self.wait_until(lambda: not self.busy())
        with self.lock:
            self.row[f"s_{name}"] = round(time.monotonic() - self.since, 2)
            self.step = None

    def open(self, cycle, path):
        work = os.path.join(OUT, f"{WORK}{cycle}")
        copy = shutil.copy(path, OUT)                           # one level below the repository, like projects/
        self.mode = "act"
        assert self.w.open_project(copy), f"{path} did not open"
        p = self.w.project
        p.settings["common"]["maxIterations"] = ITERATIONS
        if self.args.small:
            p.n_dof = SMALL_N_DOF
        assert self.w.save_project_as(os.path.join(work, "P", "P.mmto")), "the copy could not be saved"
        os.remove(copy)

    def side_buttons(self):
        from gui.main_window import SIDEBAR
        self.mode = "look"
        for name in SIDEBAR:
            button = self.w.sidebar_buttons.get(name)
            if button is not None and button.isEnabled():
                button.click()
                self.wait(0.5)
                self.close_windows()
        self.mode = "act"

    def run(self):
        self.w.run_button.click()
        self.wait_until(lambda: not self.busy())
        self.run_folder = self.w.run_window.save_dir                # (Analyse later starts afresh)

    def results(self):
        from PyQt5 import QtWidgets
        res = self.w.open_results_panel(self.run_folder)
        self.wait(0.5)
        camera = self.w.plotter.camera
        for _ in range(360 // TURN_STEP):
            camera.Azimuth(TURN_STEP)
            self.w.plotter.render()
            self.wait(0.02)
        pressed = set()
        while True:                                             # every Results button, each once
            left = [b for b in res.findChildren(QtWidgets.QPushButton)
                    if b.isVisible() and b.isEnabled() and b.text() and b.text() not in SKIP
                    and b.text() not in pressed]
            if not left:
                break
            pressed.add(left[0].text())
            left[0].click()
            self.wait(0.3)
            self.wait_until(lambda: not self.busy())

    def analyse(self):
        self.w.analyse_button.click()

    def export_stl(self):
        """Export STL of the cycle's run (after a few iterations its design may hold no material: the message
        then) and of the newest analysis (the whole part: STL files)."""
        from lsr_mmto.problem.project_folder import runs
        run = self.run_folder
        newest = runs(self.w.project)[0]
        outcomes = []
        for name, folder in (("run", run), ("analysis", newest)):
            if name == "analysis" and folder == run:
                outcomes.append("analysis: none saved")
                continue
            self.w.open_results_panel(folder)
            self.wait(0.5)
            if not self.w.export_button.isEnabled():
                outcomes.append(f"{name}: Export STL not available")
                continue
            self.w.exported_last = None
            self.w.export_button.click()
            self.wait(0.3)
            self.wait_until(lambda: not self.busy())
            last = getattr(self.w, "exported_last", None)
            outcomes.append(f"{name}: {1 + len(last[1].pieces)} files" if last and last[0] == folder
                            else f"{name}: refused")
        self.row["exports"] = "; ".join(outcomes)

    def close_windows(self):
        from gui.main_window import Strip
        for strip in self.w.findChildren(Strip):
            if strip.isVisible():
                strip.close()
        for top in self.app.topLevelWidgets():
            if top is not self.w and top.isVisible() and top.isWindow():
                top.close()
        self.wait(0.2)

    # one cycle --------------------------------------------------------------------------------------------
    def cycle(self, k):
        path = projects()[k % len(projects())]
        self.row = dict(cycle=k + 1, time=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        project=os.path.basename(path)[:-5])
        self.pop_ups = []
        self.timed("open", lambda: self.open(k, path))
        self.timed("side_buttons", self.side_buttons)
        self.timed("run", self.run)
        self.timed("results", self.results)
        self.timed("analyse", self.analyse)
        self.timed("export_stl", self.export_stl)
        self.timed("close", self.close_windows)
        self.row.update(measures())
        self.row["windows"] = sum(1 for t in self.app.topLevelWidgets() if t.isVisible())
        self.row["objects_3d"] = len(self.w.plotter.actors)
        self.row["pop_ups"] = "; ".join(self.pop_ups)
        self.row["status"] = "ok"
        if isinstance(self.row["memory_mb"], float) and self.row["memory_mb"] > ram_mb() / 2:
            self.row["status"] = self.stopped = "stopped: memory over half the computer's RAM"
        self.write(self.row)
        for old in glob.glob(os.path.join(OUT, f"{WORK}*")):   # earlier cycles' copies (this one stays open)
            if old != os.path.join(OUT, f"{WORK}{k}"):
                shutil.rmtree(old, ignore_errors=True)

    def write(self, row):
        new = not os.path.exists(self.csv_path)
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            out = csv.DictWriter(f, COLUMNS)
            if new:
                out.writeheader()
            out.writerow({c: row.get(c, "") for c in COLUMNS})

    # the watchdog --------------------------------------------------------------------------------------------
    def watch(self):
        """A step longer than --step-limit: a last row, every thread's stack in the crash report, and the end."""
        from gui import crash_report
        while True:
            time.sleep(5)
            with self.lock:
                step, since = self.step, self.since
                if step is None or time.monotonic() - since < self.args.step_limit * 60:
                    continue
                row = dict(self.row, status=f"hung in {step} (over {self.args.step_limit} min)",
                           pop_ups="; ".join(self.pop_ups))
                self.write(row)
            crash_report.write(f"Soak test: hung in {step}; every thread's stack follows")
            if crash_report.started():
                faulthandler.dump_traceback(file=crash_report._state["file"], all_threads=True)
            print(f"Stopped: hung in {step}. See {self.csv_path} and the crash report.", flush=True)
            os._exit(2)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    how_long = parser.add_mutually_exclusive_group(required=True)
    how_long.add_argument("--hours", type=float, help="run cycles until this many hours are up")
    how_long.add_argument("--cycles", type=int, help="run this many cycles")
    parser.add_argument("--small", action="store_true", help=f"every project on a small mesh ({SMALL_N_DOF})")
    parser.add_argument("--step-limit", type=float, default=20, help="minutes a step may take (default 20)")
    args = parser.parse_args(argv)

    import torch  # noqa: F401 (before PyQt5, as run_gui.py)
    from PyQt5 import QtWidgets
    from PyQt5.QtGui import QIcon
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    from gui import crash_report
    from gui import main_window as G
    G.apply_style(app)
    app.setWindowIcon(QIcon(G.ICON))
    crash_report.start()
    os.makedirs(OUT, exist_ok=True)
    for old in glob.glob(os.path.join(OUT, f"{WORK}*")) + glob.glob(os.path.join(OUT, "*.mmto")):
        shutil.rmtree(old, ignore_errors=True) if os.path.isdir(old) else os.remove(old)   # a killed soak's
    w = G.MMTOWindow()
    G.install_error_handler(w)
    w.show()
    csv_path = os.path.join(OUT, f"soak_{datetime.datetime.now():%Y-%m-%d_%H-%M-%S}.csv")
    soak = Soak(app, w, args, csv_path)
    crash_report.write(f"Soak test started: {' '.join(sys.argv[1:])}, writing {csv_path}")
    print(f"Soak test: writing {csv_path}", flush=True)
    end = time.monotonic() + (args.hours or 0) * 3600
    k = 0
    while (k < args.cycles) if args.cycles else (time.monotonic() < end):
        try:
            soak.cycle(k)
        except Exception as e:                                      # noqa: BLE001 (the row says why, then the end)
            import traceback
            crash_report.write("Soak test error:\n" + traceback.format_exc().rstrip())
            soak.stopped = f"error in {soak.step}: {type(e).__name__}: {e}"
            soak.write(dict(soak.row, status=soak.stopped, pop_ups="; ".join(soak.pop_ups)))
            print(f"Stopped: {soak.stopped}", flush=True)
            break
        print(f"cycle {k + 1}: {soak.row['project']}, {soak.row['memory_mb']} MB, {soak.row['status']}", flush=True)
        k += 1
        if soak.stopped:
            break
    crash_report.write(f"Soak test ended after {k} cycles")
    soak.answer.stop()
    w.close()
    crash_report.stop()
    for old in glob.glob(os.path.join(OUT, f"{WORK}*")):           # the last cycle's copy
        shutil.rmtree(old, ignore_errors=True)
    print(f"Done: {k} cycles in {csv_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
