"""tests/gui_soak.py, the soak test, without soaking (a few seconds): pytest never collects it, it drives the
app's own lists, and its measurements give numbers here."""
import configparser
import fnmatch
import glob
import importlib.util
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def soak():
    """tests/gui_soak.py, loaded from its file (tests/ is not a package)."""
    spec = importlib.util.spec_from_file_location("gui_soak", os.path.join(ROOT, "tests", "gui_soak.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_pytest_never_collects_the_soak_test():
    ini = configparser.ConfigParser()
    ini.read(os.path.join(ROOT, "pytest.ini"))
    patterns = ini.get("pytest", "python_files", fallback="test_*.py *_test.py").split()
    assert not any(fnmatch.fnmatch("gui_soak.py", p) for p in patterns), patterns


def test_the_soak_test_uses_the_apps_own_lists():
    gui_soak = soak()
    assert gui_soak.projects() == sorted(glob.glob(os.path.join(ROOT, "projects", "*.mmto")))
    assert len(gui_soak.projects()) >= 1
    assert set(gui_soak.STEPS) <= {c[2:] for c in gui_soak.COLUMNS if c.startswith("s_")}


def test_the_soak_tests_measurements_give_numbers():
    gui_soak = soak()
    m = gui_soak.measures()
    assert m["memory_mb"] > 0 and m["os_threads"] >= 1 and m["python_threads"] >= 1, m
    assert gui_soak.ram_mb() > m["memory_mb"]
