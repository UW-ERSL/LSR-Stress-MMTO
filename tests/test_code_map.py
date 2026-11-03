"""
test_code_map.py
================
docs/code_map.md names files and folders; each one exists, so the map cannot drift from the code
when something is renamed or moved. A name with <name> in it must exist for the structural physics
and for the template.

    python tests/test_code_map.py        (or: python -m pytest tests)
"""
import os
import re
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _names():
    text = open(os.path.join(ROOT, "docs", "code_map.md"), encoding="utf-8").read()
    return set(re.findall(r"[\w<>./-]*\w\.(?:py|md|qss)\b|[\w<>./-]+/", text)) - {"../", "./"}


def test_every_name_in_the_code_map_exists():
    files = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True).stdout.split()
    folders = {"/".join(f.split("/")[:i]) for f in files for i in range(1, f.count("/") + 1)}
    basenames = {os.path.basename(f) for f in files}
    missing = []
    for name in _names():
        variants = [name.replace("<name>", p) for p in ("structural", "template")] if "<name>" in name else [name]
        for n in variants:
            n = n[3:] if n.startswith("../") else n
            if n.endswith("/"):
                ok = any(f == n[:-1] or f.endswith("/" + n[:-1]) for f in folders)
            elif "/" in n:
                ok = any(f == n or f.endswith("/" + n) for f in files)
            else:
                ok = n in basenames
            if not ok:
                missing.append(n)
    assert not missing, f"docs/code_map.md names what does not exist: {sorted(set(missing))}"


if __name__ == "__main__":
    test_every_name_in_the_code_map_exists()
    print("ok")
