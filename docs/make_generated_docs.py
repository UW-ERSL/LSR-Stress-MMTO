"""
make_generated_docs.py
======================
Writes every generated document in one go, so a change to the code needs one command, not three:

    docs/reference.md               python docs/make_reference.py
    docs/user_guide.md (its lists)  python -m gui.user_guide_window --write
    docs/testing_the_codebase.md    python docs/make_testing_doc.py

    python docs/make_generated_docs.py        (from the repository folder)

Each runs as its own program, exactly as its command above, and the first that fails stops the rest. The user
guide's lists need Qt; with no screen (QT_QPA_PLATFORM unset on a computer without one) Qt draws off-screen.
"""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

COMMANDS = [
    ["docs/make_reference.py"],
    ["-m", "gui.user_guide_window", "--write"],
    ["docs/make_testing_doc.py"],
]


if __name__ == "__main__":
    env = dict(os.environ)
    if sys.platform.startswith("linux") and not env.get("DISPLAY"):
        env.setdefault("QT_QPA_PLATFORM", "offscreen")
    for command in COMMANDS:
        subprocess.run([sys.executable, *command], cwd=ROOT, env=env, check=True)
