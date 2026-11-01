"""
run_gui.py
==========
Start the LSR MMTO GUI, from the repository folder, in the lsr-stress-mmto-ad environment:

    python run_gui.py                       an empty window: File > New or File > Open
    python run_gui.py projects/beam.mmto    with a project open
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import project_paths  # noqa: E402

project_paths.bootstrap()

from gui.main_window import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
