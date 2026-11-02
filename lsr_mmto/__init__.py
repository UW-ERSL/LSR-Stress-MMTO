"""
lsr_mmto
========
LSR MMTO: multi-material topology optimization with a learned material space. A project is set
up and run with:

    from lsr_mmto import Project, run, face, box
    p = Project.load("my project.mmto")
    run(p)

    problem/     a project: the part, its regions, supports and loads, settings, run(p)
    materials/   material databases and the material network
    physics/     one iteration's physics: solve, stress, objective, gradients
    optimize/    the optimization loop, its history, the final check, saving a run
    results/     a saved run read back: tables, figures, 3D views, STL export
    pyto_ad.py   every name taken from PyTO-AD
"""
from lsr_mmto.problem.project import Project, run
from lsr_mmto.problem.selections import (box, cylinder, difference, face, function, halfspaces, ids, inside_stl,
                                         intersect, near_face, nearest, plane, rows, selection, sphere, stl_faces,
                                         union)
