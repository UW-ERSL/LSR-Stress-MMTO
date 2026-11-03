"""
lsr_mmto
========
LSR MMTO: multi-material topology optimization with a learned material space. A project is set
up and run with:

    from lsr_mmto import Project, run, face, box
    p = Project.load("my project.mmto")
    run(p)

    problem/      a project: the part, its selections, supports and loads, settings, run(p)
    materials/    material databases and the material network
    elements/     the finite element: strain matrix and Lame parameters
    design/       density filter and projection, latent space to properties, latent penalties
    formulation/  the quantities an objective or constraint can name, and the formula text
    physics/      the physics a project may name, one folder each (structural/)
    optimize/     the optimization loop, its history, the final design analysis, saving a run
    results/      a saved run read back: tables, figures, 3D views, STL export
    pyto_ad_imports.py   every name taken from PyTO-AD
"""
from lsr_mmto.problem.project import Project, run
from lsr_mmto.problem.selections import (box, cylinder, difference, face, function, halfspaces, ids, inside_stl,
                                         intersect, near_face, nearest, plane, rows, selection, sphere, stl_faces,
                                         union)
