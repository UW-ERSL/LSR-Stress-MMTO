# LSR MMTO

Multi-material topology optimization in a learned material space, with a GUI that takes a part from its STL to the optimized design.

This is the `autodiff` branch, built on [PyTO-AD](https://github.com/UW-ERSL/PyTO-AD). The `main` branch holds older code and is not developed further.

Install below, then open LSR MMTO from the Start menu. Everything, from your part's STL to the optimized design, is done in the GUI.

## What LSR MMTO is

LSR MMTO designs parts made of several materials: it decides where material goes in the part and which material from a database goes where. The materials sit in a learned material space, so the optimization can move from one material to another smoothly. A problem is set up, run and exported in the LSR MMTO GUI, from the part's STL to the optimized design as STL, or in a short Python script.

Today it solves structural problems, with stiffness, mass, cost, volume and stress as the objective or the constraints. Thermal problems, and thermoelastic problems with temperature-dependent material properties, are in development.

## Installation

First, download the code: on this page, switch the branch menu (top left, it says `main`) to **autodiff**, then click the green **Code** button, then **Download ZIP**, and unzip it anywhere. With git instead: `git clone -b autodiff https://github.com/UW-ERSL/LSR-Stress-MMTO.git`.

If you already have conda (Miniforge, Miniconda or Anaconda), skip step 1. If you already have an environment that runs this code, you can keep using it; `requirements.txt` lists what it needs. To set everything up yourself instead, see [Setting up by hand](#setting-up-by-hand) further down.

1. **Install Miniforge.** Download the Windows installer, `Miniforge3-Windows-x86_64.exe`, from [github.com/conda-forge/miniforge](https://github.com/conda-forge/miniforge/releases/latest), run it, and accept the defaults.

2. **Open the Miniforge Prompt.** Start menu, type `Miniforge Prompt`, open it. With Miniconda or Anaconda it is the `Anaconda Prompt`.

3. **Run the setup.** In that prompt, type `python` and a space, **without pressing Enter**, then drag `setup_environment.py` into the window, and only then press Enter. It first asks one question, whether to put an LSR MMTO icon on the desktop (it always goes in the Start menu), and then runs on its own. It takes roughly 10 minutes and ends with `Done.`

   This also downloads [PyTO-AD](https://github.com/UW-ERSL/PyTO-AD), the differentiable topology optimization code this branch builds on (finite element analysis, filters, MMA, automatic differentiation), into a folder next to this one, unless it is already there. It creates its own environment, `lsr-stress-mmto-ad`, so it never touches the one `main` uses.

4. **Open LSR MMTO.** Start menu, type LSR, and open it, or use the desktop icon. The start page offers the four examples and New project from an STL.

## Your first project

1. **New project from an STL** on the start page, or File > New.
2. Work down the sidebar: **Units**, **Materials**, **Supports and loads**, **Formulation**, **Settings**.
3. **Analyse** to check the setup on the solid part, then **Run**.
4. Look at the design in **Results**.
5. **Export STL**.

Each panel's (i) button explains that panel, and Help > User guide covers every step.

## Setting up by hand

In a Miniforge Prompt (or Anaconda Prompt), from inside this folder, run these one at a time.

**Create the environment:**

```
conda create -n lsr-stress-mmto-ad -c conda-forge --override-channels python=3.11
```

**Activate it:**

```
conda activate lsr-stress-mmto-ad
```

**Install the packages:**

```
pip install -r requirements.txt
```

**Install this repository as the package `lsr_mmto`**, so a script in any folder can import it:

```
pip install --no-deps -e .
```

**Fetch PyTO-AD next to this folder**, at the commit this branch was tested with:

```
git clone https://github.com/UW-ERSL/PyTO-AD.git ../PyTO-AD
```

```
git -C ../PyTO-AD checkout 790d935
```

To use a PyTO-AD you already have instead, it must sit next to this folder and be named `PyTO-AD`. If yours has another name, run `mklink /J PyTO-AD <its name>` in the folder that holds both.

## Files

The pages in `docs/` and `CONTRIBUTING.md` can also be read inside the GUI: the user guide under Help > User guide, and the others under Help > Researcher guide, so no Markdown viewer is needed.

### Top level

| File | Purpose |
|---|---|
| `run_gui.py` | starts the LSR MMTO GUI: `python run_gui.py`, or `python run_gui.py projects/beam.mmto` |
| `make_shortcut.py` | adds LSR MMTO to the Start menu and the desktop (Windows); the setup runs it, and running it again (`python make_shortcut.py`, with the environment active) repairs the shortcuts after the folder moves |
| `projects/` | the four examples as project files: geometry, supports, loads, materials, objective, constraints, settings |
| `docs/how_the_code_works.md` | how the code is laid out, and what happens in one run |
| `docs/code_map.md` | the map of the code: what is where, and where a new feature goes |
| `docs/project_file_reference.md` | everything a project file can hold, generated from the code |
| `docs/how_to_add_features.md` | how to add a quantity, a selection type, a kind of support or load, a setting, a material attribute, a physics or a database |
| `CONTRIBUTING.md` | how changes to the code are made and checked |
| `docs/make_project_file_reference.py` | writes `docs/project_file_reference.md` from the code |
| `setup_environment.py` | installs everything; see Installation |
| `requirements.txt` | Python packages the setup installs |
| `pyproject.toml` | makes this folder installable as the package `lsr_mmto` (`pip install -e .`) |

### `gui/`

| File | Purpose |
|---|---|
| `main_window.py` | the window: New, Open and Save of .mmto projects, units, geometry, the top row (Run, Analyse, Export STL, Part and Design, light and dark) and the status bar (Snap, Measure, Ruler) |
| `materials_panel.py` | the Materials panel |
| `formulation_panel.py` | the Formulation panel: objective and constraints |
| `settings_panel.py` | the Settings panel |
| `supports_and_loads_panel.py` | the Supports and loads panel: selections, supports, loads, keep solid, preview |
| `database_editor.py` | making and changing a material database in the app |
| `run_panel.py` | the Current run panel: a run's progress, its live design and latent space, its log; the Analyse window |
| `results_panel.py` | the Results panel: a finished run's tables, plots and 3D design |
| `draw_design.py` | draws a design in the 3D view |
| `dark_and_light_3d_view.py` | the 3D view light or dark, and its drawings readable on either |
| `panel_widgets.py` | the small pieces every panel shares: number box, (i) button, OK and Cancel, colour swatch |
| `pyto_ad_imports.py` | everything the GUI takes from the PyTO-AD GUI |
| `assets/` | the app's icon, the button pictures (`icons/`) and the style sheet (`app_style.qss`) |

### `lsr_mmto/`

The package a script imports (`from lsr_mmto import Project, run`). `pyto_ad_imports.py` imports PyTO-AD: finite element analysis, filters and the MMA optimizer.

#### `lsr_mmto/problem/`

| File | Purpose |
|---|---|
| `project.py` | a project: saves and loads it, checks it, builds the problem, runs it (`run(p)`) |
| `settings.py` | every setting, one entry each: its default, label, help and place in the Settings panel |
| `selections.py` | named selections of the mesh that supports, loads and keep-solid regions act on |
| `boundary_conditions.py` | what a kind of support or load is (`BCKind`), and the list of kinds of every physics |
| `project_folder.py` | keeps everything a project uses in its folder |
| `mesh.py` | turns the part's STL into the voxel mesh |

#### `lsr_mmto/materials/`

| File | Purpose |
|---|---|
| `databases.py` | the attributes a material database may hold, the check of a database, the colour and label of each material, the VAE settings of each shipped database |
| `encoder.py` | reads a material database, trains or loads the VAE |
| `networks.py` | the VAE network |

#### `lsr_mmto/elements/`

| File | Purpose |
|---|---|
| `hex_element.py` | the hexahedral element: Lame parameters and the strain matrix at its centre |

#### `lsr_mmto/design/`

| File | Purpose |
|---|---|
| `density_filter_and_projection.py` | the density filter and the Heaviside projection |
| `latent_to_properties.py` | each element's material properties from its point in the latent space |
| `latent_penalties.py` | the terms added to the objective: latent smoothing, the distance penalty, and those of a physics |

#### `lsr_mmto/formulation/`

| File | Purpose |
|---|---|
| `quantities.py` | every objective and constraint a formula can name, as differentiable functions; autograd gives their gradients |
| `formulation_text.py` | reads the formula text of the objective and the constraints |

#### `lsr_mmto/physics/`

| File | Purpose |
|---|---|
| `available_physics.py` | the list of physics a project may name, and what each one's front page gives |
| `structural/structural_module.py` | the structural module's front page: everything the rest of the code takes from it |
| `structural/structural_physics.py` | the structural solver, one iteration's solve, element stress, von Mises, p-norm |
| `structural/structural_supports_and_loads.py` | the structural supports and loads: fixed, displacement, force, traction, and the check that they hold the part |
| `structural/structural_quantities.py` | compliance, stress and failure factor, and the structural measurements of each iteration and of the final design |
| `template/` | a small working steady diffusion, commented line by line: the place to start a physics of your own |

#### `lsr_mmto/optimize/`

| File | Purpose |
|---|---|
| `optimization_loop.py` | the optimization loop; internal, a project runs it only through `run(p)` |
| `design_model.py` | one iteration: design, properties, the physics' solve, objective and constraints |
| `mmto_to_params.py` | the parameter container the loop reads |
| `filters.py` | the density and material filters |
| `history.py` | what each iteration records and prints |
| `final_design_analysis.py` | the final design analysed three ways, and the run saved |
| `solid_part_analysis.py` | Analyse: the whole part solid in one material, solved once without optimizing |
| `save_results.py` | saves a finished run to disk |
| `gradient_check.py` | the autodiff gradient against central finite differences |

#### `lsr_mmto/results/`

| File | Purpose |
|---|---|
| `load_results.py` | a finished run, read back for plotting |
| `run_plots.py` | design, latent space, convergence, boundary conditions |
| `tables.py` | run summary and material database tables |
| `design_3d.py` | 2D panels and 3D voxel renders |
| `stl_export.py` | a run's design as STL files |
| `compare.py` | two runs side by side |

### `data/`

| File | Purpose |
|---|---|
| `databases/` | material databases (`.xlsx`) and trained VAEs (`.nt`) |
| `models/` | geometry of the design domains, as STL |
