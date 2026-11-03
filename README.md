# LSR-Stress-MMTO (autodiff branch)

Stress-based multi-material topology optimization in a learned latent space.

**This is the `autodiff` branch.** The gradients of the objective and constraints come from automatic differentiation (PyTorch autograd) through [PyTO-AD](https://github.com/UW-ERSL/PyTO-AD), instead of the hand-derived sensitivities on `main`. The `main` branch is the code the paper used; see [Differences from main](#differences-from-main).

Install below, then open `Tutorial.ipynb` to reproduce the paper's examples.

## Installation

First, download the code: on this page, switch the branch menu (top left, it says `main`) to **autodiff**, then click the green **Code** button, then **Download ZIP**, and unzip it anywhere. With git instead: `git clone -b autodiff https://github.com/UW-ERSL/LSR-Stress-MMTO.git`.

If you already have conda (Miniforge, Miniconda or Anaconda), skip step 1. If you already have an environment that runs this code, you can keep using it; `requirements.txt` lists what it needs. To set everything up yourself instead, see [Setting up by hand](#setting-up-by-hand) further down.

1. **Install Miniforge.** Download the Windows installer, `Miniforge3-Windows-x86_64.exe`, from [github.com/conda-forge/miniforge](https://github.com/conda-forge/miniforge/releases/latest), run it, and accept the defaults.

2. **Open the Miniforge Prompt.** Start menu, type `Miniforge Prompt`, open it. With Miniconda or Anaconda it is the `Anaconda Prompt`.

3. **Run the setup.** In that prompt, type `python` and a space, **without pressing Enter**, then drag `setup_environment.py` into the window, and only then press Enter. It first asks one question, whether to put an LSR MMTO icon on the desktop (it always goes in the Start menu), and then runs on its own. It takes roughly 10 minutes and ends with `Done.`

   This also downloads [PyTO-AD](https://github.com/UW-ERSL/PyTO-AD), the differentiable topology optimization code this branch builds on (finite element analysis, filters, MMA, automatic differentiation), into a folder next to this one, unless it is already there. It creates its own environment, `lsr-stress-mmto-ad`, so it never touches the one `main` uses.

4. **Open the tutorial.** Open `Tutorial.ipynb` in your code editor and select the kernel **Python (lsr-stress-mmto-ad)**.

To use your browser instead of an editor, run these in the same prompt, one at a time:

```
conda activate lsr-stress-mmto-ad
```

```
pip install jupyterlab
```

Then type `jupyter lab` and a space, **without pressing Enter**, drag `Tutorial.ipynb` into the window, and press Enter.

## Running the paper's examples

`Tutorial.ipynb` reproduces each of the four examples in the paper and, in doing so, walks through the whole workflow: the material database, choosing the latent dimension, the design problem, the trained VAE, the optimization, and the resulting design.

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

**Register it as a notebook kernel:**

```
python -m ipykernel install --user --name lsr-stress-mmto-ad --display-name "Python (lsr-stress-mmto-ad)"
```

**Fetch PyTO-AD next to this folder**, at the commit this branch was tested with:

```
git clone https://github.com/UW-ERSL/PyTO-AD.git ../PyTO-AD
```

```
git -C ../PyTO-AD checkout 790d935
```

To use a PyTO-AD you already have instead, it must sit next to this folder and be named `PyTO-AD`. If yours has another name, run `mklink /J PyTO-AD <its name>` in the folder that holds both.

## Differences from main

| | `main` | `autodiff` |
|---|---|---|
| Solver library | PyTO | PyTO-AD only (PyTO is not used) |
| Gradients | hand-derived adjoints, `SensitivitiesPureStructural.py` | autograd through the forward pass, `lsr_mmto/physics/structural.py` |
| Environment | `lsr-stress-mmto` | `lsr-stress-mmto-ad` |

Everything else (examples, material databases, VAEs, optimizer settings, schedules) is the same, so a run on either branch can be compared one to one.

## Files

### Top level

| File | Purpose |
|---|---|
| `Tutorial.ipynb` | runs one example end to end |
| `run_gui.py` | starts the LSR MMTO GUI: `python run_gui.py`, or `python run_gui.py projects/beam.mmto` |
| `make_shortcut.py` | adds LSR MMTO to the Start menu and the desktop (Windows); the setup runs it, and running it again (`python make_shortcut.py`, with the environment active) repairs the shortcuts after the folder moves |
| `projects/` | the four examples as project files: geometry, supports, loads, materials, objective, constraints, settings |
| `docs/project_file_reference.md` | everything a project file can hold, generated from the code |
| `docs/how_to_add_features.md` | how to add a quantity, a selection type, a boundary condition, a setting, a material attribute or a database |
| `docs/make_project_file_reference.py` | writes `docs/project_file_reference.md` from the code |
| `setup_environment.py` | installs everything; see Installation |
| `requirements.txt` | Python packages the setup installs |
| `pyproject.toml` | makes this folder installable as the package `lsr_mmto` (`pip install -e .`) |

### `gui/`

| File | Purpose |
|---|---|
| `main_window.py` | the window: New, Open and Save of .mmto projects, units, geometry |
| `panels.py` | the Materials, Formulation and Settings panels |
| `supports.py` | the Supports and loads panel: regions, supports, loads, keep solid, preview |
| `pyto_ad.py` | everything the GUI takes from the PyTO-AD GUI |
| `database_editor.py` | making and changing a material database in the app |
| `assets/` | the app's icon |

### `lsr_mmto/`

The package a script imports (`from lsr_mmto import Project, run`). `pyto_ad.py` imports PyTO-AD: finite element analysis, filters and the MMA optimizer.

#### `lsr_mmto/problem/`

| File | Purpose |
|---|---|
| `project.py` | a project: saves and loads it, checks it, builds the problem, runs it (`run(p)`) |
| `settings.py` | every setting, one entry each: its default, label, help and place in the Settings panel |
| `selections.py` | named regions of the mesh that supports, loads and keep-solid regions act on |
| `project_folder.py` | keeps everything a project uses in its folder |
| `mesh.py` | turns the part's STL into the voxel mesh |

#### `lsr_mmto/materials/`

| File | Purpose |
|---|---|
| `databases.py` | the attributes a material database may hold, the check of a database, the colour and label of each material, the VAE settings of each shipped database |
| `encoder.py` | reads a material database, trains or loads the VAE |
| `networks.py` | the VAE network |
| `colors.py` | the default colours, for materials the database's Display sheet gives none |

#### `lsr_mmto/physics/`

| File | Purpose |
|---|---|
| `__init__.py` | the list of physics a project may name, and the file of each |
| `structural.py` | the objective and constraints as differentiable functions; autograd gives their gradients |
| `gradcheck.py` | the autodiff gradient against central finite differences |

#### `lsr_mmto/optimize/`

| File | Purpose |
|---|---|
| `run.py` | the optimization loop; internal, a project runs it only through `run(p)` |
| `parameters.py` | the parameter container shared by the examples |
| `filters.py` | the density and material filters |
| `history.py` | what each iteration records and prints: history, log, stress diagnostics |
| `final_check.py` | the final design analysed three ways, and the run saved |
| `save.py` | saves a finished run to disk |

#### `lsr_mmto/results/`

| File | Purpose |
|---|---|
| `records.py` | a finished run, read back for plotting |
| `run_plots.py` | design, latent space, convergence, boundary conditions |
| `tables.py` | run summary and material database tables |
| `render.py` | 2D panels and 3D voxel renders |
| `stl_export.py` | a run's design as STL files |
| `compare.py` | two runs side by side |
| `display.py` | tables formatted for the notebook |

### `data/`

| File | Purpose |
|---|---|
| `databases/` | material databases (`.xlsx`) and trained VAEs (`.nt`) |
| `models/` | geometry of the design domains, as STL |
