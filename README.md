# LSR-Stress-MMTO (autodiff branch)

Stress-based multi-material topology optimization in a learned latent space.

**This is the `autodiff` branch.** The gradients of the objective and constraints come from automatic differentiation (PyTorch autograd) through [PyTO-AD](https://github.com/UW-ERSL/PyTO-AD), instead of the hand-derived sensitivities on `main`. The `main` branch is the code the paper used; see [Differences from main](#differences-from-main).

Install below, then open `Tutorial.ipynb` to reproduce the paper's examples.

## Installation

First, download the code: on this page, switch the branch menu (top left, it says `main`) to **autodiff**, then click the green **Code** button, then **Download ZIP**, and unzip it anywhere. With git instead: `git clone -b autodiff https://github.com/UW-ERSL/LSR-Stress-MMTO.git`.

If you already have conda (Miniforge, Miniconda or Anaconda), skip step 1. If you already have an environment that runs this code, you can keep using it; `requirements.txt` lists what it needs. To set everything up yourself instead, see [Setting up by hand](#setting-up-by-hand) further down.

1. **Install Miniforge.** Download the Windows installer, `Miniforge3-Windows-x86_64.exe`, from [github.com/conda-forge/miniforge](https://github.com/conda-forge/miniforge/releases/latest), run it, and accept the defaults.

2. **Open the Miniforge Prompt.** Start menu, type `Miniforge Prompt`, open it. With Miniconda or Anaconda it is the `Anaconda Prompt`.

3. **Run the setup.** In that prompt, type `python` and a space, **without pressing Enter**, then drag `setup_environment.py` into the window, and only then press Enter. It takes roughly 10 minutes and ends with `Done.`

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

**Register it as a notebook kernel:**

```
python -m ipykernel install --user --name lsr-stress-mmto-ad --display-name "Python (lsr-stress-mmto-ad)"
```

**Fetch PyTO-AD next to this folder**, at the commit this branch was tested with:

```
git clone https://github.com/UW-ERSL/PyTO-AD.git ../PyTO-AD
```

```
git -C ../PyTO-AD checkout e52704a
```

To use a PyTO-AD you already have instead, it must sit next to this folder and be named `PyTO-AD`. If yours has another name, run `mklink /J PyTO-AD <its name>` in the folder that holds both.

## Differences from main

| | `main` | `autodiff` |
|---|---|---|
| Solver library | PyTO | PyTO-AD only (PyTO is not used) |
| Gradients | hand-derived adjoints, `SensitivitiesPureStructural.py` | autograd through the forward pass, `AutodiffPureStructural.py` |
| Environment | `lsr-stress-mmto` | `lsr-stress-mmto-ad` |

Everything else (examples, material databases, VAEs, optimizer settings, schedules) is the same, so a run on either branch can be compared one to one.

## Files

### Top level

| File | Purpose |
|---|---|
| `Tutorial.ipynb` | runs one example end to end |
| `projects/` | the four examples as project files: geometry, supports, loads, materials, objective, constraints, settings |
| `docs/project_reference.md` | everything a project file can hold, generated from the code |
| `docs/extending.md` | how to add a quantity, a selection type, a boundary condition, a setting, a material attribute or a database |
| `setup_environment.py` | installs everything; see Installation |
| `requirements.txt` | Python packages the setup installs |

### `0-Common/`

| File | Purpose |
|---|---|
| `Models/` | geometry of the four design domains, as STL |
| `project.py` | a project: saves and loads it, checks it, builds the problem, runs it |
| `selections.py` | named regions of the mesh that supports, loads and keep-solid regions act on |
| `pyto_project.py` | meshing, and the supports and loads picked in the PyTO-AD GUI |
| `databases.py` | the attributes a material database may hold, the check of a database, the VAE settings of each shipped database |
| `materialEncoder.py` | reads a material database, trains or loads the VAE |
| `networks.py` | the VAE network |
| `materialColors.py` | the colour each material is drawn in |
| `mmto_common.py` | the parameter container shared by the examples |
| `PyTOImports.py` | imports PyTO-AD: finite element analysis, filters and the MMA optimizer |

### `1-PureStructural/`

| File | Purpose |
|---|---|
| `MaterialDataPureStructural/` | material databases (`.xlsx`) and trained VAEs (`.nt`) |
| `MainPureStructural.py` | the optimization driver |
| `AutodiffPureStructural.py` | the objective and constraints as differentiable functions; autograd gives their gradients |
| `mmto_utility_functions.py` | filters, stress and mass diagnostics, run bookkeeping |
| `run_io.py` | saves a finished run to disk |

### `utils/`

| File | Purpose |
|---|---|
| `records.py` | a finished run, read back for plotting |
| `figures.py` | design, latent space, convergence, boundary conditions |
| `tables.py` | run summary and material database tables |
| `render.py` | 2D panels and 3D voxel renders |
| `display.py` | tables formatted for the notebook |
| `project_paths.py` | locates the repository and sets up imports |
| `reference.py` | writes `docs/project_reference.md` |
| `database.py` | checks a material database, and writes an empty one to fill in |
