# LSR-Stress-MMTO

Stress-based multi-material topology optimization in a learned latent space.

Install below, then open `Tutorial.ipynb` to reproduce the paper's examples.

## Installation

First, download the code: on this page, click the green **Code** button, then **Download ZIP**, and unzip it anywhere.

If you already have conda, skip step 1. To set everything up yourself instead, see [Setting up by hand](#setting-up-by-hand) further down.

1. **Install Miniconda.** Download the Windows installer from [anaconda.com/download](https://www.anaconda.com/download/success), run it, and accept the defaults.

2. **Open the Anaconda Prompt.** Start menu, type `Anaconda Prompt`, open it.

3. **Run the setup.** In the Anaconda Prompt, type `python` and a space, **without pressing Enter**, then drag `setup_environment.py` into the window, and only then press Enter. It takes roughly 10 minutes and ends with `Done.`

   This also downloads [PyTO](https://github.com/UW-ERSL/PyTO), the single-material topology optimization code this builds on (finite element analysis, filters, MMA), into a folder next to this one, unless it is already there.

4. **Open the tutorial.** Open `Tutorial.ipynb` in your code editor and select the kernel **Python (lsr-stress-mmto)**.

To use your browser instead of an editor, run these in the Anaconda Prompt, one at a time:

```
conda activate lsr-stress-mmto
```

```
pip install jupyterlab
```

Then type `jupyter lab` and a space, **without pressing Enter**, drag `Tutorial.ipynb` into the window, and press Enter.

## Running the paper's examples

`Tutorial.ipynb` reproduces each of the four examples in the paper and, in doing so, walks through the whole workflow: the material database, choosing the latent dimension, the design problem, the trained VAE, the optimization, and the resulting design.

## Setting up by hand

In an Anaconda Prompt, from inside this folder, run these one at a time.

**Create the environment:**

```
conda create -n lsr-stress-mmto -c conda-forge --override-channels python=3.11
```

**Activate it:**

```
conda activate lsr-stress-mmto
```

**Install the packages:**

```
pip install -r requirements.txt
```

**Register it as a notebook kernel:**

```
python -m ipykernel install --user --name lsr-stress-mmto --display-name "Python (lsr-stress-mmto)"
```

**Fetch PyTO next to this folder**, at the commit the paper used:

```
git clone https://github.com/UW-ERSL/PyTO.git ../PyTO
```

```
git -C ../PyTO checkout b109a169
```

To use a PyTO you already have instead, it must sit next to this folder and be named `PyTO`. If yours has another name, run `mklink /J PyTO <its name>` in the folder that holds both.

## Files

### Top level

| File | Purpose |
|---|---|
| `Tutorial.ipynb` | runs one example end to end |
| `setup_environment.py` | installs everything; see Installation |
| `requirements.txt` | Python packages the setup installs |

### `0-Common/`

| File | Purpose |
|---|---|
| `Models/` | geometry of the four design domains, as STL |
| `StructuralFEAExamples.py` | mesh, loads and supports for each domain |
| `materialEncoder.py` | reads a material database, trains or loads the VAE |
| `networks.py` | the VAE network |
| `materialColors.py` | the colour each material is drawn in |
| `mmto_common.py` | the parameter container shared by the examples |
| `PyTOImports.py` | imports PyTO: finite element analysis, filters and the MMA optimizer |

### `1-PureStructural/`

| File | Purpose |
|---|---|
| `MaterialDataPureStructural/` | material databases (`.xlsx`) and trained VAEs (`.nt`) |
| `ExamplesPureStructural.py` | the four examples: objective, constraints, settings |
| `MainPureStructural.py` | the optimization driver |
| `SensitivitiesPureStructural.py` | gradients of the objective and constraints |
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
