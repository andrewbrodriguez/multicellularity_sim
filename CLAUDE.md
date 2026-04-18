# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CS2212 final project by Andrew Rodriguez. A physics-based evolutionary simulation of the emergence of multicellularity, modelling the **defector problem**: cells that join cooperative clusters but withhold computation and reap shared rewards. The core question is what conditions allow cooperation to persist against parasitic free-riders.

## Running the Simulation

```bash
# Activate the virtual environment first
source sim_venv/bin/activate

# Headless run with console stats (no visualization)
python main.py

# Full Napari GUI visualization (default: 40 cells, 500 ticks, sample every 10)
python visualize.py

# Common visualize.py flags
python visualize.py --initial-cells 100 --ticks 600 --sample-every 5 --seed 42
python visualize.py --task-flip-period 100   # Enable MVG (Modularly Varying Goals) extension
python visualize.py --sample-every 1 --quiet  # Every tick, no console noise
```

There are no tests and no linter configuration. Install dependencies with:
```bash
pip install -r requirements.txt
```

## Architecture

The simulation is a **multilevel selection model** — individual selection within clusters favors defectors; group selection between clusters favors cooperators. These two pressures create the oscillating dynamics the sim explores.

### Data flow each tick (`Environment.tick`)

1. `_refresh_env()` — draw random 8-bit integers A, B, C; optionally flip MVG task
2. `_apply_forces()` — vectorized numpy physics (Brownian jitter → repulsion → adhesion springs → wall reflection) using `scipy.cKDTree` for neighbour queries
3. Fitness evaluation — lone cells earn `BASE_REWARD`; clusters attempt the current logic task; success grants `COMPLEX_REWARD − DEFECTOR_DRAIN × n_defectors` split across all members
4. `cell.tick()` / `cluster.tick()` — age increment, cooldown decrement
5. `_try_adhesion()` — KDTree search to bond adhesive lone cells into clusters or join existing ones
6. Death — old age (`≥150` ticks), starvation (`fitness/age < 0.04` after grace period), or age-based competition at `MAX_CELLS = 900`
7. Replication — lone cells at `fitness ≥ 18.0`; clusters at `per_cell ≥ 10.0` (whole cluster replicates as a unit, propagating defectors into offspring)

### Genome encoding (`src/cell.py`)

8-bit bitstring; only bits 0–3 are active (bits 4–7 reserved):

| Bits | Field | Effect |
|------|-------|--------|
| 0–1 | Operation | `00`=AND `01`=OR `10`=XOR `11`=NAND |
| 2 | Adhesion | Can form/join clusters; pays `ADHESION_COST = 0.3/tick` |
| 3 | Cooperator | `0` + adhesion = **defector** (free-rider phenotype) |

### Task computation (`src/cluster.py`)

`Cluster.compute_task(a, b, c, op1, op2)` executes 1-step (primitive) or 2-step (complex) tasks. A 2-step task requires one cooperator with `op1` AND one with `op2` — this forced division of labour is what selects for multicellularity. Defectors (`has_adhesion=True, is_cooperator=False`) receive the per-cell reward share but contribute nothing.

### MVG Extension (`src/environment.py`)

Partially implemented. When `task_flip_period` is set, `_refresh_env()` cycles through four task structures every N ticks:
```python
self.task_states = [("AND", "XOR"), ("OR",), ("NAND", "OR"), ("XOR",)]
```
The hypothesis: dynamic environments force modular networks that can identify and exclude defectors.

### Visualization (`src/visualizer.py` + `visualize.py`)

`build_napari_data()` builds an `(T, 400, 400, 3)` uint8 image stack (Gaussian density heatmap) plus point clouds per cell type. Colors: gray = lone, blue = cooperator, red = defector. The Napari title bar updates live with tick stats and current task string as you scrub the time slider.

`Simulation.history` is a list of dicts with keys: `tick`, `total_cells`, `lone_cells`, `clustered_cells`, `num_clusters`, `avg_cluster_size`, `cooperators`, `defectors`, `cooperator_pct`, `defector_pct`, `mean_fitness`, `current_task`, `cell_records`.

## Key Constants (all in `src/environment.py`)

```python
COMPLEX_REWARD      = 5.0    # NOTE: SIM_DESIGN.md says 20.0 — current value is conservative
DEFECTOR_DRAIN      = 1.5    # per-defector reduction in gross cluster reward
COOP_COST           = 0.3    # extra metabolic cost paid only by cooperators in successful clusters
BASE_REWARD         = 1.0    # lone cell baseline per tick
MAX_CLUSTER_SIZE    = 12
MAX_CELLS           = 900
ADHESION_BOND_PROB  = 0.20
```

**Important:** `COMPLEX_REWARD` is currently `5.0` in code but the design document specifies `20.0`. With `5.0`, clusters have very little fitness headroom before defector drain collapses them. Increasing to `20.0` dramatically increases cluster viability and makes multicellular emergence more observable.

## Design Decisions Worth Knowing

- **Age-based competition at carrying capacity** (not fitness-based): intentional. Fitness-based killing would bias against defectors — suppressing the very individual selection pressure the model is designed to study.
- **Defectors pay only 25% adhesion cost when lone**: keeps adhesive defectors alive while they search for clusters to infiltrate.
- **Cluster replication copies defectors**: models how cancer-like cells propagate within multicellular lineages.
- **No wrapping** — elastic wall reflection keeps spatial structure intact for cluster cohesion.
