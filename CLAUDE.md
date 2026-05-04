# CLAUDE.md

Guidance for Claude Code working in this repository. See `SIM_DESIGN.md` for the full simulation spec.

## Project Overview

CS2212 final project by Andrew Rodriguez. A physics-based evolutionary simulation of the emergence of multicellularity, modelling the **defector problem**: cells that join cooperative clusters but withhold computation and reap shared rewards. The core question is what conditions allow cooperation to persist against parasitic free-riders.

## Running

```bash
source sim_venv/bin/activate
pip install -r requirements.txt    # if first time

python main.py                                        # headless smoke run
python visualize.py                                    # full Napari GUI
python visualize.py --task-flip-period 100             # MVG mode
python experiment.py --sweep coop_cost --quiet         # one sweep → CSVs in results/
python experiment.py --sweep all --ticks 500 --quiet   # all sweeps
python graph_results.py --sweep coop_cost --save-dir figures/
```

No tests, no linter.

## Architecture

The simulation is a **multilevel selection model** — individual selection within clusters favors defectors; group selection between clusters favors cooperators. These two pressures create the oscillating dynamics the sim explores.

### Data flow each tick (`Environment.tick`)

1. (MVG) if `task_flip_period` is set and we hit the period, re-roll the regional reward landscape via `_generate_regional_rewards()`
2. `_apply_forces()` — Taichi physics (Brownian + repulsion + adhesion + inter-cluster attraction + wall reflection); numpy/cKDTree fallback in `_apply_forces_numpy_fallback`
3. Fitness evaluation — **no basal reward**; every cell pays `MAINTENANCE_COST = 0.02` + `adhesion_cost` every tick, and health (fitness) only goes up via task completion. Atrophy → starvation kills cells with no income source.
   - Lone cooperator with op X in a tile rewarding 1-step task X: earns `rvec[X] × coop_reward_scale`, pays `coop_cost`.
   - Lone defectors and lone cooperators in tiles that don't reward their op: earn nothing, atrophy at `MAINTENANCE_COST + adhesion_cost`/tick.
   - Clusters: sum across **all 12 tasks in `ALL_TASKS`** of `max(0, rvec[t] × scale − DEFECTOR_DRAIN × n_defectors)` whenever `cluster.can_complete_task(t)` is True; the sum splits across the **entire** cluster (both cooperators and defectors share). Cooperators additionally pay `coop_cost` only when `total_gross > 0`. Every cell pays `MAINTENANCE_COST + adhesion_cost` regardless. Health can go negative.
   - The cluster loop covers both "cluster does a multi-step task" and "cluster has only one cooperator who can do a 1-step task in this tile" — both paths feed `total_gross` and both split across the entire cluster.
4. `cell.tick()` / `cluster.tick()` — age increment, cooldown decrement
5. `_break_stretched_bonds()` — cluster members > `BOND_BREAK_DIST = 60` µm from centroid detach stochastically (probability scales with excess distance, capped at 0.05/tick). Then `_try_adhesion()` — kin-Hamming-gated bond formation (lone↔lone or lone→cluster)
6. **Probabilistic** death — old age (ramp from `MAX_CELL_AGE = 500` to 1.5×), starvation (`fitness/age < 0.02` after grace, p ≤ 15%/tick), or oldest-near-spawn culling at `MAX_CELLS = 10_000`
7. Replication — lone cells at `fitness ≥ 28.0` AND `size ≥ 2.0` (cooperators additionally need ≥1 reward point in their op's regional vector); clusters at per-cell `≥ 18.0` AND mean member `size ≥ 2.0`. Both gated by stochastic probabilities (`LONE_REPL_PROB = 0.10`, `CLUSTER_REPL_PROB = 0.05`). After division, all involved cells reset to `size = 1.0`

### Genome (`src/cell.py`)

8-bit bitstring; only bits 0–3 active. **Cooperators force adhesion** (`has_adhesion = genome[2] OR genome[3]`). Mutation is **asymmetric on the cooperator bit**: `coop→def` at `2.0 × rate`, `def→coop` at `0.3 × rate`. All other bits flip symmetrically.

### Cell cycle (`src/cell.py`)

Cells carry a `size` attribute (`NEWBORN_SIZE = 1.0` → `DIVISION_SIZE = 2.0`) that grows by `GROWTH_RATE = 0.015`/tick **only when `fitness > 0`**. Replication requires the cell-cycle gate (`is_division_ready`) in addition to fitness. After division both halves reset to size 1.0. The cluster path requires the *mean* member size to be division-ready.

### Reward landscape (`src/environment.py`)

The world is tiled in `REGION_SIZE = 100` µm squares. Each tile has a 12-element reward vector — one per task in `ALL_TASKS` (4 one-step + 4 two-step + 4 three-step). Mass is allocated per tier (`SINGLE/DOUBLE/TRIPLE_MASS = 3/8/12`) and split across the four slots in each tier by `Dirichlet(task_alpha)`. Low alpha = harsh specialisation; high alpha = uniform.

### Population seeding (`src/simulation.py`)

By default, **40% of initial cells are seeded as pre-formed 2–4 cell cooperator clusters with distinct operations**. This skips the slow startup phase where random adhesion would have to assemble a working cluster from scratch.

### Sweep + plot pipeline

`experiment.py` defines `SWEEPS` dict of `{name: [{param: value}, ...]}`. Each trial runs `Simulation.run`, builds two DataFrames (`cell_df`, `history_df`), tags them with config metadata, and concats across configs+seeds. Saves to `results/{sweep}_cell_df.csv` and `results/{sweep}_history_df.csv`. `graph_results.py` reads those and emits a 4–5 figure set per sweep into `figures/`.

`Simulation.history` rows have keys: `tick`, `total_cells`, `lone_cells`, `clustered_cells`, `num_clusters`, `avg_cluster_size`, `cooperators`, `defectors`, `cooperator_pct`, `defector_pct`, `coop_genome_pct`, `mean_fitness`, `multi_advantage` (cluster_rate − lone_rate), `coop_advantage` (clustered-coop rate − clustered-def rate), `current_task`, `cell_records`, `cluster_groups`.

## Key Constants (`src/environment.py`)

```python
COOP_REWARD_SCALE   = 0.10    # global multiplier on regional reward vectors
MAINTENANCE_COST    = 0.02    # per-tick metabolic drain on EVERY cell — no basal income
DEFECTOR_DRAIN      = 1.5     # per-defector drain on each completed task
COOP_COST           = 0.1     # extra metabolic cost for cooperators in successful clusters
ADHESION_COST       = 0.1     # paid every tick by adhesion-expressing cells (cell.py)
# BASE_REWARD / SIMPLE_REWARD / COMPLEX_REWARD / TRIPLE_REWARD: kept as 0.0 for
# backward compat with visualize.py imports; no longer used in fitness eval.

LONE_REPL_THRESH    = 28.0
CLUSTER_REPL_THRESH = 18.0
LONE_REPL_COOLDOWN  = 50
CLUSTER_REPL_COOLDOWN = 50
LONE_REPL_PROB      = 0.10
CLUSTER_REPL_PROB   = 0.05

MAX_CELL_AGE        = 500
GRACE_PERIOD        = 20
SURVIVAL_RATE       = 0.02
MAX_CLUSTER_SIZE    = 12
MAX_CELLS           = 10_000

ADHESION_BOND_PROB   = 0.20
ADHESION_FORM_RADIUS = 10.0   # µm
REGION_SIZE          = 100    # µm

BOND_BREAK_DIST      = 60.0   # µm — cluster centroid distance beyond which bonds yield
BOND_BREAK_PROB_MAX  = 0.05   # per-tick detach probability cap
```

`src/cell.py`:
```python
NEWBORN_SIZE   = 1.0
DIVISION_SIZE  = 2.0
GROWTH_RATE    = 0.015   # ~67 ticks newborn → division-ready
```

## Design Decisions Worth Knowing

- **No basal reward / atrophy**: every cell pays `MAINTENANCE_COST + adhesion_cost` every tick. Health goes up only via task completion. Cells without any income source drift below zero fitness and die via the existing starvation path. This is what makes the spatial reward landscape and complementary-cooperation incentives selective rather than decorative.
- **Cooperators force adhesion**: a cooperator that doesn't bond can't share rewards, so the bit gets OR'd onto the adhesion bit at parse time.
- **Asymmetric coop-bit mutation**: cheating must be biologically easier to evolve than cooperation; the genetic filter has to overcome the bias.
- **Kin-gated adhesion**: bond probability scales with genomic similarity on bits 0–3, making defector infiltration of established cooperator clusters harder than ad-hoc lone-on-lone bonding.
- **Seeded cooperator clusters at startup**: avoids waiting tens of thousands of ticks for random adhesion to find a working complementary pair. Bias the initial conditions, then let evolution operate.
- **Probabilistic age/starvation death** (not deterministic): smooths population dynamics; avoids synchronized cohort die-offs.
- **Stochastic replication gates**: even when fitness threshold is crossed, replication only fires with probability `LONE_REPL_PROB` / `CLUSTER_REPL_PROB`. Prevents simultaneous division pulses across the whole population.
- **Cluster replication copies defectors**: models cancer-like propagation within multicellular lineages.
- **Cooperator-only lone replication gate**: lone cooperators must be in a tile that pays for their op before they can reproduce; defectors face no such gate (they free-ride wherever they land).
- **Age-based competition at carrying capacity** (not fitness-based): fitness-based killing biases against defectors whose accumulated fitness is diluted by cluster infiltration — that would suppress the very individual selection the model studies.
- **No wrapping** — elastic wall reflection keeps spatial structure intact for cluster cohesion.

## Status of sweeps (`results/`, `figures/`)

All eight sweeps in `SWEEPS` have CSV outputs in `results/`. Figures exist in `figures/` for: `population`, `mutation_penalty`, `task_flip` (re-run after MVG was wired in), `coop_bonus`, `task_distribution`, `coop_cost`. **`grid_size` and `mutation_rate` have CSVs but no plotted figures yet** — run `graph_results.py --sweep grid_size --save-dir figures/` to produce them.
