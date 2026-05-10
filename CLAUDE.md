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
python experiment.py --sweep all --ticks 1500 --quiet  # all sweeps
python graph_results.py --sweep coop_cost --save-dir figures/

# 3D phase-diagram sweeps (slow)
python 3d_experiment.py --plot-only       # mutation × task_flip surface
python 3d_experiment2.py --plot-only      # mass partition × Dirichlet sparsity
```

No tests, no linter.

## Architecture

The simulation is a **multilevel selection model** — individual selection within clusters favors defectors; group selection between clusters favors cooperators. These two pressures create the oscillating dynamics the sim explores.

### Data flow each tick (`Environment.tick`)

1. (MVG) every `task_flip_period` ticks (default 200), re-roll the regional reward landscape via `_generate_regional_rewards()`.
2. `_apply_forces()` — Taichi physics on GPU/CPU (Brownian + Gaussian repulsion + intra-cluster attraction). Falls back to CPU if no Metal/CUDA.
3. `_break_stretched_bonds()` — cluster members > `BOND_BREAK_DIST` µm from centroid detach stochastically.
4. Fitness eval. **No basal income**; every cell pays `MAINTENANCE_COST` + `adhesion_cost` every tick. Health (fitness) only goes up via task completion.
   - Lone cell with op X in a tile rewarding 1-step task X earns `rvec[X] × coop_reward_scale`.
   - Clusters: sum across **all 12 tasks in `ALL_TASKS`** of `max(0, rvec[t] × scale − DEFECTOR_DRAIN × n_defectors)` whenever `cluster.can_complete_task(t)`. The total splits across the **entire cluster** (cooperators and defectors). Cooperators additionally pay `coop_cost` only when `total_gross > 0`.
5. `cell.tick()` — age++, accumulate `health` when net-positive (defectors in clusters compound at half-rate even when fitness is flat). `cluster.tick()` — age++.
6. `_try_adhesion()` — kin-Hamming-gated bond formation. Both endpoints must carry adhesion AND cooperator bits; defectors arise only via mutation **after** the bond has formed.
7. **Probabilistic** death — old age (ramp from `MAX_CELL_AGE = 500` to 1.5×), starvation (`fitness/age < SURVIVAL_RATE` after grace, p ≤ 15%/tick), or oldest-near-spawn culling at `MAX_CELLS = 10_000`.
8. Replication — lone cells at `health ≥ LONE_REPL_THRESH` (cooperators additionally need ≥1 reward point in their op's regional vector); in-cluster defectors at half threshold (`DEFECTOR_REPL_THRESH`); whole clusters when mean(member health) ≥ `CLUSTER_REPL_THRESH`. Lone/cluster gates also stochastic (`LONE_REPL_PROB`, `CLUSTER_REPL_PROB`). After division, all involved cells reset health to 0.

### Genome (`src/cell.py`)

8-bit bitstring; only bits 0–3 active. **Cooperators force adhesion via the bond gate** (`can_form_bond` requires both `is_cooperator` AND `has_adhesion`). Mutation is **asymmetric on the cooperator bit**: `coop→def` at `8.0 × rate`, `def→coop` at `0.3 × rate`. All other bits flip symmetrically.

### Reward landscape (`src/environment.py`)

The world is tiled in `REGION_SIZE = 100` µm squares. Each tile has a 12-element reward vector — one per task in `ALL_TASKS` (4 one-step + 4 two-step + 4 three-step). Mass is allocated per tier (`SINGLE/DOUBLE/TRIPLE_MASS = 6/8/11`) and split across the four slots in each tier by `Dirichlet(task_alpha)`. Low alpha = harsh specialisation; high alpha = uniform.

### Population seeding (`src/simulation.py`)

By default, **40% of initial cells are seeded as pre-formed 2–4 cell cooperator clusters with distinct operations**, with ~20% defectors mixed in. This skips the slow startup phase where random adhesion would have to assemble a working cluster from scratch.

### Sweep + plot pipeline

`experiment.py` defines `SWEEPS` dict of `{name: [{param: value}, ...]}`. Each trial runs `Simulation.run`, builds two DataFrames (`cell_df`, `history_df`), tags them with config metadata, and concats across configs+seeds. Saves to `results/{sweep}_cell_df.csv` and `results/{sweep}_history_df.csv`. `graph_results.py` reads those and emits a 4–6 figure set per sweep into `figures/`.

`Simulation.history` rows have keys: `tick`, `total_cells`, `lone_cells`, `clustered_cells`, `num_clusters`, `avg_cluster_size`, `cooperators`, `defectors`, `cooperator_pct`, `defector_pct`, `coop_genome_pct`, `mean_fitness`, `multi_advantage` (cluster_rate − lone_rate), `coop_advantage` (clustered-coop rate − clustered-def rate), `current_task`, `cell_records`, `cluster_groups`, `regional_rewards`.

## Key Constants (`src/environment.py`)

```python
COOP_REWARD_SCALE   = 1.5     # global multiplier on regional reward vectors
MAINTENANCE_COST    = 0.01    # per-tick metabolic drain on EVERY cell — no basal income
DEFECTOR_DRAIN      = 1.5     # per-defector drain on each completed task
COOP_COST           = 0.5     # extra metabolic cost paid only by cooperators in successful clusters
ADHESION_COST       = 0.1     # paid per tick by adhesion-expressing cooperators (cell.py)

LONE_REPL_THRESH      = 100.0
CLUSTER_REPL_THRESH   = 100.0   # PER-CELL; cluster divides on mean(health)
DEFECTOR_REPL_THRESH  = 50.0    # in-cluster defectors split at half cost
LONE_REPL_COOLDOWN    = 10
CLUSTER_REPL_COOLDOWN = 10
LONE_REPL_PROB        = 0.5
CLUSTER_REPL_PROB     = 0.25

MAX_CELL_AGE   = 500
GRACE_PERIOD   = 10
SURVIVAL_RATE  = 0.02
MAX_CLUSTER_SIZE = 10
MAX_CELLS        = 10_000

ADHESION_BOND_PROB    = 0.5
ADHESION_FORM_RADIUS  = 20.0   # µm
REGION_SIZE           = 100    # µm

BOND_BREAK_DIST     = 40.0   # µm — cluster centroid distance beyond which bonds yield
BOND_BREAK_PROB_MAX = 0.25   # per-tick detach probability cap
```

`src/cell.py`:
```python
HEALTH_GROWTH_RATE = 1.0   # health gained per tick when net-positive (×0.5 for in-cluster defectors)
```

## Design Decisions Worth Knowing

- **No basal reward / atrophy**: every cell pays `MAINTENANCE_COST + adhesion_cost` every tick. Health goes up only via task completion. Cells without any income source drift below zero fitness and die via the existing starvation path. This is what makes the spatial reward landscape and complementary-cooperation incentives selective rather than decorative.
- **Cooperators force adhesion via the bond gate**: a cell can only form bonds if it carries BOTH adhesion AND cooperator bits — defectors cannot infiltrate from outside; they only arise via cooperator → defector mutation **after** the bond has formed.
- **Asymmetric coop-bit mutation**: cheating must be biologically easier to evolve than cooperation; the genetic filter has to overcome the bias.
- **Kin-gated adhesion**: bond probability scales with genomic similarity on bits 0–3, making defector spread between clusters harder than within-cluster mutation drift.
- **Seeded cooperator clusters at startup**: avoids waiting tens of thousands of ticks for random adhesion to find a working complementary pair. Bias the initial conditions, then let evolution operate.
- **Probabilistic age/starvation death** (not deterministic): smooths population dynamics; avoids synchronized cohort die-offs.
- **Stochastic replication gates**: even when health threshold is crossed, replication only fires with probability `LONE_REPL_PROB` / `CLUSTER_REPL_PROB`. Prevents simultaneous division pulses across the whole population.
- **Defector-in-cluster replication at half threshold**: models cancer-like clonal expansion — parasitic mutants compound from the cluster's shared income while the host cluster continues operating.
- **Cooperator-only lone replication gate**: lone cooperators must be in a tile that pays for their op before they can reproduce; defectors face no such gate (they free-ride wherever they land).
- **Age-based competition at carrying capacity** (not fitness-based): fitness-based killing biases against defectors whose accumulated fitness is diluted by cluster infiltration — that would suppress the very individual selection the model studies.
- **No wrapping** — elastic wall reflection keeps spatial structure intact for cluster cohesion.

## Status of sweeps (`results/`, `figures/`)

The eight sweeps in `SWEEPS` plus the two 3D sweeps have CSVs in `results/`. The headline `figures/` are: `coop_cost_final_bars`, `reward_scale_final_bars`, `3d_mass_heatmap/surface`, `3d_mutation_flip_heatmap/surface`. Re-run any 1D sweep figure with `python graph_results.py --sweep <name> --save-dir figures/`.
