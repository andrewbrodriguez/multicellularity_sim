# Multicellularity Simulation — Design Document

**CS2212 Final Project · Andrew Rodriguez**

---

## Overview

This simulation models the computational leap from single-celled to multicellular life. Independent digital organisms evolve on a 2D spatial plane, competing for resources by executing bitwise logic tasks drawn from a heterogeneous reward landscape. The core tension is the **defector problem**: a mutant free-rider that joins a cooperative cluster, reaps shared rewards without paying the metabolic cost of computation, and mathematically should spread — yet real multicellular organisms somehow suppress this.

The simulation is built to explore what environmental and algorithmic conditions allow cooperation to emerge and persist against parasitic invasion.

---

## Architecture

```
multicellularity_sim/
├── src/
│   ├── cell.py              # genome encoding, mutation, computation
│   ├── cluster.py           # cooperative group, multi-step task execution
│   ├── environment.py       # reward landscape, fitness, death, replication, MVG
│   ├── physics_taichi.py    # GPU-friendly Taichi physics kernel
│   ├── simulation.py        # tick loop + history collection
│   └── visualizer.py        # Napari heatmap + point-cloud builder
├── main.py                  # headless smoke run
├── visualize.py             # full GUI run via Napari
├── experiment.py            # batch parameter sweeps → CSV
├── graph_results.py         # plot CSV outputs → PNG figure sets
└── SIM_DESIGN.md            # this document
```

---

## The Genome

Each cell carries an **8-bit bitstring genome** that fully determines its behavior.

| Bits | Field | Values |
|------|-------|--------|
| 0–1  | Operation | `00`=AND · `01`=OR · `10`=XOR · `11`=NAND |
| 2    | Adhesion allele | `1` = expresses adhesion (metabolic cost) |
| 3    | Cooperator allele | `1` = contributes computation to cluster; `0` = defector |
| 4–7  | Reserved | Available for future traits (signalling, apoptosis, etc.) |

**Cooperators force adhesion.** A cooperator that doesn't physically bond can't share the reward, so `cell.has_adhesion = genome[2] OR genome[3]`. This means defectors are the only cells that can carry adhesion without contributing computation.

**Mutation is asymmetric on the cooperator bit:**
- `cooperator → defector` flips at `2.0 × mutation_rate`
- `defector → cooperator` flips at `0.3 × mutation_rate`

This bakes in the assumption that cheating is biologically easier to evolve than cooperation — the ~6.7× bias the genetic filter must overcome. All other bits flip symmetrically at the base `mutation_rate`.

---

## The Environment

Cells live on a **continuous 2D plane** of `width × height` µm (default `500 × 500`) with **hard elastic walls** — positions reflect off boundaries instead of wrapping.

### The reward landscape

The world is partitioned into a grid of `REGION_SIZE = 100` µm tiles. Each tile carries an independent **12-element reward vector** — one entry per task in `ALL_TASKS`:

```python
ALL_TASKS = [
    ("AND",), ("OR",), ("XOR",), ("NAND",),                          # 1-step  (idx 0–3)
    ("AND", "XOR"), ("NAND", "OR"), ("OR", "AND"), ("XOR", "NAND"),  # 2-step  (idx 4–7)
    ("AND", "OR", "XOR"), ("OR", "NAND", "AND"),                     # 3-step  (idx 8–11)
    ("XOR", "AND", "OR"), ("NAND", "XOR", "AND"),
]
```

Mass is distributed per tile as multinomial draws within each complexity tier:

| Tier | Mass per tile | Weight class |
|------|---------------|--------------|
| 1-step | `SINGLE_MASS = 3` | abundant, low payoff |
| 2-step | `DOUBLE_MASS = 8` | lucrative, requires 2 cooperators |
| 3-step | `TRIPLE_MASS = 12` | rare, requires 3 cooperators |

Within each tier the mass is split across the four task slots by `Dirichlet(task_alpha)`. Low `task_alpha` (~0.05, the default) yields **specialised tiles** that pay only one task per tier — clusters must migrate or carry the right operations to survive. High alpha (~5.0) flattens the landscape so every task pays everywhere.

This replaces the original v1 design's single hard-coded `(A AND B) XOR C` task with a spatially heterogeneous, multi-tier reward field.

---

## Physics

Every tick, vectorised forces are applied via a Taichi kernel (with a numpy/cKDTree fallback in `_apply_forces_numpy_fallback`):

| Force | Rule |
|-------|------|
| **Brownian motion** | Gaussian displacement σ = `BROWNIAN_SIGMA = 1.0` µm/tick |
| **Short-range repulsion** | Cells inside `REPULSION_RADIUS = 8.0` µm push apart with force ∝ `(1 − d/r)²`. Inside a cluster, the radius shrinks to `CLUSTER_REPULSION_RADIUS = 5.0` µm so members pack tightly |
| **Adhesion spring** | Same-cluster cells beyond `ADHESION_REST_DIST = 5.0` µm are pulled back with `k = 0.90` |
| **Inter-cluster attraction** | Different clusters within `CLUSTER_ATTRACT_RADIUS = 30.0` µm feel a weak draw (`strength = 0.8`), encouraging mergers |

Per-tick displacement is hard-capped at `MAX_DISPLACEMENT = 4.0` µm to prevent tunnelling, and positions reflect off the four walls.

---

## Cooperation and the Task

### Lone cells
A lone cell earns a survival stipend `BASE_REWARD = 0.2 − 0.25 × adhesion_cost`. A lone **cooperator** additionally collects the regional 1-step reward for its operation:

```
fitness += BASE + max(0, regional_rewards[op] × coop_reward_scale − coop_cost)
```

A lone defector gets only the stipend — cooperation requires being in a cluster to share rewards.

### Clusters
A cluster's fitness is the sum across **all 12 tasks** of (regional reward × scale − defector drain), each only counted if the cluster has the required cooperators:

```
total_gross = Σ over tasks t:
    if cluster.can_complete(t):
        max(0, rvec[t] × coop_reward_scale − DEFECTOR_DRAIN × n_defectors)
per_cell = total_gross / cluster_size
```

A 2-step task requires one cooperator with `op1` AND one with `op2`. A 3-step task requires three. This forced **division of computational labour** is what selects for multicellularity.

---

## Defector Economics (the Public Goods Dilemma)

Three interacting mechanisms create the spatial public-goods game:

### 1. Cooperation cost
Cooperators in a successful cluster pay `COOP_COST = 0.1` per tick on top of their `ADHESION_COST = 0.1`. Defectors pay only adhesion.

### 2. Defector drain
Each defector in a cluster reduces every completed task's gross reward by `DEFECTOR_DRAIN = 1.5` (per task, not per cluster — clusters in tiles with many active tasks pay drain on all of them).

### 3. Within-cluster fitness advantage
Inside a successful cluster the defector earns `per_cell − adhesion_cost` while the cooperator earns `per_cell − adhesion_cost − COOP_COST`. The defector has a `COOP_COST = 0.1`/tick fitness advantage — the individual selection pressure that should drive defector spread.

The simulation is designed to show **multilevel selection**: individual selection favours defectors *within* clusters, group selection favours cooperator-only clusters *between* groups.

---

## Cell Death

Three independent death paths, all probabilistic:

| Cause | Condition |
|-------|-----------|
| **Old age** | At `age ≥ MAX_CELL_AGE = 500`, kill probability ramps from 0 → 1 over the next 250 ticks |
| **Starvation** | After `GRACE_PERIOD = 20` ticks, if `fitness/age < SURVIVAL_RATE = 0.02`, kill probability scales with the deficit (max 15%/tick) |
| **Out-competed** | At carrying capacity (`MAX_CELLS = 10_000`), the **oldest** nearby cell is killed to make room for a new offspring |

> **Why age-based competition?** Fitness-based competition strongly biases against defectors whose accumulated fitness is diluted by the clusters they infiltrate — effectively suppressing the very individual selection the model aims to study. Age-based culling is neutral w.r.t. cooperator/defector strategy.

---

## Cell cycle

Cells don't divide instantly when they hit the fitness threshold — they have to **grow to division size first**. Each cell carries a `size` attribute that starts at `NEWBORN_SIZE = 1.0` and grows by `GROWTH_RATE = 0.015`/tick (≈67 ticks to mature) **only when fitness > 0** — starving cells can't accumulate biomass. Replication is gated on `size ≥ DIVISION_SIZE = 2.0`, so the cell cycle paces division independently of fitness.

After successful replication, both parent and offspring reset to `size = 1.0` (the biomass has been split). For cluster-level replication, the **mean** member size must be division-ready, so a single fast-growing cooperator can't trigger cluster division while siblings are still juvenile.

## Replication

### Lone cells
A lone cell needs `fitness ≥ LONE_REPL_THRESH = 28.0`, `size ≥ 2.0`, and a passed stochastic gate (`LONE_REPL_PROB = 0.10`). **Cooperators additionally need ≥1 reward point** for their operation in the current tile — they can't blindly multiply where the landscape doesn't pay them. Defectors face no such gate (they free-ride wherever they land).

After replication, parent and offspring enter a `LONE_REPL_COOLDOWN = 50` tick lockout.

### Cluster-level replication
This is the key evolutionary transition modelled: **the unit of selection shifts from individual cells to the cooperative group.**

When a cluster's per-cell fitness exceeds `CLUSTER_REPL_THRESH = 18.0`, mean member size ≥ 2.0, and a `CLUSTER_REPL_PROB = 0.05` stochastic gate fires, the entire cluster replicates as a unit — every member cell mutates, and a daughter cluster is spawned ringed around the parent's anchor cell. Both clusters enter a `CLUSTER_REPL_COOLDOWN = 50` tick lockout.

Defectors inside a replicating cluster are copied into the offspring, allowing the defector lineage to spread through the cluster — mirroring how cancer-like cells arise within multicellular organisms.

---

## Bond breakage under strain

Cluster membership is not permanent. After every physics step, `_break_stretched_bonds` scans each cluster: any member whose distance to the cluster's centroid exceeds `BOND_BREAK_DIST = 60.0` µm (well beyond the attractive Gaussian's tail at 35 µm) detaches stochastically. The per-tick detach probability scales linearly with how far past the threshold the cell has drifted, capped at `BOND_BREAK_PROB_MAX = 0.05`. Detached cells revert to lone status and may rejoin a cluster via the normal kin-gated adhesion path. Empty clusters (size 0 after detachments) are pruned.

This models real cell-cell adhesions yielding under mechanical strain — without it, a cluster member dragged across the world by the random walk would remain logically bonded forever, sharing rewards from a centroid it can never reach.

## Adhesion Formation (kin-gated)

Cells with `has_adhesion = True` can form bonds within `ADHESION_FORM_RADIUS = 10.0` µm. The base bond probability `ADHESION_BOND_PROB = 0.20` is **scaled by genomic similarity**:

```
hamming = popcount(genomeA[:4] XOR genomeB[:4])    # over the 4 active bits
scale   = max(1.0 − 0.225 × hamming, 0.1)          # 0 diff → 1.0×, 4 diff → 0.1×
bond_prob = ADHESION_BOND_PROB × scale
```

Two paths:
- **Lone ↔ Lone** → form a new 2-cell cluster (kin-scaled)
- **Lone → existing cluster** → join up to `MAX_CLUSTER_SIZE = 12`, scaled by mean Hamming distance to existing members; base join probability is `ADHESION_BOND_PROB × 0.5 × scale`

Kin-gating makes pure-defector clusters easier to form (defectors are similar to defectors) but harder for an isolated defector to *infiltrate* an established cooperator cluster — the bond probability drops sharply with Hamming distance.

---

## Population seeding

Simulations don't start from random genomes alone. By default, **40% of the initial population is seeded as pre-formed cooperator clusters** of 2–4 cells with *distinct* operations (sampled without replacement from {AND, OR, XOR, NAND}). This guarantees seeds with genuine complementarity that can immediately attempt multi-step tasks, instead of waiting tens of thousands of ticks for random adhesion to assemble a working cluster. The remaining 60% spawn as lone cells with random genomes.

---

## MVG Extension (implemented)

When `task_flip_period` is set on the `Environment`, the entire reward landscape regenerates every N ticks via a fresh `Dirichlet`/`multinomial` draw:

```python
if self.task_flip_period is not None and self.tick_count % self.task_flip_period == 0:
    self._generate_regional_rewards()
```

This is the minimal MVG implementation: the structure of the reward landscape is preserved (12 tasks, three tiers, same masses), but *which* task each tile pays is re-rolled.

**Hypothesis tested**: dynamic environments select for clusters with more diverse operation portfolios.

**Empirical result (500-tick sweep, 3 seeds, periods ∈ {None, 50, 100, 250, 500})**:
- Cooperator fraction: **flat at ~96%** across all periods — MVG does not change the cooperate/defect equilibrium in this model.
- Average cluster size: **monotonic decrease with flip period** (3.91 at period=50 → 3.38 at period=500). Frequent flips slightly favor larger / more operation-diverse clusters that can survive landscape shifts.

---

## Visualization (Napari)

```bash
python visualize.py                         # 40 cells, 500 ticks, sample every 10
python visualize.py --sample-every 1        # every tick (memory-intensive)
python visualize.py --initial-cells 100     # denser start
python visualize.py --task-flip-period 100  # MVG mode
```

| Layer | Colour | Meaning |
|-------|--------|---------|
| `environment` | RGB heatmap | Gaussian density field — blobs show where each cell type is concentrated |
| `Lone cells` | Gray points | Cells not yet in any cluster |
| `Cooperators` | Blue points | Active cooperator inside a cluster |
| `Defectors` | Red points | Free-riders inside a cluster |

The time slider scrubs through recorded frames. The title bar updates live with tick number, total cell count, cluster count, cooperator %, and defector %.

---

## Key Parameters (current defaults — `src/environment.py`)

| Parameter | Default | Effect if increased |
|-----------|---------|---------------------|
| `coop_reward_scale` | 0.10 | Linearly scales every regional reward; clusters more profitable |
| `task_alpha` | 0.05 | Higher → flatter landscape (every task pays everywhere); lower → harsher specialisation |
| `coop_cost` | 0.1 | Stronger individual selection for defectors within clusters |
| `DEFECTOR_DRAIN` | 1.5 | Fewer defectors needed to crash a cluster |
| `ADHESION_BOND_PROB` | 0.20 | Faster cluster growth + more defector infiltration |
| `MAX_CLUSTER_SIZE` | 12 | Larger clusters; allows more defectors to accumulate before collapse |
| `mutation_rate` | 0.005 | More defector mutations arising within cluster lineages (asymmetric: 2× to defect) |
| `task_flip_period` | None | If set, re-rolls landscape every N ticks (MVG) |
| `MAX_CELLS` | 10 000 | Soft carrying capacity; oldest-near-spawn culling triggers above |

---

## Sweeps run (`results/`, `figures/`)

| Sweep | Axis | Configs | Seeds | Ticks |
|-------|------|---------|-------|-------|
| `population` | `initial_cells` | 5 | 3 | 1000 |
| `mutation_rate` | `mutation_rate` | 5 | 3 | 1000 |
| `mutation_penalty` | `mutation_rate` (extended range) | 8 | 3 | 1000 |
| `coop_bonus` | `coop_reward_scale` | 8 | 3 | 1000 |
| `coop_cost` | `coop_cost` | 8 | 3 | 1000 |
| `task_distribution` | `task_alpha` | 8 | 3 | 1000 |
| `grid_size` | `grid_size` | 3 | 3 | 1000 |
| `task_flip` | `task_flip_period` | 5 | 3 | 500 |

---

## References

- Lenski et al. — Avida digital evolution platform
- Kashtan & Alon (2005) — Spontaneous evolution of modularity and network motifs
- Chastain et al. — Multiplicative weights updates in game theory and evolution
- Nowak & May (1992) — Evolutionary games and spatial chaos
