# Multicellularity Simulation — Design Document

**CS2212 Final Project · Andrew Rodriguez**

---

## Overview

This simulation models the computational leap from single-celled to multicellular life. Independent digital organisms evolve on a 2D spatial plane, competing for resources by executing bitwise logic tasks. The core tension is the **defector problem**: a mutant free-rider that joins a cooperative cluster, reaps shared rewards without paying the metabolic cost of computation, and mathematically should spread — yet real multicellular organisms somehow suppress this.

The simulation is built to explore what environmental and algorithmic conditions allow cooperation to emerge and persist against parasitic invasion.

---

## Architecture

```
multicellularity_sim/
├── src/
│   ├── cell.py          # genome encoding, physics state, mutation
│   ├── cluster.py       # cooperative group, task computation, replication
│   ├── environment.py   # physics engine, fitness evaluation, death, replication
│   └── visualizer.py    # Napari heatmap builder and viewer launcher
├── main.py              # headless run with console stats
├── visualize.py         # full GUI run via Napari
└── SIM_DESIGN.md        # this document
```

---

## The Genome

Each cell carries an **8-bit bitstring genome** that fully determines its behavior.

| Bits | Field | Values |
|------|-------|--------|
| 0–1  | Operation | `00`=AND · `01`=OR · `10`=XOR · `11`=NAND |
| 2    | Adhesion allele | `1` = can bond to adjacent cells (pays metabolic cost) |
| 3    | Cooperator allele | `1` = contributes computation to cluster; `0` = defector |
| 4–7  | Reserved | Available for future traits (signalling, apoptosis, etc.) |

Mutation flips each bit independently with probability `mutation_rate` (default 2%) at every replication event.

---

## The Environment

Cells live on a **continuous 2D plane** (default 50 × 50 world units) with **hard elastic walls** — positions reflect off boundaries instead of wrapping. Each tick the environment draws three random 8-bit integers **A, B, C** and sets the complex task target:

```
target = (A AND B) XOR C
```

---

## Physics

Every tick, three forces are applied to all cells simultaneously via vectorised numpy operations (scipy KDTree for neighbour queries):

| Force | Rule |
|-------|------|
| **Brownian motion** | Gaussian displacement σ = 0.18 world units/tick — constant thermal jitter |
| **Short-range repulsion** | Cells closer than 1.4 units push apart with force ∝ (1 − d/r)² — prevents overlap |
| **Adhesion spring** | Cells in the same cluster are pulled together when d > 1.45 units, with spring constant k = 0.55 — keeps clusters compact |

Forces are capped at 1.6 units/tick to prevent tunnelling. After force application, positions are reflected off walls.

---

## Cooperation and the Task

### Simple task (lone cells)
A lone cell earns `BASE_REWARD = 1.0` per tick just for existing. Adhesive cells pay a small maintenance cost (25% of full bond cost) even when not bonded.

### Complex task (clusters)
A cluster earns `COMPLEX_REWARD = 20.0` per tick **only if** it contains at least one AND-cooperator and one XOR-cooperator:

```
step 1: AND-cooperator computes  A AND B  → intermediate
step 2: XOR-cooperator computes  intermediate XOR C  → output
        if output == target: cluster earns full reward
```

This two-step task is deliberately too complex for a single cell — it requires **division of computational labour** across a minimum of two specialised cooperators. This is the algorithmic pressure that selects for multicellularity.

---

## Defector Economics (the Public Goods Dilemma)

The simulation implements a spatial public goods game with three interacting mechanisms:

### 1. Cooperation cost
When a cluster successfully solves the task, cooperators pay an extra `COOP_COST = 0.3` per tick for the shared computation. Defectors pay nothing extra — they receive the same per-cell share of the gross reward without contributing.

### 2. Defector drain
Each defector in a cluster reduces the gross reward pool by `DEFECTOR_DRAIN = 1.5`:

```
gross = max(0,  COMPLEX_REWARD  −  DEFECTOR_DRAIN × n_defectors)
per_cell = gross / cluster_size
```

Collapse timeline for a cluster with 2 cooperators:

| Defectors | Cluster size | Per-cell (cooperator) | State |
|-----------|-------------|----------------------|-------|
| 0 | 2 | 9.7 / tick | thriving |
| 2 | 4 | 4.25 / tick | profitable |
| 4 | 6 | 2.5 / tick | marginal |
| 7 | 9 | 1.1 / tick | cooperators starving |
| 10 | 12 | 0.4 / tick | full collapse |

### 3. Within-cluster fitness advantage
Inside a successful cluster the defector earns `per_cell − adhesion_cost` while the cooperator earns `per_cell − adhesion_cost − COOP_COST`. The defector has a **~0.3 units/tick fitness advantage** — the individual selection pressure that should drive defector spread.

The simulation is designed to show multilevel selection: individual selection favours defectors *within* clusters, group selection favours cooperator-only clusters *between* groups.

---

## Cell Death

Three independent death paths are active each tick:

| Cause | Condition |
|-------|-----------|
| **Old age** | `cell.age ≥ 150` ticks |
| **Starvation** | `cell.fitness / cell.age < 0.04` after a 12-tick grace period |
| **Out-competed** | At carrying capacity (`MAX_CELLS = 900`), the **oldest** nearby cell is killed to make room for a new offspring — age-based death is neutral w.r.t. the cooperator/defector strategy so within-cluster selection can actually operate |

> **Why age-based competition?** Fitness-based competition at carrying capacity strongly biases against defectors whose accumulated fitness is diluted by the clusters they infiltrate — effectively suppressing the very individual selection the model aims to study.

---

## Replication

### Lone cells
Accumulate fitness until they reach `LONE_REPL_THRESH = 18.0`, then place a mutated offspring within radius 3 of the parent. Both parent and offspring enter a `LONE_REPL_COOLDOWN = 20` tick lockout.

### Cluster-level replication
This is the key evolutionary transition modelled: **the unit of selection shifts from individual cells to the cooperative group.**

When a cluster's per-cell fitness exceeds `CLUSTER_REPL_THRESH = 10.0`, the entire cluster replicates as a unit — every member cell mutates and a daughter cluster is spawned nearby. Both parent and offspring clusters enter a `CLUSTER_REPL_COOLDOWN = 15` tick lockout.

Defectors inside a replicating cluster are copied into the offspring, allowing the defector genome to spread through the cluster lineage — mirroring how cancer-like cells can arise within multicellular organisms.

---

## Adhesion Formation

Cells with the adhesion allele (`genome[2] = 1`) can form clusters when they come within `ADHESION_FORM_RADIUS = 3.5` world units of another adhesive cell. Bond formation fires with probability `ADHESION_BOND_PROB = 0.20` per eligible pair per tick.

Two paths:
- **Lone ↔ Lone**: two unattached adhesive cells found by KDTree → form a new 2-cell cluster
- **Lone → Cluster**: a lone adhesive cell joins an existing cluster (probability 0.10/tick) if the cluster has not reached `MAX_CLUSTER_SIZE = 12`

Defectors can join clusters through both paths. They infiltrate from outside as well as arising from mutation within cluster offspring.

---

## Visualization (Napari)

```bash
python visualize.py                         # 40 cells, 500 ticks, sample every 10
python visualize.py --sample-every 1        # every tick (memory-intensive)
python visualize.py --initial-cells 100     # denser start
```

| Layer | Colour | Meaning |
|-------|--------|---------|
| `environment` | RGB heatmap | Gaussian density field — blobs show where each cell type is concentrated |
| `Lone cells` | Gray points | Cells not yet in any cluster |
| `Cooperators` | Blue points | AND/XOR contributors inside clusters |
| `Defectors` | Red points | Free-riders inside clusters |

The time slider at the bottom of Napari scrubs through recorded frames. The title bar updates live with tick number, total cell count, cluster count, cooperator %, and defector %.

**Controls:**
- Scrub bottom slider → move through time
- Eye icon on layer → toggle cell type visibility
- Scroll / pinch → zoom in on cluster detail
- Click-drag → pan

---

## Key Parameters (tuning guide)

| Parameter | Default | Effect if increased |
|-----------|---------|---------------------|
| `COMPLEX_REWARD` | 20.0 | Clusters more attractive; more headroom before defectors crash them |
| `DEFECTOR_DRAIN` | 1.5 | Fewer defectors needed to crash a cluster |
| `COOP_COST` | 0.3 | Stronger individual selection for defectors within clusters |
| `ADHESION_BOND_PROB` | 0.20 | Faster cluster growth; more defector infiltration |
| `MAX_CLUSTER_SIZE` | 12 | Larger clusters; allows more defectors to accumulate before collapse |
| `CLUSTER_REPL_COOLDOWN` | 15 | Slower cluster growth; more observable dynamics |
| `mutation_rate` | 0.02 | More defector mutations arising within cluster lineages |

---

## MVG Extension (planned)

The Modularly Varying Goals (MVG) extension periodically flips the structure of the complex task — e.g., switching from `(A AND B) XOR C` to `(A OR B) XNOR C`. This forces clusters to rapidly rewire their cooperative networks.

The hypothesis to test: **static environments produce brittle, easily exploited clusters, while fluctuating environments force the evolution of modular adhesion networks that can identify and exclude defectors.**

To enable: add a `task_flip_period` parameter to `Environment` and alternate between two task structures on the given interval.

---

## References

- Lenski et al. — Avida digital evolution platform
- Kashtan & Alon (2005) — Spontaneous evolution of modularity and network motifs
- Chastain et al. — Multiplicative weights updates in game theory and evolution
- Nowak & May (1992) — Evolutionary games and spatial chaos
