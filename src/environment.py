"""
Physics-based environment: cells have continuous (x, y) float positions.

Each tick applies:
  1. Brownian motion   — thermal jitter
  2. Short-range repulsion — cells push apart when closer than REPULSION_RADIUS
  3. Adhesion springs  — same-cluster cells pulled back to ADHESION_REST_DIST
  4. Hard walls        — elastic reflection at all four boundaries (no wrapping)

Defector economics
------------------
  cooperators   pay COOP_COST on top of adhesion_cost when cluster succeeds
  defectors     do NOT pay COOP_COST  → per-tick fitness advantage within cluster
  each defector drains DEFECTOR_DRAIN from the cluster's gross reward
  → a cluster with many defectors produces less total reward AND cooperators
    are hit by both the diluted share AND COOP_COST; their per-tick fitness
    drops below BASE_REWARD, making the cluster worse than being alone

Replication cooldown
--------------------
  After any replication event (cell or cluster), the offspring and parent
  both enter a cooldown window (LONE_REPL_COOLDOWN / CLUSTER_REPL_COOLDOWN
  ticks) during which replication is blocked, preventing instant cascades.
"""

import random
import numpy as np
from scipy.spatial import cKDTree
from typing import Dict, List, Optional, Tuple
from .cell import Cell, _apply_op
from .cluster import Cluster

from .cell import Cell
from .cluster import Cluster

# ── physics ───────────────────────────────────────────────────────────────────
BROWNIAN_SIGMA      = 0.1
REPULSION_RADIUS    = 1.0
REPULSION_STRENGTH  = 20.0
ADHESION_REST_DIST  = 1.25   # just above REPULSION_RADIUS — bonded cells sit nearly touching
ADHESION_SPRING_K   = 0.40   # must be <0.5 for Euler stability
MAX_FORCE           = 4.0

# ── biology ───────────────────────────────────────────────────────────────────
MAX_CELL_AGE        = 500    # longer lifespan — cells persist through lean patches
GRACE_PERIOD        = 120    # more time before starvation check kicks in
SURVIVAL_RATE       = 0.02   # minimum fitness/age ratio to avoid starvation

COOP_REWARD_SCALE = 0.15  # global multiplier on all regional vector values (vector entries sum to 15,
                          # so max single-task reward = 15 × scale; tune this to adjust cooperation incentive)

BASE_REWARD     = 0.2    # survival stipend — NOT scaled, stays as absolute floor
SIMPLE_REWARD   = 2.0    # kept for reference; actual rewards now come from scaled vector entries
COMPLEX_REWARD  = 7.0
TRIPLE_REWARD   = 15.0

# Task-complexity sampling weights [1-step, 2-step, 3-step].
# 1-step tasks are most common; 3-step tasks are rare but lucrative.
TASK_STEP_WEIGHTS = [0.55, 0.35, 0.10]

# Defector economics — these create the public-goods dilemma:
#   cooperators pay COOP_COST for the shared computation
#   each defector drains DEFECTOR_DRAIN from each task's gross reward
COOP_COST           = 0.1    # metabolic cost paid ONLY by cooperators when cluster succeeds
DEFECTOR_DRAIN      = 0.8    # per-defector drain applied to each completed task's gross reward

LONE_REPL_THRESH    = 28.0   # higher bar → slower lone-cell reproduction
CLUSTER_REPL_THRESH = 18.0   # higher bar → slower cluster reproduction
LONE_REPL_COOLDOWN  = 50     # ticks between lone-cell replications
CLUSTER_REPL_COOLDOWN = 50   # ticks between cluster replications
LONE_REPL_PROB      = 0.10   # stochastic: probability of actually dividing each tick when ready
CLUSTER_REPL_PROB   = 0.05   # stochastic: probability of cluster dividing each tick when ready

MAX_CLUSTER_SIZE    = 12
MAX_CELLS           = 10_000
ADHESION_BOND_PROB  = 0.20
ADHESION_FORM_RADIUS = 1.2
ADHESION_SNAP_DIST  = 1.8

REGION_SIZE         = 10    # world-unit side length of each spatial task tile

# Ordered list of all 12 task tuples — index into the regional reward vector
ALL_TASKS: List[tuple] = [
    ("AND",), ("OR",), ("XOR",), ("NAND",),                          # 1-step (indices 0-3)
    ("AND", "XOR"), ("NAND", "OR"), ("OR", "AND"), ("XOR", "NAND"),  # 2-step (indices 4-7)
    ("AND", "OR", "XOR"), ("OR", "NAND", "AND"),                     # 3-step (indices 8-11)
    ("XOR", "AND", "OR"),  ("NAND", "XOR", "AND"),
]
_OP_TO_IDX = {"AND": 0, "OR": 1, "XOR": 2, "NAND": 3}  # 1-step op → vector index


class Environment:
    def __init__(
            self,
            width: int = 250,
            height: int = 250,
            seed: Optional[int] = None,
            task_flip_period: Optional[int] = None,  # <-- New parameter
        ) -> None:
            if seed is not None:
                np.random.seed(seed)
                random.seed(seed)

            self.width  = width
            self.height = height
            self.cells:    Dict[int, Cell]    = {}
            self.clusters: Dict[int, Cluster] = {}
            self.tick_count = 0
            
            self.task_flip_period = task_flip_period

            # Task pools, partitioned by complexity
            self._task_pool = {
                1: [("AND",), ("OR",), ("XOR",), ("NAND",)],
                2: [("AND", "XOR"), ("NAND", "OR"), ("OR", "AND"), ("XOR", "NAND")],
                3: [("AND", "OR", "XOR"), ("OR", "NAND", "AND"),
                    ("XOR", "AND", "OR"),  ("NAND", "XOR", "AND")],
            }

            # Spatial reward landscape — fixed at startup
            self._generate_regional_rewards()

    # ── environment inputs ────────────────────────────────────────────────────

    def _generate_regional_rewards(self) -> None:
        """
        Each tile gets a reward vector of length 12 (one entry per task in ALL_TASKS)
        with integer values summing to 15, drawn from a Dirichlet-Multinomial:
          θ ~ Dirichlet(α=0.5)   ← sparse prior → regions specialise in a few tasks
          v ~ Multinomial(15, θ)
        """
        n_rows = max(1, int(np.ceil(self.width  / REGION_SIZE)))
        n_cols = max(1, int(np.ceil(self.height / REGION_SIZE)))
        n_tasks = len(ALL_TASKS)
        alpha = np.full(n_tasks, 0.5)
        self.regional_rewards = np.zeros((n_rows, n_cols, n_tasks), dtype=np.float32)
        for r in range(n_rows):
            for c in range(n_cols):
                probs = np.random.dirichlet(alpha)
                self.regional_rewards[r, c] = np.random.multinomial(15, probs).astype(np.float32)

    def _get_regional_rewards(self, pos: Tuple[float, float]) -> np.ndarray:
        """Return the (12,) reward vector for the tile containing pos."""
        x, y   = pos
        n_rows = self.regional_rewards.shape[0]
        n_cols = self.regional_rewards.shape[1]
        row = min(int(x / REGION_SIZE), n_rows - 1)
        col = min(int(y / REGION_SIZE), n_cols - 1)
        return self.regional_rewards[row, col]

    # ── placement helpers ─────────────────────────────────────────────────────

    def _rand_pos(self) -> Tuple[float, float]:
        return (
            random.uniform(0.5, self.width  - 0.5),
            random.uniform(0.5, self.height - 0.5),
        )

    def _rand_pos_seeding(self) -> Tuple[float, float]:
        """Central 40% of each axis — leaves room to expand toward edges."""
        margin = 0.30
        return (
            random.uniform(self.width  * margin, self.width  * (1.0 - margin)),
            random.uniform(self.height * margin, self.height * (1.0 - margin)),
        )

    def _pos_near(self, pos: Tuple[float, float], r: float = 3.0
                  ) -> Tuple[float, float]:
        angle = random.uniform(0, 2 * np.pi)
        dist  = random.uniform(0.8, r)
        x = float(np.clip(pos[0] + dist * np.cos(angle), 0.5, self.width  - 0.5))
        y = float(np.clip(pos[1] + dist * np.sin(angle), 0.5, self.height - 0.5))
        return (x, y)

    def place_cell(self, cell: Cell, pos: Tuple[float, float]) -> bool:
        cell.position = pos
        self.cells[cell.cell_id] = cell
        return True
    
    @property
    def current_task_str(self) -> str:
        nr, nc = self.regional_rewards.shape[:2]
        return f"Reward vectors ({nr}×{nc} tiles, {REGION_SIZE}wu, Σ=15 each)"

    def find_empty_position(self, seeding: bool = False) -> Tuple[float, float]:
        rand_fn = self._rand_pos_seeding if seeding else self._rand_pos
        if not self.cells:
            return rand_fn()
        pos_arr = np.array([c.position for c in self.cells.values()])
        for _ in range(30):
            p = np.array(rand_fn())
            if np.linalg.norm(pos_arr - p, axis=1).min() > 1.0:
                return (float(p[0]), float(p[1]))
        return rand_fn()

    def kill_cell(self, cell: Cell) -> None:
        if cell.cluster_id is not None:
            cl = self.clusters.get(cell.cluster_id)
            if cl is not None:
                cl.remove_cell(cell)
                if cl.size == 0:
                    del self.clusters[cl.cluster_id]
        cell.position = None
        self.cells.pop(cell.cell_id, None)

    # ── physics ───────────────────────────────────────────────────────────────

    def _apply_forces(self) -> None:
        cell_list = list(self.cells.values())
        N = len(cell_list)
        if N == 0:
            return

        pos    = np.array([c.position for c in cell_list], dtype=np.float64)
        forces = np.random.normal(0, BROWNIAN_SIGMA, (N, 2))

        if N > 1:
            query_r = max(REPULSION_RADIUS, ADHESION_SNAP_DIST)
            tree    = cKDTree(pos)
            pairs   = tree.query_pairs(query_r, output_type='ndarray')

            if len(pairs):
                ii, jj = pairs[:, 0], pairs[:, 1]
                delta  = pos[jj] - pos[ii]                   # (P, 2)
                d      = np.linalg.norm(delta, axis=1)        # (P,)

                degen = d < 1e-9
                if degen.any():
                    delta[degen] = np.random.normal(0, 0.1, (degen.sum(), 2))
                    d[degen] = np.maximum(
                        np.linalg.norm(delta[degen], axis=1), 1e-9
                    )

                unit = delta / d[:, np.newaxis]               # (P, 2)

                # repulsion
                rep_m  = d < REPULSION_RADIUS
                f_rep  = np.zeros(len(pairs))
                f_rep[rep_m] = (
                    REPULSION_STRENGTH * (1.0 - d[rep_m] / REPULSION_RADIUS) ** 2
                )
                fv_rep = f_rep[:, np.newaxis] * unit
                np.add.at(forces, ii, -fv_rep)
                np.add.at(forces, jj, +fv_rep)

                # adhesion spring (same cluster, stretched beyond rest dist)
                cids  = np.array(
                    [c.cluster_id if c.cluster_id is not None else -1
                     for c in cell_list]
                )
                same  = (cids[ii] >= 0) & (cids[ii] == cids[jj])
                adh_m = same & (d > ADHESION_REST_DIST)
                f_adh = np.zeros(len(pairs))
                f_adh[adh_m] = ADHESION_SPRING_K * (d[adh_m] - ADHESION_REST_DIST)
                fv_adh = f_adh[:, np.newaxis] * unit
                np.add.at(forces, ii, +fv_adh)
                np.add.at(forces, jj, -fv_adh)

        # clamp
        mags  = np.linalg.norm(forces, axis=1)
        safe  = np.maximum(mags, 1e-10)
        scale = np.where(mags > MAX_FORCE, MAX_FORCE / safe, 1.0)
        forces *= scale[:, np.newaxis]

        pos += forces

        # ── elastic wall reflection (no wrapping) ────────────────────────────
        for axis, limit in ((0, self.width), (1, self.height)):
            lo  = pos[:, axis] < 0.0
            hi  = pos[:, axis] > limit
            pos[lo, axis] =  -pos[lo, axis]
            pos[hi, axis] = 2.0 * limit - pos[hi, axis]
            # hard clamp in case of overshoots beyond far wall
            np.clip(pos[:, axis], 0.0, limit, out=pos[:, axis])

        for k, cell in enumerate(cell_list):
            cell.position = (float(pos[k, 0]), float(pos[k, 1]))

    # ── bond snapping ─────────────────────────────────────────────────────────

    def _snap_broken_bonds(self) -> None:
        """
        After physics, detach any cell that has drifted farther than
        ADHESION_SNAP_DIST from every other member of its cluster.
        Clusters reduced to ≤1 cell are disbanded.
        """
        for cl in list(self.clusters.values()):
            if cl.cluster_id not in self.clusters:
                continue
            positioned = [c for c in cl.cells if c.position is not None]
            if len(positioned) < 2:
                continue
            pos_arr = np.array([c.position for c in positioned])

            to_detach = []
            for i, cell in enumerate(positioned):
                dists = np.linalg.norm(pos_arr - pos_arr[i], axis=1)
                dists[i] = np.inf
                if dists.min() > ADHESION_SNAP_DIST:
                    to_detach.append(cell)

            for cell in to_detach:
                cl.remove_cell(cell)

            # Disband if too small to cooperate
            if cl.size <= 1:
                for remaining in list(cl.cells):
                    cl.remove_cell(remaining)
                if cl.cluster_id in self.clusters:
                    del self.clusters[cl.cluster_id]

    # ── adhesion formation ────────────────────────────────────────────────────

    @staticmethod
    def _kin_bond_prob(cell_a: "Cell", cell_b: "Cell", base_prob: float) -> float:
        """
        Scale base_prob by genomic similarity on the 4 active bits (0-3).
        Hamming distance 0 → 1.0× base; distance 4 → 0.1× base.
        This lets kin cluster together and makes defector infiltration harder.
        """
        hamming = int(np.sum(cell_a.genome[:4] != cell_b.genome[:4]))
        scale   = 1.0 - 0.225 * hamming   # 0 diff→1.0, 4 diff→0.1
        return base_prob * max(scale, 0.1)

    def _try_adhesion(self) -> None:
        adhesive_lone = [
            c for c in self.cells.values()
            if c.has_adhesion and c.cluster_id is None
        ]
        if len(adhesive_lone) < 1:
            return

        lone_pos = np.array([c.position for c in adhesive_lone])

        # lone ↔ lone → new cluster
        if len(adhesive_lone) >= 2:
            tree  = cKDTree(lone_pos)
            pairs = tree.query_pairs(ADHESION_FORM_RADIUS, output_type='ndarray')
            for i, j in pairs:
                ci, cj = adhesive_lone[i], adhesive_lone[j]
                if ci.cluster_id is not None or cj.cluster_id is not None:
                    continue
                bond_prob = self._kin_bond_prob(ci, cj, ADHESION_BOND_PROB)
                if random.random() > bond_prob:
                    continue
                cl = Cluster()
                cl.add_cell(ci)
                cl.add_cell(cj)
                self.clusters[cl.cluster_id] = cl

        # lone → existing cluster
        cluster_cells = [
            c for cl in self.clusters.values()
            for c in cl.cells if c.position is not None
        ]
        if cluster_cells:
            cl_pos  = np.array([c.position for c in cluster_cells])
            cl_tree = cKDTree(cl_pos)
            for lone in adhesive_lone:
                if lone.cluster_id is not None:
                    continue
                idxs = cl_tree.query_ball_point(lone.position, ADHESION_FORM_RADIUS)
                if not idxs:
                    continue
                target = cluster_cells[idxs[0]]
                if target.cluster_id is None:
                    continue
                cl = self.clusters.get(target.cluster_id)
                if cl is None or cl.size >= MAX_CLUSTER_SIZE:
                    continue
                # Kin check: compare lone cell against the nearest cluster member
                mean_hamming = float(np.mean([
                    np.sum(lone.genome[:4] != c.genome[:4]) for c in cl.cells
                ]))
                scale    = max(1.0 - 0.225 * mean_hamming, 0.1)
                join_prob = ADHESION_BOND_PROB * 0.5 * scale
                if random.random() <= join_prob:
                    cl.add_cell(lone)

    # ── fitness evaluation ────────────────────────────────────────────────────

    def _lone_cells(self) -> List[Cell]:
        return [c for c in self.cells.values() if c.cluster_id is None]

    def _eval_lone_cell(self, cell: Cell) -> float:
        lone_cost = cell.adhesion_cost * 0.25
        if cell.is_cooperator and cell.position is not None:
            rvec   = self._get_regional_rewards(cell.position)
            r_val  = float(rvec[_OP_TO_IDX[cell.operation]]) * COOP_REWARD_SCALE
            if r_val > 0:
                return max(0.0, r_val - lone_cost)
        return max(0.0, BASE_REWARD - lone_cost)

    def _eval_cluster(self, cluster: Cluster) -> None:
        positioned = [c for c in cluster.cells if c.position is not None]
        if not positioned:
            return
        cx   = float(np.mean([c.position[0] for c in positioned]))
        cy   = float(np.mean([c.position[1] for c in positioned]))
        rvec = self._get_regional_rewards((cx, cy))

        total_gross = 0.0
        for i, task in enumerate(ALL_TASKS):
            if rvec[i] == 0:
                continue
            op1 = task[0]
            op2 = task[1] if len(task) > 1 else None
            op3 = task[2] if len(task) > 2 else None
            if cluster.can_complete_task(op1, op2, op3):
                total_gross += max(0.0, float(rvec[i]) * COOP_REWARD_SCALE - DEFECTOR_DRAIN * cluster.defector_count)

        if total_gross > 0:
            per_cell = total_gross / cluster.size
            for cell in cluster.cells:
                coop_cost = COOP_COST if cell.is_cooperator else 0.0
                cell.fitness += max(0.0, per_cell - cell.adhesion_cost - coop_cost)
            cluster.fitness += total_gross
        else:
            for cell in cluster.cells:
                cell.fitness += max(0.0, BASE_REWARD - cell.adhesion_cost * 0.25)

    # ── replication ───────────────────────────────────────────────────────────

    def _kill_weakest_near(self, ref: Tuple[float, float]) -> None:
        """
        Kill the oldest nearby cell to make population room.
        Age-based death is neutral w.r.t. the cooperator/defector strategy
        so within-cluster selection pressure (COOP_COST) can actually operate.
        Fitness-based killing biases strongly against defectors whose fitness
        is diluted by the very clusters they infiltrate.
        """
        cell_list = list(self.cells.values())
        if not cell_list:
            return
        pos_arr = np.array([c.position for c in cell_list])
        dists   = np.linalg.norm(pos_arr - np.array(ref), axis=1)
        near    = np.where(dists < 15.0)[0]
        if len(near) == 0:
            near = np.arange(len(cell_list))
        ages = np.array([cell_list[k].age for k in near])
        self.kill_cell(cell_list[near[np.argmax(ages)]])

    def _replicate_lone(self, cell: Cell) -> None:
        if cell.position is None or cell.replication_cooldown > 0:
            return
        if cell.fitness < LONE_REPL_THRESH:
            return
        # Defectors replicate freely; cooperators need ≥1 reward point for their op in this region.
        if not cell.is_defector:
            rvec = self._get_regional_rewards(cell.position)
            if float(rvec[_OP_TO_IDX[cell.operation]]) < 1.0:
                return
        # Stochastic gate: division timing has biological noise
        if random.random() > LONE_REPL_PROB:
            return
        if len(self.cells) >= MAX_CELLS:
            self._kill_weakest_near(cell.position)
            if len(self.cells) >= MAX_CELLS or cell.position is None:
                return
        cell.fitness -= LONE_REPL_THRESH
        child = cell.mutate()
        self.place_cell(child, self._pos_near(cell.position, r=3.0))
        cell.replication_cooldown  = LONE_REPL_COOLDOWN
        child.replication_cooldown = LONE_REPL_COOLDOWN

    def _replicate_cluster(self, cluster: Cluster) -> None:
        if cluster.replication_cooldown > 0:
            return
        per_cell = cluster.fitness / max(cluster.size, 1)
        if per_cell < CLUSTER_REPL_THRESH:
            return
        # Stochastic gate: prevents synchronized division pulses across clusters
        if random.random() > CLUSTER_REPL_PROB:
            return
        ref = cluster.cells[0].position if cluster.cells else self._rand_pos()
        if ref is None:
            return
        if len(self.cells) >= MAX_CELLS:
            self._kill_weakest_near(ref)
            if len(self.cells) >= MAX_CELLS:
                return
        cluster.fitness -= CLUSTER_REPL_THRESH * cluster.size
        offspring = cluster.replicate()
        # Place offspring in a tight ring so they stay within ADHESION_SNAP_DIST
        # and the spring can pull them to ADHESION_REST_DIST immediately.
        n = len(offspring.cells)
        for i, child_cell in enumerate(offspring.cells):
            angle = 2 * np.pi * i / max(n, 1)
            ox = float(np.clip(ref[0] + ADHESION_REST_DIST * np.cos(angle), 0.5, self.width  - 0.5))
            oy = float(np.clip(ref[1] + ADHESION_REST_DIST * np.sin(angle), 0.5, self.height - 0.5))
            self.place_cell(child_cell, (ox, oy))
        if any(c.position is not None for c in offspring.cells):
            self.clusters[offspring.cluster_id] = offspring
        cluster.replication_cooldown   = CLUSTER_REPL_COOLDOWN
        offspring.replication_cooldown = CLUSTER_REPL_COOLDOWN

    # ── main tick ─────────────────────────────────────────────────────────────

    def tick(self) -> None:
        self.tick_count += 1
        self._apply_forces()
        self._snap_broken_bonds()

        for cell in self._lone_cells():
            cell.fitness += self._eval_lone_cell(cell)
        for cluster in list(self.clusters.values()):
            self._eval_cluster(cluster)

        for cell in list(self.cells.values()):
            cell.tick()
            if cell.replication_cooldown > 0:
                cell.replication_cooldown -= 1
        for cluster in list(self.clusters.values()):
            cluster.tick()
            if cluster.replication_cooldown > 0:
                cluster.replication_cooldown -= 1

        self._try_adhesion()

        for cell in list(self.cells.values()):
            if cell.cell_id not in self.cells:
                continue
            if cell.age >= MAX_CELL_AGE:
                # Old-age death: probability ramps from 0 at MAX_CELL_AGE to 1 at 1.5× that age
                p_age = (cell.age - MAX_CELL_AGE) / (MAX_CELL_AGE * 0.5)
                if random.random() < min(p_age, 1.0):
                    self.kill_cell(cell)
                continue
            if cell.age > GRACE_PERIOD:
                deficit = SURVIVAL_RATE - (cell.fitness / cell.age)
                if deficit > 0:
                    # Starvation: probability scales with how far below the threshold
                    p_starve = min(deficit / SURVIVAL_RATE, 1.0) * 0.15
                    if random.random() < p_starve:
                        self.kill_cell(cell)

        for cell in list(self._lone_cells()):
            if cell.cell_id in self.cells:
                self._replicate_lone(cell)
        for cluster in list(self.clusters.values()):
            if cluster.cluster_id in self.clusters:
                self._replicate_cluster(cluster)
