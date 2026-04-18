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

from .cell import Cell
from .cluster import Cluster

# ── physics ───────────────────────────────────────────────────────────────────
BROWNIAN_SIGMA      = 0.18
REPULSION_RADIUS    = 1.4
REPULSION_STRENGTH  = 1.6
ADHESION_REST_DIST  = 1.45
ADHESION_SPRING_K   = 0.55
MAX_FORCE           = 1.6

# ── biology ───────────────────────────────────────────────────────────────────
MAX_CELL_AGE        = 150
GRACE_PERIOD        = 12
SURVIVAL_RATE       = 0.04

BASE_REWARD         = 1.0    # lone-cell reward per tick
COMPLEX_REWARD      = 20.0   # cluster reward per tick when task is solved

# Defector economics — these create the public-goods dilemma:
#   cooperators pay COOP_COST for the shared computation
#   each defector drains DEFECTOR_DRAIN from the gross reward pool
#
# With DRAIN=1.5 and COMPLEX_REWARD=20, cluster collapse timeline:
#   0 defectors  (2-cell pure) : per_cell = 9.7  → thriving
#   2 defectors  (4-cell mixed): per_cell = 4.25 → still profitable
#   4 defectors  (6-cell mixed): per_cell = 2.5  → marginal
#   7 defectors  (9-cell mixed): per_cell = 1.1  → cooperators starving
#   10 defectors (12-cell)     : per_cell = 0.4  → full collapse
COOP_COST           = 0.3    # metabolic cost paid ONLY by cooperators in a successful cluster
DEFECTOR_DRAIN      = 1.5    # per-defector reduction in gross cluster reward

LONE_REPL_THRESH    = 18.0
CLUSTER_REPL_THRESH = 10.0
LONE_REPL_COOLDOWN  = 20     # ticks between lone-cell replications
CLUSTER_REPL_COOLDOWN = 15   # ticks between cluster replications

MAX_CLUSTER_SIZE    = 12   # larger clusters → more room for defectors to accumulate
MAX_CELLS           = 900
ADHESION_BOND_PROB  = 0.20  # higher join rate so clusters actually grow large
ADHESION_FORM_RADIUS = 3.5  # slightly wider search radius


class Environment:
    def __init__(
        self,
        width: int = 50,
        height: int = 50,
        seed: Optional[int] = None,
    ) -> None:
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        self.width  = width
        self.height = height
        self.cells:    Dict[int, Cell]    = {}
        self.clusters: Dict[int, Cluster] = {}
        self.tick_count = 0
        self._refresh_env()

    # ── environment inputs ────────────────────────────────────────────────────

    def _refresh_env(self) -> None:
        self.env_a = random.randint(0, 255)
        self.env_b = random.randint(0, 255)
        self.env_c = random.randint(0, 255)
        self.task_target = (self.env_a & self.env_b) ^ self.env_c

    # ── placement helpers ─────────────────────────────────────────────────────

    def _rand_pos(self) -> Tuple[float, float]:
        return (
            random.uniform(0.5, self.width  - 0.5),
            random.uniform(0.5, self.height - 0.5),
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

    def find_empty_position(self) -> Tuple[float, float]:
        if not self.cells:
            return self._rand_pos()
        pos_arr = np.array([c.position for c in self.cells.values()])
        for _ in range(30):
            p = np.array(self._rand_pos())
            if np.linalg.norm(pos_arr - p, axis=1).min() > 1.0:
                return (float(p[0]), float(p[1]))
        return self._rand_pos()

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
            query_r = max(REPULSION_RADIUS, ADHESION_REST_DIST + 0.5)
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

    # ── adhesion formation ────────────────────────────────────────────────────

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
                if random.random() > ADHESION_BOND_PROB:
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
                if random.random() > ADHESION_BOND_PROB * 0.5:
                    continue
                idxs = cl_tree.query_ball_point(lone.position, ADHESION_FORM_RADIUS)
                if not idxs:
                    continue
                target = cluster_cells[idxs[0]]
                if target.cluster_id is None:
                    continue
                cl = self.clusters.get(target.cluster_id)
                if cl is not None and cl.size < MAX_CLUSTER_SIZE:
                    cl.add_cell(lone)

    # ── fitness evaluation ────────────────────────────────────────────────────

    def _lone_cells(self) -> List[Cell]:
        return [c for c in self.cells.values() if c.cluster_id is None]

    def _eval_lone_cell(self, cell: Cell) -> float:
        # Lone adhesive cells pay only 25% of full bond-maintenance cost —
        # adhesion molecules are only metabolically expensive when actively
        # holding cells together.  This keeps adhesive defectors alive while
        # they search for a cluster to infiltrate.
        lone_cost = cell.adhesion_cost * 0.25
        return max(0.0, BASE_REWARD - lone_cost)

    def _eval_cluster(self, cluster: Cluster) -> None:
        result = cluster.compute_task(self.env_a, self.env_b, self.env_c)
        if result is not None and result == self.task_target:
            # Each defector drains the reward pool, but cooperators also pay
            # an extra metabolic cost for the computation — giving defectors
            # a clear per-cell fitness advantage that drives their spread.
            gross    = max(0.0, COMPLEX_REWARD - DEFECTOR_DRAIN * cluster.defector_count)
            per_cell = gross / cluster.size
            for cell in cluster.cells:
                coop_cost = COOP_COST if cell.is_cooperator else 0.0
                cell.fitness += max(0.0, per_cell - cell.adhesion_cost - coop_cost)
            cluster.fitness += gross
        else:
            for cell in cluster.cells:
                cell.fitness += self._eval_lone_cell(cell)

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
        if len(self.cells) >= MAX_CELLS:
            self._kill_weakest_near(cell.position)
            if len(self.cells) >= MAX_CELLS or cell.position is None:
                return  # cell may have been the one killed
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
        ref = cluster.cells[0].position if cluster.cells else self._rand_pos()
        if ref is None:
            return
        if len(self.cells) >= MAX_CELLS:
            self._kill_weakest_near(ref)
            if len(self.cells) >= MAX_CELLS:
                return
        cluster.fitness -= CLUSTER_REPL_THRESH * cluster.size
        offspring = cluster.replicate()
        for child_cell in offspring.cells:
            self.place_cell(child_cell, self._pos_near(ref, r=4.0))
        if any(c.position is not None for c in offspring.cells):
            self.clusters[offspring.cluster_id] = offspring
        cluster.replication_cooldown   = CLUSTER_REPL_COOLDOWN
        offspring.replication_cooldown = CLUSTER_REPL_COOLDOWN

    # ── main tick ─────────────────────────────────────────────────────────────

    def tick(self) -> None:
        self.tick_count += 1
        self._refresh_env()

        self._apply_forces()

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
                self.kill_cell(cell)
                continue
            if cell.age > GRACE_PERIOD and (cell.fitness / cell.age) < SURVIVAL_RATE:
                self.kill_cell(cell)

        for cell in list(self._lone_cells()):
            if cell.cell_id in self.cells:
                self._replicate_lone(cell)
        for cluster in list(self.clusters.values()):
            if cluster.cluster_id in self.clusters:
                self._replicate_cluster(cluster)
