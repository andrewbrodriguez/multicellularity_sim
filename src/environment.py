"""
Physics-based environment. Cells have continuous (x, y) float positions.

Each tick:
  1. Brownian motion + short-range repulsion + intra-cluster adhesion
  2. Hard wall reflection at the world edges
  3. Fitness eval against the regional reward landscape (no basal income)
  4. Stretched-bond breakage, then kin-gated adhesion attempts
  5. Probabilistic age/starvation death, then replication

Fitness/health rules:
  - Every cell pays MAINTENANCE_COST + adhesion_cost every tick.
  - Lone cell with op X in a tile rewarding 1-step task X earns rvec[X] * scale.
  - For each task t in ALL_TASKS rewarded by the cluster's tile that the cluster
    can compute, the cluster earns max(0, rvec[t]*scale - DEFECTOR_DRAIN*n_def);
    that sum splits across the entire cluster (cooperators AND defectors).
    Cooperators additionally pay coop_cost when total_gross > 0.
  - Cells with no income source atrophy and starve out.
"""

import random
import numpy as np
from scipy.spatial import cKDTree
from typing import Dict, List, Optional, Tuple

from .cell import Cell, _apply_op
from .cluster import Cluster
from . import physics_taichi as _ti_physics


# physics
BROWNIAN_SIGMA           = 1.0    # µm/tick thermal jitter
REPULSION_RADIUS         = 8.0    # µm cell diameter
CLUSTER_REPULSION_RADIUS = 5.0    # µm
REPULSION_STRENGTH       = 20.0
ADHESION_REST_DIST       = 5.0
ADHESION_SPRING_K        = 0.90
MAX_INTERACTION_RADIUS   = 35.0   # µm — hard cutoff
CLUSTER_ATTRACT_RADIUS   = 30.0   # µm — long-range inter-cluster onset
CLUSTER_ATTRACT_STRENGTH = 0.4
MAX_DISPLACEMENT         = 4.0    # µm/tick

# biology
MAX_CELL_AGE  = 500
GRACE_PERIOD  = 10
SURVIVAL_RATE = 0.02   # min fitness/age ratio to avoid starvation

COOP_REWARD_SCALE = 1.5
MAINTENANCE_COST  = 0.01

# Reward mass per tile, partitioned by task complexity
SINGLE_MASS = 6
DOUBLE_MASS = 8
TRIPLE_MASS = 11

# Defector economics — the public-goods dilemma
COOP_COST      = 0.5     # paid by cooperators only when cluster earned
DEFECTOR_DRAIN = 1.5     # subtracted per-defector from each completed task

LONE_REPL_THRESH       = 100.0
CLUSTER_REPL_THRESH    = 100.0   # PER-CELL; cluster divides on mean(health)
DEFECTOR_REPL_THRESH   = 50.0    # in-cluster defectors halve the cost — cancer-like
LONE_REPL_COOLDOWN     = 10
CLUSTER_REPL_COOLDOWN  = 10
LONE_REPL_PROB         = 0.5
CLUSTER_REPL_PROB      = 0.25

MAX_CLUSTER_SIZE     = 10
MAX_CELLS            = 10_000
ADHESION_BOND_PROB   = 0.5
ADHESION_FORM_RADIUS = 20.0

BOND_BREAK_DIST     = 40.0
BOND_BREAK_PROB_MAX = 0.25

REGION_SIZE = 100   # µm — task region tile size

ALL_TASKS: List[tuple] = [
    ("AND",), ("OR",), ("XOR",), ("NAND",),                          # 1-step (idx 0-3)
    ("AND", "XOR"), ("NAND", "OR"), ("OR", "AND"), ("XOR", "NAND"),  # 2-step (idx 4-7)
    ("AND", "OR", "XOR"), ("OR", "NAND", "AND"),                     # 3-step (idx 8-11)
    ("XOR", "AND", "OR"),  ("NAND", "XOR", "AND"),
]
_OP_TO_IDX = {"AND": 0, "OR": 1, "XOR": 2, "NAND": 3}


class Environment:
    def __init__(
        self,
        width: int = 500,
        height: int = 500,
        seed: Optional[int] = None,
        task_flip_period: Optional[int] = None,
        coop_reward_scale: float = COOP_REWARD_SCALE,
        task_alpha: float = 0.5,
        coop_cost: float = COOP_COST,
    ) -> None:
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        self.width  = width
        self.height = height
        self.cells:    Dict[int, Cell]    = {}
        self.clusters: Dict[int, Cluster] = {}
        self.tick_count = 0

        self.task_flip_period  = task_flip_period
        self.coop_reward_scale = coop_reward_scale
        self.task_alpha        = task_alpha
        self.coop_cost         = coop_cost

        self._task_pool = {
            1: [("AND",), ("OR",), ("XOR",), ("NAND",)],
            2: [("AND", "XOR"), ("NAND", "OR"), ("OR", "AND"), ("XOR", "NAND")],
            3: [("AND", "OR", "XOR"), ("OR", "NAND", "AND"),
                ("XOR", "AND", "OR"),  ("NAND", "XOR", "AND")],
        }

        self._generate_regional_rewards()

    def _generate_regional_rewards(self) -> None:
        # Multi-step tasks get more mass than 1-step tasks so that complementary
        # cooperation pays comfortably better than a single cell's op alone.
        n_rows  = max(1, int(np.ceil(self.width  / REGION_SIZE)))
        n_cols  = max(1, int(np.ceil(self.height / REGION_SIZE)))
        n_tasks = len(ALL_TASKS)
        self.regional_rewards = np.zeros((n_rows, n_cols, n_tasks), dtype=np.float32)
        tiers = ((0, SINGLE_MASS), (4, DOUBLE_MASS), (8, TRIPLE_MASS))
        for r in range(n_rows):
            for c in range(n_cols):
                for start, mass in tiers:
                    probs  = np.random.dirichlet(np.full(4, self.task_alpha))
                    counts = np.random.multinomial(mass, probs).astype(np.float32)
                    self.regional_rewards[r, c, start:start + 4] = counts

    def _get_regional_rewards(self, pos: Tuple[float, float]) -> np.ndarray:
        x, y   = pos
        n_rows = self.regional_rewards.shape[0]
        n_cols = self.regional_rewards.shape[1]
        row = min(int(x / REGION_SIZE), n_rows - 1)
        col = min(int(y / REGION_SIZE), n_cols - 1)
        return self.regional_rewards[row, col]

    def _rand_pos(self) -> Tuple[float, float]:
        return (
            random.uniform(0.5, self.width  - 0.5),
            random.uniform(0.5, self.height - 0.5),
        )

    def _rand_pos_seeding(self) -> Tuple[float, float]:
        # Central 40% of each axis — leaves room to expand toward edges.
        margin = 0.30
        return (
            random.uniform(self.width  * margin, self.width  * (1.0 - margin)),
            random.uniform(self.height * margin, self.height * (1.0 - margin)),
        )

    def _pos_near(self, pos: Tuple[float, float], r: float = 3.0) -> Tuple[float, float]:
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
            if np.linalg.norm(pos_arr - p, axis=1).min() > REPULSION_RADIUS:
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

    def _apply_forces(self) -> None:
        cell_list = list(self.cells.values())
        N = len(cell_list)
        if N == 0:
            return

        pos  = np.array([c.position for c in cell_list], dtype=np.float64)
        vel  = np.array([c.velocity  for c in cell_list], dtype=np.float64)
        cids = np.array(
            [c.cluster_id if c.cluster_id is not None else -1 for c in cell_list],
            dtype=np.int32,
        )

        new_pos, new_vel = _ti_physics.step(pos, vel, cids, float(self.width), float(self.height))

        for i, cell in enumerate(cell_list):
            cell.position = (float(new_pos[i, 0]), float(new_pos[i, 1]))
            cell.velocity = (float(new_vel[i, 0]), float(new_vel[i, 1]))

    def _break_stretched_bonds(self) -> None:
        for cluster in list(self.clusters.values()):
            positioned = [c for c in cluster.cells if c.position is not None]
            if len(positioned) < 2:
                continue
            cx = float(np.mean([c.position[0] for c in positioned]))
            cy = float(np.mean([c.position[1] for c in positioned]))
            for cell in list(cluster.cells):
                if cell.position is None:
                    continue
                # Newly-divided cells get a grace period — give adhesion time
                # to lock them in before bond-break checks apply.
                if cell.replication_cooldown > 0:
                    continue
                d = float(np.hypot(cell.position[0] - cx, cell.position[1] - cy))
                if d <= BOND_BREAK_DIST:
                    continue
                excess = (d - BOND_BREAK_DIST) / BOND_BREAK_DIST
                p = min(BOND_BREAK_PROB_MAX, BOND_BREAK_PROB_MAX * excess)
                if random.random() < p:
                    cluster.remove_cell(cell)
            if cluster.size == 0:
                self.clusters.pop(cluster.cluster_id, None)

    @staticmethod
    def _kin_bond_prob(cell_a: "Cell", cell_b: "Cell", base_prob: float) -> float:
        # Scale base_prob by genomic similarity on the 4 active bits.
        # 0 differences → 1.0 ;  4 differences → 0.1
        hamming = int(np.sum(cell_a.genome[:4] != cell_b.genome[:4]))
        scale   = 1.0 - 0.225 * hamming
        return base_prob * max(scale, 0.1)

    def _try_adhesion(self) -> None:
        bondable_lone = [
            c for c in self.cells.values()
            if c.can_form_bond and c.cluster_id is None
        ]
        if len(bondable_lone) < 1:
            return

        lone_pos = np.array([c.position for c in bondable_lone])

        # lone ↔ lone → new cluster
        if len(bondable_lone) >= 2:
            tree  = cKDTree(lone_pos)
            pairs = tree.query_pairs(ADHESION_FORM_RADIUS, output_type='ndarray')
            for i, j in pairs:
                ci, cj = bondable_lone[i], bondable_lone[j]
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
            for c in cl.cells if c.position is not None and c.can_form_bond
        ]
        if cluster_cells:
            cl_pos  = np.array([c.position for c in cluster_cells])
            cl_tree = cKDTree(cl_pos)
            for lone in bondable_lone:
                if lone.cluster_id is not None:
                    continue
                idxs = cl_tree.query_ball_point(lone.position, ADHESION_FORM_RADIUS)
                if not idxs:
                    continue
                target = cluster_cells[idxs[0]]
                if target.cluster_id is None or not target.can_form_bond:
                    continue
                cl = self.clusters.get(target.cluster_id)
                if cl is None or cl.size >= MAX_CLUSTER_SIZE:
                    continue
                mean_hamming = float(np.mean([
                    np.sum(lone.genome[:4] != c.genome[:4]) for c in cl.cells
                ]))
                scale     = max(1.0 - 0.225 * mean_hamming, 0.1)
                join_prob = ADHESION_BOND_PROB * 0.5 * scale
                if random.random() <= join_prob:
                    cl.add_cell(lone)

    def _lone_cells(self) -> List[Cell]:
        return [c for c in self.cells.values() if c.cluster_id is None]

    def _eval_lone_cell(self, cell: Cell) -> float:
        income = 0.0
        if cell.position is not None:
            rvec      = self._get_regional_rewards(cell.position)
            task_mass = float(rvec[_OP_TO_IDX[cell.operation]])
            if task_mass > 0:
                income = task_mass
        return income - MAINTENANCE_COST

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
                total_gross += max(
                    0.0,
                    float(rvec[i]) * self.coop_reward_scale
                    - DEFECTOR_DRAIN * cluster.defector_count,
                )

        per_cell = total_gross / max(cluster.size, 1)
        for cell in cluster.cells:
            coop_paid = self.coop_cost if (cell.is_cooperator and total_gross > 0) else 0.0
            cell.fitness += per_cell - cell.adhesion_cost - coop_paid - MAINTENANCE_COST
        cluster.fitness += total_gross

    def _kill_weakest_near(self, ref: Tuple[float, float]) -> None:
        # Age-based, not fitness-based: fitness-based killing biases against
        # defectors whose fitness is diluted by the very clusters they infiltrate,
        # which would suppress the within-cluster selection pressure we study.
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
        if cell.health < LONE_REPL_THRESH:
            return
        # Cooperators must be in a tile that pays for their op; defectors free-ride.
        if not cell.is_defector:
            rvec = self._get_regional_rewards(cell.position)
            if float(rvec[_OP_TO_IDX[cell.operation]]) < 1.0:
                return
        if random.random() > LONE_REPL_PROB:
            return
        if len(self.cells) >= MAX_CELLS:
            self._kill_weakest_near(cell.position)
            if len(self.cells) >= MAX_CELLS or cell.position is None:
                return
        child = cell.mutate()
        cell.health  = 0.0
        child.health = 0.0
        self.place_cell(child, self._pos_near(cell.position, r=3.0))
        cell.replication_cooldown  = LONE_REPL_COOLDOWN
        child.replication_cooldown = LONE_REPL_COOLDOWN

    def _replicate_defector_in_cluster(self, cluster: Cluster) -> None:
        # Defectors divide on their own health — independent of cluster mean.
        # Models cancer-like clonal expansion: parasitic mutants compound while
        # the host cluster continues operating.
        for cell in list(cluster.cells):
            if not cell.is_defector or cell.position is None:
                continue
            if cell.replication_cooldown > 0:
                continue
            if cell.health < DEFECTOR_REPL_THRESH:
                continue
            if cluster.size >= MAX_CLUSTER_SIZE:
                return
            if len(self.cells) >= MAX_CELLS:
                return
            # Place child near centroid (NOT next to parent) so it doesn't
            # land inside the repulsion radius and get ejected before adhesion locks in.
            positioned = [c for c in cluster.cells if c.position is not None]
            cx = float(np.mean([c.position[0] for c in positioned]))
            cy = float(np.mean([c.position[1] for c in positioned]))
            angle = random.uniform(0, 2 * np.pi)
            child_pos = (
                float(np.clip(cx + ADHESION_REST_DIST * np.cos(angle), 0.5, self.width  - 0.5)),
                float(np.clip(cy + ADHESION_REST_DIST * np.sin(angle), 0.5, self.height - 0.5)),
            )
            child = cell.mutate()
            cell.health  = 0.0
            child.health = 0.0
            self.place_cell(child, child_pos)
            cluster.add_cell(child)
            cell.replication_cooldown  = LONE_REPL_COOLDOWN
            child.replication_cooldown = LONE_REPL_COOLDOWN

    def _replicate_cluster(self, cluster: Cluster) -> None:
        if cluster.replication_cooldown > 0:
            return
        avg_health = float(np.mean([c.health for c in cluster.cells])) if cluster.cells else 0.0
        if avg_health < CLUSTER_REPL_THRESH:
            return
        if random.random() > CLUSTER_REPL_PROB:
            return
        ref = cluster.cells[0].position if cluster.cells else self._rand_pos()
        if ref is None:
            return
        if len(self.cells) + cluster.size > MAX_CELLS:
            return
        offspring = cluster.replicate()
        n = len(offspring.cells)
        placed = []
        for i, child_cell in enumerate(offspring.cells):
            if len(self.cells) >= MAX_CELLS:
                break
            angle = 2 * np.pi * i / max(n, 1)
            ox = float(np.clip(ref[0] + ADHESION_REST_DIST * np.cos(angle), 0.5, self.width  - 0.5))
            oy = float(np.clip(ref[1] + ADHESION_REST_DIST * np.sin(angle), 0.5, self.height - 0.5))
            self.place_cell(child_cell, (ox, oy))
            placed.append(child_cell)
        if placed:
            self.clusters[offspring.cluster_id] = offspring
        for c in cluster.cells:
            c.health = 0.0
        for c in offspring.cells:
            c.health = 0.0
        cluster.replication_cooldown   = CLUSTER_REPL_COOLDOWN
        offspring.replication_cooldown = CLUSTER_REPL_COOLDOWN

    def tick(self) -> None:
        self.tick_count += 1
        flip_period = self.task_flip_period if self.task_flip_period is not None else 200
        if self.tick_count % flip_period == 0:
            self._generate_regional_rewards()

        self._apply_forces()
        # Run before fitness eval so a cell that just left doesn't earn cluster reward.
        self._break_stretched_bonds()

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
                # Old-age death ramps from 0 at MAX_CELL_AGE to 1 at 1.5×.
                p_age = (cell.age - MAX_CELL_AGE) / (MAX_CELL_AGE * 0.5)
                if random.random() < min(p_age, 1.0):
                    self.kill_cell(cell)
                continue
            if cell.age > GRACE_PERIOD:
                deficit = SURVIVAL_RATE - (cell.fitness / cell.age)
                if deficit > 0:
                    p_starve = min(deficit / SURVIVAL_RATE, 1.0) * 0.15
                    if random.random() < p_starve:
                        self.kill_cell(cell)

        for cell in list(self._lone_cells()):
            if cell.cell_id in self.cells:
                self._replicate_lone(cell)
        for cluster in list(self.clusters.values()):
            if cluster.cluster_id in self.clusters:
                self._replicate_defector_in_cluster(cluster)
                self._replicate_cluster(cluster)
