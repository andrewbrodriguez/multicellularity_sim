"""
Physics-based environment cells have continuous (x, y) float positions

Each tick applies 4 main things
  1. Brownian motion
  2. Short range repulsion, cells push apart when closer than REPULSION_RADIUS
  3. Adhesion, same cluster cells pulled back to ADHESION_REST_DIST
  4. Hard walls where cells hit the edge

Cell Health

Every cell pays MAINTENANCE_COST + adhesion_cost every tick there is mo
basal income. Health only goes up when a cell or its cluster
completes a rewarded task. A cell that cannot complete any task atrophies
and eventually dies

Lone cells:
- any cooperator with op X in a tile rewarding task X earns rvec[X] * 
coop_reward_scale and pays coop_cost.

- defectors and cooperators with no matching task earn nothing and drain at
MAINTENANCE_COST + adhesion_cost per tick.

Clusters:
for every task t in ALL_TASKS where rvec[t] > 0 and the cluster has the 
cooperators to compute it (1-step needs one matching cooperator; 2- and 3-step 
need complementary cooperators), the cluster earns max(0, rvec[t] * scale  - 
DEFECTOR_DRAIN * n_defectors). the sum is split evenly across the ENTIRE 
cluster, including defectors.  This handles both:
    (a) cluster has cooperators for a complex task -> splits the complex
    reward across everyone.
    (b) cluster lacks cooperators for any complex task but one cooperator can 
    do a 1-step task that's rewarded here -> that 1-step reward enters 
    total_gross and is split across everyone.
ALSO
cooperators pay coop_cost per tick only when the cluster earned.
every cell pays MAINTENANCE_COST + adhesion_cost regardless.

Replication cooldown:

After any replication event (cell or cluster), the offspring and parent
both enter a cooldown window (LONE_REPL_COOLDOWN / CLUSTER_REPL_COOLDOWN
ticks) during which replication is blocked
"""

import random
import numpy as np
from scipy.spatial import cKDTree
from typing import Dict, List, Optional, Tuple
from .cell import Cell, _apply_op
from .cluster import Cluster
from . import physics_taichi as _ti_physics

from .cell import Cell
from .cluster import Cluster

# physics
BROWNIAN_SIGMA = 1.0 # µm/tick thermal jitter
REPULSION_RADIUS = 8.0 # µm cell diameter
CLUSTER_REPULSION_RADIUS = 5.0 # µm 
REPULSION_STRENGTH = 20.0
ADHESION_REST_DIST = 5.0 
ADHESION_SPRING_K = 0.90 
MAX_INTERACTION_RADIUS = 35.0 # µm hard cutoff: no cell interacts this far
CLUSTER_ATTRACT_RADIUS = 30.0 # µm long-range inter-cluster attraction onset 
CLUSTER_ATTRACT_STRENGTH = 0.4 # force amplitude drawing clusters together
MAX_DISPLACEMENT = 4.0 # µm/tick


# biology
MAX_CELL_AGE = 500 
GRACE_PERIOD = 10
SURVIVAL_RATE = 0.02 # minimum fitness/age ratio to avoid starvation

COOP_REWARD_SCALE = 1.5

# Per tick metabolic drain on every cell
MAINTENANCE_COST = 0.01

# Reward mass distributed per tile, partitioned by task complexity
SINGLE_MASS = 6
DOUBLE_MASS = 8
TRIPLE_MASS = 11

# Defector economics, these create the public-goods dilemma
COOP_COST = 0.5    # metabolic cost paid ONLY by cooperators when cluster succeeds
DEFECTOR_DRAIN = 1.5 

LONE_REPL_THRESH = 100.0     # health needed for a lone cell to divide
CLUSTER_REPL_THRESH = 100.0  # PER-CELL threshold; cluster divides when mean(health) ≥ 100
DEFECTOR_REPL_THRESH = 50.0  # in-cluster defectors divide on their own at half the cost —
                             # they free-ride on cluster income to compound faster than the host
LONE_REPL_COOLDOWN = 10  
CLUSTER_REPL_COOLDOWN = 10
LONE_REPL_PROB = 0.5
CLUSTER_REPL_PROB = 0.25  

MAX_CLUSTER_SIZE = 10
MAX_CELLS = 10_000
ADHESION_BOND_PROB = 0.5
ADHESION_FORM_RADIUS = 20.0

BOND_BREAK_DIST = 40.0
BOND_BREAK_PROB_MAX = 0.25

REGION_SIZE = 100 # the task regions

ALL_TASKS: List[tuple] = [
    ("AND",), ("OR",), ("XOR",), ("NAND",),                          # 1-step (indices 0-3)
    ("AND", "XOR"), ("NAND", "OR"), ("OR", "AND"), ("XOR", "NAND"),  # 2-step (indices 4-7)
    ("AND", "OR", "XOR"), ("OR", "NAND", "AND"),                     # 3-step (indices 8-11)
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

            self.task_flip_period = task_flip_period
            self.coop_reward_scale = coop_reward_scale
            self.task_alpha = task_alpha
            self.coop_cost = coop_cost

            self._task_pool = {
                1: [("AND",), ("OR",), ("XOR",), ("NAND",)],
                2: [("AND", "XOR"), ("NAND", "OR"), ("OR", "AND"), ("XOR", "NAND")],
                3: [("AND", "OR", "XOR"), ("OR", "NAND", "AND"),
                    ("XOR", "AND", "OR"),  ("NAND", "XOR", "AND")],
            }

            self._generate_regional_rewards()


    def _generate_regional_rewards(self) -> None:
        """
        Rewards are distributed across all task tiers, but multi step tasks
        receive more mass than 1 step tasks so cooperation through complementary
        operations pays comfortably better than a single cell's op alone:
        """
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
        """Return the (12,) reward vector for the tile containing pos."""
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
        """
        Cells too far from their cluster's centroid detach 
        """
        for cluster in list(self.clusters.values()):
            positioned = [c for c in cluster.cells if c.position is not None]
            if len(positioned) < 2:
                continue
            cx = float(np.mean([c.position[0] for c in positioned]))
            cy = float(np.mean([c.position[1] for c in positioned]))
            for cell in list(cluster.cells):
                if cell.position is None:
                    continue
                # Newly-divided cells get a grace period — give adhesion time to
                # lock them into the cluster before bond-break checks apply.
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
        """
        Scale base_prob by genomic similarity on the 4 active bits (0-3)
        """
        hamming = int(np.sum(cell_a.genome[:4] != cell_b.genome[:4]))
        scale   = 1.0 - 0.225 * hamming   # 0 diff→1.0, 4 diff→0.1
        return base_prob * max(scale, 0.1)

    def _try_adhesion(self) -> None:
        # Bond formation requires BOTH endpoints to carry the cooperator bit
        # AND the adhesion bit. Cells missing either bit cannot form bonds —
        # defectors only arise via cooperator→defector mutation after the
        # bond is already established.
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

        # lone → existing cluster (target cluster member must also be bondable)
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
                # Kin check: compare lone cell against the nearest cluster member
                mean_hamming = float(np.mean([
                    np.sum(lone.genome[:4] != c.genome[:4]) for c in cl.cells
                ]))
                scale    = max(1.0 - 0.225 * mean_hamming, 0.1)
                join_prob = ADHESION_BOND_PROB * 0.5 * scale
                if random.random() <= join_prob:
                    cl.add_cell(lone)


    def _lone_cells(self) -> List[Cell]:
        return [c for c in self.cells.values() if c.cluster_id is None]

    def _eval_lone_cell(self, cell: Cell) -> float:
        """
        Per-tick health change for a lone cell.

        ANY lone cell (cooperator or defector) with op X standing on a tile
        that rewards the 1-step task X earns rvec[X] * coop_reward_scale.
        Cooperators additionally pay coop_cost when they earn (the cost of
        wiring up the cooperative pathway); defectors keep the full reward.

        Every cell pays MAINTENANCE_COST + adhesion_cost every tick.  The
        return value can be NEGATIVE — cells without an income source
        atrophy and eventually starve.
        """
        income    = 0.0
        if cell.position is not None:
            rvec      = self._get_regional_rewards(cell.position)
            task_mass = float(rvec[_OP_TO_IDX[cell.operation]])
            if task_mass > 0:
                income = task_mass
        cost = MAINTENANCE_COST 
        return income - cost

    def _eval_cluster(self, cluster: Cluster) -> None:
        """
        Per-tick health update for every member of a cluster.

        For every task t in ALL_TASKS where the tile rewards t AND the
        cluster has the cooperators required to compute it, the cluster
        earns max(0, rvec[t] * scale - DEFECTOR_DRAIN * n_defectors).  This
        sum is split evenly across the ENTIRE cluster (cooperators AND
        defectors).  Two important cases are handled by the same loop:

          (a) The cluster has the cooperators for a multi-step task in this
              tile → that complex reward enters total_gross and splits.

          (b) The cluster lacks the multi-step cooperators but one
              cooperator can do a rewarded 1-step task → that 1-step reward
              enters total_gross and splits across the entire cluster.

        Every cell pays MAINTENANCE_COST + adhesion_cost every tick.
        Cooperators additionally pay coop_cost when the cluster earned.
        Health can go negative — cells in a cluster that can complete
        nothing in its current tile atrophy and eventually starve.
        """
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
        if cell.health < LONE_REPL_THRESH:
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
        child = cell.mutate()
        cell.health  = 0.0
        child.health = 0.0
        self.place_cell(child, self._pos_near(cell.position, r=3.0))
        cell.replication_cooldown  = LONE_REPL_COOLDOWN
        child.replication_cooldown = LONE_REPL_COOLDOWN

    def _replicate_defector_in_cluster(self, cluster: Cluster) -> None:
        """
        A defector inside a cluster can divide on its OWN health (>=100),
        independent of the cluster-mean gate.  Both daughters stay in the
        cluster.  This models cancer-like clonal expansion: parasitic mutants
        hijack the cluster's shared income to copy themselves while the host
        cluster continues operating.
        """
        # Iterate over a snapshot — we may add cells to the cluster
        for cell in list(cluster.cells):
            if not cell.is_defector or cell.position is None:
                continue
            if cell.replication_cooldown > 0:
                continue
            if cell.health < DEFECTOR_REPL_THRESH:
                continue
            # No stochastic gate — defectors divide reliably once they hit the
            # threshold, modelling unchecked parasitic proliferation.
            if cluster.size >= MAX_CLUSTER_SIZE:
                return
            if len(self.cells) >= MAX_CELLS:
                return
            # Place child near the cluster centroid at one rest-length offset —
            # NOT next to the parent (would overlap repulsion radius and get
            # ejected before adhesion locks the child into the cluster).
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
        # Stochastic gate: prevents synchronized division pulses across clusters
        if random.random() > CLUSTER_REPL_PROB:
            return
        ref = cluster.cells[0].position if cluster.cells else self._rand_pos()
        if ref is None:
            return
        # Need room for the whole offspring cluster; abort if not enough space
        slots_needed = cluster.size
        if len(self.cells) + slots_needed > MAX_CELLS:
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

    # ── main tick ─────────────────────────────────────────────────────────────

    def tick(self) -> None:
        self.tick_count += 1
        flip_period = self.task_flip_period if self.task_flip_period is not None else 200
        if self.tick_count % flip_period == 0:
            self._generate_regional_rewards()
        self._apply_forces()

        # Cells dragged too far from their cluster's centroid detach.
        # Run before fitness eval so a cell that just left doesn't earn
        # this tick's cluster reward.
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
                self._replicate_defector_in_cluster(cluster)
                self._replicate_cluster(cluster)
