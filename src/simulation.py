import numpy as np
from typing import Dict, List, Optional

from .environment import Environment, LONE_REPL_THRESH, CLUSTER_REPL_THRESH
from .cell import Cell


class Simulation:
    def __init__(
            self,
            grid_size: int = 250,
            initial_cells: int = 10,
            mutation_rate: float = 0.005,
            seed: Optional[int] = None,
            task_flip_period: Optional[int] = None, # <-- New MVG parameter
        ) -> None:
            self.env = Environment(
                width=grid_size, 
                height=grid_size, 
                seed=seed,
                task_flip_period=task_flip_period
            )
            self.mutation_rate = mutation_rate
            
            # These two lines were accidentally dropped! 
            # They are required to track stats and spawn the starting cells.
            self.history: List[Dict] = []
            self._seed_population(initial_cells)

    def _seed_population(self, n: int) -> None:
        from .cluster import Cluster
        import random, numpy as np

        # Seed ~40% of cells as pre-formed clusters of 2-4 cooperators
        n_cluster_cells = int(n * 0.4)
        placed = 0
        # All four operations as (OP_HIGH, OP_LOW) bit pairs
        ALL_OPS = [(0, 0), (0, 1), (1, 0), (1, 1)]  # AND, OR, XOR, NAND

        while placed < n_cluster_cells:
            size = random.randint(2, 4)
            if placed + size > n_cluster_cells:
                break
            # Assign distinct operations to each cell — no two members share an op.
            # This seeds genuine complementarity: the cluster can immediately attempt
            # multi-step tasks that require different operations.
            ops = random.sample(ALL_OPS, size)
            anchor = self.env.find_empty_position(seeding=True)
            cl = Cluster()
            for i in range(size):
                genome = np.zeros(8, dtype=np.uint8)
                genome[0], genome[1] = ops[i]   # distinct operation per cell
                genome[2] = 1                   # adhesion on
                genome[3] = 1                   # cooperator
                cell = Cell(genome=genome, mutation_rate=self.mutation_rate)
                angle = 2 * np.pi * i / size
                pos = (
                    float(np.clip(anchor[0] + 8.0 * np.cos(angle), 0.5, self.env.width  - 0.5)),
                    float(np.clip(anchor[1] + 8.0 * np.sin(angle), 0.5, self.env.height - 0.5)),
                )
                self.env.place_cell(cell, pos)
                cl.add_cell(cell)
            self.env.clusters[cl.cluster_id] = cl
            placed += size

        # Remaining cells seeded as lone individuals (mixed random genomes)
        for _ in range(n - placed):
            cell = Cell(mutation_rate=self.mutation_rate)
            self.env.place_cell(cell, self.env.find_empty_position(seeding=True))

    def run(self, ticks: int, record_every: int = 10) -> List[Dict]:
        for t in range(ticks):
            self.env.tick()
            if t % record_every == 0:
                stats = self._collect_stats(t)
                self.history.append(stats)
                self._print_stats(stats)
        return self.history

    def _collect_stats(self, tick: int) -> Dict:
        all_cells = list(self.env.cells.values())
        n = len(all_cells)

        # Per-cell records consumed by the visualizer.
        # type codes: 0=lone, 1=cooperator-in-cluster, 2=defector-in-cluster
        cell_records = []
        for cell in all_cells:
            if cell.position is None:
                continue
            if cell.cluster_id is not None:
                ctype = 1 if cell.is_cooperator else 2
                thresh = CLUSTER_REPL_THRESH
            else:
                ctype = 0
                thresh = LONE_REPL_THRESH
            div_progress = float(np.clip(cell.fitness / thresh, 0.0, 1.0))
            cell_records.append({
                "pos":          cell.position,
                "type":         ctype,
                "fitness":      cell.fitness,
                "op":           cell.operation,
                "div_progress": div_progress,
            })

        # Positions of cells in each cluster (≥2 members) for adhesion bond drawing
        cluster_groups = [
            [c.position for c in cl.cells if c.position is not None]
            for cl in self.env.clusters.values()
            if sum(1 for c in cl.cells if c.position is not None) >= 2
        ]

        if n == 0:
            return {"tick": tick, "total_cells": 0, "num_clusters": 0,
                    "cooperator_pct": 0.0, "defector_pct": 0.0,
                    "coop_genome_pct": 0.0,
                    "mean_fitness": 0.0, "avg_cluster_size": 0.0,
                    "coop_rate_clustered": 0.0, "def_rate_clustered": 0.0,
                    "coop_rate_lone": 0.0, "def_rate_lone": 0.0,
                    "cluster_rate": 0.0, "lone_rate": 0.0,
                    "multi_advantage": 0.0, "coop_advantage": 0.0,
                    "cell_records": cell_records, "cluster_groups": []}

        lone         = sum(1 for c in all_cells if c.cluster_id is None)
        # Active phenotypes: only count cells inside clusters.
        # A lone cell with cooperator-bit=1 is not yet cooperating with anyone.
        cooperators  = sum(1 for c in all_cells if c.cluster_id is not None and c.is_cooperator)
        defectors    = sum(1 for c in all_cells if c.cluster_id is not None and c.is_defector)
        # Genome-level statistic: how many cells carry the cooperator allele at all
        coop_genome  = sum(1 for c in all_cells if c.is_cooperator)
        mean_fit     = float(np.mean([c.fitness for c in all_cells]))
        num_clusters = len(self.env.clusters)
        avg_cl_size  = (
            float(np.mean([cl.size for cl in self.env.clusters.values()]))
            if self.env.clusters else 0.0
        )

        # Selection-pressure metrics: fitness-per-tick by phenotype.
        # These make the two selection gradients explicit:
        #   multi_advantage = (clustered coop) vs (lone)    → group selection
        #   coop_advantage  = (clustered coop) vs (defector) → individual selection
        def _rate(cells):
            if not cells:
                return 0.0
            return float(np.mean([c.fitness / max(c.age, 1) for c in cells]))

        clustered_coops = [c for c in all_cells
                           if c.cluster_id is not None and c.is_cooperator]
        clustered_defs  = [c for c in all_cells
                           if c.cluster_id is not None and c.is_defector]
        lone_coops      = [c for c in all_cells
                           if c.cluster_id is None and c.is_cooperator]
        lone_defs       = [c for c in all_cells
                           if c.cluster_id is None and not c.is_cooperator]

        coop_rate_clustered = _rate(clustered_coops)
        def_rate_clustered  = _rate(clustered_defs)
        coop_rate_lone      = _rate(lone_coops)
        def_rate_lone       = _rate(lone_defs)
        cluster_rate        = _rate(clustered_coops + clustered_defs)
        lone_rate           = _rate(lone_coops + lone_defs)

        # Positive → multicellularity pays; negative → going solo is better.
        multi_advantage = cluster_rate - lone_rate
        # Positive → cooperation pays inside clusters; negative → defection wins.
        coop_advantage  = coop_rate_clustered - def_rate_clustered

        clustered = n - lone
        return {
                    "tick":             tick,
                    "total_cells":      n,
                    "lone_cells":       lone,
                    "clustered_cells":  clustered,
                    "num_clusters":     num_clusters,
                    "avg_cluster_size": avg_cl_size,
                    "cooperators":      cooperators,
                    "defectors":        defectors,
                    # % of ALL cells actively cooperating / defecting in a cluster
                    "cooperator_pct":   100.0 * cooperators / n,
                    "defector_pct":     100.0 * defectors / n,
                    # % of all cells carrying the cooperator allele (genome level)
                    "coop_genome_pct":  100.0 * coop_genome / n,
                    "mean_fitness":     mean_fit,
                    "current_task":     self.env.current_task_str,
                    "coop_rate_clustered": coop_rate_clustered,
                    "def_rate_clustered":  def_rate_clustered,
                    "coop_rate_lone":      coop_rate_lone,
                    "def_rate_lone":       def_rate_lone,
                    "cluster_rate":        cluster_rate,
                    "lone_rate":           lone_rate,
                    "multi_advantage":     multi_advantage,
                    "coop_advantage":      coop_advantage,
                    "cell_records":     cell_records,
                    "cluster_groups":   cluster_groups,
                }
    def _print_stats(self, s: Dict) -> None:
        if s["total_cells"] == 0:
            print(f"Tick {s['tick']:5d} | POPULATION EXTINCT")
            return
        print(
            f"Tick {s['tick']:5d} | "
            f"Cells: {s['total_cells']:4d} (lone {s.get('lone_cells',0):3d}) | "
            f"Clusters: {s['num_clusters']:3d} (avg sz {s['avg_cluster_size']:.1f}) | "
            f"ActvCoop: {s['cooperator_pct']:5.1f}% | "
            f"ActvDef: {s['defector_pct']:5.1f}% | "
            f"CoopAllele: {s.get('coop_genome_pct',0.0):5.1f}% | "
            f"AvgFit: {s['mean_fitness']:6.2f}"
        )
