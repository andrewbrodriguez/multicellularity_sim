import numpy as np
from typing import Dict, List, Optional

from .environment import Environment
from .cell import Cell


class Simulation:
    def __init__(
            self,
            grid_size: int = 250,
            initial_cells: int = 10,
            mutation_rate: float = 0.02,
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
        while placed < n_cluster_cells:
            size = random.randint(2, 4)
            if placed + size > n_cluster_cells:
                break
            # All cooperators with the same random operation so they can contribute
            op_bits = np.array([random.randint(0, 1), random.randint(0, 1)], dtype=np.uint8)
            anchor = self.env.find_empty_position(seeding=True)
            cl = Cluster()
            for i in range(size):
                genome = np.zeros(8, dtype=np.uint8)
                genome[0], genome[1] = op_bits  # same operation
                genome[2] = 1                   # adhesion on
                genome[3] = 1                   # cooperator
                cell = Cell(genome=genome, mutation_rate=self.mutation_rate)
                # Pack tightly around anchor within ADHESION_REST_DIST
                angle = 2 * np.pi * i / size
                pos = (
                    float(np.clip(anchor[0] + 0.75 * np.cos(angle), 0.5, self.env.width  - 0.5)),
                    float(np.clip(anchor[1] + 0.75 * np.sin(angle), 0.5, self.env.height - 0.5)),
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
            else:
                ctype = 0
            cell_records.append({
                "pos":     cell.position,
                "type":    ctype,
                "fitness": cell.fitness,
                "op":      cell.operation,
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
                    "mean_fitness": 0.0, "avg_cluster_size": 0.0,
                    "cell_records": cell_records, "cluster_groups": []}

        cooperators  = sum(1 for c in all_cells if c.is_cooperator)
        defectors    = sum(1 for c in all_cells if c.is_defector)
        lone         = sum(1 for c in all_cells if c.cluster_id is None)
        mean_fit     = float(np.mean([c.fitness for c in all_cells]))
        num_clusters = len(self.env.clusters)
        avg_cl_size  = (
            float(np.mean([cl.size for cl in self.env.clusters.values()]))
            if self.env.clusters else 0.0
        )

        return {
                    "tick":             tick,
                    "total_cells":      n,
                    "lone_cells":       lone,
                    "clustered_cells":  n - lone,
                    "num_clusters":     num_clusters,
                    "avg_cluster_size": avg_cl_size,
                    "cooperators":      cooperators,
                    "defectors":        defectors,
                    "cooperator_pct":   100.0 * cooperators / n,
                    "defector_pct":     100.0 * defectors / n,
                    "mean_fitness":     mean_fit,
                    "current_task":     self.env.current_task_str,
                    "cell_records":     cell_records,
                    "cluster_groups":   cluster_groups,
                }
    def _print_stats(self, s: Dict) -> None:
        if s["total_cells"] == 0:
            print(f"Tick {s['tick']:5d} | POPULATION EXTINCT")
            return
        print(
            f"Tick {s['tick']:5d} | "
            f"Cells: {s['total_cells']:4d} | "
            f"Clusters: {s['num_clusters']:3d} (avg sz {s['avg_cluster_size']:.1f}) | "
            f"Coop: {s['cooperator_pct']:5.1f}% | "
            f"Def: {s['defector_pct']:5.1f}% | "
            f"AvgFit: {s['mean_fitness']:6.2f}"
        )
