import numpy as np
from typing import Dict, List, Optional

from .environment import Environment
from .cell import Cell


class Simulation:
    def __init__(
        self,
        grid_size: int = 50,
        initial_cells: int = 200,
        mutation_rate: float = 0.02,
        seed: Optional[int] = None,
    ) -> None:
        self.env = Environment(width=grid_size, height=grid_size, seed=seed)
        self.mutation_rate = mutation_rate
        self.history: List[Dict] = []
        self._seed_population(initial_cells)

    def _seed_population(self, n: int) -> None:
        for _ in range(n):
            cell = Cell(mutation_rate=self.mutation_rate)
            self.env.place_cell(cell, self.env.find_empty_position())

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
                "pos":     cell.position,   # (x, y) as stored in environment
                "type":    ctype,
                "fitness": cell.fitness,
            })

        if n == 0:
            return {"tick": tick, "total_cells": 0, "num_clusters": 0,
                    "cooperator_pct": 0.0, "defector_pct": 0.0,
                    "mean_fitness": 0.0, "avg_cluster_size": 0.0,
                    "cell_records": cell_records}

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
            "cell_records":     cell_records,
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
