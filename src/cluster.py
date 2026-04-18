from typing import List, Optional
from .cell import Cell


class Cluster:
    _next_id: int = 0

    def __init__(self) -> None:
        Cluster._next_id += 1
        self.cluster_id:           int        = Cluster._next_id
        self.cells:                List[Cell] = []
        self.fitness:              float      = 0.0
        self.age:                  int        = 0
        self.replication_cooldown: int        = 0

    # ── membership ──────────────────────────────────────────────────────────

    def add_cell(self, cell: Cell) -> None:
        cell.cluster_id = self.cluster_id
        self.cells.append(cell)

    def remove_cell(self, cell: Cell) -> None:
        self.cells = [c for c in self.cells if c.cell_id != cell.cell_id]
        cell.cluster_id = None

    # ── computation ─────────────────────────────────────────────────────────

    def can_complete_task(self) -> bool:
        has_and = any(c.is_cooperator and c.operation == "AND" for c in self.cells)
        has_xor = any(c.is_cooperator and c.operation == "XOR" for c in self.cells)
        return has_and and has_xor

    def compute_task(self, a: int, b: int, c: int) -> Optional[int]:
        """
        Two-step division of labour:
          step 1 — AND cooperator computes  a AND b  → intermediate
          step 2 — XOR cooperator computes  intermediate XOR c  → output
        Only cooperators (is_cooperator=True) contribute; defectors free-ride.
        Returns None when the cluster lacks the required pair.
        """
        and_cell = next(
            (cell for cell in self.cells if cell.is_cooperator and cell.operation == "AND"),
            None,
        )
        xor_cell = next(
            (cell for cell in self.cells if cell.is_cooperator and cell.operation == "XOR"),
            None,
        )
        if and_cell is None or xor_cell is None:
            return None
        intermediate = and_cell.compute(a, b)
        return xor_cell.compute(intermediate, c)

    # ── lifecycle ────────────────────────────────────────────────────────────

    def tick(self) -> None:
        """Age the cluster; individual cells are ticked separately in Environment."""
        self.age += 1

    def replicate(self) -> "Cluster":
        """
        Cluster-level replication: every member cell copies and mutates together.
        This is the unit-of-selection shift from individual to group.
        """
        offspring = Cluster()
        for cell in self.cells:
            offspring.add_cell(cell.mutate())
        return offspring

    # ── properties ───────────────────────────────────────────────────────────

    @property
    def size(self) -> int:
        return len(self.cells)

    @property
    def cooperator_count(self) -> int:
        return sum(1 for c in self.cells if c.is_cooperator)

    @property
    def defector_count(self) -> int:
        return sum(1 for c in self.cells if c.is_defector)

    def __repr__(self) -> str:
        return (f"Cluster({self.cluster_id} sz={self.size} "
                f"coop={self.cooperator_count} def={self.defector_count} "
                f"fit={self.fitness:.2f})")
