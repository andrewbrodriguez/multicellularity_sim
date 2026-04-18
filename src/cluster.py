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

    def can_complete_task(self, op1: str, op2: Optional[str] = None, op3: Optional[str] = None) -> bool:
        has_op1 = any(c.is_cooperator and c.operation == op1 for c in self.cells)
        if op2 is None:
            return has_op1
        has_op2 = any(c.is_cooperator and c.operation == op2 for c in self.cells)
        if op3 is None:
            return has_op1 and has_op2
        has_op3 = any(c.is_cooperator and c.operation == op3 for c in self.cells)
        return has_op1 and has_op2 and has_op3

    def compute_task(
        self,
        a: int, b: int, c: int,
        op1: str, op2: Optional[str] = None, op3: Optional[str] = None,
        d: int = 0,
    ) -> Optional[int]:
        """1-, 2-, or 3-step task. Each step requires a cooperator with the matching operation."""
        cell1 = next((cell for cell in self.cells if cell.is_cooperator and cell.operation == op1), None)
        if cell1 is None:
            return None
        result1 = cell1.compute(a, b)
        if op2 is None:
            return result1

        cell2 = next((cell for cell in self.cells if cell.is_cooperator and cell.operation == op2), None)
        if cell2 is None:
            return None
        result2 = cell2.compute(result1, c)
        if op3 is None:
            return result2

        cell3 = next((cell for cell in self.cells if cell.is_cooperator and cell.operation == op3), None)
        if cell3 is None:
            return None
        return cell3.compute(result2, d)
    

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
