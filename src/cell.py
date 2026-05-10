import numpy as np
from typing import Optional, Tuple

GENOME_LENGTH = 8

# Genome bit indices
OP_HIGH    = 0  # operation MSB \  00=AND  01=OR
OP_LOW     = 1  # operation LSB  > 10=XOR  11=NAND
ADHESION   = 2
COOPERATOR = 3
# bits 4-7: reserved

_OP_MAP = {
    (0, 0): "AND",
    (0, 1): "OR",
    (1, 0): "XOR",
    (1, 1): "NAND",
}

ADHESION_COST = 0.1

NEWBORN_HEALTH     = 0.0
HEALTH_GROWTH_RATE = 1.0


def _apply_op(op: str, a: int, b: int) -> int:
    a, b = a & 0xFF, b & 0xFF
    if op == "AND":  return a & b
    if op == "OR":   return a | b
    if op == "XOR":  return a ^ b
    if op == "NAND": return (~(a & b)) & 0xFF
    return a & b


class Cell:
    _next_id: int = 0

    def __init__(
        self,
        genome: Optional[np.ndarray] = None,
        parent_id: Optional[int] = None,
        mutation_rate: float = 0.01,
    ) -> None:
        Cell._next_id += 1
        self.cell_id     = Cell._next_id
        self.parent_id   = parent_id
        self.mutation_rate = mutation_rate

        self.genome: np.ndarray = (
            np.random.randint(0, 2, GENOME_LENGTH, dtype=np.uint8)
            if genome is None
            else genome.copy()
        )

        self.age:                  int   = 0
        self.fitness:              float = 0.0
        self.health:               float = NEWBORN_HEALTH
        self.cluster_id:           Optional[int]                 = None
        self.position:             Optional[Tuple[float, float]] = None
        self.velocity:             Tuple[float, float]           = (0.0, 0.0)
        self.replication_cooldown: int   = 0

        self._parse_genome()

    def _parse_genome(self) -> None:
        self.operation:     str  = _OP_MAP[(int(self.genome[OP_HIGH]), int(self.genome[OP_LOW]))]
        self.is_cooperator: bool = bool(self.genome[COOPERATOR])
        self.has_adhesion:  bool = bool(self.genome[ADHESION])

    # Bonds require BOTH endpoints to carry adhesion AND cooperator bits;
    # defectors only arise via post-bond mutation, never by joining as defectors.
    @property
    def can_form_bond(self) -> bool:
        return self.is_cooperator and self.has_adhesion

    @property
    def is_defector(self) -> bool:
        return self.has_adhesion and not self.is_cooperator

    @property
    def adhesion_cost(self) -> float:
        # Defectors free-ride on cluster cohesion: they express adhesion but
        # don't pay its upkeep — strictly cheaper to maintain than cooperators.
        if not self.has_adhesion:
            return 0.0
        return 0.0 if self.is_defector else ADHESION_COST

    def compute(self, a: int, b: int) -> int:
        return _apply_op(self.operation, a, b)

    def tick(self) -> None:
        self.age += 1
        if self.fitness > 0:
            self.health += HEALTH_GROWTH_RATE
        elif self.is_defector and self.cluster_id is not None:
            # Defectors compound from the cluster's pool even when net fitness
            # is flat — the cancer-like property of dividing regardless of host state.
            self.health += HEALTH_GROWTH_RATE * 0.5

    def mutate(self) -> "Cell":
        # Asymmetric coop-bit mutation (8× toward defection, 0.3× back) forces
        # the genetic filter to do real work — otherwise defectors emerge constantly.
        new_genome = self.genome.copy()

        if new_genome[COOPERATOR] == 1:
            if np.random.random() < self.mutation_rate * 8.0:
                new_genome[COOPERATOR] = 0
        else:
            if np.random.random() < self.mutation_rate * 0.3:
                new_genome[COOPERATOR] = 1

        for i in range(GENOME_LENGTH):
            if i == COOPERATOR:
                continue
            if np.random.random() < self.mutation_rate:
                new_genome[i] ^= 1

        return Cell(genome=new_genome, parent_id=self.cell_id,
                    mutation_rate=self.mutation_rate)

    def __repr__(self) -> str:
        g = "".join(map(str, self.genome))
        return (f"Cell({self.cell_id} g={g} op={self.operation} "
                f"adh={int(self.has_adhesion)} coop={int(self.is_cooperator)} "
                f"age={self.age} fit={self.fitness:.2f})")
