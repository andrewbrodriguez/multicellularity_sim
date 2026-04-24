import numpy as np
from typing import Optional, Tuple

GENOME_LENGTH = 8

# Genome bit indices
OP_HIGH    = 0  # operation MSB  \  00=AND  01=OR
OP_LOW     = 1  # operation LSB   > 10=XOR  11=NAND
ADHESION   = 2  # 1 = adhesion allele expressed (metabolic cost)
COOPERATOR = 3  # 1 = contributes computation to cluster; 0 = defector
# bits 4-7: reserved for future traits (e.g. signalling, apoptosis)

_OP_MAP = {
    (0, 0): "AND",
    (0, 1): "OR",
    (1, 0): "XOR",
    (1, 1): "NAND",
}

ADHESION_COST = 0.1   # metabolic penalty per tick for expressing adhesion


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
        mutation_rate: float = 0.02,
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
        self.cluster_id:           Optional[int]             = None
        self.position:             Optional[Tuple[float, float]] = None
        self.velocity:             Tuple[float, float]           = (0.0, 0.0)
        self.replication_cooldown: int   = 0

        self._parse_genome()

    def _parse_genome(self) -> None:
        self.operation:    str  = _OP_MAP[(int(self.genome[OP_HIGH]), int(self.genome[OP_LOW]))]
        self.is_cooperator: bool = bool(self.genome[COOPERATOR])
        # Cooperation requires physical contact — cooperators always express adhesion
        self.has_adhesion: bool = bool(self.genome[ADHESION]) or self.is_cooperator

    # A defector: joins clusters via adhesion but withholds computation
    @property
    def is_defector(self) -> bool:
        return self.has_adhesion and not self.is_cooperator

    @property
    def adhesion_cost(self) -> float:
        return ADHESION_COST if self.has_adhesion else 0.0

    def compute(self, a: int, b: int) -> int:
        return _apply_op(self.operation, a, b)

    def tick(self) -> None:
        self.age += 1

    def mutate(self) -> "Cell":
        # Asymmetric mutation on the cooperator bit: cheating is biologically
        # easier to evolve than cooperation, so the genetic filter must be
        # strong enough to overcome a 2×/0.3× bias toward defection.
        new_genome = self.genome.copy()

        if new_genome[COOPERATOR] == 1:
            if np.random.random() < self.mutation_rate * 2.0:
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
