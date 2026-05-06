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

# Cell health: accumulates when a cell is net-positive; replication is gated
# on health reaching the division threshold.  Resets to 0 after division.
# Growth rate of 1.0/tick means a lone cell (~50 threshold) divides in ~50 ticks
# when earning steadily; clusters (~100/cell) take proportionally longer.
NEWBORN_HEALTH  = 0.0
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
        self.cluster_id:           Optional[int]             = None
        self.position:             Optional[Tuple[float, float]] = None
        self.velocity:             Tuple[float, float]           = (0.0, 0.0)
        self.replication_cooldown: int   = 0

        self._parse_genome()

    def _parse_genome(self) -> None:
        self.operation:    str  = _OP_MAP[(int(self.genome[OP_HIGH]), int(self.genome[OP_LOW]))]
        self.is_cooperator: bool = bool(self.genome[COOPERATOR])
        # Adhesion is now strictly the adhesion bit — cooperators no longer force it.
        self.has_adhesion: bool = bool(self.genome[ADHESION])

    # A bond between two cells requires BOTH endpoints to express
    # adhesion AND carry the cooperator bit.  Defectors arise only via
    # cooperator → defector mutation after the bond has already formed.
    @property
    def can_form_bond(self) -> bool:
        return self.is_cooperator and self.has_adhesion

    # A defector: in a cluster but withholds computation (cooperator bit off).
    @property
    def is_defector(self) -> bool:
        return self.has_adhesion and not self.is_cooperator

    @property
    def adhesion_cost(self) -> float:
        # Defectors free-ride on cluster cohesion too — they express adhesion
        # but don't pay the metabolic upkeep.  This makes them strictly cheaper
        # to maintain than cooperators, which is the whole point of the cheating.
        if not self.has_adhesion:
            return 0.0
        return 0.0 if self.is_defector else ADHESION_COST

    def compute(self, a: int, b: int) -> int:
        return _apply_op(self.operation, a, b)

    def tick(self) -> None:
        self.age += 1
        # Health accumulates when net-positive.  Defectors are an exception:
        # they parasitically draw from the cluster's shared pool and accumulate
        # health even when the cluster's net-fitness is flat — the cancer-like
        # property of dividing regardless of host metabolic state.
        if self.fitness > 0:
            self.health += HEALTH_GROWTH_RATE
        elif self.is_defector and self.cluster_id is not None:
            self.health += HEALTH_GROWTH_RATE * 0.5

    def mutate(self) -> "Cell":
        # Strongly asymmetric mutation on the cooperator bit: cheating is
        # biologically far easier to evolve than cooperation.  The 8×/0.3×
        # bias forces the genetic filter to do real work to keep clusters
        # cooperative — otherwise defectors emerge constantly.
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
