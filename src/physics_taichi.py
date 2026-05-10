"""
GPU-accelerated physics (Taichi). Falls back to CPU if no Metal/CUDA device.

Per tick:
  1. Build a 10 µm spatial-hash grid
  2. update_velocity — sum pairwise Gaussian forces over the 11×11 bucket
     neighbourhood, normalise → scale by motility (nearest-neighbour/2) → damp
  3. update_position — apply velocity + Brownian displacement, then elastic
     wall reflection

Force model (Cell_Sim_2, Rodriguez et al. 2026):
    F(d) = A(d) − REPULSE_SCALE · R(d)
    A(d) = exp(−0.5·((d − ATTRACT_MU) / ATTRACT_SIGMA)²)   ← attractive well ~35 µm
    R(d) = exp(−0.5·((d − REPULSE_MU) / REPULSE_SIGMA)²)   ← repulsive core  ~0 µm
A(d) is included only when both cells are in the SAME cluster — cross-cluster
attraction would wrongly fuse independent cooperative groups together.
"""

import numpy as np
import taichi as ti

try:
    ti.init(arch=ti.gpu, log_level=ti.WARN)
    _BACKEND = "gpu"
except Exception:
    ti.init(arch=ti.cpu, log_level=ti.WARN)
    _BACKEND = "cpu"

# physics parameters
ALPHA            = 0.12    # velocity retention (damping)
DIFFUSION_SIGMA  = 0.492   # √(2·D·Δt), D = 1.21e-7 mm²/min
ATTRACT_MU       = 35.0    # µm
ATTRACT_SIGMA    = 5.0     # µm
REPULSE_MU       = -12.5   # µm
REPULSE_SIGMA    = 10.0    # µm
REPULSE_SCALE    = 50.0
MAX_DISPLACEMENT = 4.0     # µm/tick hard cap

# spatial hash: 10 µm grid cells, 11×11 neighbourhood
HASH_CELL    = 10.0
MAX_GRID_DIM = 200          # max buckets/axis → world up to 2000 µm/side
MAX_BUCKETS  = MAX_GRID_DIM * MAX_GRID_DIM
MAX_PER_BKT  = 32
NBHD_HALF    = 5
MAX_CELLS    = 12_000       # headroom above environment.MAX_CELLS for burst replication

_pos       = ti.Vector.field(2, dtype=ti.f32, shape=MAX_CELLS)
_vel       = ti.Vector.field(2, dtype=ti.f32, shape=MAX_CELLS)
_cid       = ti.field(dtype=ti.i32, shape=MAX_CELLS)
_bkt_count = ti.field(dtype=ti.i32, shape=MAX_BUCKETS)
_bkt_cells = ti.field(dtype=ti.i32, shape=(MAX_BUCKETS, MAX_PER_BKT))

_pos_buf = np.zeros((MAX_CELLS, 2), dtype=np.float32)
_vel_buf = np.zeros((MAX_CELLS, 2), dtype=np.float32)
_cid_buf = np.full(MAX_CELLS, -1,   dtype=np.int32)


@ti.kernel
def _clear_hash():
    for b in range(MAX_BUCKETS):
        _bkt_count[b] = 0


@ti.kernel
def _build_hash(n: int, grid_w: int, grid_h: int):
    for i in range(n):
        gx = int(ti.floor(_pos[i][0] / HASH_CELL))
        gy = int(ti.floor(_pos[i][1] / HASH_CELL))
        gx = ti.max(0, ti.min(gx, grid_w - 1))
        gy = ti.max(0, ti.min(gy, grid_h - 1))
        b  = gx * grid_h + gy
        slot = ti.atomic_add(_bkt_count[b], 1)
        if slot < MAX_PER_BKT:
            _bkt_cells[b, slot] = i


@ti.kernel
def _update_velocity(n: int, grid_w: int, grid_h: int):
    for i in range(n):
        xi = _pos[i]
        ci = _cid[i]
        fi = ti.Vector([0.0, 0.0])
        min_d = 1e9

        gx = int(ti.floor(xi[0] / HASH_CELL))
        gy = int(ti.floor(xi[1] / HASH_CELL))

        for dgx in ti.static(range(-NBHD_HALF, NBHD_HALF + 1)):
            for dgy in ti.static(range(-NBHD_HALF, NBHD_HALF + 1)):
                nx = gx + dgx
                ny = gy + dgy
                if 0 <= nx < grid_w and 0 <= ny < grid_h:
                    b      = nx * grid_h + ny
                    n_in_b = _bkt_count[b]
                    for k in range(MAX_PER_BKT):
                        if k < n_in_b:
                            j = _bkt_cells[b, k]
                            if j != i:
                                xj    = _pos[j]
                                cj    = _cid[j]
                                delta = xj - xi
                                d     = delta.norm()

                                if d > 1e-6:
                                    unit = delta / d

                                    if d < min_d:
                                        min_d = d

                                    r_val = ti.exp(
                                        -0.5 * ((d - REPULSE_MU) / REPULSE_SIGMA) ** 2
                                    )
                                    f_scalar = -REPULSE_SCALE * r_val

                                    # Attraction only between same-cluster cells
                                    if ci >= 0 and ci == cj:
                                        a_val = ti.exp(
                                            -0.5 * ((d - ATTRACT_MU) / ATTRACT_SIGMA) ** 2
                                        )
                                        f_scalar += a_val

                                    fi += f_scalar * unit

        nn_d     = min_d if min_d < 1e8 else 100.0
        motility = nn_d * 0.5

        f_mag  = fi.norm()
        f_safe = f_mag if f_mag > 1e-10 else 1.0
        fi_hat = fi * (1.0 / f_safe) * (1.0 if f_mag > 1e-10 else 0.0)

        v_new = ALPHA * _vel[i] + fi_hat * motility

        v_mag = v_new.norm()
        if v_mag > MAX_DISPLACEMENT:
            v_new = v_new * (MAX_DISPLACEMENT / v_mag)

        _vel[i] = v_new


@ti.kernel
def _update_position(n: int, world_w: float, world_h: float):
    for i in range(n):
        # Box-Muller → two independent Gaussians for Brownian displacement
        u1 = ti.max(ti.random(ti.f32), 1e-7)
        u2 = ti.random(ti.f32)
        u3 = ti.max(ti.random(ti.f32), 1e-7)
        u4 = ti.random(ti.f32)
        bx = ti.sqrt(-2.0 * ti.log(u1)) * ti.cos(2.0 * 3.14159265 * u2)
        by = ti.sqrt(-2.0 * ti.log(u3)) * ti.cos(2.0 * 3.14159265 * u4)
        brownian = ti.Vector([bx * DIFFUSION_SIGMA, by * DIFFUSION_SIGMA])

        p  = _pos[i] + _vel[i] + brownian
        px = p[0]
        py = p[1]

        # elastic wall reflection
        if px < 0.0:
            px = -px
        if px > world_w:
            px = 2.0 * world_w - px
        px = ti.max(0.001, ti.min(px, world_w - 0.001))

        if py < 0.0:
            py = -py
        if py > world_h:
            py = 2.0 * world_h - py
        py = ti.max(0.001, ti.min(py, world_h - 0.001))

        _pos[i] = ti.Vector([px, py])


def step(
    positions:   np.ndarray,   # (n, 2) world µm coords
    velocities:  np.ndarray,   # (n, 2) µm/tick (persistent across ticks)
    cluster_ids: np.ndarray,   # (n,) int32; -1 = lone cell
    world_w:     float,
    world_h:     float,
) -> tuple:
    """Run one physics tick. Returns (new_positions, new_velocities) as float64."""
    n = len(positions)
    if n == 0:
        return positions, velocities
    if n > MAX_CELLS:
        raise RuntimeError(f"Cell count {n} exceeds physics_taichi.MAX_CELLS={MAX_CELLS}; raise the constant")

    grid_w = int(np.ceil(world_w / HASH_CELL))
    grid_h = int(np.ceil(world_h / HASH_CELL))
    if grid_w > MAX_GRID_DIM or grid_h > MAX_GRID_DIM:
        raise RuntimeError(
            f"World {world_w}×{world_h} µm needs {grid_w}×{grid_h} buckets but "
            f"MAX_GRID_DIM={MAX_GRID_DIM} (≈{MAX_GRID_DIM*HASH_CELL:.0f} µm/side); raise the constant"
        )

    _pos_buf[:n] = positions.astype(np.float32)
    _vel_buf[:n] = velocities.astype(np.float32)
    _cid_buf[:n] = cluster_ids.astype(np.int32)

    _pos.from_numpy(_pos_buf)
    _vel.from_numpy(_vel_buf)
    _cid.from_numpy(_cid_buf)

    _clear_hash()
    _build_hash(n, grid_w, grid_h)
    _update_velocity(n, grid_w, grid_h)
    _update_position(n, float(world_w), float(world_h))

    new_pos = _pos.to_numpy()[:n].astype(np.float64)
    new_vel = _vel.to_numpy()[:n].astype(np.float64)
    return new_pos, new_vel
