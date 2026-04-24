"""
GPU-accelerated physics following Cell_Sim_2 (Rodriguez et al. 2026).

Architecture (Fig. 4 / Fig. 13 of the paper):
  Every tick:
    1. Build spatial hash grid (10µm × 10µm cells)
    2. update_velocity  — sum Gaussian pairwise forces → normalise → scale by motility → damp
    3. update_position  — apply velocity + Brownian displacement → elastic wall reflection

Force function (§3.2):
    F(d) = A(d) − REPULSE_SCALE × R(d)
    A(d) = exp(−0.5·((d − ATTRACT_MU)  / ATTRACT_SIGMA)²)   ← attractive well ~35 µm
    R(d) = exp(−0.5·((d − REPULSE_MU)  / REPULSE_SIGMA)²)   ← repulsive core  ~0 µm

    A(d) is included only when BOTH cells are in clusters
    (paper: attraction is dropped when either cell is "differentiated").

Velocity integration (§2.3):
    v_new = α · v_old  +  M_i · F̂_sum
    x_new = x_old + v_new + √(2D) · ξ
    where M_i = d_nearest_neighbour / 2  (motility limit, §3.3)

Falls back silently to CPU if no Metal/CUDA device is found.
"""

import numpy as np
import taichi as ti

# ── one-time Taichi initialisation ────────────────────────────────────────────
try:
    ti.init(arch=ti.gpu, log_level=ti.WARN)
    _BACKEND = "gpu"
except Exception:
    ti.init(arch=ti.cpu, log_level=ti.WARN)
    _BACKEND = "cpu"

# ── physics parameters (§2.3, §3.2, Appendix A) ──────────────────────────────
ALPHA            = 0.12    # velocity retention (damping), learned param from paper
DIFFUSION_SIGMA  = 0.492   # √(2·D·Δt), D=1.21e-7 mm²/min → 0.121 µm²/tick
ATTRACT_MU       = 35.0    # µm — attractive Gaussian centre
ATTRACT_SIGMA    = 5.0     # µm
REPULSE_MU       = -12.5   # µm — repulsive Gaussian centre (strong at d≈0)
REPULSE_SIGMA    = 10.0    # µm
REPULSE_SCALE    = 50.0    # learned parameter: ratio of repulsive to attractive magnitude
MAX_DISPLACEMENT = 4.0     # µm/tick hard cap (our constraint)

# ── spatial hash (§2.1): 10 µm grid cells, 11×11 neighbourhood ───────────────
HASH_CELL      = 10.0   # µm — matches paper's 10 µm × 10 µm grid
GRID_W         = 25     # ceil(250 / 10)
GRID_H         = 25
N_BUCKETS      = GRID_W * GRID_H   # 625
MAX_PER_BKT    = 32     # generous upper bound per bucket at our densities
NBHD_HALF      = 5      # ±5 buckets → 11×11 = 110 µm neighbourhood
MAX_CELLS      = 12_000   # headroom above environment.py MAX_CELLS=10_000 for burst replication

# ── Taichi fields ─────────────────────────────────────────────────────────────
_pos       = ti.Vector.field(2, dtype=ti.f32, shape=MAX_CELLS)
_vel       = ti.Vector.field(2, dtype=ti.f32, shape=MAX_CELLS)  # persistent velocity
_cid       = ti.field(dtype=ti.i32, shape=MAX_CELLS)            # cluster id; -1 = lone
_bkt_count = ti.field(dtype=ti.i32, shape=N_BUCKETS)
_bkt_cells = ti.field(dtype=ti.i32, shape=(N_BUCKETS, MAX_PER_BKT))

# numpy staging buffers (reused each tick to avoid reallocation)
_pos_buf = np.zeros((MAX_CELLS, 2), dtype=np.float32)
_vel_buf = np.zeros((MAX_CELLS, 2), dtype=np.float32)
_cid_buf = np.full(MAX_CELLS, -1,   dtype=np.int32)


# ── kernels ───────────────────────────────────────────────────────────────────

@ti.kernel
def _clear_hash():
    for b in range(N_BUCKETS):
        _bkt_count[b] = 0


@ti.kernel
def _build_hash(n: int):
    """Place each living cell into its 10 µm grid bucket (O(N), fully parallel)."""
    for i in range(n):
        gx = int(ti.floor(_pos[i][0] / HASH_CELL))
        gy = int(ti.floor(_pos[i][1] / HASH_CELL))
        gx = ti.max(0, ti.min(gx, GRID_W - 1))
        gy = ti.max(0, ti.min(gy, GRID_H - 1))
        b  = gx * GRID_H + gy
        slot = ti.atomic_add(_bkt_count[b], 1)
        if slot < MAX_PER_BKT:
            _bkt_cells[b, slot] = i


@ti.kernel
def _update_velocity(n: int):
    """
    For each cell i, sum pairwise Gaussian forces from all neighbours
    within the 11×11 bucket neighbourhood (≤ 110 µm).

    Then:
      v_new = α·v_old + M_i · F̂_sum
    where M_i = nearest_neighbour_dist / 2  (§3.3 motility limit).
    Velocity is clamped to MAX_DISPLACEMENT before writing.
    """
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
                if 0 <= nx < GRID_W and 0 <= ny < GRID_H:
                    b      = nx * GRID_H + ny
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

                                    # ── Gaussian force function (§3.2) ────────
                                    # Repulsive component: always applied
                                    r_val = ti.exp(
                                        -0.5 * ((d - REPULSE_MU) / REPULSE_SIGMA) ** 2
                                    )
                                    f_scalar = -REPULSE_SCALE * r_val

                                    # Attractive component: only between cells in
                                    # the SAME cluster — intra-cluster cohesion only.
                                    # Cross-cluster attraction would wrongly pull
                                    # independent cooperative groups together.
                                    if ci >= 0 and ci == cj:
                                        a_val = ti.exp(
                                            -0.5 * ((d - ATTRACT_MU) / ATTRACT_SIGMA) ** 2
                                        )
                                        f_scalar += a_val

                                    fi += f_scalar * unit

        # motility limit M_i = nearest-neighbour dist / 2 (§3.3)
        nn_d = min_d if min_d < 1e8 else 100.0
        motility = nn_d * 0.5

        # normalise cumulative force to a unit vector, then scale by M_i
        f_mag  = fi.norm()
        f_safe = f_mag if f_mag > 1e-10 else 1.0
        fi_hat = fi * (1.0 / f_safe) * (1.0 if f_mag > 1e-10 else 0.0)

        # velocity update with damping (§2.3): v_new = α·v_old + M_i·F̂
        v_new = ALPHA * _vel[i] + fi_hat * motility

        # clamp velocity magnitude to MAX_DISPLACEMENT µm/tick
        v_mag = v_new.norm()
        if v_mag > MAX_DISPLACEMENT:
            v_new = v_new * (MAX_DISPLACEMENT / v_mag)

        _vel[i] = v_new


@ti.kernel
def _update_position(n: int, world_w: float, world_h: float):
    """
    Integrate position: x_new = x + v + √(2D)·ξ  (§2.3 Brownian term).
    Elastic wall reflection applied after (§2.4).
    """
    for i in range(n):
        # Brownian displacement: Box-Muller → two independent Gaussians
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

        # elastic wall reflection (§2.4)
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


# ── public API ────────────────────────────────────────────────────────────────

def step(
    positions:   np.ndarray,   # (n, 2) float64, world µm coords
    velocities:  np.ndarray,   # (n, 2) float64, µm/tick (persistent across ticks)
    cluster_ids: np.ndarray,   # (n,)   int32; -1 = lone cell
    world_w:     float,
    world_h:     float,
) -> tuple:
    """
    Run one physics tick on the GPU.

    Returns (new_positions (n,2) float64, new_velocities (n,2) float64).
    Caller is responsible for storing velocities and passing them back each tick.
    """
    n = len(positions)
    if n == 0:
        return positions, velocities
    if n > MAX_CELLS:
        raise RuntimeError(f"Cell count {n} exceeds physics_taichi.MAX_CELLS={MAX_CELLS}; raise the constant")

    _pos_buf[:n] = positions.astype(np.float32)
    _vel_buf[:n] = velocities.astype(np.float32)
    _cid_buf[:n] = cluster_ids.astype(np.int32)

    _pos.from_numpy(_pos_buf)
    _vel.from_numpy(_vel_buf)
    _cid.from_numpy(_cid_buf)

    _clear_hash()
    _build_hash(n)
    _update_velocity(n)                                 # writes _vel
    _update_position(n, float(world_w), float(world_h)) # reads _vel, writes _pos

    new_pos = _pos.to_numpy()[:n].astype(np.float64)
    new_vel = _vel.to_numpy()[:n].astype(np.float64)
    return new_pos, new_vel
