"""
3D surface sweep: mass partition × landscape sparsity → multicellularity.

X = single_frac (fraction of tile mass in 1-step tasks; remainder split equally
across 2-step and 3-step tiers). Y = task_alpha (Dirichlet concentration inside
each tier — low α = one slot hoards mass, high α = uniform). Z = final %
multicellular. Total mass per tile is held fixed at TOTAL_MASS so we isolate
the *shape* of the landscape from its richness.

    python 3d_experiment2.py                      # default 21×21, 3 seeds, 500 ticks
    python 3d_experiment2.py --ticks 1000 --steps 15 --seeds 5
    python 3d_experiment2.py --plot-only
"""

import argparse
import itertools
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm

from src.simulation import Simulation


TOTAL_MASS = 20

DEFAULTS = dict(
    grid_size         = 1000,
    initial_cells     = 50,
    coop_reward_scale = 1.5,
    coop_cost         = 0.5,
    mutation_rate     = 0.0001,
)

# Axis ranges
SINGLE_FRAC_MIN, SINGLE_FRAC_MAX = 0.0, 1.0
ALPHA_MIN,       ALPHA_MAX       = 0.1, 3.0


def _frac_axis(n: int) -> np.ndarray:
    """Linear axis of single_frac values in [0, 1]."""
    return np.round(np.linspace(SINGLE_FRAC_MIN, SINGLE_FRAC_MAX, n), 4)


def _alpha_axis(n: int) -> np.ndarray:
    """Log-spaced axis of task_alpha values."""
    return np.round(np.logspace(np.log10(ALPHA_MIN), np.log10(ALPHA_MAX), n), 4)


def _partition(single_frac: float, total: int = TOTAL_MASS) -> tuple:
    """Split total tile mass into (single_mass, multi_mass) integer counts."""
    sm = int(round(total * single_frac))
    remaining = max(0, total - sm)
    mm = remaining // 2
    return sm, mm


# ── runner ────────────────────────────────────────────────────────────────────

def run_trial(single_frac: float, task_alpha: float, ticks: int, seed: int) -> float:
    """Patch reward-mass module constants, run sim, return % multicellular."""
    import src.environment as _env
    saved = (_env.SINGLE_MASS, _env.DOUBLE_MASS, _env.TRIPLE_MASS)
    sm, mm = _partition(single_frac)
    _env.SINGLE_MASS = sm
    _env.DOUBLE_MASS = mm
    _env.TRIPLE_MASS = mm

    try:
        sim = Simulation(seed=seed, task_alpha=float(task_alpha), **DEFAULTS)
        sim._print_stats = lambda _: None
        sim.run(ticks=ticks, record_every=ticks)

        cells = [c for c in sim.env.cells.values() if c.position is not None]
        if not cells:
            return 0.0
        clustered = sum(1 for c in cells if c.cluster_id is not None)
        return clustered / len(cells)
    finally:
        _env.SINGLE_MASS, _env.DOUBLE_MASS, _env.TRIPLE_MASS = saved


def run_sweep(
    single_fracs: np.ndarray,
    alphas:       np.ndarray,
    ticks:        int,
    seeds:        list,
    quiet:        bool,
) -> pd.DataFrame:
    rows  = []
    total = len(single_fracs) * len(alphas) * len(seeds)
    done  = 0
    t0    = time.time()

    for sf, alpha in itertools.product(single_fracs, alphas):
        sm, mm = _partition(sf)
        frac_sum = 0.0
        for seed in seeds:
            frac = run_trial(sf, alpha, ticks, seed)
            frac_sum += frac
            done += 1
            if not quiet:
                elapsed = time.time() - t0
                eta = elapsed / done * (total - done) if done < total else 0
                print(f"  [{done}/{total}]  single_frac={sf:.3f}  α={alpha:.3f}  "
                      f"(sm={sm:2d},mm={mm:2d})  seed={seed}  "
                      f"multicellular={frac:.3f}  ETA {eta:.0f}s")
        rows.append({
            "single_frac":      float(sf),
            "task_alpha":       float(alpha),
            "single_mass":      sm,
            "multi_mass":       mm,
            "total_mass":       int(sm + 2 * mm),
            "multicellular_pct": frac_sum / len(seeds),
        })

    return pd.DataFrame(rows)


# ── plotting ──────────────────────────────────────────────────────────────────

def plot_surface(df: pd.DataFrame, save_dir: Path) -> None:
    sf_vals    = np.sort(df["single_frac"].unique())
    alpha_vals = np.sort(df["task_alpha"].unique())

    Z = np.zeros((len(alpha_vals), len(sf_vals)))
    for i, a in enumerate(alpha_vals):
        for j, sf in enumerate(sf_vals):
            mask = (np.isclose(df["single_frac"], sf)
                    & np.isclose(df["task_alpha"], a))
            Z[i, j] = df.loc[mask, "multicellular_pct"].mean() * 100 if mask.any() else 0.0

    X, Y = np.meshgrid(sf_vals, alpha_vals)

    # ── 3D surface ────────────────────────────────────────────────
    fig = plt.figure(figsize=(11, 8))
    ax  = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(X, Y, Z, cmap=cm.plasma, linewidth=0.3,
                           edgecolor="grey", alpha=0.92, antialiased=True)
    fig.colorbar(surf, ax=ax, shrink=0.45, pad=0.08, label="% multicellular")
    ax.set_xlabel("single_frac (1-step share of mass)", labelpad=12)
    ax.set_ylabel("task_alpha (Dirichlet sparsity)", labelpad=12)
    ax.set_zlabel("% multicellular", labelpad=10)
    ax.set_zlim(max(0, Z.min() - 2), min(100, Z.max() + 2))
    ax.set_title(
        f"Mass partition × landscape sparsity → multicellularity\n"
        f"(total mass per tile fixed at {TOTAL_MASS})",
        pad=14,
    )
    ax.contourf(X, Y, Z, zdir="z", offset=max(0, Z.min() - 2),
                cmap=cm.plasma, alpha=0.25, levels=12)

    save_dir.mkdir(parents=True, exist_ok=True)
    out = save_dir / "3d_mass_surface.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)

    # ── flat heatmap ──────────────────────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    # log y-axis is more readable for a log-spaced alpha sweep, so
    # use pcolormesh on the actual coordinates instead of imshow
    pcm = ax2.pcolormesh(sf_vals, alpha_vals, Z, cmap="plasma",
                         vmin=0, vmax=100, shading="auto")
    fig2.colorbar(pcm, ax=ax2, label="% multicellular")
    ax2.set_xlabel("single_frac (1-step share of mass)")
    ax2.set_ylabel("task_alpha (Dirichlet sparsity)")
    ax2.set_yscale("log")
    ax2.set_title(f"Phase diagram (total mass = {TOTAL_MASS})")

    out2 = save_dir / "3d_mass_heatmap.png"
    fig2.savefig(out2, dpi=150, bbox_inches="tight")
    print(f"Saved: {out2}")
    plt.close(fig2)


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="3D surface sweep: single/multi mass partition × landscape sparsity"
    )
    parser.add_argument("--ticks",       type=int, default=500)
    parser.add_argument("--steps",       type=int, default=21,
                        help="Points per axis (default: 21; 21×21 = 441 configs)")
    parser.add_argument("--seeds",       type=int, default=3)
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--figures-dir", type=str, default="figures")
    parser.add_argument("--quiet",       action="store_true")
    parser.add_argument("--plot-only",   action="store_true")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    figures_dir = Path(args.figures_dir)
    csv_path    = results_dir / "3d_mass.csv"

    if args.plot_only:
        if not csv_path.exists():
            raise FileNotFoundError(f"{csv_path} not found — run without --plot-only first")
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} rows from {csv_path}")
    else:
        single_fracs = _frac_axis(args.steps)
        alphas       = _alpha_axis(args.steps)
        seeds        = list(range(args.seeds))

        print(
            f"3D surface sweep: single_frac × task_alpha   (total mass = {TOTAL_MASS})\n"
            f"  single_frac  : {len(single_fracs)} values  "
            f"[{single_fracs[0]:.2f} → {single_fracs[-1]:.2f}]\n"
            f"  task_alpha   : {len(alphas)} values  "
            f"[{alphas[0]:.3f} → {alphas[-1]:.3f}]  (log-spaced)\n"
            f"  seeds        : {seeds}\n"
            f"  ticks/trial  : {args.ticks}\n"
            f"  total trials : {len(single_fracs) * len(alphas) * len(seeds)}\n"
            f"{'─'*60}"
        )

        df = run_sweep(single_fracs, alphas, args.ticks, seeds, args.quiet)

        results_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, index=False)
        print(f"\nSaved: {csv_path}")

    plot_surface(df, figures_dir)


if __name__ == "__main__":
    main()
