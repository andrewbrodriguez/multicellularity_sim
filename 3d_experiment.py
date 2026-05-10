"""
3D surface sweep: mutation_rate × task_flip_period → % multicellular at end of run.

    python 3d_experiment.py                  # default 10×10 grid, 100 ticks, 2 seeds
    python 3d_experiment.py --ticks 1000 --mut-steps 15 --flip-steps 15 --seeds 3
    python 3d_experiment.py --plot-only      # re-plot from cached CSV
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


# ── default sim params (everything except the two sweep axes) ─────────────────

DEFAULTS = dict(
    grid_size      = 1000,
    initial_cells  = 50,
    coop_reward_scale = 1.5,
    task_alpha     = 0.5,
    coop_cost      = 0.5,
)

# ── sweep axes ────────────────────────────────────────────────────────────────

def _mut_axis(n: int) -> np.ndarray:
    """Log-spaced mutation rates from 0.0005 to 0.05."""
    return np.logspace(np.log10(0.000001), np.log10(1.0), n)


def _flip_axis(n: int) -> np.ndarray:
    """Flip periods: 50 ticks (rapid churn) → 1000 (near-static)."""
    return np.array(sorted({
        int(v) for v in np.logspace(np.log10(1), np.log10(500), n)
    }))


# ── runner ────────────────────────────────────────────────────────────────────

def run_trial(mutation_rate: float, task_flip_period: int, ticks: int, seed: int) -> float:
    """Return fraction of population that is multicellular (clustered) at end."""
    sim = Simulation(
        mutation_rate     = mutation_rate,
        task_flip_period  = task_flip_period,
        seed              = seed,
        **DEFAULTS,
    )
    sim._print_stats = lambda _: None
    sim.run(ticks=ticks, record_every=ticks)   # only need final snapshot

    cells = [c for c in sim.env.cells.values() if c.position is not None]
    if not cells:
        return 0.0
    clustered = sum(1 for c in cells if c.cluster_id is not None)
    return clustered / len(cells)


def run_sweep(
    mut_values: np.ndarray,
    flip_values: np.ndarray,
    ticks: int,
    seeds: list[int],
    quiet: bool,
) -> pd.DataFrame:
    rows = []
    total = len(mut_values) * len(flip_values) * len(seeds)
    done  = 0
    t0    = time.time()

    for mut, flip in itertools.product(mut_values, flip_values):
        frac_sum = 0.0
        for seed in seeds:
            frac = run_trial(mut, int(flip), ticks, seed)
            frac_sum += frac
            done += 1
            if not quiet:
                elapsed = time.time() - t0
                eta = elapsed / done * (total - done)
                print(f"  [{done}/{total}]  mut={mut:.5f}  flip={int(flip):4d}  "
                      f"seed={seed}  multicellular={frac:.3f}  "
                      f"ETA {eta:.0f}s")
        rows.append({
            "mutation_rate":    mut,
            "task_flip_period": int(flip),
            "multicellular_pct": frac_sum / len(seeds),
        })
        print(mut,flip, "done")

    return pd.DataFrame(rows)


# ── plotting ──────────────────────────────────────────────────────────────────

def plot_surface(df: pd.DataFrame, save_dir: Path) -> None:
    mut_vals  = np.sort(df["mutation_rate"].unique())
    flip_vals = np.sort(df["task_flip_period"].unique())

    Z = np.zeros((len(flip_vals), len(mut_vals)))
    for i, flip in enumerate(flip_vals):
        for j, mut in enumerate(mut_vals):
            row = df[(df["task_flip_period"] == flip) & (df["mutation_rate"].round(6) == round(mut, 6))]
            Z[i, j] = row["multicellular_pct"].mean() if not row.empty else 0.0

    log_mut  = np.log10(mut_vals)
    log_flip = np.log10(flip_vals)
    X, Y     = np.meshgrid(log_mut, log_flip)

    fig = plt.figure(figsize=(10, 7))
    ax  = fig.add_subplot(111, projection="3d")

    surf = ax.plot_surface(X, Y, Z * 100, cmap=cm.viridis, linewidth=0, antialiased=True, alpha=0.9)
    fig.colorbar(surf, ax=ax, shrink=0.5, pad=0.1, label="% multicellular")

    ax.set_xlabel("log₁₀(mutation rate)", labelpad=10)
    ax.set_ylabel("log₁₀(flip period / ticks)", labelpad=10)
    ax.set_zlabel("% multicellular", labelpad=10)
    ax.set_title("Multicellularity as a function of\nmutation rate and environmental volatility")

    # readable x-tick labels
    ax.set_xticks(log_mut[::max(1, len(log_mut)//5)])
    ax.set_xticklabels([f"{v:.4f}" for v in mut_vals[::max(1, len(mut_vals)//5)]], fontsize=7)
    ax.set_yticks(log_flip[::max(1, len(log_flip)//5)])
    ax.set_yticklabels([str(int(v)) for v in flip_vals[::max(1, len(flip_vals)//5)]], fontsize=7)

    save_dir.mkdir(parents=True, exist_ok=True)
    out = save_dir / "3d_mutation_flip_surface.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)

    # also emit a flat heatmap for the paper
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    im = ax2.imshow(
        Z * 100,
        origin="lower",
        aspect="auto",
        cmap="viridis",
        vmin=0, vmax=100,
        extent=[log_mut[0], log_mut[-1], log_flip[0], log_flip[-1]],
    )
    fig2.colorbar(im, ax=ax2, label="% multicellular")
    ax2.set_xlabel("log₁₀(mutation rate)")
    ax2.set_ylabel("log₁₀(flip period / ticks)")
    ax2.set_title("Multicellularity landscape (heatmap view)")

    xt = log_mut[::max(1, len(log_mut)//6)]
    ax2.set_xticks(xt)
    ax2.set_xticklabels([f"{10**v:.5f}" for v in xt], rotation=30, ha="right", fontsize=7)
    yt = log_flip[::max(1, len(log_flip)//6)]
    ax2.set_yticks(yt)
    ax2.set_yticklabels([str(int(10**v)) for v in yt], fontsize=7)

    out2 = save_dir / "3d_mutation_flip_heatmap.png"
    fig2.savefig(out2, dpi=150, bbox_inches="tight")
    print(f"Saved: {out2}")
    plt.close(fig2)


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="3D surface sweep: mutation * flip → multicellularity")
    parser.add_argument("--ticks",       type=int, default=100,  help="Ticks per trial (default: 500)")
    parser.add_argument("--mut-steps",   type=int, default=10,   help="Points along mutation axis (default: 10)")
    parser.add_argument("--flip-steps",  type=int, default=10,   help="Points along flip-period axis (default: 10)")
    parser.add_argument("--seeds",       type=int, default=2,    help="Seeds to average over (default: 2)")
    parser.add_argument("--results-dir", type=str, default="results", help="Where to save CSV (default: results/)")
    parser.add_argument("--figures-dir", type=str, default="figures", help="Where to save plots (default: figures/)")
    parser.add_argument("--quiet",       action="store_true",    help="Suppress per-trial output")
    parser.add_argument("--plot-only",   action="store_true",    help="Skip simulation, just re-plot existing CSV")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    figures_dir = Path(args.figures_dir)
    csv_path    = results_dir / "3d_surface.csv"

    if args.plot_only:
        if not csv_path.exists():
            raise FileNotFoundError(f"{csv_path} not found — run without --plot-only first")
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} rows from {csv_path}")
    else:
        mut_values  = _mut_axis(args.mut_steps)
        flip_values = _flip_axis(args.flip_steps)
        seeds       = list(range(args.seeds))

        print(
            f"3D surface sweep\n"
            f"  mutation_rate : {len(mut_values)} points  [{mut_values[0]:.5f} → {mut_values[-1]:.5f}]\n"
            f"  flip_period   : {len(flip_values)} points  [{flip_values[0]} → {flip_values[-1]}]\n"
            f"  seeds         : {seeds}\n"
            f"  ticks/trial   : {args.ticks}\n"
            f"  total trials  : {len(mut_values) * len(flip_values) * len(seeds)}\n"
            f"{'─'*60}"
        )

        df = run_sweep(mut_values, flip_values, args.ticks, seeds, args.quiet)

        results_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, index=False)
        print(f"\nSaved: {csv_path}")

    plot_surface(df, figures_dir)


if __name__ == "__main__":
    main()
