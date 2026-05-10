"""Interactive 3D surface viewer for the mutation × flip-period sweep CSV (3d_experiment.py output)."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def load_grid(csv_path: Path):
    df = pd.read_csv(csv_path)
    mut_vals  = np.sort(df["mutation_rate"].unique())
    flip_vals = np.sort(df["task_flip_period"].unique())

    Z = np.zeros((len(flip_vals), len(mut_vals)))
    for i, flip in enumerate(flip_vals):
        for j, mut in enumerate(mut_vals):
            mask = (df["task_flip_period"] == flip) & (np.isclose(df["mutation_rate"], mut))
            Z[i, j] = df.loc[mask, "multicellular_pct"].mean() if mask.any() else 0.0

    return mut_vals, flip_vals, Z * 100  # convert to %


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive 3D surface viewer")
    parser.add_argument("--csv",    type=str, default="results/3d_surface.csv")
    parser.add_argument("--smooth", action="store_true", help="Gaussian-smooth the surface")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"{csv_path} not found — run 3d_experiment.py first")

    mut_vals, flip_vals, Z = load_grid(csv_path)

    if args.smooth:
        from scipy.ndimage import gaussian_filter
        Z = gaussian_filter(Z, sigma=1.0)

    log_mut  = np.log10(mut_vals)
    log_flip = np.log10(flip_vals)
    X, Y     = np.meshgrid(log_mut, log_flip)

    fig = plt.figure(figsize=(11, 8))
    ax  = fig.add_subplot(111, projection="3d")

    surf = ax.plot_surface(
        X, Y, Z,
        cmap=cm.viridis,
        linewidth=0.3,
        edgecolor="grey",
        alpha=0.92,
        antialiased=True,
    )
    fig.colorbar(surf, ax=ax, shrink=0.45, pad=0.08, label="% multicellular")

    # Axis labels
    ax.set_xlabel("mutation rate (log₁₀)", labelpad=12)
    ax.set_ylabel("flip period / ticks (log₁₀)", labelpad=12)
    ax.set_zlabel("% multicellular", labelpad=10)
    ax.set_zlim(max(0, Z.min() - 2), min(100, Z.max() + 2))
    ax.set_title(
        "Multicellularity landscape\nmutation rate × environmental volatility",
        pad=14,
    )

    # Readable tick labels on the log axes
    stride_x = max(1, len(log_mut)  // 6)
    stride_y = max(1, len(log_flip) // 6)
    ax.set_xticks(log_mut[::stride_x])
    ax.set_xticklabels([f"{v:.4f}" for v in mut_vals[::stride_x]], fontsize=7)
    ax.set_yticks(log_flip[::stride_y])
    ax.set_yticklabels([str(int(v)) for v in flip_vals[::stride_y]], fontsize=7)

    # Contour projected onto the floor
    ax.contourf(X, Y, Z, zdir="z", offset=0, cmap=cm.viridis, alpha=0.3, levels=12)

    print("Drag to rotate  |  scroll to zoom  |  close window to exit")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
