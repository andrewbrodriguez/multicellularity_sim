"""
Interactive 3D viewer for the single_frac × task_alpha sweep.

Usage
-----
python 3d_grapher2.py
python 3d_grapher2.py --csv results/3d_mass.csv
python 3d_grapher2.py --smooth
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def load_grid(csv_path: Path):
    df = pd.read_csv(csv_path)
    sf_vals    = np.sort(df["single_frac"].unique())
    alpha_vals = np.sort(df["task_alpha"].unique())

    Z = np.zeros((len(alpha_vals), len(sf_vals)))
    for i, a in enumerate(alpha_vals):
        for j, sf in enumerate(sf_vals):
            mask = (np.isclose(df["single_frac"], sf)
                    & np.isclose(df["task_alpha"], a))
            Z[i, j] = df.loc[mask, "multicellular_pct"].mean() * 100 if mask.any() else 0.0

    return sf_vals, alpha_vals, Z


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive 3D viewer: single_frac × task_alpha")
    parser.add_argument("--csv",    type=str, default="results/3d_mass.csv")
    parser.add_argument("--smooth", action="store_true")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"{csv_path} not found — run 3d_experiment2.py first")

    sf_vals, alpha_vals, Z = load_grid(csv_path)

    if args.smooth:
        from scipy.ndimage import gaussian_filter
        Z = gaussian_filter(Z, sigma=1.0)

    X, Y = np.meshgrid(sf_vals, alpha_vals)

    fig = plt.figure(figsize=(11, 8))
    ax  = fig.add_subplot(111, projection="3d")

    surf = ax.plot_surface(
        X, Y, Z,
        cmap=cm.plasma,
        linewidth=0.3,
        edgecolor="grey",
        alpha=0.92,
        antialiased=True,
    )
    fig.colorbar(surf, ax=ax, shrink=0.45, pad=0.08, label="% multicellular")

    ax.set_xlabel("single_frac (1-step share of mass)", labelpad=12)
    ax.set_ylabel("task_alpha (Dirichlet sparsity)", labelpad=12)
    ax.set_zlabel("% multicellular", labelpad=10)
    ax.set_zlim(max(0, Z.min() - 2), min(100, Z.max() + 2))
    ax.set_title(
        "Mass partition × landscape sparsity → multicellularity\n"
        "(total mass per tile held fixed)",
        pad=14,
    )

    ax.contourf(X, Y, Z, zdir="z", offset=max(0, Z.min() - 2),
                cmap=cm.plasma, alpha=0.25, levels=12)

    print("Drag to rotate  |  scroll to zoom  |  close window to exit")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
