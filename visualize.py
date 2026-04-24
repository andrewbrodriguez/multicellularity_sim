"""
Entry point for the multicellularity simulation.

Examples
--------
# default: sparse start, Napari viewer opens automatically
python visualize.py

# headless — no viewer, dumps CSVs at the end
python visualize.py --viewer off --ticks 2000 --sample-every 50

# record every tick, full temporal resolution
python visualize.py --sample-every 1 --ticks 300

# denser start (less empty space, faster cluster formation)
python visualize.py --initial-cells 100 --ticks 600

# MVG extension: test environment fluctuation
python visualize.py --task-flip-period 100 --initial-cells 100 --ticks 600
"""

import argparse
import numpy as np
import pandas as pd
from src.simulation import Simulation
from src.visualizer import launch_viewer
from src.environment import BASE_REWARD, SIMPLE_REWARD, COMPLEX_REWARD, TRIPLE_REWARD, DEFECTOR_DRAIN, COOP_COST, REPULSION_RADIUS


# ── output helpers ────────────────────────────────────────────────────────────

def _build_cell_df(sim: Simulation) -> pd.DataFrame:
    """
    Build a flat DataFrame of every living cell at the END of the run.
    Pulls directly from sim.env.cells for full genome access.
    """
    rows = []
    for cell in sim.env.cells.values():
        if cell.position is None:
            continue
        genome_str = "".join(map(str, cell.genome))
        cid = cell.cluster_id if cell.cluster_id is not None else -1
        if cid == -1:
            ctype = "lone"
        elif cell.is_cooperator:
            ctype = "cooperator"
        else:
            ctype = "defector"
        rows.append({
            "cell_id":       cell.cell_id,
            "genome":        genome_str,
            "op":            cell.operation,
            "is_cooperator": int(cell.is_cooperator),
            "is_defector":   int(cell.is_defector),
            "has_adhesion":  int(cell.has_adhesion),
            "cluster_id":    cid,
            "type":          ctype,
            "fitness":       round(cell.fitness, 4),
            "age":           cell.age,
            "x":             round(cell.position[0], 3),
            "y":             round(cell.position[1], 3),
        })
    return pd.DataFrame(rows)


def _build_history_df(sim: Simulation) -> pd.DataFrame:
    """
    Build a per-frame summary DataFrame from sim.history (one row per recorded tick).
    Excludes the raw cell_records and cluster_groups blobs.
    """
    scalar_keys = [
        "tick", "total_cells", "lone_cells", "clustered_cells",
        "num_clusters", "avg_cluster_size",
        "cooperators", "defectors",
        "cooperator_pct", "defector_pct", "coop_genome_pct",
        "mean_fitness",
        "multi_advantage", "coop_advantage",
        "coop_rate_clustered", "def_rate_clustered",
        "coop_rate_lone", "def_rate_lone",
        "cluster_rate", "lone_rate",
    ]
    rows = []
    for snap in sim.history:
        rows.append({k: snap.get(k) for k in scalar_keys})
    return pd.DataFrame(rows)


def _print_final_summary(cell_df: pd.DataFrame, history_df: pd.DataFrame) -> None:
    final_tick = int(history_df["tick"].iloc[-1])
    print(f"\n{'─'*75}")
    print(f"Final state (tick {final_tick})  —  {len(cell_df)} living cells\n")

    # Population breakdown
    counts = cell_df["type"].value_counts()
    for t in ("cooperator", "defector", "lone"):
        n = counts.get(t, 0)
        pct = 100.0 * n / max(len(cell_df), 1)
        print(f"  {t:<12}: {n:5d}  ({pct:5.1f}%)")

    # Operation distribution
    print(f"\n  Operation frequencies (all cells):")
    for op, cnt in cell_df["op"].value_counts().items():
        pct = 100.0 * cnt / max(len(cell_df), 1)
        print(f"    {op:<6}: {cnt:5d}  ({pct:5.1f}%)")

    # Cluster genome diversity: mean unique ops per cluster
    clustered = cell_df[cell_df["cluster_id"] >= 0]
    if not clustered.empty:
        diversity = clustered.groupby("cluster_id")["op"].nunique().mean()
        print(f"\n  Avg distinct operations per cluster: {diversity:.2f}")

    # Selection pressures from final history row
    last = history_df.iloc[-1]
    print(f"\n  multi_advantage : {last['multi_advantage']:+.4f}  "
          f"(+= multicellularity pays)")
    print(f"  coop_advantage  : {last['coop_advantage']:+.4f}  "
          f"(+= cooperation pays inside clusters)")
    print(f"{'─'*75}\n")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run multicellularity simulation and optionally open Napari viewer."
    )
    parser.add_argument("--grid-size",      type=int,   default=250,
                        help="Width and height of the 2-D spatial grid in µm (default: 250)")
    parser.add_argument("--initial-cells",  type=int,   default=40,
                        help="Starting cell count (default: 40)")
    parser.add_argument("--mutation-rate",  type=float, default=0.005,
                        help="Per-bit flip probability per replication (default: 0.005)")
    parser.add_argument("--ticks",          type=int,   default=500,
                        help="Total simulation ticks to run (default: 500)")
    parser.add_argument("--sample-every",   type=int,   default=10,
                        help="Record a frame every N ticks (default: 10)")
    parser.add_argument("--seed",           type=int,   default=42,
                        help="RNG seed for reproducibility (default: 42)")
    parser.add_argument("--quiet",          action="store_true",
                        help="Suppress per-tick console output")
    parser.add_argument("--task-flip-period", type=int, default=None,
                        help="Ticks between environmental task structure flips (MVG extension)")
    parser.add_argument("--viewer",         choices=["on", "off"], default="on",
                        help="Launch Napari viewer after run (default: on)")
    parser.add_argument("--save-csv",       type=str,   default=None,
                        help="Directory to save cell_df.csv and history_df.csv (default: none)")

    args = parser.parse_args()

    flip_status = (f"  task_flip_period={args.task_flip_period}"
                   if args.task_flip_period else "  task_flip_period=None (Static)")
    viewer_status = f"  viewer={args.viewer}"

    print(
        f"Multicellularity Simulation\n"
        f"  grid={args.grid_size}×{args.grid_size}  "
        f"cells={args.initial_cells}  "
        f"ticks={args.ticks}  "
        f"sample_every={args.sample_every}  "
        f"seed={args.seed}\n"
        f"{flip_status}\n"
        f"{viewer_status}\n"
        f"{'─' * 75}"
    )

    sim = Simulation(
        grid_size=args.grid_size,
        initial_cells=args.initial_cells,
        mutation_rate=args.mutation_rate,
        seed=args.seed,
        task_flip_period=args.task_flip_period,
    )

    if args.quiet:
        sim._print_stats = lambda _s: None

    sim.run(ticks=args.ticks, record_every=args.sample_every)

    # ── post-run analysis ──────────────────────────────────────────────────────
    cell_df    = _build_cell_df(sim)
    history_df = _build_history_df(sim)

    _print_final_summary(cell_df, history_df)

    if args.save_csv:
        import os
        os.makedirs(args.save_csv, exist_ok=True)
        cell_path    = os.path.join(args.save_csv, "cell_df.csv")
        history_path = os.path.join(args.save_csv, "history_df.csv")
        cell_df.to_csv(cell_path,    index=False)
        history_df.to_csv(history_path, index=False)
        print(f"Saved:\n  {cell_path}\n  {history_path}\n")

    # ── viewer ─────────────────────────────────────────────────────────────────
    if args.viewer == "off":
        return

    n_frames = len(sim.history)
    print(f"{'─' * 75}")
    print(f"Collected {n_frames} frames.  Launching Napari…\n")
    print("Controls:")
    print("  • Scrub the bottom slider to move through time")
    print("  • Toggle layer visibility (eye icon) to isolate lone/coop/defector cells")
    print("  • Scroll / pinch to zoom; click-drag to pan")

    launch_viewer(
        snapshots=sim.history,
        grid_width=args.grid_size,
        grid_height=args.grid_size,
        cell_radius=REPULSION_RADIUS,
        regional_tasks=sim.env.regional_rewards,
        reward_params={
            "base":     BASE_REWARD,
            "simple":   SIMPLE_REWARD,
            "complex":  COMPLEX_REWARD,
            "triple":   TRIPLE_REWARD,
            "drain":    DEFECTOR_DRAIN,
            "coop_cost": COOP_COST,
        },
    )


if __name__ == "__main__":
    main()
