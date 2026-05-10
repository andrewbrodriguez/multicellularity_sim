"""
Run the simulation with the Napari viewer.

Examples:
    python visualize.py
    python visualize.py --viewer off --ticks 2000 --sample-every 50
    python visualize.py --task-flip-period 100 --initial-cells 100 --ticks 600
"""

import argparse
import os

import pandas as pd

from src.simulation import Simulation
from src.visualizer import launch_viewer
from src.environment import (
    MAINTENANCE_COST, DEFECTOR_DRAIN, COOP_COST, COOP_REWARD_SCALE, REPULSION_RADIUS,
)
from src.cell import ADHESION_COST


def _build_cell_df(sim: Simulation) -> pd.DataFrame:
    rows = []
    for cell in sim.env.cells.values():
        if cell.position is None:
            continue
        cid = cell.cluster_id if cell.cluster_id is not None else -1
        if cid == -1:
            ctype = "lone"
        elif cell.is_cooperator:
            ctype = "cooperator"
        else:
            ctype = "defector"
        rows.append({
            "cell_id":       cell.cell_id,
            "genome":        "".join(map(str, cell.genome)),
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
    return pd.DataFrame([{k: s.get(k) for k in scalar_keys} for s in sim.history])


def _print_final_summary(cell_df: pd.DataFrame, history_df: pd.DataFrame) -> None:
    final_tick = int(history_df["tick"].iloc[-1])
    print(f"\n{'─'*75}")
    print(f"Final state (tick {final_tick})  —  {len(cell_df)} living cells\n")

    counts = cell_df["type"].value_counts()
    for t in ("cooperator", "defector", "lone"):
        n = counts.get(t, 0)
        pct = 100.0 * n / max(len(cell_df), 1)
        print(f"  {t:<12}: {n:5d}  ({pct:5.1f}%)")

    print(f"\n  Operation frequencies (all cells):")
    for op, cnt in cell_df["op"].value_counts().items():
        pct = 100.0 * cnt / max(len(cell_df), 1)
        print(f"    {op:<6}: {cnt:5d}  ({pct:5.1f}%)")

    clustered = cell_df[cell_df["cluster_id"] >= 0]
    if not clustered.empty:
        diversity = clustered.groupby("cluster_id")["op"].nunique().mean()
        print(f"\n  Avg distinct operations per cluster: {diversity:.2f}")

    last = history_df.iloc[-1]
    print(f"\n  multi_advantage : {last['multi_advantage']:+.4f}  (+= multicellularity pays)")
    print(f"  coop_advantage  : {last['coop_advantage']:+.4f}  (+= cooperation pays inside clusters)")
    print(f"{'─'*75}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run multicellularity simulation and optionally open Napari viewer."
    )
    parser.add_argument("--grid-size",        type=int,   default=250)
    parser.add_argument("--initial-cells",    type=int,   default=40)
    parser.add_argument("--mutation-rate",    type=float, default=0.005)
    parser.add_argument("--ticks",            type=int,   default=500)
    parser.add_argument("--sample-every",     type=int,   default=10)
    parser.add_argument("--seed",             type=int,   default=42)
    parser.add_argument("--quiet",            action="store_true")
    parser.add_argument("--task-flip-period", type=int,   default=None,
                        help="Ticks between reward-landscape re-rolls (MVG mode)")
    parser.add_argument("--viewer",           choices=["on", "off"], default="on")
    parser.add_argument("--save-csv",         type=str,   default=None,
                        help="Directory to save cell_df.csv and history_df.csv")
    args = parser.parse_args()

    flip_status = (f"  task_flip_period={args.task_flip_period}"
                   if args.task_flip_period else "  task_flip_period=None (Static)")
    print(
        f"Multicellularity Simulation\n"
        f"  grid={args.grid_size}×{args.grid_size}  "
        f"cells={args.initial_cells}  ticks={args.ticks}  "
        f"sample_every={args.sample_every}  seed={args.seed}\n"
        f"{flip_status}\n"
        f"  viewer={args.viewer}\n"
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

    cell_df    = _build_cell_df(sim)
    history_df = _build_history_df(sim)
    _print_final_summary(cell_df, history_df)

    if args.save_csv:
        os.makedirs(args.save_csv, exist_ok=True)
        cell_path    = os.path.join(args.save_csv, "cell_df.csv")
        history_path = os.path.join(args.save_csv, "history_df.csv")
        cell_df.to_csv(cell_path,       index=False)
        history_df.to_csv(history_path, index=False)
        print(f"Saved:\n  {cell_path}\n  {history_path}\n")

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
            "maintenance": MAINTENANCE_COST,
            "adhesion":    ADHESION_COST,
            "coop_cost":   COOP_COST,
            "drain":       DEFECTOR_DRAIN,
            "scale":       COOP_REWARD_SCALE,
        },
    )


if __name__ == "__main__":
    main()
