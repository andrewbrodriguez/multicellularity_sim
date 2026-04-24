"""
Batch experimentation script.

Each SWEEP dict below defines a parameter axis to vary. The script runs every
configuration, collects per-tick history and final cell snapshots, and writes
combined CSVs plus a printed comparison table.

Usage
-----
# run the default sweep (initial population size)
python experiment.py

# run a specific sweep by name
python experiment.py --sweep mutation_rate
python experiment.py --sweep task_flip
python experiment.py --sweep population

# run all sweeps back-to-back
python experiment.py --sweep all

# override common settings
python experiment.py --ticks 3000 --seeds 3 --quiet
"""

import argparse
import itertools
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.simulation import Simulation


# ── sweep definitions ─────────────────────────────────────────────────────────
# Each sweep is a list of dicts; every key maps directly to a Simulation /
# Environment constructor argument.  Add new sweeps here.

SWEEPS = {
    "population": [
        {"initial_cells": 20},
        {"initial_cells": 50},
        {"initial_cells": 100},
        {"initial_cells": 200},
        {"initial_cells": 400},
    ],
    "mutation_rate": [
        {"mutation_rate": 0.001},
        {"mutation_rate": 0.005},
        {"mutation_rate": 0.01},
        {"mutation_rate": 0.02},
        {"mutation_rate": 0.05},
    ],
    "task_flip": [
        {"task_flip_period": None},
        {"task_flip_period": 50},
        {"task_flip_period": 100},
        {"task_flip_period": 250},
        {"task_flip_period": 500},
    ],
    "grid_size": [
        {"grid_size": 150},
        {"grid_size": 250},
        {"grid_size": 400},
    ],
}

# Defaults used for every trial (overridden by sweep values)
DEFAULTS = {
    "grid_size":        250,
    "initial_cells":    100,
    "mutation_rate":    0.005,
    "task_flip_period": None,
}


# ── trial runner ──────────────────────────────────────────────────────────────

def run_trial(
    config: dict,
    ticks: int,
    sample_every: int,
    seed: int,
    quiet: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run one simulation trial and return (cell_df, history_df).
    Both DataFrames carry all config values as extra columns for easy grouping.
    """
    params = {**DEFAULTS, **config}

    sim = Simulation(
        grid_size=params["grid_size"],
        initial_cells=params["initial_cells"],
        mutation_rate=params["mutation_rate"],
        seed=seed,
        task_flip_period=params["task_flip_period"],
    )
    if quiet:
        sim._print_stats = lambda _: None

    sim.run(ticks=ticks, record_every=sample_every)

    cell_df    = _build_cell_df(sim)
    history_df = _build_history_df(sim)

    # Tag both DataFrames with config metadata
    meta = {
        "seed":             seed,
        "ticks":            ticks,
        **params,
    }
    for k, v in meta.items():
        cell_df[k]    = v
        history_df[k] = v

    return cell_df, history_df


# ── DataFrame builders (mirrors visualize.py, kept self-contained) ────────────

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


# ── summary table ─────────────────────────────────────────────────────────────

def _print_comparison(all_history: pd.DataFrame, sweep_key: str) -> None:
    """Print a compact comparison table grouped by the sweep variable."""
    group_col = sweep_key if sweep_key in all_history.columns else "initial_cells"

    agg = (
        all_history
        .groupby([group_col, "seed"])
        .last()                             # final tick per trial
        .groupby(group_col)
        .agg(
            trials=("tick", "count"),
            cells_mean=("total_cells", "mean"),
            cells_std=("total_cells", "std"),
            coop_pct_mean=("cooperator_pct", "mean"),
            def_pct_mean=("defector_pct", "mean"),
            coop_genome_mean=("coop_genome_pct", "mean"),
            num_clusters_mean=("num_clusters", "mean"),
            avg_cluster_sz=("avg_cluster_size", "mean"),
            multi_adv_mean=("multi_advantage", "mean"),
            coop_adv_mean=("coop_advantage", "mean"),
            mean_fitness=("mean_fitness", "mean"),
        )
        .reset_index()
    )

    col_w = 12
    headers = [
        group_col, "trials", "cells", "±",
        "coop%", "def%", "coopAllele%",
        "nClust", "avgSz",
        "multiAdv", "coopAdv", "avgFit",
    ]
    print("\n" + "  ".join(h.rjust(col_w) for h in headers))
    print("  " + ("─" * col_w + "  ") * len(headers))

    for _, row in agg.iterrows():
        vals = [
            str(row[group_col]),
            f"{int(row['trials'])}",
            f"{row['cells_mean']:.0f}",
            f"{row['cells_std']:.0f}" if not np.isnan(row["cells_std"]) else "—",
            f"{row['coop_pct_mean']:.1f}",
            f"{row['def_pct_mean']:.1f}",
            f"{row['coop_genome_mean']:.1f}",
            f"{row['num_clusters_mean']:.1f}",
            f"{row['avg_cluster_sz']:.2f}",
            f"{row['multi_adv_mean']:+.4f}",
            f"{row['coop_adv_mean']:+.4f}",
            f"{row['mean_fitness']:.2f}",
        ]
        print("  ".join(v.rjust(col_w) for v in vals))
    print()


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Batch experimentation for multicellularity sim.")
    parser.add_argument("--sweep",        default="population",
                        help=f"Sweep to run: {list(SWEEPS)} or 'all' (default: population)")
    parser.add_argument("--ticks",        type=int,   default=1000,
                        help="Ticks per trial (default: 1000)")
    parser.add_argument("--sample-every", type=int,   default=50,
                        help="Record every N ticks (default: 50)")
    parser.add_argument("--seeds",        type=int,   default=3,
                        help="Number of random seeds per config (default: 3)")
    parser.add_argument("--base-seed",    type=int,   default=0,
                        help="Seeds used are base_seed + 0..seeds-1 (default: 0)")
    parser.add_argument("--save-csv",     type=str,   default="results",
                        help="Output directory for CSVs (default: results/)")
    parser.add_argument("--quiet",        action="store_true",
                        help="Suppress per-tick output (strongly recommended for batch runs)")
    args = parser.parse_args()

    sweep_names = list(SWEEPS.keys()) if args.sweep == "all" else [args.sweep]
    for name in sweep_names:
        if name not in SWEEPS:
            parser.error(f"Unknown sweep '{name}'. Choose from: {list(SWEEPS)} or 'all'")

    out_dir = Path(args.save_csv)
    out_dir.mkdir(parents=True, exist_ok=True)

    seeds = [args.base_seed + i for i in range(args.seeds)]

    for sweep_name in sweep_names:
        configs = SWEEPS[sweep_name]
        total = len(configs) * len(seeds)
        print(f"\n{'═'*75}")
        print(f"  Sweep: {sweep_name}  |  {len(configs)} configs × {len(seeds)} seeds = {total} trials")
        print(f"  ticks={args.ticks}  sample_every={args.sample_every}  seeds={seeds}")
        print(f"{'═'*75}\n")

        all_cells   : list[pd.DataFrame] = []
        all_history : list[pd.DataFrame] = []

        trial_num = 0
        for config, seed in itertools.product(configs, seeds):
            trial_num += 1
            label = "  ".join(f"{k}={v}" for k, v in config.items())
            print(f"[{trial_num}/{total}]  {label}  seed={seed}", flush=True)

            t0 = time.perf_counter()
            cell_df, history_df = run_trial(
                config=config,
                ticks=args.ticks,
                sample_every=args.sample_every,
                seed=seed,
                quiet=args.quiet,
            )
            elapsed = time.perf_counter() - t0

            final = history_df.iloc[-1]
            print(
                f"         cells={int(final['total_cells'])}  "
                f"coop%={final['cooperator_pct']:.1f}  "
                f"def%={final['defector_pct']:.1f}  "
                f"clusters={int(final['num_clusters'])}  "
                f"multiAdv={final['multi_advantage']:+.3f}  "
                f"({elapsed:.1f}s)\n"
            )

            all_cells.append(cell_df)
            all_history.append(history_df)

        combined_cells   = pd.concat(all_cells,   ignore_index=True)
        combined_history = pd.concat(all_history, ignore_index=True)

        # Print comparison table
        _print_comparison(combined_history, sweep_name)

        # Save
        cells_path   = out_dir / f"{sweep_name}_cell_df.csv"
        history_path = out_dir / f"{sweep_name}_history_df.csv"
        combined_cells.to_csv(cells_path,   index=False)
        combined_history.to_csv(history_path, index=False)
        print(f"Saved:\n  {cells_path}\n  {history_path}\n")


if __name__ == "__main__":
    main()
