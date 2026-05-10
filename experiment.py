"""
Batch experimentation. Each entry in SWEEPS is a list of config dicts whose
keys map directly onto Simulation/Environment kwargs. For each config the
script runs `seeds` trials, collects per-tick history and final cell
snapshots, writes combined CSVs to results/, and prints a comparison table.

Usage:
    python experiment.py --sweep coop_cost
    python experiment.py --sweep all --ticks 1500 --seeds 3 --quiet
"""

import argparse
import itertools
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.simulation import Simulation


SWEEPS = {
    # H1: at what metabolic cost does cooperation collapse?
    "coop_cost": [
        {"coop_cost": 0.00},
        {"coop_cost": 0.10},
        {"coop_cost": 0.25},
        {"coop_cost": 0.50},
        {"coop_cost": 1.00},
        {"coop_cost": 2.00},
        {"coop_cost": 4.00},
        {"coop_cost": 7.50},
        {"coop_cost": 20.0},
    ],
    # H2: at what reward scale does cooperation become net-positive?
    "reward_scale": [
        {"coop_reward_scale": 0.01},
        {"coop_reward_scale": 0.1},
        {"coop_reward_scale": 0.25},
        {"coop_reward_scale": 0.5},
        {"coop_reward_scale": 1.0},
        {"coop_reward_scale": 1.5},
        {"coop_reward_scale": 2.5},
        {"coop_reward_scale": 5.0},
    ],
    # H3: does spatial specialisation favor cooperation?
    # low alpha = harsh specialisation; high alpha = uniform.
    "task_alpha": [
        {"task_alpha": 0.01},
        {"task_alpha": 0.05},
        {"task_alpha": 0.25},
        {"task_alpha": 1.25},
        {"task_alpha": 2.0},
        {"task_alpha": 5.00},
        {"task_alpha": 8.00},
        {"task_alpha": 10.00},
    ],
    # H4: where does mutational drift overwhelm selection?
    "mutation_rate": [
        {"mutation_rate": 0.0001},
        {"mutation_rate": 0.0003},
        {"mutation_rate": 0.0009},
        {"mutation_rate": 0.0027},
        {"mutation_rate": 0.0081},
        {"mutation_rate": 0.0243},
        {"mutation_rate": 0.0729},
        {"mutation_rate": 0.2187},
    ],
    # H5 (MVG): does environmental fluctuation affect the equilibrium?
    "task_flip": [
        {"task_flip_period": 500},
        {"task_flip_period": 250},
        {"task_flip_period": 100},
        {"task_flip_period": 50},
        {"task_flip_period": 30},
        {"task_flip_period": 15},
        {"task_flip_period": 5},
        {"task_flip_period": 3},
        {"task_flip_period": 1},
    ],
    # 2D phase diagram on (coop_cost, coop_reward_scale)
    "phase_diagram": [
        {"coop_cost": cc, "coop_reward_scale": rs}
        for cc in (0.1, 0.5, 2.0, 4.0)
        for rs in (0.2, 0.5, 1.0, 2.5)
    ],
}

DEFAULTS = {
    "grid_size":          250,
    "initial_cells":      100,
    "mutation_rate":      0.005,
    "task_flip_period":   None,
    "coop_reward_scale":  None,   # None → Environment default
    "task_alpha":         None,
    "coop_cost":          None,
}


def run_trial(
    config: dict,
    ticks: int,
    sample_every: int,
    seed: int,
    quiet: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run one trial; both returned DataFrames carry config values as extra columns."""
    params = {**DEFAULTS, **config}

    sim_kwargs = dict(
        grid_size=params["grid_size"],
        initial_cells=params["initial_cells"],
        mutation_rate=params["mutation_rate"],
        seed=seed,
        task_flip_period=params["task_flip_period"],
    )
    if params.get("coop_reward_scale") is not None:
        sim_kwargs["coop_reward_scale"] = params["coop_reward_scale"]
    if params.get("task_alpha") is not None:
        sim_kwargs["task_alpha"] = params["task_alpha"]
    if params.get("coop_cost") is not None:
        sim_kwargs["coop_cost"] = params["coop_cost"]
    sim = Simulation(**sim_kwargs)
    if quiet:
        sim._print_stats = lambda _: None

    sim.run(ticks=ticks, record_every=sample_every)

    cell_df    = _build_cell_df(sim)
    history_df = _build_history_df(sim)

    meta = {"seed": seed, "ticks": ticks, **params}
    for k, v in meta.items():
        cell_df[k]    = v
        history_df[k] = v

    return cell_df, history_df


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


def _print_comparison(all_history: pd.DataFrame, sweep_key: str) -> None:
    """Compact final-state comparison table grouped by the sweep variable."""
    group_map = {
        "coop_cost":      "coop_cost",
        "reward_scale":   "coop_reward_scale",
        "task_alpha":     "task_alpha",
        "mutation_rate":  "mutation_rate",
        "task_flip":      "task_flip_period",
    }
    group_col = group_map.get(sweep_key)
    if group_col is None or group_col not in all_history.columns:
        # 2D sweeps have no single varying axis — skip the rollup
        print(f"\n[skipping comparison table for multi-axis sweep '{sweep_key}' "
              f"— see CSV for results]\n")
        return

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


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch experimentation for multicellularity sim.")
    parser.add_argument("--sweep",        default="coop_cost",
                        help=f"Sweep name from {list(SWEEPS)}, or 'all'")
    parser.add_argument("--ticks",        type=int, default=1000)
    parser.add_argument("--sample-every", type=int, default=50)
    parser.add_argument("--seeds",        type=int, default=3,
                        help="Random seeds per config")
    parser.add_argument("--base-seed",    type=int, default=0)
    parser.add_argument("--save-csv",     type=str, default="results",
                        help="Output directory for CSVs")
    parser.add_argument("--quiet",        action="store_true")
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

        _print_comparison(combined_history, sweep_name)

        cells_path   = out_dir / f"{sweep_name}_cell_df.csv"
        history_path = out_dir / f"{sweep_name}_history_df.csv"
        combined_cells.to_csv(cells_path,   index=False)
        combined_history.to_csv(history_path, index=False)
        print(f"Saved:\n  {cells_path}\n  {history_path}\n")


if __name__ == "__main__":
    main()
