"""
Graph results from experiment.py CSV outputs.

Usage
-----
# graph a specific sweep (looks in results/ by default)
python graph_results.py --sweep population
python graph_results.py --sweep mutation_rate
python graph_results.py --sweep task_flip

# custom results directory
python graph_results.py --sweep population --results-dir my_results/

# save figures instead of showing interactively
python graph_results.py --sweep population --save-dir figures/
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec

# ── style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":  "#0e1117",
    "axes.facecolor":    "#1a1d27",
    "axes.edgecolor":    "#3a3d4a",
    "axes.labelcolor":   "#d0d3e0",
    "axes.titlecolor":   "#ffffff",
    "axes.grid":         True,
    "grid.color":        "#2a2d3a",
    "grid.linewidth":    0.6,
    "xtick.color":       "#9a9db0",
    "ytick.color":       "#9a9db0",
    "text.color":        "#d0d3e0",
    "legend.facecolor":  "#1a1d27",
    "legend.edgecolor":  "#3a3d4a",
    "legend.labelcolor": "#d0d3e0",
    "lines.linewidth":   1.8,
    "figure.dpi":        110,
})

CMAP = plt.cm.plasma
OP_COLORS = {"AND": "#4fc3f7", "OR": "#81c784", "XOR": "#ffb74d", "NAND": "#f06292"}
TYPE_COLORS = {"cooperator": "#4fc3f7", "defector": "#f06292", "lone": "#9a9db0"}


# ── helpers ───────────────────────────────────────────────────────────────────

def _sweep_col(history: pd.DataFrame, sweep: str) -> str:
    """Return the column name that varies in this sweep."""
    candidates = {
        "population":       "initial_cells",
        "mutation_rate":    "mutation_rate",
        "mutation_penalty": "mutation_rate",
        "task_flip":        "task_flip_period",
        "grid_size":        "grid_size",
        "coop_cost":        "coop_cost",
    }
    col = candidates.get(sweep, sweep)
    if col in history.columns:
        return col
    config_cols = ["initial_cells", "mutation_rate", "task_flip_period",
                   "grid_size", "coop_cost"]
    for c in config_cols:
        if c in history.columns and history[c].nunique() > 1:
            return c
    return config_cols[0]


def _palette(n: int):
    return [CMAP(i / max(n - 1, 1)) for i in range(n)]


def _fmt_val(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "None"
    if isinstance(v, float):
        return f"{v:g}"
    return str(v)


def _mean_band(ax, group_df, x_col, y_col, color, label):
    """Plot mean line ± 1 std band across seeds."""
    agg = group_df.groupby(x_col)[y_col].agg(["mean", "std"]).reset_index()
    ax.plot(agg[x_col], agg["mean"], color=color, label=label)
    ax.fill_between(
        agg[x_col],
        agg["mean"] - agg["std"].fillna(0),
        agg["mean"] + agg["std"].fillna(0),
        color=color, alpha=0.18,
    )


# ── figure 1: time-series panel ───────────────────────────────────────────────

def fig_timeseries(history: pd.DataFrame, sweep_col: str, sweep_name: str) -> plt.Figure:
    """8-panel time-series: one line per sweep value, mean ± std across seeds."""
    groups = sorted(history[sweep_col].unique(), key=lambda x: (x is None, x))
    palette = _palette(len(groups))

    metrics = [
        ("total_cells",       "Total cells",             None),
        ("cooperator_pct",    "Active cooperators (%)",  (0, 100)),
        ("defector_pct",      "Active defectors (%)",    (0, 100)),
        ("coop_genome_pct",   "Coop allele (%)",         (0, 100)),
        ("num_clusters",      "Cluster count",           None),
        ("avg_cluster_size",  "Avg cluster size",        None),
        ("multi_advantage",   "Multi advantage",         None),
        ("coop_advantage",    "Coop advantage",          None),
        ("mean_fitness",      "Mean fitness",            None),
    ]

    ncols = 3
    nrows = int(np.ceil(len(metrics) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, nrows * 3.4))
    axes = axes.flatten()

    for ax, (col, title, ylim) in zip(axes, metrics):
        for color, val in zip(palette, groups):
            sub = history[history[sweep_col] == val]
            _mean_band(ax, sub, "tick", col, color, _fmt_val(val))
        ax.set_title(title, fontsize=10, pad=4)
        ax.set_xlabel("tick", fontsize=8)
        if ylim:
            ax.set_ylim(*ylim)
        ax.tick_params(labelsize=7)

    # hide unused axes
    for ax in axes[len(metrics):]:
        ax.set_visible(False)

    # shared legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, title=sweep_col, loc="lower right",
               fontsize=8, title_fontsize=9, ncol=min(len(groups), 6),
               bbox_to_anchor=(0.99, 0.01))

    fig.suptitle(f"Time series — sweep: {sweep_name}", fontsize=13, y=1.01)
    fig.tight_layout()
    return fig


# ── figure 2: final-state comparison bars ────────────────────────────────────

def fig_final_bars(history: pd.DataFrame, sweep_col: str, sweep_name: str) -> plt.Figure:
    """Bar charts of key final-state metrics grouped by sweep value."""
    final = (
        history.groupby([sweep_col, "seed"]).last().reset_index()
        .groupby(sweep_col).agg(
            cells_mean=("total_cells",      "mean"),
            cells_std=("total_cells",       "std"),
            coop_mean=("cooperator_pct",    "mean"),
            coop_std=("cooperator_pct",     "std"),
            def_mean=("defector_pct",       "mean"),
            def_std=("defector_pct",        "std"),
            genome_mean=("coop_genome_pct", "mean"),
            genome_std=("coop_genome_pct",  "std"),
            multi_mean=("multi_advantage",  "mean"),
            multi_std=("multi_advantage",   "std"),
            coop_adv_mean=("coop_advantage","mean"),
            coop_adv_std=("coop_advantage", "std"),
            fit_mean=("mean_fitness",       "mean"),
            fit_std=("mean_fitness",        "std"),
        ).reset_index()
    )

    x_labels = [_fmt_val(v) for v in final[sweep_col]]
    x = np.arange(len(x_labels))
    palette = _palette(len(x))

    panels = [
        ("cells_mean",    "cells_std",    "Final cell count",        None),
        ("coop_mean",     "coop_std",     "Active cooperators (%)",  (0, 100)),
        ("def_mean",      "def_std",      "Active defectors (%)",    (0, 100)),
        ("genome_mean",   "genome_std",   "Coop allele (%)",         (0, 100)),
        ("multi_mean",    "multi_std",    "Multi advantage",         None),
        ("coop_adv_mean", "coop_adv_std", "Coop advantage",          None),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 7))
    axes = axes.flatten()

    for ax, (m_col, s_col, title, ylim) in zip(axes, panels):
        bars = ax.bar(x, final[m_col], color=palette,
                      yerr=final[s_col].fillna(0), capsize=4,
                      error_kw={"ecolor": "#ffffff44", "lw": 1.2})
        ax.set_title(title, fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, rotation=30, ha="right", fontsize=8)
        ax.set_xlabel(sweep_col, fontsize=8)
        if ylim:
            ax.set_ylim(*ylim)
        ax.tick_params(labelsize=7)
        # zero-line for advantage plots
        if "advantage" in title.lower():
            ax.axhline(0, color="#ffffff44", lw=1, ls="--")

    fig.suptitle(f"Final-state summary — sweep: {sweep_name}", fontsize=13)
    fig.tight_layout()
    return fig


# ── figure 3: genome / cell-type breakdown ────────────────────────────────────

def fig_genome(cells: pd.DataFrame, sweep_col: str, sweep_name: str) -> plt.Figure:
    """Operation frequency + type distribution + cluster diversity from cell_df."""
    groups = sorted(cells[sweep_col].unique(), key=lambda x: (x is None, x))
    g_labels = [_fmt_val(v) for v in groups]
    x = np.arange(len(groups))
    ops = ["AND", "OR", "XOR", "NAND"]
    types = ["cooperator", "defector", "lone"]

    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.35)

    # ── (0,0) stacked bar: operation mix ──────────────────────────────────────
    ax0 = fig.add_subplot(gs[0, 0])
    bottoms = np.zeros(len(groups))
    for op in ops:
        vals = []
        for val in groups:
            sub = cells[cells[sweep_col] == val]
            vals.append(100.0 * (sub["op"] == op).sum() / max(len(sub), 1))
        ax0.bar(x, vals, bottom=bottoms, color=OP_COLORS[op], label=op, width=0.6)
        bottoms += np.array(vals)
    ax0.set_title("Operation mix (all cells)", fontsize=10)
    ax0.set_xticks(x); ax0.set_xticklabels(g_labels, rotation=30, ha="right", fontsize=8)
    ax0.set_ylabel("%"); ax0.set_ylim(0, 100)
    ax0.legend(fontsize=7, loc="upper right")

    # ── (0,1) stacked bar: cell type mix ─────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 1])
    bottoms = np.zeros(len(groups))
    for ctype in types:
        vals = []
        for val in groups:
            sub = cells[cells[sweep_col] == val]
            vals.append(100.0 * (sub["type"] == ctype).sum() / max(len(sub), 1))
        ax1.bar(x, vals, bottom=bottoms, color=TYPE_COLORS[ctype], label=ctype, width=0.6)
        bottoms += np.array(vals)
    ax1.set_title("Cell type mix (final)", fontsize=10)
    ax1.set_xticks(x); ax1.set_xticklabels(g_labels, rotation=30, ha="right", fontsize=8)
    ax1.set_ylabel("%"); ax1.set_ylim(0, 100)
    ax1.legend(fontsize=7, loc="upper right")

    # ── (0,2) cluster genome diversity ───────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 2])
    div_means, div_stds = [], []
    for val in groups:
        sub = cells[(cells[sweep_col] == val) & (cells["cluster_id"] >= 0)]
        if sub.empty:
            div_means.append(0); div_stds.append(0)
            continue
        div = sub.groupby("cluster_id")["op"].nunique()
        div_means.append(div.mean())
        div_stds.append(div.std())
    palette = _palette(len(groups))
    ax2.bar(x, div_means, color=palette, yerr=div_stds,
            capsize=4, error_kw={"ecolor": "#ffffff44"}, width=0.6)
    ax2.axhline(1, color="#ffffff44", lw=1, ls="--", label="homogeneous")
    ax2.set_title("Avg distinct ops / cluster", fontsize=10)
    ax2.set_xticks(x); ax2.set_xticklabels(g_labels, rotation=30, ha="right", fontsize=8)
    ax2.set_ylabel("unique operations"); ax2.legend(fontsize=7)

    # ── (1,0) operation mix — cooperators only ────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    bottoms = np.zeros(len(groups))
    for op in ops:
        vals = []
        for val in groups:
            sub = cells[(cells[sweep_col] == val) & (cells["is_cooperator"] == 1)]
            vals.append(100.0 * (sub["op"] == op).sum() / max(len(sub), 1))
        ax3.bar(x, vals, bottom=bottoms, color=OP_COLORS[op], label=op, width=0.6)
        bottoms += np.array(vals)
    ax3.set_title("Operation mix (cooperators only)", fontsize=10)
    ax3.set_xticks(x); ax3.set_xticklabels(g_labels, rotation=30, ha="right", fontsize=8)
    ax3.set_ylabel("%"); ax3.set_ylim(0, 100)
    ax3.legend(fontsize=7, loc="upper right")

    # ── (1,1) fitness distribution violin ────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    data_by_group = [
        cells[cells[sweep_col] == val]["fitness"].dropna().values
        for val in groups
    ]
    data_by_group = [d if len(d) > 1 else np.array([0.0, 0.0]) for d in data_by_group]
    parts = ax4.violinplot(data_by_group, positions=x, showmedians=True, widths=0.6)
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(palette[i])
        pc.set_alpha(0.7)
    parts["cmedians"].set_color("#ffffff")
    ax4.set_title("Fitness distribution (final)", fontsize=10)
    ax4.set_xticks(x); ax4.set_xticklabels(g_labels, rotation=30, ha="right", fontsize=8)
    ax4.set_ylabel("fitness")

    # ── (1,2) cluster size histogram overlay ─────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 2])
    for color, val in zip(palette, groups):
        sub = cells[(cells[sweep_col] == val) & (cells["cluster_id"] >= 0)]
        if sub.empty:
            continue
        sizes = sub.groupby("cluster_id").size().values
        ax5.hist(sizes, bins=range(1, 14), density=True, histtype="step",
                 color=color, label=_fmt_val(val), linewidth=1.5)
    ax5.set_title("Cluster size distribution", fontsize=10)
    ax5.set_xlabel("cluster size"); ax5.set_ylabel("density (normalized)")
    ax5.legend(title=sweep_col, fontsize=7, title_fontsize=8)

    fig.suptitle(f"Genome & cell composition — sweep: {sweep_name}", fontsize=13)
    return fig


# ── figure 4: selection pressure heatmap ─────────────────────────────────────

def fig_selection_heatmap(history: pd.DataFrame, sweep_col: str, sweep_name: str) -> plt.Figure:
    """Heatmap of multi_advantage and coop_advantage over time × sweep value."""
    groups = sorted(history[sweep_col].unique(), key=lambda x: (x is None, x))
    ticks  = sorted(history["tick"].unique())

    def _build_matrix(col):
        mat = np.full((len(groups), len(ticks)), np.nan)
        for gi, val in enumerate(groups):
            sub = history[history[sweep_col] == val].groupby("tick")[col].mean()
            for ti, t in enumerate(ticks):
                if t in sub.index:
                    mat[gi, ti] = sub[t]
        return mat

    fig, axes = plt.subplots(1, 2, figsize=(15, max(3, len(groups) * 0.9 + 2)))

    for ax, col, title in [
        (axes[0], "multi_advantage",  "Multi advantage\n(+= multicellularity pays)"),
        (axes[1], "coop_advantage",   "Coop advantage\n(+= cooperation beats defection)"),
    ]:
        mat = _build_matrix(col)
        vmax = np.nanpercentile(np.abs(mat), 95)
        im = ax.imshow(mat, aspect="auto", cmap="RdYlGn",
                       vmin=-vmax, vmax=vmax,
                       extent=[ticks[0], ticks[-1], len(groups) - 0.5, -0.5])
        ax.set_yticks(range(len(groups)))
        ax.set_yticklabels([_fmt_val(v) for v in groups], fontsize=8)
        ax.set_xlabel("tick"); ax.set_ylabel(sweep_col)
        ax.set_title(title, fontsize=10)
        fig.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle(f"Selection pressure over time — sweep: {sweep_name}", fontsize=13)
    fig.tight_layout()
    return fig


# ── figure 5: coop_bonus scatter (coop_reward_scale vs cooperator_pct) ───────

def fig_coop_bonus_scatter(history: pd.DataFrame, cells: pd.DataFrame) -> plt.Figure:
    """
    x-axis : coop_reward_scale (the cooperation bonus multiplier)
    y-axis : cooperator_pct, defector_pct, coop_genome_pct at end of run
    Each seed is a translucent dot; the bold line is the mean across seeds.
    A second panel shows cluster genome diversity vs the bonus.
    """
    sweep_col = "coop_reward_scale"

    final = (
        history.groupby([sweep_col, "seed"])
        .last()
        .reset_index()
    )

    xs = sorted(final[sweep_col].unique())

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # ── panel A: cooperator / defector / allele pct ───────────────────────────
    ax = axes[0]
    for col, color, label in [
        ("cooperator_pct",  "#4fc3f7", "active cooperators"),
        ("defector_pct",    "#f06292", "active defectors"),
        ("coop_genome_pct", "#81c784", "coop allele"),
    ]:
        # individual seed dots
        ax.scatter(final[sweep_col], final[col], color=color, alpha=0.35, s=30, zorder=2)
        # mean line
        means = final.groupby(sweep_col)[col].mean()
        ax.plot(means.index, means.values, color=color, label=label,
                marker="o", markersize=5, zorder=3)

    ax.set_xscale("log")
    ax.set_xlabel("coop_reward_scale  (log)", fontsize=10)
    ax.set_ylabel("% of all cells", fontsize=10)
    ax.set_ylim(0, 100)
    ax.set_title("Cooperation vs reward bonus", fontsize=11)
    ax.legend(fontsize=8)
    ax.axvline(0.15, color="#ffffff33", lw=1, ls="--", label="default (0.15)")

    # ── panel B: selection pressures vs bonus ─────────────────────────────────
    ax = axes[1]
    for col, color, label in [
        ("multi_advantage", "#ffb74d", "multi advantage"),
        ("coop_advantage",  "#ce93d8", "coop advantage"),
    ]:
        ax.scatter(final[sweep_col], final[col], color=color, alpha=0.35, s=30, zorder=2)
        means = final.groupby(sweep_col)[col].mean()
        ax.plot(means.index, means.values, color=color, label=label,
                marker="o", markersize=5, zorder=3)

    ax.set_xscale("log")
    ax.axhline(0, color="#ffffff44", lw=1, ls="--")
    ax.axvline(0.15, color="#ffffff33", lw=1, ls="--")
    ax.set_xlabel("coop_reward_scale  (log)", fontsize=10)
    ax.set_ylabel("fitness/tick differential", fontsize=10)
    ax.set_title("Selection pressures vs reward bonus", fontsize=11)
    ax.legend(fontsize=8)

    # ── panel C: cluster genome diversity vs bonus ────────────────────────────
    ax = axes[2]
    if sweep_col in cells.columns:
        div_mean, div_std, div_xs = [], [], []
        for x in xs:
            sub = cells[(cells[sweep_col] == x) & (cells["cluster_id"] >= 0)]
            if sub.empty:
                continue
            d = sub.groupby("cluster_id")["op"].nunique()
            div_mean.append(d.mean())
            div_std.append(d.std(ddof=0))
            div_xs.append(x)
        div_xs    = np.array(div_xs)
        div_mean  = np.array(div_mean)
        div_std   = np.array(div_std)
        ax.fill_between(div_xs, div_mean - div_std, div_mean + div_std,
                        color="#4fc3f7", alpha=0.2)
        ax.plot(div_xs, div_mean, color="#4fc3f7", marker="o", markersize=5,
                label="mean ± std")
        ax.axhline(1, color="#ffffff33", lw=1, ls="--", label="homogeneous")
        ax.axvline(0.15, color="#ffffff33", lw=1, ls="--")
        ax.set_xscale("log")
        ax.set_xlabel("coop_reward_scale  (log)", fontsize=10)
        ax.set_ylabel("distinct ops / cluster", fontsize=10)
        ax.set_title("Cluster genome diversity vs reward bonus", fontsize=11)
        ax.legend(fontsize=8)

    fig.suptitle("Effect of cooperation bonus (coop_reward_scale)", fontsize=13)
    fig.tight_layout()
    return fig


# ── figure 6: task_distribution scatter ──────────────────────────────────────

def fig_task_distribution(history: pd.DataFrame, cells: pd.DataFrame) -> plt.Figure:
    """
    Four-panel figure showing how task_alpha (Dirichlet concentration) shapes
    population composition.

    Panel A — population mix vs alpha
        y: cooperator_pct, defector_pct, coop_genome_pct
        Interpretation: does enforced spatial specialisation drive more cooperation?

    Panel B — selection pressures vs alpha
        y: multi_advantage, coop_advantage
        Interpretation: at what alpha does group selection dominate?

    Panel C — cluster genome diversity vs alpha
        y: mean distinct ops per cluster
        Interpretation: concentrated tasks → clusters must be more diverse to cover them.

    Panel D — operation evenness vs alpha (Shannon entropy of op distribution)
        y: Shannon entropy of operation frequencies across all cells
        Interpretation: uniform tasks → all ops equally viable; sparse tasks → one op dominates.
    """
    sweep_col = "task_alpha"

    final = (
        history.groupby([sweep_col, "seed"])
        .last()
        .reset_index()
    )
    xs = np.array(sorted(final[sweep_col].unique()))

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # ── A: population mix ────────────────────────────────────────────────────
    ax = axes[0]
    for col, color, label in [
        ("cooperator_pct",  "#4fc3f7", "active cooperators"),
        ("defector_pct",    "#f06292", "active defectors"),
        ("coop_genome_pct", "#81c784", "coop allele"),
    ]:
        ax.scatter(final[sweep_col], final[col], color=color, alpha=0.35, s=30)
        means = final.groupby(sweep_col)[col].mean()
        ax.plot(means.index, means.values, color=color, marker="o",
                markersize=5, label=label)
    ax.set_xscale("log")
    ax.set_xlabel("task_alpha  (log)\n← specialised   uniform →", fontsize=9)
    ax.set_ylabel("% of all cells", fontsize=9)
    ax.set_ylim(0, 100)
    ax.set_title("Population mix", fontsize=11)
    ax.legend(fontsize=7)
    ax.axvline(0.05, color="#ffffff33", lw=1, ls="--")

    # ── B: selection pressures ───────────────────────────────────────────────
    ax = axes[1]
    for col, color, label in [
        ("multi_advantage", "#ffb74d", "multi advantage"),
        ("coop_advantage",  "#ce93d8", "coop advantage"),
    ]:
        ax.scatter(final[sweep_col], final[col], color=color, alpha=0.35, s=30)
        means = final.groupby(sweep_col)[col].mean()
        ax.plot(means.index, means.values, color=color, marker="o",
                markersize=5, label=label)
    ax.axhline(0, color="#ffffff44", lw=1, ls="--")
    ax.axvline(0.05, color="#ffffff33", lw=1, ls="--")
    ax.set_xscale("log")
    ax.set_xlabel("task_alpha  (log)\n← specialised   uniform →", fontsize=9)
    ax.set_ylabel("fitness/tick differential", fontsize=9)
    ax.set_title("Selection pressures", fontsize=11)
    ax.legend(fontsize=7)

    # ── C: cluster genome diversity ──────────────────────────────────────────
    ax = axes[2]
    if sweep_col in cells.columns:
        div_xs, div_mean, div_std = [], [], []
        for x in xs:
            sub = cells[(cells[sweep_col] == x) & (cells["cluster_id"] >= 0)]
            if sub.empty:
                continue
            d = sub.groupby("cluster_id")["op"].nunique()
            div_xs.append(x); div_mean.append(d.mean()); div_std.append(d.std(ddof=0))
        div_xs   = np.array(div_xs)
        div_mean = np.array(div_mean)
        div_std  = np.array(div_std)
        ax.fill_between(div_xs, div_mean - div_std, div_mean + div_std,
                        color="#4fc3f7", alpha=0.2)
        ax.plot(div_xs, div_mean, color="#4fc3f7", marker="o", markersize=5,
                label="mean ± std")
        ax.axhline(1, color="#ffffff33", lw=1, ls="--", label="homogeneous")
        ax.axvline(0.05, color="#ffffff33", lw=1, ls="--")
        ax.set_xscale("log")
        ax.set_xlabel("task_alpha  (log)\n← specialised   uniform →", fontsize=9)
        ax.set_ylabel("distinct ops / cluster", fontsize=9)
        ax.set_title("Cluster genome diversity", fontsize=11)
        ax.legend(fontsize=7)

    # ── D: Shannon entropy of operation distribution ──────────────────────────
    ax = axes[3]
    if sweep_col in cells.columns:
        ent_xs, ent_mean, ent_std = [], [], []
        ops = ["AND", "OR", "XOR", "NAND"]
        for x in xs:
            entropies = []
            for seed_val in cells["seed"].unique():
                sub = cells[(cells[sweep_col] == x) & (cells["seed"] == seed_val)]
                if sub.empty:
                    continue
                freqs = np.array([(sub["op"] == op).sum() for op in ops], dtype=float)
                total = freqs.sum()
                if total == 0:
                    continue
                p = freqs / total
                p = p[p > 0]
                entropies.append(-np.sum(p * np.log2(p)))
            if not entropies:
                continue
            ent_xs.append(x)
            ent_mean.append(np.mean(entropies))
            ent_std.append(np.std(entropies, ddof=0))
        ent_xs   = np.array(ent_xs)
        ent_mean = np.array(ent_mean)
        ent_std  = np.array(ent_std)
        ax.fill_between(ent_xs, ent_mean - ent_std, ent_mean + ent_std,
                        color="#ffb74d", alpha=0.2)
        ax.plot(ent_xs, ent_mean, color="#ffb74d", marker="o", markersize=5,
                label="mean ± std")
        # max entropy for 4 ops = log2(4) = 2.0
        ax.axhline(2.0, color="#ffffff33", lw=1, ls="--", label="max (uniform)")
        ax.axvline(0.05, color="#ffffff33", lw=1, ls="--")
        ax.set_xscale("log")
        ax.set_ylim(0, 2.2)
        ax.set_xlabel("task_alpha  (log)\n← specialised   uniform →", fontsize=9)
        ax.set_ylabel("Shannon entropy (bits)", fontsize=9)
        ax.set_title("Op distribution evenness", fontsize=11)
        ax.legend(fontsize=7)

    fig.suptitle(
        "Effect of task distribution (task_alpha)\n"
        "low α = each region rewards 1–2 tasks only   |   high α = all tasks equally rewarded",
        fontsize=12,
    )
    fig.tight_layout()
    return fig


# ── figure 7: coop_cost scatter ──────────────────────────────────────────────

def fig_coop_cost_scatter(history: pd.DataFrame, cells: pd.DataFrame) -> plt.Figure:
    """
    x-axis : coop_cost (metabolic penalty cooperators pay each tick)
    Three panels:
      A — cooperator_pct, defector_pct, coop_genome_pct (final state)
      B — multi_advantage, coop_advantage (selection pressures, final state)
      C — coop_advantage - (coop_cost × mean_fitness proxy) breakeven line
    Each seed = translucent dot; bold line = mean across seeds.
    """
    sweep_col = "coop_cost"
    final = history.groupby([sweep_col, "seed"]).last().reset_index()
    xs    = np.array(sorted(final[sweep_col].unique()))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # ── A: population mix ────────────────────────────────────────────────────
    ax = axes[0]
    for col, color, label in [
        ("cooperator_pct",  "#4fc3f7", "active cooperators"),
        ("defector_pct",    "#f06292", "active defectors"),
        ("coop_genome_pct", "#81c784", "coop allele"),
    ]:
        ax.scatter(final[sweep_col], final[col], color=color, alpha=0.35, s=30, zorder=2)
        means = final.groupby(sweep_col)[col].mean()
        ax.plot(means.index, means.values, color=color, marker="o",
                markersize=5, label=label, zorder=3)

    ax.set_xlabel("coop_cost  (per tick)", fontsize=10)
    ax.set_ylabel("% of all cells", fontsize=10)
    ax.set_ylim(0, 100)
    ax.set_title("Cooperation vs metabolic cost", fontsize=11)
    ax.legend(fontsize=8)
    ax.axvline(0.10, color="#ffffff33", lw=1, ls="--", label="default (0.10)")

    # ── B: selection pressures ────────────────────────────────────────────────
    ax = axes[1]
    for col, color, label in [
        ("multi_advantage", "#ffb74d", "multi advantage\n(cluster vs lone)"),
        ("coop_advantage",  "#ce93d8", "coop advantage\n(coop vs def in cluster)"),
    ]:
        ax.scatter(final[sweep_col], final[col], color=color, alpha=0.35, s=30, zorder=2)
        means = final.groupby(sweep_col)[col].mean()
        ax.plot(means.index, means.values, color=color, marker="o",
                markersize=5, label=label, zorder=3)
    ax.axhline(0, color="#ffffff44", lw=1, ls="--")
    ax.axvline(0.10, color="#ffffff33", lw=1, ls="--")
    ax.set_xlabel("coop_cost  (per tick)", fontsize=10)
    ax.set_ylabel("fitness/tick differential", fontsize=10)
    ax.set_title("Selection pressure vs cost", fontsize=11)
    ax.legend(fontsize=8)

    # ── C: timeseries of cooperator_pct, one line per cost level ─────────────
    ax = axes[2]
    palette = _palette(len(xs))
    for color, x_val in zip(palette, xs):
        sub = history[history[sweep_col] == x_val]
        _mean_band(ax, sub, "tick", "cooperator_pct", color, f"cost={x_val:g}")
    ax.set_xlabel("tick", fontsize=10)
    ax.set_ylabel("active cooperators (%)", fontsize=10)
    ax.set_ylim(0, 100)
    ax.set_title("Cooperator % trajectory by cost", fontsize=11)
    ax.legend(fontsize=7, ncol=2)

    fig.suptitle(
        "Effect of cooperation penalty (coop_cost)\n"
        "low cost = free-riding nearly impossible  |  high cost = cooperators out-competed",
        fontsize=12,
    )
    fig.tight_layout()
    return fig


# ── figure 8: mutation_penalty scatter ───────────────────────────────────────

def fig_mutation_scatter(history: pd.DataFrame, cells: pd.DataFrame) -> plt.Figure:
    """
    x-axis : mutation_rate (log scale)
    The built-in 2× coop→def / 0.3× def→coop asymmetry means higher mutation
    rate = stronger drift toward defection.

    Three panels:
      A — cooperator_pct, defector_pct, coop_genome_pct (final state)
      B — cluster genome diversity (unique ops / cluster) vs mutation_rate
      C — timeseries of cooperator_pct at each mutation level
    """
    sweep_col = "mutation_rate"
    final = history.groupby([sweep_col, "seed"]).last().reset_index()
    xs    = np.array(sorted(final[sweep_col].unique()))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # ── A: population mix ────────────────────────────────────────────────────
    ax = axes[0]
    for col, color, label in [
        ("cooperator_pct",  "#4fc3f7", "active cooperators"),
        ("defector_pct",    "#f06292", "active defectors"),
        ("coop_genome_pct", "#81c784", "coop allele"),
    ]:
        ax.scatter(final[sweep_col], final[col], color=color, alpha=0.35, s=30, zorder=2)
        means = final.groupby(sweep_col)[col].mean()
        ax.plot(means.index, means.values, color=color, marker="o",
                markersize=5, label=label, zorder=3)
    ax.set_xscale("log")
    ax.set_xlabel("mutation_rate  (log)", fontsize=10)
    ax.set_ylabel("% of all cells", fontsize=10)
    ax.set_ylim(0, 100)
    ax.set_title("Cooperation vs mutation rate", fontsize=11)
    ax.legend(fontsize=8)
    ax.axvline(0.005, color="#ffffff33", lw=1, ls="--", label="default (0.005)")

    # ── B: cluster genome diversity ───────────────────────────────────────────
    ax = axes[1]
    if sweep_col in cells.columns:
        div_xs, div_mean, div_std = [], [], []
        for x_val in xs:
            sub = cells[(cells[sweep_col] == x_val) & (cells["cluster_id"] >= 0)]
            if sub.empty:
                continue
            d = sub.groupby("cluster_id")["op"].nunique()
            div_xs.append(x_val); div_mean.append(d.mean()); div_std.append(d.std(ddof=0))
        div_xs   = np.array(div_xs)
        div_mean = np.array(div_mean)
        div_std  = np.array(div_std)
        ax.fill_between(div_xs, div_mean - div_std, div_mean + div_std,
                        color="#4fc3f7", alpha=0.2)
        ax.plot(div_xs, div_mean, color="#4fc3f7", marker="o", markersize=5,
                label="distinct ops / cluster")
        ax.axhline(1, color="#ffffff33", lw=1, ls="--", label="homogeneous")
        ax.axvline(0.005, color="#ffffff33", lw=1, ls="--")
        ax.set_xscale("log")
        ax.set_xlabel("mutation_rate  (log)", fontsize=10)
        ax.set_ylabel("mean distinct ops / cluster", fontsize=10)
        ax.set_title("Cluster genome diversity", fontsize=11)
        ax.legend(fontsize=8)

    # ── C: selection pressure trajectories ───────────────────────────────────
    ax = axes[2]
    palette = _palette(len(xs))
    for color, x_val in zip(palette, xs):
        sub = history[history[sweep_col] == x_val]
        _mean_band(ax, sub, "tick", "coop_advantage", color, f"μ={x_val:g}")
    ax.axhline(0, color="#ffffff44", lw=1, ls="--")
    ax.set_xlabel("tick", fontsize=10)
    ax.set_ylabel("coop advantage (coop−def in cluster)", fontsize=10)
    ax.set_title("Coop advantage trajectory by μ", fontsize=11)
    ax.legend(fontsize=7, ncol=2)

    fig.suptitle(
        "Effect of mutation rate — 2× coop→def bias built in\n"
        "high μ = stronger drift toward defection regardless of selection",
        fontsize=12,
    )
    fig.tight_layout()
    return fig


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Graph experiment CSV results.")
    parser.add_argument("--sweep",       required=True,
                        help="Sweep name (e.g. population, mutation_rate, task_flip, grid_size)")
    parser.add_argument("--results-dir", default="results",
                        help="Directory containing the CSVs (default: results/)")
    parser.add_argument("--save-dir",    default=None,
                        help="Save figures as PNGs here instead of showing interactively")
    args = parser.parse_args()

    results = Path(args.results_dir)
    hist_path = results / f"{args.sweep}_history_df.csv"
    cell_path = results / f"{args.sweep}_cell_df.csv"

    if not hist_path.exists():
        raise FileNotFoundError(f"No history CSV found at {hist_path}\n"
                                f"Run: python experiment.py --sweep {args.sweep}")
    if not cell_path.exists():
        raise FileNotFoundError(f"No cell CSV found at {cell_path}")

    print(f"Loading {hist_path} …")
    history = pd.read_csv(hist_path)
    print(f"Loading {cell_path} …")
    cells   = pd.read_csv(cell_path)

    sweep_col = _sweep_col(history, args.sweep)
    print(f"Sweep column: '{sweep_col}'  |  "
          f"{history[sweep_col].nunique()} values  |  "
          f"{history['seed'].nunique()} seeds\n")

    figures = [
        ("timeseries",         fig_timeseries(history, sweep_col, args.sweep)),
        ("final_bars",         fig_final_bars(history, sweep_col, args.sweep)),
        ("genome_composition", fig_genome(cells, sweep_col, args.sweep)),
        ("selection_heatmap",  fig_selection_heatmap(history, sweep_col, args.sweep)),
    ]
    if args.sweep == "coop_bonus":
        figures.append(("coop_bonus_scatter", fig_coop_bonus_scatter(history, cells)))
    if args.sweep == "task_distribution":
        figures.append(("task_distribution", fig_task_distribution(history, cells)))
    if args.sweep == "coop_cost":
        figures.append(("coop_cost_scatter", fig_coop_cost_scatter(history, cells)))
    if args.sweep in ("mutation_penalty", "mutation_rate"):
        figures.append(("mutation_scatter", fig_mutation_scatter(history, cells)))

    if args.save_dir:
        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        for name, fig in figures:
            path = save_dir / f"{args.sweep}_{name}.png"
            fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
            print(f"Saved: {path}")
        plt.close("all")
    else:
        plt.show()


if __name__ == "__main__":
    main()
