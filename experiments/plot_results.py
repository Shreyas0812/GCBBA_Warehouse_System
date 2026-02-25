"""
Plot Generator for GCBBA Integration Experiments
=================================================

Plots generated:

  === Batch Mode (initial_tasks > 0) ===
  1.  Makespan comparison bar chart (canonical methods, fully connected graph)
  2.  Makespan vs communication range -- key thesis plot
  3.  Improvement ratio vs SGA
  4.  GCBBA computation time and rerun frequency
  5.  Agent utilization and task balance
  6.  Task completion throughput curves
  7.  Collision and deadlock summary
  8.  Completion rate by communication range
  9.  Completion rate broken down by task load
  10. Graph connectivity vs performance (scatter)
  11. Sub-optimality ratio vs SGA (makespan-based)
  12. Rerun interval sensitivity sweep
  13. GCBBA trigger reason breakdown
  14. LaTeX results table (batch)
  15. Charging overhead
  16. Energy-performance scatter

  === Steady-State Mode (task_arrival_rate > 0) ===
  NOTE: Plots 17-22 use SS_PRIMARY_ARRIVAL_RATES=[0.01,0.02,0.03,0.05,0.1,0.15,0.2].
        All rates now have complete uniform coverage — C1 timeout + C2 capped timesteps
        ensure cbba/sga always finish at high arrival rates.
  17. Throughput vs arrival rate (primary steady-state metric)
  18. Task wait time vs arrival rate
  19. Queue depth and saturation metrics vs arrival rate
  20. Throughput vs communication range (fixed arrival rate)
  21. Steady-state optimality ratio vs SGA (throughput-based)
  22. Steady-state LaTeX table
  23. Allocation scalability (SS) -- timing vs arrival rate (ALL data incl. ar=0.2)
      Left: log scale all methods. Right: linear GCBBA vs SGA (CBBA off-chart).
  24. Batch allocation scalability -- timing vs task count (log scale)
      Same structure as 23 but for batch mode; shows SGA 3x worse than GCBBA at tpi=20.

  === New Metric Plots (both modes) ===
  25. Task execution quality -- distance per task + charging abort rate
      Distance/task measures assignment tightness; abort rate measures energy disruption.
  26. Agent time breakdown -- stacked bar: working / idle / charging fractions
      Shows how effectively each method utilises agent capacity.
  27. Scaling exponent -- power-law fit (time ∝ n^k), annotated k per method
      Rigorous empirical O(n^k) argument; GCBBA should show smallest k.
  28. Decentralization penalty -- (SGA_metric - method_metric) / SGA_metric vs CR
      Quantifies graceful degradation; CBBA collapses at low CR, GCBBA degrades gently.
  29. Peak vs avg allocation time + CV -- max/avg/std_gcbba_time_ms, both modes.
      Three panels: avg, peak, and CV (std/mean×100%). Low CV = predictable latency.
  30. Queue drop rate (SS) -- tasks_dropped / tasks_injected vs arrival rate.
      Shows overload threshold; GCBBA should stay lower (faster allocation clears queue).
  31. Batch failure heatmap -- comm_range × tasks_per_induct completion rate grid.
      Red cells = method failed; GCBBA should stay green across more of the space.
  32. Throughput ramp-up curve -- rolling derivative of tasks_completed_over_time (json).
      Shows how quickly each method reaches steady state after t=0.
  33. Allocation time distribution -- violin + log-box from gcbba_run_durations_ms (json).
      Full empirical spread; tight distribution = predictable real-time behaviour.
  34. Per-timestep wall time -- avg_step_time_ms / max_step_time_ms per method vs load.
      Real-time feasibility: if avg < 1 s/tick the system can run at 1 Hz in deployment.
  35. Computational budget breakdown -- stacked bar: allocation time + path planning time.
      Shows what fraction of total compute is allocation vs A* path planning per method.
  36. Energy safety margin (D3) -- avg_final / min_final / min_energy_ever per method.
      Gap between min_final and min_ever reveals mid-run brownout events.
  37. Queue depth time series (D4) -- rolling mean queue depth over time from JSON.
      Shows whether a method clears the queue fast enough or lets it grow unbounded.

Usage:
  python plot_results.py <path_to_experiment_dir>
  python plot_results.py          # plots ALL experiment directories
"""

import os
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    from matplotlib.patches import Patch
except ImportError:
    print("ERROR: matplotlib and pandas required.")
    print("  pip install matplotlib pandas --break-system-packages")
    sys.exit(1)


# -----------------------------------------------------------------
#  Global style
# -----------------------------------------------------------------

plt.rcParams.update({
    "font.size": 12,
    "font.family": "serif",
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "legend.fontsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "figure.figsize": (7, 5),
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "lines.linewidth": 2,
    "lines.markersize": 7,
})

# -- Colors ----------------------------------------------------------
COLORS = {
    "static":        "#1f77b4",   # blue
    "dynamic":       "#d62728",   # red  (canonical ri=50)
    "cbba":          "#ff7f0e",   # orange
    "sga":           "#2ca02c",   # green
    # Batch variants -- same hue as steady-state counterparts
    "static_batch":  "#1f77b4",
    "dynamic_batch": "#d62728",
    "cbba_batch":    "#ff7f0e",
    "sga_batch":     "#2ca02c",
    # Sensitivity sweep -- shades of red/pink for dynamic_ri* configs
    "dynamic_ri10":  "#8B0000",  # dark red
    "dynamic_ri25":  "#c0392b",  # red-crimson
    "dynamic_ri50":  "#d62728",  # same as 'dynamic'
    "dynamic_ri100": "#e07b54",  # salmon
    "dynamic_ri200": "#f2a07b",  # light salmon
}

# -- Labels ----------------------------------------------------------
LABELS = {
    "static":        "Static GCBBA (one-shot)",
    "dynamic":       "Dynamic GCBBA (ri=50)",
    "cbba":          "CBBA (standard)",
    "sga":           "SGA (centralized upper bound)",
    # Batch variants
    "static_batch":  "Static GCBBA (batch)",
    "dynamic_batch": "Dynamic GCBBA (batch, ri=50)",
    "cbba_batch":    "CBBA (batch)",
    "sga_batch":     "SGA (batch, centralized)",
    # ri sensitivity
    "dynamic_ri10":  "Dynamic GCBBA (ri=10)",
    "dynamic_ri25":  "Dynamic GCBBA (ri=25)",
    "dynamic_ri50":  "Dynamic GCBBA (ri=50)",
    "dynamic_ri100": "Dynamic GCBBA (ri=100)",
    "dynamic_ri200": "Dynamic GCBBA (ri=200)",
}


def get_label(config_name: str) -> str:
    """Return label for any config_name, with fallback for unknown names."""
    return LABELS.get(config_name, config_name)


def get_color(config_name: str) -> str:
    """Return color for any config_name, with fallback for unknown names."""
    if config_name in COLORS:
        return COLORS[config_name]
    if config_name.startswith("dynamic_ri"):
        return "#e07b54"
    return "#888888"


# -- Method sets ----------------------------------------------------
CANONICAL_SS_METHODS    = ["static", "dynamic", "cbba", "sga"]
CANONICAL_BATCH_METHODS = ["static_batch", "dynamic_batch", "cbba_batch", "sga_batch"]
CANONICAL_METHODS       = CANONICAL_SS_METHODS   # backward-compat alias

# Arrival rates with complete, uniform data (all methods × all CRs × all seeds).
# All 7 rates are now primary: C1 (timeout) + C2 (capped timesteps) ensure cbba/sga
# complete at high loads.  Throughput is tasks/ss_ts so the shorter cbba/sga window
# (800 ts at ar>=0.1) still produces comparable rates.
# Use _ss_clean_df() for method-comparison plots.
# Use _ss_df() only for scalability/timing plots that include all data.
SS_PRIMARY_ARRIVAL_RATES = [0.01, 0.02, 0.03, 0.05, 0.1, 0.15, 0.2]

# Connectivity threshold (cr < this -> disconnected at t=0)
CONNECTIVITY_THRESHOLD = 13

# Connectivity annotations for x-axis labels
CR_ANNOTATIONS = {
    3:  "Fully\nisolated",
    5:  "Disconnected\n(4 comp.)",
    8:  "Disconnected\n(2 clusters)",
    13: "Barely\nconnected",
    20: "Well\nconnected",
    45: "Fully\nconnected",
}


# -----------------------------------------------------------------
#  Data loading
# -----------------------------------------------------------------

def load_data(exp_dir: str) -> pd.DataFrame:
    """
    Load summary CSV.  Enriched CSV (summary_with_optimality.csv) is used
    when present.  Falls back to computing optimality ratios inline.

    Derived columns added here:
      is_batch          -- True when task_arrival_rate == 0
      tasks_per_induct  -- initial_tasks // 8  (batch) or 0 (steady-state)
      effective_makespan -- makespan if > 0, else total_steps
      effective_throughput -- num_tasks_completed / total_steps (batch proxy)
    """
    summary_path = os.path.join(exp_dir, "summary.csv")
    if not os.path.exists(summary_path):
        print(f"ERROR: summary.csv not found in {exp_dir}")
        sys.exit(1)

    enriched_path = os.path.join(exp_dir, "summary_with_optimality.csv")
    if os.path.exists(enriched_path):
        df = pd.read_csv(enriched_path)
        print(f"  Loaded enriched data ({len(df)} runs) from {enriched_path}")
    else:
        df = pd.read_csv(summary_path)
        print(f"  Loaded {len(df)} runs from {summary_path}")
        print("  (No summary_with_optimality.csv -- computing optimality ratios inline)")
        df = _compute_optimality_inline(df)

    # --- Backward-compat: add task_arrival_rate column if missing ---
    if "task_arrival_rate" not in df.columns:
        df["task_arrival_rate"] = 0.0

    if "initial_tasks" not in df.columns:
        df["initial_tasks"] = 0

    # --- Derived columns ---
    df["is_batch"] = df["task_arrival_rate"] == 0.0

    # tasks_per_induct: meaningful only for batch runs
    if "tasks_per_induct" not in df.columns:
        df["tasks_per_induct"] = (
            df["initial_tasks"].where(df["is_batch"], 0) // 8
        ).astype(int)

    # effective_makespan: use total_steps for DNF or steady-state runs
    df["effective_makespan"] = df.apply(
        lambda r: r["makespan"] if r["makespan"] > 0 else r["total_steps"],
        axis=1,
    )

    # effective_throughput: for batch, proxy = completed / total_steps
    if "throughput" in df.columns:
        df["effective_throughput"] = df["throughput"]
        # For batch rows, fill in a proxy if throughput is 0/missing
        mask = df["is_batch"] & (df["effective_throughput"] == 0)
        if mask.any() and "num_tasks_completed" in df.columns:
            df.loc[mask, "effective_throughput"] = (
                df.loc[mask, "num_tasks_completed"]
                / df.loc[mask, "total_steps"].replace(0, 1)
            )
    elif "num_tasks_completed" in df.columns:
        df["effective_throughput"] = (
            df["num_tasks_completed"] / df["total_steps"].replace(0, 1)
        )
    else:
        df["effective_throughput"] = 0.0

    # --- New derived metrics ---

    # Distance per completed task: measures path quality of allocation.
    # Lower = tighter assignments, less wasted travel.
    if "total_distance_all_agents" in df.columns:
        df["distance_per_task"] = (
            df["total_distance_all_agents"]
            / df["num_tasks_completed"].replace(0, 1)
        )
    else:
        df["distance_per_task"] = np.nan

    # Task abort rate: fraction of tasks abandoned mid-execution for charging.
    # High rate = energy management is disruptive to throughput.
    if "num_tasks_aborted_for_charging" in df.columns:
        df["task_abort_rate"] = (
            df["num_tasks_aborted_for_charging"]
            / df["num_tasks_completed"].replace(0, 1)
        )
    else:
        df["task_abort_rate"] = np.nan

    # Working fraction: agent-timesteps actually executing tasks.
    # working + idle + charging = 1  (approximately; clipped to [0,1]).
    if "avg_idle_ratio" in df.columns and "charging_time_fraction" in df.columns:
        df["working_fraction"] = (
            1.0 - df["avg_idle_ratio"] - df["charging_time_fraction"]
        ).clip(lower=0.0, upper=1.0)
    else:
        df["working_fraction"] = np.nan

    # Queue drop rate: fraction of injected tasks rejected due to full queue (SS).
    # High rate = system is overloaded; low rate = queue has headroom.
    if "tasks_dropped_by_queue_cap" in df.columns and "total_tasks_injected" in df.columns:
        df["tasks_drop_rate"] = (
            df["tasks_dropped_by_queue_cap"] / df["total_tasks_injected"].replace(0, 1)
        )
    else:
        df["tasks_drop_rate"] = np.nan

    return df


def _compute_optimality_inline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute sub-optimality ratios inline for both batch and steady-state runs.

    Batch:         optimality_ratio = effective_makespan / sga_makespan
                   (lower is better; 1.0 = matches SGA)
    Steady-state:  throughput_ratio = method_throughput / sga_throughput
                   (higher is better; 1.0 = matches SGA)
    """
    df = df.copy()

    # Ensure derived columns exist
    if "effective_makespan" not in df.columns:
        df["effective_makespan"] = df.apply(
            lambda r: r["makespan"] if r["makespan"] > 0 else r["total_steps"],
            axis=1,
        )
    if "task_arrival_rate" not in df.columns:
        df["task_arrival_rate"] = 0.0
    if "initial_tasks" not in df.columns:
        df["initial_tasks"] = 0
    if "is_batch" not in df.columns:
        df["is_batch"] = df["task_arrival_rate"] == 0.0

    # --- Batch: makespan-based optimality ---
    df["sga_reference_makespan"] = np.nan
    df["optimality_ratio"]       = np.nan
    df["optimality_ratio_valid"] = False

    batch_df = df[df["is_batch"]]
    if not batch_df.empty and "tasks_per_induct" in df.columns:
        for _, group_df in batch_df.groupby(["seed", "tasks_per_induct", "comm_range"]):
            sga_rows = group_df[
                (group_df["config_name"] == "sga_batch")
                & (group_df["all_tasks_completed"] == True)
            ]
            if sga_rows.empty:
                # Also try bare "sga" in case of legacy data
                sga_rows = group_df[
                    (group_df["config_name"] == "sga")
                    & (group_df["all_tasks_completed"] == True)
                ]
            if sga_rows.empty:
                continue
            sga_makespan = float(sga_rows["effective_makespan"].iloc[0])
            if sga_makespan <= 0:
                continue
            df.loc[group_df.index, "sga_reference_makespan"] = sga_makespan
            for idx in group_df.index:
                row = df.loc[idx]
                completed = bool(row["all_tasks_completed"])
                df.loc[idx, "optimality_ratio_valid"] = completed
                if completed:
                    df.loc[idx, "optimality_ratio"] = round(
                        float(row["effective_makespan"]) / sga_makespan, 4
                    )

    # --- Steady-state: throughput-based optimality ---
    df["sga_reference_throughput"] = np.nan
    df["throughput_ratio"]         = np.nan
    df["throughput_ratio_valid"]   = False

    if "throughput" in df.columns:
        ss_df = df[~df["is_batch"]]
        for _, group_df in ss_df.groupby(["seed", "task_arrival_rate", "comm_range"]):
            sga_rows = group_df[group_df["config_name"] == "sga"]
            if sga_rows.empty:
                continue
            sga_tp = float(sga_rows["throughput"].iloc[0])
            if sga_tp <= 0:
                continue
            df.loc[group_df.index, "sga_reference_throughput"] = sga_tp
            for idx in group_df.index:
                method_tp = float(df.loc[idx, "throughput"])
                df.loc[idx, "throughput_ratio_valid"] = True
                df.loc[idx, "throughput_ratio"] = round(method_tp / sga_tp, 4)

    return df


# -----------------------------------------------------------------
#  Helpers
# -----------------------------------------------------------------

def _savefig(fig, plot_dir: str, name: str) -> None:
    path = os.path.join(plot_dir, f"{name}.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  [OK] {name}.png")


def _canonical_methods_present(df: pd.DataFrame, method_set: list) -> list:
    """Return methods from method_set that actually appear in the dataframe."""
    return [m for m in method_set if m in df["config_name"].unique()]


def _get_max_connected_cr(df: pd.DataFrame) -> int:
    return int(df["comm_range"].max())


def _get_connected_df(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["comm_range"] >= CONNECTIVITY_THRESHOLD].copy()


def _batch_df(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to batch runs with canonical batch method names."""
    return df[
        df["is_batch"] & df["config_name"].isin(CANONICAL_BATCH_METHODS)
    ].copy()


def _ss_df(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to steady-state runs with canonical SS method names (all arrival rates)."""
    return df[
        (~df["is_batch"]) & df["config_name"].isin(CANONICAL_SS_METHODS)
    ].copy()


def _ss_clean_df(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to steady-state runs at SS_PRIMARY_ARRIVAL_RATES only.

    At ar=0.2 cbba and sga have fewer comm_range samples than static/dynamic,
    so comparing them on the same plot would be misleading.  This helper restricts
    to the three arrival rates where every method has complete, uniform coverage
    (all 6 comm_ranges × all 5 seeds).  Use this for every method-comparison plot.
    Use _ss_df() only for the allocation-scalability plot where partial data is valid.
    """
    return df[
        (~df["is_batch"])
        & df["config_name"].isin(CANONICAL_SS_METHODS)
        & df["task_arrival_rate"].isin(SS_PRIMARY_ARRIVAL_RATES)
    ].copy()


# -----------------------------------------------------------------
#  Plot 1: Makespan bar chart (batch) -- canonical methods, fully connected
# -----------------------------------------------------------------

def plot_makespan_bar(df: pd.DataFrame, plot_dir: str) -> None:
    """
    Grouped bar chart: all canonical batch methods, grouped by task count.
    Uses only fully-connected graph runs (highest comm_range).
    """
    dfc = _batch_df(df)
    if dfc.empty:
        print("  [!] Skipping makespan_bar (no batch runs)")
        return

    max_cr = _get_max_connected_cr(dfc)
    df_fc = dfc[dfc["comm_range"] == max_cr].copy()
    if df_fc.empty:
        df_fc = _get_connected_df(dfc)
    if df_fc.empty:
        print("  [!] Skipping makespan_bar (no connected batch runs)")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    task_counts = sorted(df_fc["tasks_per_induct"].unique())
    x = np.arange(len(task_counts))
    width = 0.2
    methods = _canonical_methods_present(df_fc, CANONICAL_BATCH_METHODS)

    for i, cfg in enumerate(methods):
        means, stds = [], []
        for tpi in task_counts:
            vals = df_fc[
                (df_fc["config_name"] == cfg) & (df_fc["tasks_per_induct"] == tpi)
            ]["effective_makespan"]
            means.append(vals.mean() if len(vals) > 0 else 0)
            stds.append(vals.std() if len(vals) > 1 else 0)

        offset = (i - (len(methods) - 1) / 2) * width
        ax.bar(
            x + offset, means, width,
            yerr=stds, label=get_label(cfg), color=get_color(cfg),
            capsize=4, alpha=0.85, edgecolor="black", linewidth=0.5,
        )
        for j, (m, s) in enumerate(zip(means, stds)):
            ax.text(
                x[j] + offset, m + s + 5, f"{m:.0f}",
                ha="center", va="bottom", fontsize=9, fontweight="bold",
            )

    ax.set_xlabel("Initial Tasks per Induct Station")
    ax.set_ylabel("Makespan (timesteps)")
    ax.set_title(
        f"All Methods: Makespan Comparison (Batch Mode)\n"
        f"(Fully Connected Graph, comm_range={max_cr})"
    )
    ax.set_xticks(x)
    ax.set_xticklabels([f"{t}\n({t*8} total)" for t in task_counts])
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(bottom=0)
    _savefig(fig, plot_dir, "makespan_comparison")


# -----------------------------------------------------------------
#  Plot 2: Makespan vs communication range (batch)
# -----------------------------------------------------------------

def plot_makespan_vs_comm_range(df: pd.DataFrame, plot_dir: str) -> None:
    """
    One subplot per task count showing how comm range affects makespan (batch mode).
    """
    dfc = _batch_df(df)
    if dfc.empty:
        print("  [!] Skipping makespan_vs_comm_range (no batch runs)")
        return

    task_counts = sorted(dfc["tasks_per_induct"].unique())
    n_plots = len(task_counts)
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 5), sharey=False)
    if n_plots == 1:
        axes = [axes]

    for ax_idx, tpi in enumerate(task_counts):
        ax = axes[ax_idx]
        df_sub = dfc[dfc["tasks_per_induct"] == tpi]
        methods = _canonical_methods_present(df_sub, CANONICAL_BATCH_METHODS)

        for cfg in methods:
            subset = df_sub[df_sub["config_name"] == cfg]
            grouped = (
                subset.groupby("comm_range")["effective_makespan"]
                .agg(["mean", "std"])
                .reset_index()
            )
            ax.errorbar(
                grouped["comm_range"], grouped["mean"], yerr=grouped["std"],
                label=get_label(cfg), color=get_color(cfg),
                marker="o", capsize=4, linewidth=2,
            )

        comm_ranges = sorted(df_sub["comm_range"].unique())
        y_min = ax.get_ylim()[0]
        for cr in comm_ranges:
            if cr in CR_ANNOTATIONS:
                ax.annotate(
                    CR_ANNOTATIONS[cr], xy=(cr, y_min),
                    fontsize=8, ha="center", va="top",
                    color="gray", style="italic",
                    xytext=(0, -10), textcoords="offset points",
                )

        ax.set_xlabel("Communication Range")
        ax.set_ylabel("Makespan (timesteps)")
        ax.set_title(f"{tpi} tasks/induct ({tpi*8} total)")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        ax.axvspan(0, CONNECTIVITY_THRESHOLD - 1, alpha=0.08, color="red",
                   label="_nolegend_")

    fig.suptitle("Effect of Communication Range on Makespan (Batch Mode)",
                 fontsize=15, y=1.02)
    fig.tight_layout()
    _savefig(fig, plot_dir, "makespan_vs_comm_range")


# -----------------------------------------------------------------
#  Plot 3: Improvement ratio vs SGA (batch, DNF-filtered)
# -----------------------------------------------------------------

def plot_improvement_ratio(df: pd.DataFrame, plot_dir: str) -> None:
    """
    Shows (method_makespan - SGA_makespan) / SGA_makespan x 100%.
    Positive = slower than SGA.  Filtered to runs where BOTH methods completed.
    """
    dfc = _batch_df(df)
    if dfc.empty:
        print("  [!] Skipping improvement_ratio (no batch runs)")
        return

    sga_done = dfc[
        (dfc["config_name"] == "sga_batch") & (dfc["all_tasks_completed"] == True)
    ]
    sga_keys = set(
        zip(sga_done["tasks_per_induct"], sga_done["comm_range"], sga_done["seed"])
    )
    if not sga_keys:
        print("  [!] Skipping improvement_ratio (sga_batch never completed)")
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    methods_to_compare = [
        m for m in CANONICAL_BATCH_METHODS
        if m != "sga_batch" and m in dfc["config_name"].unique()
    ]
    linestyles = ["-", "--", "-."]
    markers = ["o", "s", "^"]
    max_cr = _get_max_connected_cr(dfc)

    for idx, cfg in enumerate(methods_to_compare):
        method_done = dfc[
            (dfc["config_name"] == cfg) & (dfc["all_tasks_completed"] == True)
        ]
        method_keys = set(
            zip(method_done["tasks_per_induct"], method_done["comm_range"], method_done["seed"])
        )
        valid_keys = sga_keys & method_keys

        comm_ranges = sorted(dfc["comm_range"].unique())
        gaps, gap_stds, valid_crs = [], [], []

        for cr in comm_ranges:
            cr_valid = {k for k in valid_keys if k[1] == cr}
            if not cr_valid:
                continue

            sga_vals = dfc[
                (dfc["config_name"] == "sga_batch")
                & (dfc["comm_range"] == cr)
                & dfc.apply(
                    lambda r: (r["tasks_per_induct"], r["comm_range"], r["seed"]) in cr_valid,
                    axis=1,
                )
            ]["effective_makespan"]

            cmp_vals = dfc[
                (dfc["config_name"] == cfg)
                & (dfc["comm_range"] == cr)
                & dfc.apply(
                    lambda r: (r["tasks_per_induct"], r["comm_range"], r["seed"]) in cr_valid,
                    axis=1,
                )
            ]["effective_makespan"]

            if len(sga_vals) > 0 and len(cmp_vals) > 0:
                sga_mean = sga_vals.mean()
                cmp_mean = cmp_vals.mean()
                gap_pct = ((cmp_mean - sga_mean) / max(sga_mean, 1)) * 100
                gap_std = (
                    np.sqrt(sga_vals.std() ** 2 + cmp_vals.std() ** 2)
                    / max(sga_mean, 1) * 100
                    if sga_mean > 0 else 0
                )
                gaps.append(gap_pct)
                gap_stds.append(gap_std)
                valid_crs.append(cr)

        if valid_crs:
            ax.errorbar(
                valid_crs, gaps, yerr=gap_stds,
                label=get_label(cfg),
                color=get_color(cfg), marker=markers[idx],
                linestyle=linestyles[idx], capsize=4, linewidth=2,
            )

    ax.axhline(y=0, color="green", linestyle="--", alpha=0.6,
               linewidth=1.5, label="SGA baseline (0%)")
    ax.axvspan(0, CONNECTIVITY_THRESHOLD - 1, alpha=0.08, color="red")
    ax.set_xlabel("Communication Range")
    ax.set_ylabel("Makespan Gap vs SGA (%)")
    ax.set_title(
        "Makespan Gap Relative to SGA Centralized Upper Bound (Batch)\n"
        "(positive = slower than SGA; only runs where BOTH methods completed)"
    )
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_xlim(left=0, right=max_cr + 5)
    _savefig(fig, plot_dir, "improvement_ratio")


# -----------------------------------------------------------------
#  Plot 4: GCBBA timing
# -----------------------------------------------------------------

def plot_gcbba_timing(df: pd.DataFrame, plot_dir: str) -> None:
    """
    Allocation computation time and rerun frequency.
    Uses steady-state data if available, falls back to batch.
    X-axis: task_arrival_rate (SS) or tasks_per_induct (batch).
    """
    df_ss = _ss_df(df)
    df_ba = _batch_df(df)

    has_ss    = not df_ss.empty
    has_batch = not df_ba.empty

    if not has_ss and not has_batch:
        print("  [!] Skipping gcbba_timing (no data)")
        return

    for label_prefix, dfc, methods, x_col, x_label in [
        ("Steady-State", df_ss, CANONICAL_SS_METHODS,    "task_arrival_rate",  "Arrival Rate (tasks/ts/station)"),
        ("Batch",        df_ba, CANONICAL_BATCH_METHODS,  "tasks_per_induct",   "Initial Tasks per Induct Station"),
    ]:
        if dfc.empty:
            continue
        methods_here = _canonical_methods_present(dfc, methods)
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        ax = axes[0]
        for cfg in methods_here:
            subset = dfc[dfc["config_name"] == cfg]
            grouped = (
                subset.groupby(x_col)["avg_gcbba_time_ms"]
                .agg(["mean", "std"])
                .reset_index()
            )
            ax.errorbar(
                grouped[x_col], grouped["mean"], yerr=grouped["std"],
                label=get_label(cfg), color=get_color(cfg), marker="s", capsize=4,
            )
        ax.set_xlabel(x_label)
        ax.set_ylabel("Avg Allocation Time per Call (ms)")
        ax.set_title(f"Allocation Computation Time ({label_prefix})")
        ax.legend()
        ax.grid(alpha=0.3)

        ax = axes[1]
        for cfg in methods_here:
            subset = dfc[dfc["config_name"] == cfg]
            grouped = (
                subset.groupby(x_col)["num_gcbba_runs"]
                .agg(["mean", "std"])
                .reset_index()
            )
            ax.errorbar(
                grouped[x_col], grouped["mean"], yerr=grouped["std"],
                label=get_label(cfg), color=get_color(cfg), marker="s", capsize=4,
            )
        ax.set_xlabel(x_label)
        ax.set_ylabel("Number of Allocation Runs")
        ax.set_title(f"Allocation Rerun Frequency ({label_prefix})")
        ax.legend()
        ax.grid(alpha=0.3)

        fig.tight_layout()
        suffix = "ss" if label_prefix == "Steady-State" else "batch"
        _savefig(fig, plot_dir, f"gcbba_timing_{suffix}")


# -----------------------------------------------------------------
#  Plot 5: Agent utilization
# -----------------------------------------------------------------

def plot_agent_utilization(df: pd.DataFrame, plot_dir: str) -> None:
    """Agent idle ratio and task balance. Handles both batch and steady-state."""
    for label_prefix, dfc, methods, x_col, x_label in [
        ("Steady-State", _ss_df(df),    CANONICAL_SS_METHODS,    "task_arrival_rate", "Arrival Rate (tasks/ts/station)"),
        ("Batch",        _batch_df(df), CANONICAL_BATCH_METHODS,  "tasks_per_induct",  "Initial Tasks per Induct Station"),
    ]:
        if dfc.empty:
            continue
        methods_here = _canonical_methods_present(dfc, methods)
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        ax = axes[0]
        for cfg in methods_here:
            subset = dfc[dfc["config_name"] == cfg]
            grouped = (
                subset.groupby(x_col)["avg_idle_ratio"]
                .agg(["mean", "std"])
                .reset_index()
            )
            ax.errorbar(
                grouped[x_col], grouped["mean"], yerr=grouped["std"],
                label=get_label(cfg), color=get_color(cfg), marker="o", capsize=4,
            )
        ax.set_xlabel(x_label)
        ax.set_ylabel("Average Idle Ratio")
        ax.set_title(f"Agent Idle Time ({label_prefix})")
        ax.legend()
        ax.grid(alpha=0.3)
        ax.set_ylim(bottom=0)

        ax = axes[1]
        for cfg in methods_here:
            subset = dfc[dfc["config_name"] == cfg]
            grouped = (
                subset.groupby(x_col)["task_balance_std"]
                .agg(["mean", "std"])
                .reset_index()
            )
            ax.errorbar(
                grouped[x_col], grouped["mean"], yerr=grouped["std"],
                label=get_label(cfg), color=get_color(cfg), marker="o", capsize=4,
            )
        ax.set_xlabel(x_label)
        ax.set_ylabel("Task Balance Std Dev")
        ax.set_title(f"Task Distribution Fairness ({label_prefix})")
        ax.legend()
        ax.grid(alpha=0.3)
        ax.set_ylim(bottom=0)

        fig.tight_layout()
        suffix = "ss" if label_prefix == "Steady-State" else "batch"
        _savefig(fig, plot_dir, f"agent_utilization_{suffix}")


# -----------------------------------------------------------------
#  Plot 6: Throughput curves (cumulative task completions over time)
# -----------------------------------------------------------------

def plot_throughput_curves(exp_dir: str, df: pd.DataFrame, plot_dir: str) -> None:
    """Cumulative task completion over time for lowest vs highest comm range.
    Supports both batch (tpi{N}) and steady-state (ar{rate}) run directory naming.
    """
    df_ss = _ss_df(df)
    df_ba = _batch_df(df)

    run_dirs = [
        d for d in os.listdir(exp_dir)
        if os.path.isdir(os.path.join(exp_dir, d))
    ]

    # --- Steady-state ---
    if not df_ss.empty:
        arr_rates = sorted(df_ss["task_arrival_rate"].unique())
        max_ar    = max(arr_rates)
        comm_ranges_of_interest = sorted(df_ss["comm_range"].unique())
        cr_low  = min(comm_ranges_of_interest)
        cr_high = max(comm_ranges_of_interest)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for ax_idx, cr in enumerate([cr_low, cr_high]):
            ax = axes[ax_idx]
            for cfg in CANONICAL_SS_METHODS:
                matching = [
                    d for d in run_dirs
                    if d.startswith(f"{cfg}_ar") and f"_cr{int(cr)}_" in d
                ]
                if not matching:
                    # Also try numeric ar string forms
                    matching = [
                        d for d in run_dirs
                        if d.startswith(cfg + "_") and f"_cr{int(cr)}_" in d
                        and "_it" not in d
                    ]
                if matching:
                    metrics_path = os.path.join(exp_dir, matching[0], "metrics.json")
                    if os.path.exists(metrics_path):
                        with open(metrics_path) as f:
                            m = json.load(f)
                        timeline = m.get("tasks_completed_over_time", [])
                        if timeline:
                            ax.plot(
                                range(len(timeline)), timeline,
                                label=get_label(cfg), color=get_color(cfg), alpha=0.8,
                            )
            ax.set_xlabel("Timestep")
            ax.set_ylabel("Tasks Completed (cumulative)")
            conn_str = "Disconnected" if cr < CONNECTIVITY_THRESHOLD else "Well Connected"
            ax.set_title(f"Throughput at cr={int(cr)} ({conn_str})\nSteady-State (ar={max_ar})")
            ax.legend()
            ax.grid(alpha=0.3)
        fig.tight_layout()
        _savefig(fig, plot_dir, "throughput_curves_ss")

    # --- Batch ---
    if not df_ba.empty:
        max_tpi = df_ba["tasks_per_induct"].max()
        comm_ranges_of_interest = sorted(df_ba["comm_range"].unique())
        cr_low  = min(comm_ranges_of_interest)
        cr_high = max(comm_ranges_of_interest)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for ax_idx, cr in enumerate([cr_low, cr_high]):
            ax = axes[ax_idx]
            for cfg in CANONICAL_BATCH_METHODS:
                matching = [
                    d for d in run_dirs
                    if d.startswith(f"{cfg}_it{max_tpi*8}_cr{int(cr)}_")
                    or d.startswith(f"{cfg}_tpi{max_tpi}_cr{int(cr)}_")
                ]
                if matching:
                    metrics_path = os.path.join(exp_dir, matching[0], "metrics.json")
                    if os.path.exists(metrics_path):
                        with open(metrics_path) as f:
                            m = json.load(f)
                        timeline = m.get("tasks_completed_over_time", [])
                        if timeline:
                            ax.plot(
                                range(len(timeline)), timeline,
                                label=get_label(cfg), color=get_color(cfg), alpha=0.8,
                            )
            ax.set_xlabel("Timestep")
            ax.set_ylabel("Tasks Completed (cumulative)")
            conn_str = "Disconnected" if cr < CONNECTIVITY_THRESHOLD else "Well Connected"
            ax.set_title(f"Throughput at cr={int(cr)} ({conn_str})\nBatch ({max_tpi} tasks/induct)")
            ax.legend()
            ax.grid(alpha=0.3)
        fig.tight_layout()
        _savefig(fig, plot_dir, "throughput_curves_batch")


# -----------------------------------------------------------------
#  Plot 7: Collision and deadlock summary
# -----------------------------------------------------------------

def plot_collision_deadlock(df: pd.DataFrame, plot_dir: str) -> None:
    """Works on all canonical methods (both batch and SS), grouped by comm_range."""
    all_canonical = CANONICAL_SS_METHODS + CANONICAL_BATCH_METHODS
    dfc = df[df["config_name"].isin(all_canonical)].copy()
    if dfc.empty:
        print("  [!] Skipping collision_deadlock (no runs)")
        return

    methods = _canonical_methods_present(dfc, all_canonical)
    comm_ranges = sorted(dfc["comm_range"].unique())
    n_cr = len(comm_ranges)
    x = np.arange(n_cr)
    bar_width = 0.15

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    for i, cfg in enumerate(methods):
        subset = dfc[dfc["config_name"] == cfg]
        means, stds = [], []
        for cr in comm_ranges:
            cr_data = subset[subset["comm_range"] == cr]["num_vertex_collisions"]
            means.append(cr_data.mean() if len(cr_data) > 0 else 0)
            stds.append(cr_data.std() if len(cr_data) > 1 else 0)
        offset = (i - (len(methods) - 1) / 2) * bar_width
        ax.bar(
            x + offset, means, bar_width, yerr=stds,
            label=get_label(cfg), color=get_color(cfg),
            capsize=3, alpha=0.8, edgecolor="black", linewidth=0.5,
        )
    ax.set_xlabel("Communication Range")
    ax.set_ylabel("Vertex Collisions (avg per run)")
    ax.set_title("Post-hoc Collision Detection")
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(cr)) for cr in comm_ranges])
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(bottom=0)

    ax = axes[1]
    for i, cfg in enumerate(methods):
        subset = dfc[dfc["config_name"] == cfg]
        means, stds = [], []
        for cr in comm_ranges:
            cr_data = subset[subset["comm_range"] == cr]["num_deadlocks"]
            means.append(cr_data.mean() if len(cr_data) > 0 else 0)
            stds.append(cr_data.std() if len(cr_data) > 1 else 0)
        offset = (i - (len(methods) - 1) / 2) * bar_width
        err_low = np.minimum(stds, means)
        ax.bar(
            x + offset, means, bar_width,
            yerr=[err_low, stds],
            label=get_label(cfg), color=get_color(cfg),
            capsize=3, alpha=0.8, edgecolor="black", linewidth=0.5,
        )
    ax.set_xlabel("Communication Range")
    ax.set_ylabel("Distinct Stuck Events (avg per run)")
    ax.set_title("Agent Stuck Events by Communication Range\n"
                 "(one event = agent transitions into stuck state)")
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(cr)) for cr in comm_ranges])
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    _savefig(fig, plot_dir, "collision_deadlock")


# -----------------------------------------------------------------
#  Plot 8: Completion rate by communication range (batch)
# -----------------------------------------------------------------

def plot_completion_rate(df: pd.DataFrame, plot_dir: str) -> None:
    """Task completion rate for batch runs (where all_tasks_completed is meaningful)."""
    dfc = _batch_df(df)
    if dfc.empty:
        print("  [!] Skipping completion_rate (no batch runs)")
        return

    methods = _canonical_methods_present(dfc, CANONICAL_BATCH_METHODS)
    comm_ranges = sorted(dfc["comm_range"].unique())
    n_cr = len(comm_ranges)
    x = np.arange(n_cr)
    bar_width = 0.2

    fig, ax = plt.subplots(figsize=(11, 5))

    for i, cfg in enumerate(methods):
        rates = []
        for cr in comm_ranges:
            subset = dfc[
                (dfc["config_name"] == cfg) & (dfc["comm_range"] == cr)
            ]
            n_total = len(subset)
            n_done = (subset["all_tasks_completed"] == True).sum()
            rates.append(n_done / n_total * 100 if n_total > 0 else 0)

        offset = (i - (len(methods) - 1) / 2) * bar_width
        ax.bar(
            x + offset, rates, bar_width,
            label=get_label(cfg), color=get_color(cfg),
            alpha=0.85, edgecolor="black", linewidth=0.5,
        )
        for j, rate in enumerate(rates):
            if rate > 0:
                ax.text(x[j] + offset, rate + 1.5, f"{rate:.0f}%",
                        ha="center", va="bottom", fontsize=8, fontweight="bold")
            else:
                ax.text(x[j] + offset, 2, "0%",
                        ha="center", va="bottom", fontsize=8,
                        fontweight="bold", color="gray")

    ax.axvspan(-0.5, n_cr - 3.5 - 0.01, alpha=0.06, color="red",
               label="_nolegend_")
    for j, cr in enumerate(comm_ranges):
        if cr in CR_ANNOTATIONS:
            label = CR_ANNOTATIONS[cr].replace("\n", " ")
            ax.text(j, -8, label, ha="center", va="top",
                    fontsize=8, color="gray", style="italic")

    ax.set_xlabel("Communication Range")
    ax.set_ylabel("Task Completion Rate (%)")
    ax.set_title(
        "Task Completion Rate by Communication Range (Batch Mode)\n"
        "(All task loads, across all seeds)"
    )
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(cr)) for cr in comm_ranges])
    ax.legend(loc="center right")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 115)
    fig.tight_layout()
    _savefig(fig, plot_dir, "completion_rate")


# -----------------------------------------------------------------
#  Plot 9: Completion rate broken down by task load (batch)
# -----------------------------------------------------------------

def plot_completion_rate_by_tpi(df: pd.DataFrame, plot_dir: str) -> None:
    dfc = _batch_df(df)
    if dfc.empty:
        print("  [!] Skipping completion_rate_by_tpi (no batch runs)")
        return

    task_counts = sorted(dfc["tasks_per_induct"].unique())
    n_plots = len(task_counts)
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 5), sharey=True)
    if n_plots == 1:
        axes = [axes]

    comm_ranges = sorted(dfc["comm_range"].unique())
    n_cr = len(comm_ranges)
    x = np.arange(n_cr)
    bar_width = 0.2

    for ax_idx, tpi in enumerate(task_counts):
        ax = axes[ax_idx]
        df_sub = dfc[dfc["tasks_per_induct"] == tpi]
        methods = _canonical_methods_present(df_sub, CANONICAL_BATCH_METHODS)

        for i, cfg in enumerate(methods):
            rates = []
            for cr in comm_ranges:
                subset = df_sub[
                    (df_sub["config_name"] == cfg) & (df_sub["comm_range"] == cr)
                ]
                n_total = len(subset)
                n_done = (subset["all_tasks_completed"] == True).sum()
                rates.append(n_done / n_total * 100 if n_total > 0 else 0)

            offset = (i - (len(methods) - 1) / 2) * bar_width
            ax.bar(
                x + offset, rates, bar_width,
                label=get_label(cfg), color=get_color(cfg),
                alpha=0.85, edgecolor="black", linewidth=0.5,
            )
            for j, rate in enumerate(rates):
                ax.text(
                    x[j] + offset, rate + 1.5, f"{rate:.0f}%",
                    ha="center", va="bottom", fontsize=7, fontweight="bold",
                    color="gray" if rate == 0 else "black",
                )

        ax.set_xlabel("Communication Range")
        if ax_idx == 0:
            ax.set_ylabel("Completion Rate (%)")
        ax.set_title(f"{tpi} tasks/induct ({tpi*8} total)")
        ax.set_xticks(x)
        ax.set_xticklabels([str(int(cr)) for cr in comm_ranges])
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylim(0, 120)
        ax.axvspan(-0.5, n_cr - 3.5 - 0.01, alpha=0.06, color="red",
                   label="_nolegend_")

    fig.suptitle(
        "Task Completion Rate by Communication Range and Task Load (Batch Mode)",
        fontsize=15, y=1.02,
    )
    fig.tight_layout()
    _savefig(fig, plot_dir, "completion_rate_by_tpi")


# -----------------------------------------------------------------
#  Plot 10: Graph connectivity vs performance (scatter)
# -----------------------------------------------------------------

def plot_graph_connectivity(df: pd.DataFrame, plot_dir: str) -> None:
    if "initial_num_components" not in df.columns:
        print("  [!] Skipping graph_connectivity (no initial_num_components column)")
        return

    # Try SS first, fall back to batch
    df_ss = _ss_df(df)
    df_ba = _batch_df(df)

    for label_prefix, dfc, methods, y_col, y_label in [
        ("Steady-State", df_ss, CANONICAL_SS_METHODS,    "effective_throughput", "Throughput (tasks/ts)"),
        ("Batch",        df_ba, CANONICAL_BATCH_METHODS,  "effective_makespan",   "Makespan (timesteps)"),
    ]:
        if dfc.empty:
            continue
        if dfc["initial_num_components"].isna().all():
            continue

        x_col = "task_arrival_rate" if label_prefix == "Steady-State" else "tasks_per_induct"
        max_val = dfc[x_col].max()
        df_sub = dfc[dfc[x_col] == max_val]
        methods_here = _canonical_methods_present(df_sub, methods)

        fig, ax = plt.subplots(figsize=(8, 5))
        for cfg in methods_here:
            subset = df_sub[df_sub["config_name"] == cfg]
            ax.scatter(
                subset["initial_num_components"], subset[y_col],
                label=get_label(cfg), color=get_color(cfg),
                alpha=0.6, s=60, edgecolors="black", linewidth=0.5,
            )

        ax.set_xlabel("Initial Communication Graph Components")
        ax.set_ylabel(y_label)
        x_desc = f"ar={max_val}" if label_prefix == "Steady-State" else f"{max_val} tasks/induct"
        ax.set_title(f"Graph Connectivity vs Performance ({label_prefix}, {x_desc})")
        ax.legend()
        ax.grid(alpha=0.3)
        ax.invert_xaxis()
        suffix = "ss" if label_prefix == "Steady-State" else "batch"
        _savefig(fig, plot_dir, f"graph_connectivity_{suffix}")


# -----------------------------------------------------------------
#  Plot 11: Sub-optimality ratio
# -----------------------------------------------------------------

def plot_optimality_ratio(df: pd.DataFrame, plot_dir: str) -> None:
    """
    Batch:         method_makespan / sga_makespan vs comm_range.
    Steady-State:  method_throughput / sga_throughput vs comm_range.
    """
    # -- Batch --
    dfc_b = _batch_df(df)
    if not dfc_b.empty and "optimality_ratio" in dfc_b.columns:
        valid_b = dfc_b[dfc_b["optimality_ratio_valid"] == True].copy()
        plot_methods_b = [m for m in ["static_batch", "dynamic_batch", "cbba_batch"]
                          if m in valid_b["config_name"].unique()]

        if not valid_b.empty and plot_methods_b:
            task_counts = sorted(valid_b["tasks_per_induct"].unique())
            n_plots = len(task_counts)
            fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 5), sharey=True)
            if n_plots == 1:
                axes = [axes]

            for ax_idx, tpi in enumerate(task_counts):
                ax = axes[ax_idx]
                df_tpi = valid_b[valid_b["tasks_per_induct"] == tpi]
                for cfg in plot_methods_b:
                    subset = df_tpi[df_tpi["config_name"] == cfg]
                    if subset.empty:
                        continue
                    grouped = (
                        subset.groupby("comm_range")["optimality_ratio"]
                        .agg(["mean", "std"])
                        .reset_index()
                    )
                    ax.errorbar(
                        grouped["comm_range"], grouped["mean"], yerr=grouped["std"],
                        label=get_label(cfg), color=get_color(cfg),
                        marker="o", capsize=4, linewidth=2,
                    )
                ax.axhline(y=1.0, color="black", linestyle="--", linewidth=1.5,
                           label="SGA reference (1.0)")
                ax.axhline(y=2.0, color="red", linestyle=":", linewidth=1.5,
                           label="GCBBA bound (2.0)", alpha=0.7)
                ax.set_xlabel("Communication Range")
                if ax_idx == 0:
                    ax.set_ylabel("Makespan / SGA Makespan")
                ax.set_title(f"{tpi} tasks/induct")
                ax.legend(fontsize=9)
                ax.grid(alpha=0.3)
                ax.set_ylim(bottom=0)
                xlim = ax.get_xlim()
                ax.axvspan(xlim[0], CONNECTIVITY_THRESHOLD - 0.5,
                           alpha=0.07, color="red", label="_nolegend_")

            fig.suptitle(
                "Sub-Optimality Ratio vs Communication Range (Batch)\n"
                "(1.0 = matches SGA; <1.0 = better than SGA)",
                fontsize=13, y=1.03,
            )
            fig.tight_layout()
            _savefig(fig, plot_dir, "optimality_ratio_batch")

    # -- Steady-state (clean rates only: uniform CR coverage across all methods) --
    dfc_s = _ss_clean_df(df)
    if not dfc_s.empty and "throughput_ratio" in dfc_s.columns:
        valid_s = dfc_s[dfc_s["throughput_ratio_valid"] == True].copy()
        plot_methods_s = [m for m in ["static", "dynamic", "cbba"]
                          if m in valid_s["config_name"].unique()]

        if not valid_s.empty and plot_methods_s:
            arr_rates = sorted(valid_s["task_arrival_rate"].unique())
            n_plots = len(arr_rates)
            fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 5), sharey=True)
            if n_plots == 1:
                axes = [axes]

            for ax_idx, ar in enumerate(arr_rates):
                ax = axes[ax_idx]
                df_ar = valid_s[valid_s["task_arrival_rate"] == ar]
                for cfg in plot_methods_s:
                    subset = df_ar[df_ar["config_name"] == cfg]
                    if subset.empty:
                        continue
                    grouped = (
                        subset.groupby("comm_range")["throughput_ratio"]
                        .agg(["mean", "std"])
                        .reset_index()
                    )
                    ax.errorbar(
                        grouped["comm_range"], grouped["mean"], yerr=grouped["std"],
                        label=get_label(cfg), color=get_color(cfg),
                        marker="o", capsize=4, linewidth=2,
                    )
                ax.axhline(y=1.0, color="black", linestyle="--", linewidth=1.5,
                           label="SGA reference (1.0)")
                ax.set_xlabel("Communication Range")
                if ax_idx == 0:
                    ax.set_ylabel("Throughput / SGA Throughput")
                ax.set_title(f"ar={ar}")
                ax.legend(fontsize=9)
                ax.grid(alpha=0.3)
                ax.set_ylim(bottom=0)
                xlim = ax.get_xlim()
                ax.axvspan(xlim[0], CONNECTIVITY_THRESHOLD - 0.5,
                           alpha=0.07, color="red", label="_nolegend_")

            fig.suptitle(
                "Throughput Ratio vs Communication Range (Steady-State)\n"
                "(1.0 = matches SGA; >1.0 = better than SGA)",
                fontsize=13, y=1.03,
            )
            fig.tight_layout()
            _savefig(fig, plot_dir, "optimality_ratio_ss")

    if dfc_b.empty and dfc_s.empty:
        print("  [!] Skipping optimality_ratio (no data)")


# -----------------------------------------------------------------
#  Plot 12: Rerun interval sensitivity
# -----------------------------------------------------------------

def plot_rerun_interval_sweep(df: pd.DataFrame, plot_dir: str) -> None:
    """Dynamic GCBBA rerun-interval sensitivity analysis (batch or steady-state)."""
    for label_prefix, dfc, x_col, x_label, y_col, y_label in [
        ("Steady-State", _ss_df(df),    "task_arrival_rate", "Arrival Rate",        "effective_throughput", "Throughput (tasks/ts)"),
        ("Batch",        _batch_df(df), "tasks_per_induct",  "Tasks per Induct",    "effective_makespan",   "Makespan (timesteps)"),
    ]:
        dynamic_df = dfc[
            (dfc["config_name"] == ("dynamic" if label_prefix == "Steady-State" else "dynamic_batch"))
            | (dfc["config_name"].str.startswith("dynamic_ri"))
        ].copy()

        if dynamic_df.empty:
            continue

        def get_ri(config_name):
            if config_name in ("dynamic", "dynamic_batch"):
                return 50
            try:
                return int(config_name.replace("dynamic_ri", ""))
            except ValueError:
                return -1

        dynamic_df["ri_value"] = dynamic_df["config_name"].apply(get_ri)
        ri_values = sorted(dynamic_df["ri_value"].unique())

        cmap = plt.cm.get_cmap("RdYlBu_r", len(ri_values))
        ri_colors = {ri: cmap(i) for i, ri in enumerate(ri_values)}

        x_vals = sorted(dynamic_df[x_col].unique())
        comm_ranges = sorted(dynamic_df["comm_range"].unique())
        n_x = len(x_vals)

        fig, axes = plt.subplots(2, n_x, figsize=(5 * n_x, 9))
        if n_x == 1:
            axes = axes.reshape(2, 1)

        for col, xv in enumerate(x_vals):
            df_x = dynamic_df[dynamic_df[x_col] == xv]

            ax = axes[0, col]
            for ri in ri_values:
                ri_df = df_x[df_x["ri_value"] == ri]
                if ri_df.empty:
                    continue
                grouped = (
                    ri_df.groupby("comm_range")[y_col]
                    .agg(["mean", "std"])
                    .reset_index()
                )
                label = f"ri={ri}" + (" (canonical)" if ri == 50 else "")
                lw = 2.5 if ri == 50 else 1.5
                ax.errorbar(
                    grouped["comm_range"], grouped["mean"], yerr=grouped["std"],
                    label=label, color=ri_colors[ri],
                    marker="o", capsize=3, linewidth=lw,
                    linestyle="-" if ri == 50 else "--",
                )
            ax.set_xlabel("Communication Range")
            ax.set_ylabel(y_label)
            ax.set_title(f"{y_label.split('(')[0].strip()} -- {x_label}={xv}")
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)
            ax.set_ylim(bottom=0)
            ax.axvspan(0, CONNECTIVITY_THRESHOLD - 1, alpha=0.08, color="red",
                       label="_nolegend_")

            ax = axes[1, col]
            for ri in ri_values:
                ri_df = df_x[df_x["ri_value"] == ri]
                if ri_df.empty:
                    continue
                cr_rates = []
                for cr in comm_ranges:
                    cr_df = ri_df[ri_df["comm_range"] == cr]
                    n_total = len(cr_df)
                    n_done = (cr_df["all_tasks_completed"] == True).sum() if "all_tasks_completed" in cr_df.columns else 0
                    cr_rates.append(n_done / n_total * 100 if n_total > 0 else 0)
                label = f"ri={ri}" + (" (canonical)" if ri == 50 else "")
                lw = 2.5 if ri == 50 else 1.5
                ax.plot(
                    comm_ranges, cr_rates,
                    label=label, color=ri_colors[ri],
                    marker="s", linewidth=lw,
                    linestyle="-" if ri == 50 else "--",
                )
            ax.set_xlabel("Communication Range")
            ax.set_ylabel("Completion Rate (%)")
            ax.set_title(f"Completion Rate -- {x_col}={xv}")
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)
            ax.set_ylim(0, 110)
            ax.axvspan(0, CONNECTIVITY_THRESHOLD - 1, alpha=0.08, color="red",
                       label="_nolegend_")

        fig.suptitle(
            f"Rerun Interval Sensitivity Analysis -- Dynamic GCBBA ({label_prefix})\n"
            "(ri=50 is canonical; other values show sensitivity to this design choice)",
            fontsize=13, y=1.02,
        )
        fig.tight_layout()
        suffix = "ss" if label_prefix == "Steady-State" else "batch"
        _savefig(fig, plot_dir, f"rerun_interval_sweep_{suffix}")


# -----------------------------------------------------------------
#  Plot 13: GCBBA trigger reason breakdown
# -----------------------------------------------------------------

def plot_trigger_breakdown(df: pd.DataFrame, plot_dir: str) -> None:
    """Fraction of GCBBA reruns from batch completion vs interval timer."""
    required_cols = {"num_gcbba_runs_batch_triggered", "num_gcbba_runs_interval_triggered"}
    if not required_cols.issubset(df.columns):
        print("  [!] Skipping trigger_breakdown (columns missing from summary.csv)")
        return

    for label_prefix, dfc, canon_name, x_col, x_label in [
        ("Steady-State", _ss_df(df),    "dynamic",       "task_arrival_rate", "Arrival Rate (tasks/ts/station)"),
        ("Batch",        _batch_df(df), "dynamic_batch",  "tasks_per_induct",  "Initial Tasks per Induct Station"),
    ]:
        sub = dfc[dfc["config_name"] == canon_name].copy()
        if sub.empty:
            continue

        x_vals = sorted(sub[x_col].unique())
        comm_ranges = sorted(sub["comm_range"].unique())
        n_cr = len(comm_ranges)
        x = np.arange(n_cr)

        fig, axes = plt.subplots(1, len(x_vals), figsize=(5 * len(x_vals), 5),
                                  sharey=False)
        if len(x_vals) == 1:
            axes = [axes]

        for ax_idx, xv in enumerate(x_vals):
            ax = axes[ax_idx]
            df_x = sub[sub[x_col] == xv]

            batch_means, interval_means = [], []
            for cr in comm_ranges:
                cr_df = df_x[df_x["comm_range"] == cr]
                batch_means.append(cr_df["num_gcbba_runs_batch_triggered"].mean()
                                   if len(cr_df) > 0 else 0)
                interval_means.append(cr_df["num_gcbba_runs_interval_triggered"].mean()
                                      if len(cr_df) > 0 else 0)

            width = 0.35
            ax.bar(x - width / 2, batch_means, width,
                   label="Batch (task completion)", color="#2ecc71", alpha=0.85,
                   edgecolor="black", linewidth=0.5)
            ax.bar(x + width / 2, interval_means, width,
                   label="Interval timer", color="#95a5a6", alpha=0.85,
                   edgecolor="black", linewidth=0.5)

            ax.set_xlabel("Communication Range")
            ax.set_ylabel("Avg GCBBA Reruns per Run")
            ax.set_title(f"Trigger Reasons -- {x_label}={xv}")
            ax.set_xticks(x)
            ax.set_xticklabels([str(int(cr)) for cr in comm_ranges])
            ax.legend(fontsize=9)
            ax.grid(axis="y", alpha=0.3)
            ax.set_ylim(bottom=0)
            ax.axvspan(-0.5, n_cr - 3.5 - 0.01, alpha=0.06, color="red",
                       label="_nolegend_")

        fig.suptitle(
            f"GCBBA Rerun Trigger Breakdown -- {label_prefix}\n"
            "(Batch = enough tasks completed; Interval = timer elapsed)",
            fontsize=13, y=1.02,
        )
        fig.tight_layout()
        suffix = "ss" if label_prefix == "Steady-State" else "batch"
        _savefig(fig, plot_dir, f"trigger_breakdown_{suffix}")


# -----------------------------------------------------------------
#  Plot 15: Charging overhead
# -----------------------------------------------------------------

def plot_charging_overhead(df: pd.DataFrame, plot_dir: str) -> None:
    """Charging time fraction and number of charging events."""
    required = {"charging_time_fraction", "num_charging_events"}
    if not required.issubset(df.columns):
        print("  [!] Skipping charging_overhead (energy columns missing)")
        return

    for label_prefix, dfc, methods, x_col, x_label in [
        ("Steady-State", _ss_df(df),    CANONICAL_SS_METHODS,    "task_arrival_rate", "Arrival Rate (tasks/ts/station)"),
        ("Batch",        _batch_df(df), CANONICAL_BATCH_METHODS,  "tasks_per_induct",  "Initial Tasks per Induct Station"),
    ]:
        if dfc.empty:
            continue
        methods_here = _canonical_methods_present(dfc, methods)
        x_vals = sorted(dfc[x_col].unique())
        x = np.arange(len(x_vals))
        width = 0.2

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        ax = axes[0]
        for i, cfg in enumerate(methods_here):
            means, stds = [], []
            for xv in x_vals:
                vals = dfc[
                    (dfc["config_name"] == cfg) & (dfc[x_col] == xv)
                ]["charging_time_fraction"] * 100
                means.append(vals.mean() if len(vals) > 0 else 0)
                stds.append(vals.std() if len(vals) > 1 else 0)
            offset = (i - (len(methods_here) - 1) / 2) * width
            ax.bar(x + offset, means, width,
                   yerr=stds, label=get_label(cfg), color=get_color(cfg),
                   capsize=4, alpha=0.85, edgecolor="black", linewidth=0.5)
            for j, (m, s) in enumerate(zip(means, stds)):
                if m > 0.5:
                    ax.text(x[j] + offset, m + s + 0.3, f"{m:.1f}%",
                            ha="center", va="bottom", fontsize=8)
        ax.set_xlabel(x_label)
        ax.set_ylabel("Charging + Navigation Time (%)")
        ax.set_title(f"Energy Management Overhead ({label_prefix})")
        ax.set_xticks(x)
        ax.set_xticklabels([str(xv) for xv in x_vals])
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylim(bottom=0)

        ax = axes[1]
        for i, cfg in enumerate(methods_here):
            means, stds = [], []
            for xv in x_vals:
                vals = dfc[
                    (dfc["config_name"] == cfg) & (dfc[x_col] == xv)
                ]["num_charging_events"]
                means.append(vals.mean() if len(vals) > 0 else 0)
                stds.append(vals.std() if len(vals) > 1 else 0)
            offset = (i - (len(methods_here) - 1) / 2) * width
            ax.bar(x + offset, means, width,
                   yerr=stds, label=get_label(cfg), color=get_color(cfg),
                   capsize=4, alpha=0.85, edgecolor="black", linewidth=0.5)
            for j, (m, s) in enumerate(zip(means, stds)):
                if m > 0.1:
                    ax.text(x[j] + offset, m + s + 0.1, f"{m:.1f}",
                            ha="center", va="bottom", fontsize=8)
        ax.set_xlabel(x_label)
        ax.set_ylabel("Charging Events (avg per run)")
        ax.set_title(f"Charging Frequency ({label_prefix})")
        ax.set_xticks(x)
        ax.set_xticklabels([str(xv) for xv in x_vals])
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylim(bottom=0)

        fig.suptitle(
            f"Energy Management: Charging Overhead ({label_prefix})",
            fontsize=14, y=1.02,
        )
        fig.tight_layout()
        suffix = "ss" if label_prefix == "Steady-State" else "batch"
        _savefig(fig, plot_dir, f"charging_overhead_{suffix}")


# -----------------------------------------------------------------
#  Plot 16: Energy-performance scatter
# -----------------------------------------------------------------

def plot_energy_performance(df: pd.DataFrame, plot_dir: str) -> None:
    """charging_time_fraction vs effective metric (makespan for batch, throughput for SS)."""
    if "charging_time_fraction" not in df.columns:
        print("  [!] Skipping energy_performance (energy columns missing)")
        return

    for label_prefix, dfc, methods, y_col, y_label, grp_col, grp_label in [
        ("Steady-State", _ss_df(df),    CANONICAL_SS_METHODS,    "effective_throughput", "Throughput (tasks/ts)",  "task_arrival_rate", "ar"),
        ("Batch",        _batch_df(df), CANONICAL_BATCH_METHODS,  "effective_makespan",   "Makespan (timesteps)",   "tasks_per_induct",  "tpi"),
    ]:
        if dfc.empty:
            continue
        dfc_done = dfc[dfc["all_tasks_completed"] == True].copy() if "all_tasks_completed" in dfc.columns else dfc.copy()
        if dfc_done.empty:
            continue

        grp_vals = sorted(dfc_done[grp_col].unique())
        markers = ["o", "s", "^", "D"]
        grp_marker = {gv: markers[i % len(markers)] for i, gv in enumerate(grp_vals)}
        methods_here = _canonical_methods_present(dfc_done, methods)

        fig, ax = plt.subplots(figsize=(9, 6))

        for cfg in methods_here:
            subset = dfc_done[dfc_done["config_name"] == cfg]
            for gv in grp_vals:
                pts = subset[subset[grp_col] == gv]
                if pts.empty:
                    continue
                ax.scatter(
                    pts["charging_time_fraction"] * 100,
                    pts[y_col],
                    color=get_color(cfg),
                    marker=grp_marker[gv],
                    alpha=0.65, s=55,
                    edgecolors="black", linewidth=0.4,
                )

        method_handles = [
            plt.Line2D([0], [0], marker="o", color="w",
                       markerfacecolor=get_color(cfg), markersize=10,
                       label=get_label(cfg), markeredgecolor="black", markeredgewidth=0.5)
            for cfg in methods_here
        ]
        grp_handles = [
            plt.Line2D([0], [0], marker=grp_marker[gv], color="w",
                       markerfacecolor="gray", markersize=9,
                       label=f"{grp_label}={gv}",
                       markeredgecolor="black", markeredgewidth=0.5)
            for gv in grp_vals
        ]
        first_legend = ax.legend(handles=method_handles, title="Method",
                                 loc="upper left", fontsize=9)
        ax.add_artist(first_legend)
        ax.legend(handles=grp_handles, title="Load", loc="upper right", fontsize=9)

        ax.set_xlabel("Charging Overhead (% of agent-timesteps)")
        ax.set_ylabel(y_label)
        ax.set_title(
            f"Energy Overhead vs {y_label.split('(')[0].strip()} ({label_prefix})\n"
            "(lower-left corner = ideal)"
        )
        ax.grid(alpha=0.3)
        suffix = "ss" if label_prefix == "Steady-State" else "batch"
        _savefig(fig, plot_dir, f"energy_performance_{suffix}")


# -----------------------------------------------------------------
#  Plot 14: LaTeX results table (batch)
# -----------------------------------------------------------------

def generate_latex_table(df: pd.DataFrame, plot_dir: str) -> None:
    """LaTeX table for batch runs (makespan-centric)."""
    dfc = _batch_df(df)
    if dfc.empty:
        print("  [!] Skipping results_table (no batch runs)")
        return

    has_energy = "charging_time_fraction" in dfc.columns

    agg_dict = dict(
        effective_makespan_mean=("effective_makespan", "mean"),
        effective_makespan_std=("effective_makespan", "std"),
        num_gcbba_runs_mean=("num_gcbba_runs", "mean"),
        avg_gcbba_time_ms_mean=("avg_gcbba_time_ms", "mean"),
        avg_idle_ratio_mean=("avg_idle_ratio", "mean"),
        task_balance_std_mean=("task_balance_std", "mean"),
        num_vertex_collisions_sum=("num_vertex_collisions", "sum"),
        num_deadlocks_sum=("num_deadlocks", "sum"),
        num_tasks_completed_mean=("num_tasks_completed", "mean"),
        all_tasks_completed_mean=("all_tasks_completed", "mean"),
    )
    if has_energy:
        agg_dict["charging_time_fraction_mean"] = ("charging_time_fraction", "mean")
        agg_dict["num_charging_events_mean"]    = ("num_charging_events",    "mean")

    agg = (
        dfc.groupby(["config_name", "tasks_per_induct", "comm_range"])
        .agg(**agg_dict)
        .reset_index()
    )

    energy_cols = r" & Chrg\% & ChrgEvt" if has_energy else ""
    col_spec    = r"llrrrrrrrrrr" + ("rr" if has_energy else "")

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Experimental Results (Batch Mode): GCBBA (Static \& Dynamic), CBBA, SGA}",
        r"\label{tab:results_batch}",
        r"\small",
        r"\begin{tabular}{" + col_spec + r"}",
        r"\toprule",
        r"Config & TPI & CR & Done\% & Makespan & $\sigma$ & "
        r"Alloc & Avg ms & Idle & Balance & Col & Stuck"
        + energy_cols + r" \\",
        r"\midrule",
    ]

    for _, row in agg.iterrows():
        ms_mean   = row.get("effective_makespan_mean", 0)
        ms_std    = row.get("effective_makespan_std", 0)
        comp_rate = row.get("all_tasks_completed_mean", 0) * 100
        collisions = row.get("num_vertex_collisions_sum", 0)
        stuck      = row.get("num_deadlocks_sum", 0)
        cfg_short  = row["config_name"].replace("_batch", "")
        base = (
            f"  {cfg_short} & {row['tasks_per_induct']:.0f} & "
            f"{row['comm_range']:.0f} & "
            f"{comp_rate:.0f} & "
            f"{ms_mean:.0f} & {ms_std:.0f} & "
            f"{row['num_gcbba_runs_mean']:.0f} & "
            f"{row['avg_gcbba_time_ms_mean']:.0f} & "
            f"{row['avg_idle_ratio_mean']:.3f} & "
            f"{row['task_balance_std_mean']:.2f} & "
            f"{collisions:.0f} & "
            f"{stuck:.0f}"
        )
        if has_energy:
            chrg_pct = row.get("charging_time_fraction_mean", 0) * 100
            chrg_evt = row.get("num_charging_events_mean", 0)
            base += f" & {chrg_pct:.1f} & {chrg_evt:.1f}"
        lines.append(base + r" \\")

    energy_footnote = (
        r" Chrg\% = mean fraction of agent-timesteps spent navigating to or charging; "
        r"ChrgEvt = mean number of charging trips per run;"
        if has_energy else ""
    )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\vspace{4pt}",
        r"\parbox{\textwidth}{\footnotesize "
        r"TPI = tasks per induct station; "
        r"CR = communication range; "
        r"Done\% = task completion rate; "
        r"Makespan = mean timesteps to complete all tasks; "
        r"Alloc = allocation invocations; "
        r"Avg ms = mean allocation time per call; "
        r"Idle = mean agent idle ratio; "
        r"Balance = std dev tasks per agent; "
        r"Col = post-hoc vertex collisions (sum); "
        r"Stuck = distinct stuck-state entry events (sum)."
        + energy_footnote + r"}",
        r"\end{table}",
    ]

    table_path = os.path.join(plot_dir, "results_table_batch.tex")
    with open(table_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("  [OK] results_table_batch.tex")


# -----------------------------------------------------------------
#  Plot 17: Throughput vs arrival rate (steady-state primary metric)
# -----------------------------------------------------------------

def plot_throughput_vs_arrival_rate(df: pd.DataFrame, plot_dir: str) -> None:
    """
    Primary steady-state metric plot.
    Y = throughput (tasks/timestep), X = arrival rate, lines = methods.
    One subplot per comm_range group (fully connected vs disconnected).
    Uses only SS_PRIMARY_ARRIVAL_RATES (ar<=0.1) for uniform method coverage.
    """
    dfc = _ss_clean_df(df)
    if dfc.empty or "throughput" not in dfc.columns:
        print("  [!] Skipping throughput_vs_arrival_rate (no steady-state data)")
        return

    comm_ranges = sorted(dfc["comm_range"].unique())
    # Show a few representative comm ranges rather than all
    cr_low  = min(comm_ranges)
    cr_high = max(comm_ranges)
    cr_mid  = comm_ranges[len(comm_ranges) // 2]
    show_crs = sorted(set([cr_low, cr_mid, cr_high]))
    n_plots = len(show_crs)

    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 5), sharey=False)
    if n_plots == 1:
        axes = [axes]

    methods = _canonical_methods_present(dfc, CANONICAL_SS_METHODS)

    for ax_idx, cr in enumerate(show_crs):
        ax = axes[ax_idx]
        df_cr = dfc[dfc["comm_range"] == cr]

        for cfg in methods:
            subset = df_cr[df_cr["config_name"] == cfg]
            grouped = (
                subset.groupby("task_arrival_rate")["throughput"]
                .agg(["mean", "std"])
                .reset_index()
            )
            ax.errorbar(
                grouped["task_arrival_rate"], grouped["mean"], yerr=grouped["std"],
                label=get_label(cfg), color=get_color(cfg),
                marker="o", capsize=4, linewidth=2,
            )

        # Diagonal reference line: perfect throughput = n_stations * arrival_rate
        arr_rates = sorted(df_cr["task_arrival_rate"].unique())
        n_stations = 8  # standard warehouse config
        perfect_tp = [ar * n_stations for ar in arr_rates]
        ax.plot(arr_rates, perfect_tp, "k--", linewidth=1.2, alpha=0.5,
                label=f"Max possible ({n_stations} stations)")

        conn_str = "Disconnected" if cr < CONNECTIVITY_THRESHOLD else "Well Connected"
        ax.set_xlabel("Arrival Rate (tasks/ts/station)")
        ax.set_ylabel("Throughput (tasks/ts)")
        ax.set_title(f"cr={int(cr)} ({conn_str})")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        ax.set_ylim(bottom=0)

    fig.suptitle(
        "Steady-State Throughput vs Arrival Rate\n"
        "(dashed line = theoretical maximum if no bottlenecks)",
        fontsize=13, y=1.03,
    )
    fig.tight_layout()
    _savefig(fig, plot_dir, "throughput_vs_arrival_rate")


# -----------------------------------------------------------------
#  Plot 18: Task wait time vs arrival rate
# -----------------------------------------------------------------

def plot_wait_time(df: pd.DataFrame, plot_dir: str) -> None:
    """
    Avg and max task wait time (injection -> execution start) vs arrival rate.
    High wait time at high AR = queue backpressure near saturation.
    Uses only SS_PRIMARY_ARRIVAL_RATES for uniform method coverage.
    """
    dfc = _ss_clean_df(df)
    required = {"avg_task_wait_time", "max_task_wait_time"}
    if dfc.empty or not required.issubset(dfc.columns):
        print("  [!] Skipping wait_time (no steady-state data or columns missing)")
        return

    methods = _canonical_methods_present(dfc, CANONICAL_SS_METHODS)
    max_cr = _get_max_connected_cr(dfc)
    df_fc  = dfc[dfc["comm_range"] == max_cr]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for col_idx, (metric, ylabel, title_part) in enumerate([
        ("avg_task_wait_time", "Avg Wait Time (timesteps)", "Mean"),
        ("max_task_wait_time", "Max Wait Time (timesteps)", "Maximum"),
    ]):
        ax = axes[col_idx]
        for cfg in methods:
            subset = df_fc[df_fc["config_name"] == cfg]
            grouped = (
                subset.groupby("task_arrival_rate")[metric]
                .agg(["mean", "std"])
                .reset_index()
            )
            ax.errorbar(
                grouped["task_arrival_rate"], grouped["mean"], yerr=grouped["std"],
                label=get_label(cfg), color=get_color(cfg),
                marker="o", capsize=4, linewidth=2,
            )
        ax.set_xlabel("Arrival Rate (tasks/ts/station)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{title_part} Task Wait Time vs Arrival Rate\n(fully connected, cr={max_cr})")
        ax.legend()
        ax.grid(alpha=0.3)
        ax.set_ylim(bottom=0)

    fig.suptitle(
        "Task Wait Time: Injection to Execution Start (Steady-State)\n"
        "(post-warmup; high wait = system near saturation)",
        fontsize=13, y=1.03,
    )
    fig.tight_layout()
    _savefig(fig, plot_dir, "task_wait_time")


# -----------------------------------------------------------------
#  Plot 19: Queue depth and saturation metrics
# -----------------------------------------------------------------

def plot_queue_metrics(df: pd.DataFrame, plot_dir: str) -> None:
    """
    Avg queue depth and saturation fraction vs arrival rate.
    Shows when the induct station queue starts filling up (near-capacity operation).
    Uses only SS_PRIMARY_ARRIVAL_RATES for uniform method coverage.
    """
    dfc = _ss_clean_df(df)
    required = {"avg_queue_depth", "queue_saturation_fraction"}
    if dfc.empty or not required.issubset(dfc.columns):
        print("  [!] Skipping queue_metrics (no steady-state data or columns missing)")
        return

    methods = _canonical_methods_present(dfc, CANONICAL_SS_METHODS)
    max_cr = _get_max_connected_cr(dfc)
    df_fc  = dfc[dfc["comm_range"] == max_cr]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    for cfg in methods:
        subset = df_fc[df_fc["config_name"] == cfg]
        grouped = (
            subset.groupby("task_arrival_rate")["avg_queue_depth"]
            .agg(["mean", "std"])
            .reset_index()
        )
        ax.errorbar(
            grouped["task_arrival_rate"], grouped["mean"], yerr=grouped["std"],
            label=get_label(cfg), color=get_color(cfg),
            marker="o", capsize=4, linewidth=2,
        )
    ax.set_xlabel("Arrival Rate (tasks/ts/station)")
    ax.set_ylabel("Avg Queue Depth (tasks per station)")
    ax.set_title(f"Induct Queue Depth vs Arrival Rate\n(cr={max_cr})")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_ylim(bottom=0)

    ax = axes[1]
    for cfg in methods:
        subset = df_fc[df_fc["config_name"] == cfg]
        grouped = (
            subset.groupby("task_arrival_rate")["queue_saturation_fraction"]
            .agg(["mean", "std"])
            .reset_index()
        )
        ax.errorbar(
            grouped["task_arrival_rate"], grouped["mean"] * 100, yerr=grouped["std"] * 100,
            label=get_label(cfg), color=get_color(cfg),
            marker="s", capsize=4, linewidth=2,
        )
    ax.set_xlabel("Arrival Rate (tasks/ts/station)")
    ax.set_ylabel("Queue Saturation (% of timesteps)")
    ax.set_title(f"Queue Saturation Fraction vs Arrival Rate\n(cr={max_cr})")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 105)

    fig.suptitle(
        "Induct Queue Metrics vs Arrival Rate (Steady-State)\n"
        "(saturation = fraction of time any station was at max capacity)",
        fontsize=13, y=1.03,
    )
    fig.tight_layout()
    _savefig(fig, plot_dir, "queue_metrics")


# -----------------------------------------------------------------
#  Plot 20: Throughput vs communication range (steady-state)
# -----------------------------------------------------------------

def plot_throughput_vs_comm_range_ss(df: pd.DataFrame, plot_dir: str) -> None:
    """
    Steady-state throughput vs communication range, one subplot per arrival rate.
    Shows how connectivity affects throughput (mirrors makespan_vs_comm_range for batch).
    Uses only SS_PRIMARY_ARRIVAL_RATES so all methods have full CR coverage per subplot.
    """
    dfc = _ss_clean_df(df)
    if dfc.empty or "throughput" not in dfc.columns:
        print("  [!] Skipping throughput_vs_comm_range_ss (no steady-state data)")
        return

    arr_rates = sorted(dfc["task_arrival_rate"].unique())
    n_plots = len(arr_rates)
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 5), sharey=False)
    if n_plots == 1:
        axes = [axes]

    methods = _canonical_methods_present(dfc, CANONICAL_SS_METHODS)

    for ax_idx, ar in enumerate(arr_rates):
        ax = axes[ax_idx]
        df_ar = dfc[dfc["task_arrival_rate"] == ar]

        for cfg in methods:
            subset = df_ar[df_ar["config_name"] == cfg]
            grouped = (
                subset.groupby("comm_range")["throughput"]
                .agg(["mean", "std"])
                .reset_index()
            )
            ax.errorbar(
                grouped["comm_range"], grouped["mean"], yerr=grouped["std"],
                label=get_label(cfg), color=get_color(cfg),
                marker="o", capsize=4, linewidth=2,
            )

        comm_ranges = sorted(df_ar["comm_range"].unique())
        y_min = ax.get_ylim()[0]
        for cr in comm_ranges:
            if cr in CR_ANNOTATIONS:
                ax.annotate(
                    CR_ANNOTATIONS[cr], xy=(cr, y_min),
                    fontsize=8, ha="center", va="top",
                    color="gray", style="italic",
                    xytext=(0, -10), textcoords="offset points",
                )

        ax.set_xlabel("Communication Range")
        ax.set_ylabel("Throughput (tasks/ts)")
        ax.set_title(f"ar={ar} (max {ar*8:.2f} tasks/ts total)")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        ax.set_ylim(bottom=0)
        ax.axvspan(0, CONNECTIVITY_THRESHOLD - 1, alpha=0.08, color="red",
                   label="_nolegend_")

    fig.suptitle("Effect of Communication Range on Throughput (Steady-State)",
                 fontsize=15, y=1.02)
    fig.tight_layout()
    _savefig(fig, plot_dir, "throughput_vs_comm_range_ss")


# -----------------------------------------------------------------
#  Plot 23: Allocation scalability (all SS data incl. ar=0.2)
# -----------------------------------------------------------------

def plot_allocation_scalability(df: pd.DataFrame, plot_dir: str) -> None:
    """
    Log-scale plot of allocation time vs arrival rate for all canonical SS methods.

    This is the ONE plot that uses all SS data including the partial ar=0.2 results.
    Because we are plotting *time* (not throughput), having fewer comm_range samples
    at ar=0.2 for cbba/sga does not bias the comparison -- allocation time scales
    with active task count, not with comm_range.

    Left panel:  avg_gcbba_time_ms vs arrival rate (log y-axis).
    Right panel: avg_gcbba_time_ms vs arrival rate (linear y-axis, zoomed to static/dynamic).

    The key thesis message: CBBA/SGA allocation time grows super-linearly with
    arrival rate and becomes computationally infeasible at ar=0.2, while GCBBA
    (static and dynamic) remains tractable.
    """
    # Use _ss_df (all rates) not _ss_clean_df -- ar=0.2 is intentionally included.
    dfc = _ss_df(df)
    if dfc.empty or "avg_gcbba_time_ms" not in dfc.columns:
        print("  [!] Skipping allocation_scalability (no SS timing data)")
        return

    methods = _canonical_methods_present(dfc, CANONICAL_SS_METHODS)

    # Build per-method per-arrival-rate mean/std (averaged over all available CRs & seeds)
    method_data = {}
    for cfg in methods:
        sub = dfc[dfc["config_name"] == cfg]
        grouped = (
            sub.groupby("task_arrival_rate")["avg_gcbba_time_ms"]
            .agg(["mean", "std", "count"])
            .reset_index()
        )
        method_data[cfg] = grouped

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # --- Left: log scale (shows full range including cbba/sga blow-up) ---
    ax = axes[0]
    for cfg in methods:
        g = method_data[cfg]
        ax.errorbar(
            g["task_arrival_rate"], g["mean"], yerr=g["std"],
            label=get_label(cfg), color=get_color(cfg),
            marker="o", capsize=4, linewidth=2,
        )
        # Mark partial-data points (ar=0.2) with an open marker overlay
        partial = g[~g["task_arrival_rate"].isin(SS_PRIMARY_ARRIVAL_RATES)]
        if not partial.empty:
            ax.scatter(
                partial["task_arrival_rate"], partial["mean"],
                facecolors="none", edgecolors=get_color(cfg),
                s=120, linewidths=2, zorder=5,
            )

    # Real-time feasibility reference line at 1000 ms
    ax.axhline(1000, color="red", linestyle="--", linewidth=1.2, alpha=0.7,
               label="1 s / call (feasibility threshold)")
    ax.set_yscale("log")
    ax.set_xlabel("Arrival Rate (tasks/ts/station)")
    ax.set_ylabel("Avg Allocation Time per Call (ms, log scale)")
    ax.set_title("Allocation Time vs Arrival Rate\n(log scale -- all data incl. ar=0.2)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, which="both")

    # Add note about open markers
    ax.annotate(
        "Open markers = partial data\n(fewer comm_range samples at ar=0.2)",
        xy=(0.98, 0.04), xycoords="axes fraction",
        ha="right", va="bottom", fontsize=8, color="gray",
        style="italic",
    )

    # --- Right: linear scale, GCBBA + SGA (CBBA excluded — off-chart at ar=0.2) ---
    # Shows that SGA also scales worse than GCBBA, not just CBBA.
    ax = axes[1]
    linear_methods = [m for m in methods if m in ("static", "dynamic", "sga")]
    # Restrict to primary rates only so all methods have equal sample coverage
    for cfg in linear_methods:
        g = method_data[cfg]
        g_clean = g[g["task_arrival_rate"].isin(SS_PRIMARY_ARRIVAL_RATES)]
        ax.errorbar(
            g_clean["task_arrival_rate"], g_clean["mean"], yerr=g_clean["std"],
            label=get_label(cfg), color=get_color(cfg),
            marker="o", capsize=4, linewidth=2,
        )
    ax.set_xlabel("Arrival Rate (tasks/ts/station)")
    ax.set_ylabel("Avg Allocation Time per Call (ms)")
    ax.set_title("GCBBA vs SGA (linear scale, primary rates)\n(CBBA off-chart; SGA scales worse than GCBBA)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_ylim(bottom=0)

    fig.suptitle(
        "Allocation Scalability: GCBBA vs CBBA vs SGA\n"
        "(Both CBBA and SGA scale super-linearly; GCBBA remains most tractable)",
        fontsize=13, y=1.03,
    )
    fig.tight_layout()
    _savefig(fig, plot_dir, "allocation_scalability")


# -----------------------------------------------------------------
#  Plot 22: Steady-state LaTeX table
# -----------------------------------------------------------------

def generate_latex_table_ss(df: pd.DataFrame, plot_dir: str) -> None:
    """LaTeX table for steady-state runs (throughput-centric).
    Restricted to SS_PRIMARY_ARRIVAL_RATES for uniform method coverage.
    """
    dfc = _ss_clean_df(df)
    if dfc.empty or "throughput" not in dfc.columns:
        print("  [!] Skipping results_table_ss (no steady-state data)")
        return

    agg_cols = {
        "throughput":                  ("throughput", "mean"),
        "throughput_std":              ("throughput", "std"),
        "num_gcbba_runs_mean":         ("num_gcbba_runs", "mean"),
        "avg_gcbba_time_ms_mean":      ("avg_gcbba_time_ms", "mean"),
        "avg_idle_ratio_mean":         ("avg_idle_ratio", "mean"),
        "num_vertex_collisions_sum":   ("num_vertex_collisions", "sum"),
        "num_deadlocks_sum":           ("num_deadlocks", "sum"),
    }
    optional = {
        "avg_task_wait_time_mean":       ("avg_task_wait_time", "mean"),
        "avg_queue_depth_mean":          ("avg_queue_depth", "mean"),
        "queue_saturation_pct_mean":     ("queue_saturation_fraction", "mean"),
        "tasks_dropped_by_queue_cap_sum":("tasks_dropped_by_queue_cap", "sum"),
    }
    agg_dict = agg_cols.copy()
    has_optional = {}
    for col_name, (src_col, agg_fn) in optional.items():
        if src_col in dfc.columns:
            agg_dict[col_name] = (src_col, agg_fn)
            has_optional[col_name] = True

    agg = (
        dfc.groupby(["config_name", "task_arrival_rate", "comm_range"])
        .agg(**agg_dict)
        .reset_index()
    )

    opt_header = ""
    if has_optional.get("avg_task_wait_time_mean"):
        opt_header += r" & AvgWait"
    if has_optional.get("avg_queue_depth_mean"):
        opt_header += r" & Q-Depth"
    if has_optional.get("queue_saturation_pct_mean"):
        opt_header += r" & Q-Sat\%"
    if has_optional.get("tasks_dropped_by_queue_cap_sum"):
        opt_header += r" & Dropped"

    n_opt = len(has_optional)
    col_spec = r"llrrrrrrrr" + "r" * n_opt

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Experimental Results (Steady-State Mode): GCBBA, CBBA, SGA}",
        r"\label{tab:results_ss}",
        r"\small",
        r"\begin{tabular}{" + col_spec + r"}",
        r"\toprule",
        r"Config & AR & CR & Throughput & $\sigma$ & Alloc & Avg ms & Idle & Col & Stuck"
        + opt_header + r" \\",
        r"\midrule",
    ]

    for _, row in agg.iterrows():
        base = (
            f"  {row['config_name']} & {row['task_arrival_rate']:.3f} & "
            f"{row['comm_range']:.0f} & "
            f"{row['throughput']:.4f} & {row['throughput_std']:.4f} & "
            f"{row['num_gcbba_runs_mean']:.0f} & "
            f"{row['avg_gcbba_time_ms_mean']:.0f} & "
            f"{row['avg_idle_ratio_mean']:.3f} & "
            f"{row['num_vertex_collisions_sum']:.0f} & "
            f"{row['num_deadlocks_sum']:.0f}"
        )
        if has_optional.get("avg_task_wait_time_mean"):
            base += f" & {row['avg_task_wait_time_mean']:.1f}"
        if has_optional.get("avg_queue_depth_mean"):
            base += f" & {row['avg_queue_depth_mean']:.2f}"
        if has_optional.get("queue_saturation_pct_mean"):
            base += f" & {row['queue_saturation_pct_mean']*100:.1f}"
        if has_optional.get("tasks_dropped_by_queue_cap_sum"):
            base += f" & {row['tasks_dropped_by_queue_cap_sum']:.0f}"
        lines.append(base + r" \\")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\vspace{4pt}",
        r"\parbox{\textwidth}{\footnotesize "
        r"AR = arrival rate (tasks/ts/station); "
        r"CR = communication range; "
        r"Throughput = tasks completed per timestep (post-warmup); "
        r"Alloc = allocation invocations; "
        r"Avg ms = mean allocation time; "
        r"Idle = mean agent idle ratio; "
        r"Col = vertex collisions (sum); "
        r"Stuck = distinct stuck-state events (sum); "
        r"AvgWait = mean task wait time (injection to start); "
        r"Q-Depth = mean induct queue depth; "
        r"Q-Sat = fraction of timesteps any station was at max capacity; "
        r"Dropped = tasks rejected due to full queue (sum).}",
        r"\end{table}",
    ]

    table_path = os.path.join(plot_dir, "results_table_ss.tex")
    with open(table_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("  [OK] results_table_ss.tex")


# -----------------------------------------------------------------
#  Plot 25: Task execution quality (distance per task + abort rate)
# -----------------------------------------------------------------

def plot_task_execution_quality(df: pd.DataFrame, plot_dir: str) -> None:
    """
    Two subplots per mode (SS / batch):
      Left:  distance_per_task vs x_col — lower means tighter, more efficient assignments.
      Right: task_abort_rate vs x_col   — fraction of tasks abandoned for charging.

    Both metrics are allocation quality signals: a better allocation minimises
    travel distance and avoids sending agents that are about to need charging.
    """
    for label_prefix, dfc, methods, x_col, x_label in [
        ("Steady-State", _ss_clean_df(df), CANONICAL_SS_METHODS,    "task_arrival_rate", "Arrival Rate (tasks/ts/station)"),
        ("Batch",        _batch_df(df),    CANONICAL_BATCH_METHODS,  "tasks_per_induct",  "Initial Tasks per Induct Station"),
    ]:
        if dfc.empty:
            continue
        if dfc["distance_per_task"].isna().all():
            continue
        methods_here = _canonical_methods_present(dfc, methods)
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        # Left: distance per task
        ax = axes[0]
        for cfg in methods_here:
            sub = dfc[dfc["config_name"] == cfg]
            g = sub.groupby(x_col)["distance_per_task"].agg(["mean", "std"]).reset_index()
            ax.errorbar(g[x_col], g["mean"], yerr=g["std"],
                        label=get_label(cfg), color=get_color(cfg),
                        marker="o", capsize=4, linewidth=2)
        ax.set_xlabel(x_label)
        ax.set_ylabel("Distance per Completed Task (grid steps)")
        ax.set_title(f"Path Efficiency ({label_prefix})\n(lower = tighter assignments)")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        ax.set_ylim(bottom=0)

        # Right: abort rate
        ax = axes[1]
        for cfg in methods_here:
            sub = dfc[dfc["config_name"] == cfg]
            g = sub.groupby(x_col)["task_abort_rate"].agg(["mean", "std"]).reset_index()
            ax.errorbar(g[x_col], g["mean"] * 100, yerr=g["std"] * 100,
                        label=get_label(cfg), color=get_color(cfg),
                        marker="s", capsize=4, linewidth=2)
        ax.set_xlabel(x_label)
        ax.set_ylabel("Task Abort Rate (% of completed tasks)")
        ax.set_title(f"Charging-Induced Task Aborts ({label_prefix})\n(lower = less disruptive energy management)")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        ax.set_ylim(bottom=0)

        fig.suptitle(
            f"Task Execution Quality ({label_prefix})\n"
            "(path efficiency and energy disruption by allocation method)",
            fontsize=13, y=1.03,
        )
        fig.tight_layout()
        suffix = "ss" if label_prefix == "Steady-State" else "batch"
        _savefig(fig, plot_dir, f"task_execution_quality_{suffix}")


# -----------------------------------------------------------------
#  Plot 26: Agent time breakdown (stacked bar)
# -----------------------------------------------------------------

def plot_agent_time_breakdown(df: pd.DataFrame, plot_dir: str) -> None:
    """
    Stacked bar chart: working / idle / charging fraction of agent-timesteps.

    working   = 1 - idle - charging  (executing tasks or travelling to them)
    idle      = avg_idle_ratio
    charging  = charging_time_fraction

    Shows how effectively each method utilises agents.  A good allocation
    maximises working and minimises idle; charging should be similar across
    methods (it is governed by the energy policy, not the allocator).
    One chart per mode (SS / batch) at the highest x-value (most demanding).
    """
    for label_prefix, dfc, methods, x_col, x_label in [
        ("Steady-State", _ss_clean_df(df), CANONICAL_SS_METHODS,    "task_arrival_rate", "Arrival Rate"),
        ("Batch",        _batch_df(df),    CANONICAL_BATCH_METHODS,  "tasks_per_induct",  "Tasks/Induct"),
    ]:
        if dfc.empty:
            continue
        if dfc["working_fraction"].isna().all():
            continue

        methods_here = _canonical_methods_present(dfc, methods)
        x_vals = sorted(dfc[x_col].unique())
        n_x = len(x_vals)

        fig, axes = plt.subplots(1, n_x, figsize=(4 * n_x, 5), sharey=True)
        if n_x == 1:
            axes = [axes]

        colors_stack = {"working": "#2ecc71", "idle": "#95a5a6", "charging": "#e74c3c"}

        for ax_idx, xv in enumerate(x_vals):
            ax = axes[ax_idx]
            df_x = dfc[dfc[x_col] == xv]
            x = np.arange(len(methods_here))
            width = 0.55

            working_means, idle_means, charging_means = [], [], []
            for cfg in methods_here:
                sub = df_x[df_x["config_name"] == cfg]
                working_means.append(sub["working_fraction"].mean() * 100)
                idle_means.append(sub["avg_idle_ratio"].mean() * 100)
                charging_means.append(sub["charging_time_fraction"].mean() * 100)

            ax.bar(x, working_means,  width, label="Working",  color=colors_stack["working"],  alpha=0.85)
            ax.bar(x, idle_means,     width, label="Idle",     color=colors_stack["idle"],     alpha=0.85,
                   bottom=working_means)
            ax.bar(x, charging_means, width, label="Charging", color=colors_stack["charging"], alpha=0.85,
                   bottom=[w + i for w, i in zip(working_means, idle_means)])

            ax.set_xticks(x)
            ax.set_xticklabels(
                [get_label(m).split("(")[0].strip() for m in methods_here],
                rotation=15, ha="right", fontsize=9,
            )
            ax.set_title(f"{x_label}={xv}")
            ax.set_ylim(0, 105)
            ax.grid(axis="y", alpha=0.3)
            if ax_idx == 0:
                ax.set_ylabel("Agent-Timesteps (%)")
                ax.legend(loc="upper right", fontsize=9)

        fig.suptitle(
            f"Agent Time Breakdown ({label_prefix})\n"
            "(working = executing/travelling to tasks; idle = waiting; charging = energy management)",
            fontsize=13, y=1.03,
        )
        fig.tight_layout()
        suffix = "ss" if label_prefix == "Steady-State" else "batch"
        _savefig(fig, plot_dir, f"agent_time_breakdown_{suffix}")


# -----------------------------------------------------------------
#  Plot 27: Scaling exponent (power-law fit)
# -----------------------------------------------------------------

def plot_scaling_exponent(df: pd.DataFrame, plot_dir: str) -> None:
    """
    Fits time = a * n^k to each method's allocation timing data and plots:
      Left:  data + fitted curves (log-log axes so the power law is a straight line).
      Right: bar chart of fitted exponent k per method.

    k > 1 means super-linear scaling.  GCBBA should show smaller k than CBBA/SGA,
    giving a rigorous empirical complexity argument.

    For SS: n = arrival_rate (proxy for active task density).
    For batch: n = tasks_per_induct (directly proportional to task count).
    """
    for label_prefix, dfc, methods, x_col, x_label in [
        ("Steady-State", _ss_clean_df(df), CANONICAL_SS_METHODS,    "task_arrival_rate", "Arrival Rate (tasks/ts/station)"),
        ("Batch",        _batch_df(df),    CANONICAL_BATCH_METHODS,  "tasks_per_induct",  "Initial Tasks per Induct Station"),
    ]:
        if dfc.empty:
            continue
        methods_here = _canonical_methods_present(dfc, methods)

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        # Left: log-log scatter + fitted line
        ax = axes[0]
        exponents = {}
        for cfg in methods_here:
            sub = dfc[dfc["config_name"] == cfg]
            g = (sub.groupby(x_col)["avg_gcbba_time_ms"]
                 .mean().reset_index()
                 .rename(columns={"avg_gcbba_time_ms": "mean"}))
            g = g[g["mean"] > 0]
            if len(g) < 2:
                continue

            ax.scatter(g[x_col], g["mean"], color=get_color(cfg),
                       s=60, zorder=5, edgecolors="black", linewidth=0.4)

            # Fit log(y) = k*log(x) + log(a)
            log_x = np.log(g[x_col].values.astype(float))
            log_y = np.log(g["mean"].values.astype(float))
            k, log_a = np.polyfit(log_x, log_y, 1)
            exponents[cfg] = k

            x_fit = np.linspace(g[x_col].min(), g[x_col].max(), 100)
            y_fit = np.exp(log_a) * x_fit ** k
            ax.plot(x_fit, y_fit, color=get_color(cfg), linewidth=2,
                    label=f"{get_label(cfg)}  (k={k:.2f})")

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(x_label)
        ax.set_ylabel("Avg Allocation Time (ms)")
        ax.set_title(f"Power-Law Fit: time ∝ n^k ({label_prefix})\n(log-log; slope = scaling exponent k)")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3, which="both")

        # Right: bar chart of k values
        ax = axes[1]
        if exponents:
            cfgs = list(exponents.keys())
            k_vals = [exponents[c] for c in cfgs]
            short_labels = [get_label(c).split("(")[0].strip() for c in cfgs]
            bars = ax.bar(short_labels, k_vals,
                          color=[get_color(c) for c in cfgs],
                          alpha=0.85, edgecolor="black", linewidth=0.5)
            ax.axhline(1.0, color="green", linestyle="--", linewidth=1.5,
                       label="k=1 (linear scaling)")
            ax.axhline(2.0, color="red", linestyle=":", linewidth=1.5,
                       label="k=2 (quadratic)")
            for bar, kv in zip(bars, k_vals):
                ax.text(bar.get_x() + bar.get_width() / 2, kv + 0.05,
                        f"{kv:.2f}", ha="center", va="bottom",
                        fontsize=10, fontweight="bold")
            ax.set_ylabel("Scaling Exponent k")
            ax.set_title(f"Empirical Scaling Exponent per Method ({label_prefix})\n"
                         "(k<1=sub-linear, k=1=linear, k>1=super-linear)")
            ax.legend(fontsize=9)
            ax.grid(axis="y", alpha=0.3)
            ax.set_ylim(bottom=0)
            ax.tick_params(axis="x", rotation=15)

        fig.suptitle(
            f"Allocation Scaling Exponents ({label_prefix}): time ∝ n^k\n"
            "(higher k = worse scaling; GCBBA should show smallest k)",
            fontsize=13, y=1.03,
        )
        fig.tight_layout()
        suffix = "ss" if label_prefix == "Steady-State" else "batch"
        _savefig(fig, plot_dir, f"scaling_exponent_{suffix}")


# -----------------------------------------------------------------
#  Plot 28: Decentralization penalty curve
# -----------------------------------------------------------------

def plot_decentralization_penalty(df: pd.DataFrame, plot_dir: str) -> None:
    """
    For each method (excluding SGA), plots:
      penalty(CR) = (sga_metric - method_metric) / sga_metric * 100

    Positive penalty = method is worse than SGA at that connectivity level.
    CBBA's penalty should spike at low CR (consensus fails when isolated).
    GCBBA's penalty should rise only gently (graceful degradation).
    SGA = 0% by definition (centralized, no connectivity dependency).

    SS metric:    throughput  (higher is better → penalty = (sga - method)/sga).
    Batch metric: makespan    (lower is better  → penalty = (method - sga)/sga).
    """
    # --- Steady-state ---
    dfc_ss = _ss_clean_df(df)
    if not dfc_ss.empty and "throughput" in dfc_ss.columns:
        # Use median arrival rate for clearest signal
        ar_vals = sorted(dfc_ss["task_arrival_rate"].unique())
        ar_focus = ar_vals[len(ar_vals) // 2]   # middle rate (e.g. 0.05)
        df_ar = dfc_ss[dfc_ss["task_arrival_rate"] == ar_focus]
        comm_ranges = sorted(df_ar["comm_range"].unique())

        plot_methods = [m for m in CANONICAL_SS_METHODS
                        if m != "sga" and m in df_ar["config_name"].unique()]

        if plot_methods:
            fig, ax = plt.subplots(figsize=(8, 5))
            for cfg in plot_methods:
                penalties, penalty_stds, valid_crs = [], [], []
                for cr in comm_ranges:
                    cr_df = df_ar[df_ar["comm_range"] == cr]
                    sga_tp  = cr_df[cr_df["config_name"] == "sga"]["throughput"]
                    meth_tp = cr_df[cr_df["config_name"] == cfg]["throughput"]
                    if sga_tp.empty or meth_tp.empty:
                        continue
                    pen = (sga_tp.mean() - meth_tp.mean()) / max(sga_tp.mean(), 1e-9) * 100
                    # std via error propagation (conservative)
                    pen_std = np.sqrt(sga_tp.std()**2 + meth_tp.std()**2) / max(sga_tp.mean(), 1e-9) * 100
                    penalties.append(pen)
                    penalty_stds.append(pen_std if not np.isnan(pen_std) else 0)
                    valid_crs.append(cr)
                if valid_crs:
                    ax.errorbar(valid_crs, penalties, yerr=penalty_stds,
                                label=get_label(cfg), color=get_color(cfg),
                                marker="o", capsize=4, linewidth=2)

            ax.axhline(0, color="black", linestyle="--", linewidth=1.2,
                       label="SGA baseline (0% penalty)")
            ax.axvspan(0, CONNECTIVITY_THRESHOLD - 1, alpha=0.08, color="red",
                       label="_nolegend_")
            for cr in comm_ranges:
                if cr in CR_ANNOTATIONS:
                    ax.annotate(CR_ANNOTATIONS[cr], xy=(cr, ax.get_ylim()[0]),
                                fontsize=7, ha="center", va="top", color="gray",
                                style="italic", xytext=(0, -10),
                                textcoords="offset points")
            ax.set_xlabel("Communication Range")
            ax.set_ylabel("Throughput Penalty vs SGA (%)")
            ax.set_title(
                f"Decentralization Penalty vs Communication Range (SS, ar={ar_focus})\n"
                "(0% = matches SGA; positive = below SGA; negative = beats SGA)"
            )
            ax.legend(fontsize=9)
            ax.grid(alpha=0.3)
            fig.tight_layout()
            _savefig(fig, plot_dir, "decentralization_penalty_ss")

    # --- Batch ---
    dfc_ba = _batch_df(df)
    if not dfc_ba.empty and "effective_makespan" in dfc_ba.columns:
        tpi_vals = sorted(dfc_ba["tasks_per_induct"].unique())
        tpi_focus = tpi_vals[len(tpi_vals) // 2]
        df_tpi = dfc_ba[dfc_ba["tasks_per_induct"] == tpi_focus]
        comm_ranges = sorted(df_tpi["comm_range"].unique())

        plot_methods = [m for m in CANONICAL_BATCH_METHODS
                        if m != "sga_batch" and m in df_tpi["config_name"].unique()]

        if plot_methods:
            fig, ax = plt.subplots(figsize=(8, 5))
            for cfg in plot_methods:
                penalties, penalty_stds, valid_crs = [], [], []
                for cr in comm_ranges:
                    cr_df = df_tpi[df_tpi["comm_range"] == cr]
                    sga_ms  = cr_df[cr_df["config_name"] == "sga_batch"]["effective_makespan"]
                    meth_ms = cr_df[cr_df["config_name"] == cfg]["effective_makespan"]
                    if sga_ms.empty or meth_ms.empty:
                        continue
                    pen = (meth_ms.mean() - sga_ms.mean()) / max(sga_ms.mean(), 1e-9) * 100
                    pen_std = np.sqrt(sga_ms.std()**2 + meth_ms.std()**2) / max(sga_ms.mean(), 1e-9) * 100
                    penalties.append(pen)
                    penalty_stds.append(pen_std if not np.isnan(pen_std) else 0)
                    valid_crs.append(cr)
                if valid_crs:
                    ax.errorbar(valid_crs, penalties, yerr=penalty_stds,
                                label=get_label(cfg), color=get_color(cfg),
                                marker="o", capsize=4, linewidth=2)

            ax.axhline(0, color="black", linestyle="--", linewidth=1.2,
                       label="SGA baseline (0% penalty)")
            ax.axvspan(0, CONNECTIVITY_THRESHOLD - 1, alpha=0.08, color="red",
                       label="_nolegend_")
            ax.set_xlabel("Communication Range")
            ax.set_ylabel("Makespan Penalty vs SGA (%)")
            ax.set_title(
                f"Decentralization Penalty vs Communication Range (Batch, tpi={tpi_focus})\n"
                "(0% = matches SGA; positive = worse; negative = beats SGA)"
            )
            ax.legend(fontsize=9)
            ax.grid(alpha=0.3)
            fig.tight_layout()
            _savefig(fig, plot_dir, "decentralization_penalty_batch")


# -----------------------------------------------------------------
#  Plot 24: Batch allocation scalability (timing vs task count, log scale)
# -----------------------------------------------------------------

def plot_batch_allocation_scalability(df: pd.DataFrame, plot_dir: str) -> None:
    """
    Log-scale allocation time vs tasks-per-induct for all canonical batch methods.

    The linear-scale gcbba_timing_batch plot is dominated by CBBA's 146-second bar
    at tpi=20, which hides the fact that SGA also scales 3× worse than GCBBA.
    This log-scale version makes that gap visible.

    Left panel:  all 4 methods on log scale — shows CBBA and SGA both diverging.
    Right panel: GCBBA + SGA on linear scale — directly shows SGA 3× > GCBBA at tpi=20.
    """
    dfc = _batch_df(df)
    if dfc.empty or "avg_gcbba_time_ms" not in dfc.columns:
        print("  [!] Skipping batch_allocation_scalability (no batch timing data)")
        return

    methods = _canonical_methods_present(dfc, CANONICAL_BATCH_METHODS)
    method_data = {}
    for cfg in methods:
        sub = dfc[dfc["config_name"] == cfg]
        grouped = (
            sub.groupby("tasks_per_induct")["avg_gcbba_time_ms"]
            .agg(["mean", "std"])
            .reset_index()
        )
        method_data[cfg] = grouped

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # --- Left: log scale, all methods ---
    ax = axes[0]
    for cfg in methods:
        g = method_data[cfg]
        ax.errorbar(
            g["tasks_per_induct"], g["mean"], yerr=g["std"],
            label=get_label(cfg), color=get_color(cfg),
            marker="o", capsize=4, linewidth=2,
        )
    ax.axhline(1000, color="red", linestyle="--", linewidth=1.2, alpha=0.7,
               label="1 s / call (feasibility threshold)")
    ax.set_yscale("log")
    ax.set_xlabel("Initial Tasks per Induct Station")
    ax.set_ylabel("Avg Allocation Time per Call (ms, log scale)")
    ax.set_title("Allocation Time vs Task Load (Batch)\n(log scale — all methods)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, which="both")

    # --- Right: linear scale, GCBBA + SGA only ---
    ax = axes[1]
    linear_methods = [m for m in methods if m in ("static_batch", "dynamic_batch", "sga_batch")]
    for cfg in linear_methods:
        g = method_data[cfg]
        label = get_label(cfg)
        ax.errorbar(
            g["tasks_per_induct"], g["mean"], yerr=g["std"],
            label=label, color=get_color(cfg),
            marker="o", capsize=4, linewidth=2,
        )
    ax.set_xlabel("Initial Tasks per Induct Station")
    ax.set_ylabel("Avg Allocation Time per Call (ms)")
    ax.set_title("GCBBA vs SGA (linear scale, batch)\n(CBBA off-chart; SGA scales 3× worse than GCBBA at tpi=20)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_ylim(bottom=0)

    fig.suptitle(
        "Batch Allocation Scalability: GCBBA vs CBBA vs SGA\n"
        "(Both CBBA and SGA scale super-linearly with task count; GCBBA remains tractable)",
        fontsize=13, y=1.03,
    )
    fig.tight_layout()
    _savefig(fig, plot_dir, "batch_allocation_scalability")


# -----------------------------------------------------------------
#  Plot 29: Peak vs avg allocation time
# -----------------------------------------------------------------

def plot_peak_vs_avg_allocation_time(df: pd.DataFrame, plot_dir: str) -> None:
    """
    Three-panel figure per mode (SS / batch):
      Left:   avg_gcbba_time_ms vs load (mean across seeds ± std across seeds)
      Centre: max_gcbba_time_ms vs load (worst-case latency per run)
      Right:  Coefficient of Variation (CV = std_gcbba_time_ms / avg_gcbba_time_ms × 100%)
              CV measures within-run call-to-call variability (A3 metric).
              Low CV → consistent, predictable per-call cost → better real-time guarantees.
              High CV → bursty: some calls are much slower than the mean.

    std_gcbba_time_ms is the within-run std of individual call durations (recorded
    per run), which is different from the across-seed std shown as error bars in
    the left and centre panels.
    """
    for label_prefix, dfc, methods, x_col, x_label in [
        ("Steady-State", _ss_clean_df(df), CANONICAL_SS_METHODS,    "task_arrival_rate", "Arrival Rate (tasks/ts/station)"),
        ("Batch",        _batch_df(df),    CANONICAL_BATCH_METHODS,  "tasks_per_induct",  "Initial Tasks per Induct Station"),
    ]:
        if dfc.empty:
            continue
        if "max_gcbba_time_ms" not in dfc.columns or "avg_gcbba_time_ms" not in dfc.columns:
            continue
        methods_here = _canonical_methods_present(dfc, methods)
        has_std = "std_gcbba_time_ms" in dfc.columns

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Left: avg allocation time (across-seed error bars)
        ax = axes[0]
        for cfg in methods_here:
            sub = dfc[dfc["config_name"] == cfg]
            g = sub.groupby(x_col)["avg_gcbba_time_ms"].agg(["mean", "std"]).reset_index()
            ax.errorbar(g[x_col], g["mean"], yerr=g["std"],
                        label=get_label(cfg), color=get_color(cfg),
                        marker="o", capsize=4, linewidth=2)
        ax.axhline(1000, color="red", linestyle="--", linewidth=1, alpha=0.7,
                   label="1 s threshold")
        ax.set_xlabel(x_label)
        ax.set_ylabel("Avg Allocation Time (ms)")
        ax.set_title(f"Average Allocation Time ({label_prefix})")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        ax.set_ylim(bottom=0)

        # Centre: peak (max) allocation time
        ax = axes[1]
        for cfg in methods_here:
            sub = dfc[dfc["config_name"] == cfg]
            g = sub.groupby(x_col)["max_gcbba_time_ms"].agg(["mean", "std"]).reset_index()
            ax.errorbar(g[x_col], g["mean"], yerr=g["std"],
                        label=get_label(cfg), color=get_color(cfg),
                        marker="s", capsize=4, linewidth=2, linestyle="--")
        ax.axhline(1000, color="red", linestyle="--", linewidth=1, alpha=0.7,
                   label="1 s threshold")
        ax.set_xlabel(x_label)
        ax.set_ylabel("Peak Allocation Time (ms)")
        ax.set_title(f"Worst-Case (Peak) Allocation Time ({label_prefix})")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        ax.set_ylim(bottom=0)

        # Right: Coefficient of Variation (within-run call-to-call variability)
        ax = axes[2]
        if has_std:
            for cfg in methods_here:
                sub = dfc[dfc["config_name"] == cfg]
                g = sub.groupby(x_col)[["avg_gcbba_time_ms", "std_gcbba_time_ms"]].mean().reset_index()
                # CV = within-run std / within-run mean × 100 (both averaged across seeds first)
                cv = (g["std_gcbba_time_ms"] / g["avg_gcbba_time_ms"].replace(0, np.nan) * 100).fillna(0)
                ax.plot(g[x_col], cv,
                        label=get_label(cfg), color=get_color(cfg),
                        marker="^", linewidth=2, linestyle=":")
            ax.axhline(100, color="gray", linestyle="--", linewidth=1.2, alpha=0.7,
                       label="CV = 100% (std = mean)")
        else:
            ax.text(0.5, 0.5, "std_gcbba_time_ms not in data\n(re-run experiments)",
                    ha="center", va="center", transform=ax.transAxes, fontsize=10, color="gray")
        ax.set_xlabel(x_label)
        ax.set_ylabel("CV = std / mean × 100 (%)")
        ax.set_title(f"Call-to-Call Variability ({label_prefix})\n"
                     "(low CV = consistent; high CV = bursty — bad for real-time)")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        ax.set_ylim(bottom=0)

        fig.suptitle(
            f"Allocation Time: Average · Peak · Variability ({label_prefix})\n"
            "(Peak >> Avg or high CV → unpredictable latency, harder to deploy in real time)",
            fontsize=13, y=1.03,
        )
        fig.tight_layout()
        suffix = "ss" if label_prefix == "Steady-State" else "batch"
        _savefig(fig, plot_dir, f"peak_vs_avg_allocation_time_{suffix}")


# -----------------------------------------------------------------
#  Plot 30: Queue drop rate (steady-state)
# -----------------------------------------------------------------

def plot_queue_drop_rate(df: pd.DataFrame, plot_dir: str) -> None:
    """
    Fraction of injected tasks rejected due to full queue vs arrival rate.

    At low arrival rates, drop rate should be ~0 (queue has headroom).
    At high rates, CBBA/SGA may drop more if allocation is slow and queue fills.
    GCBBA should maintain low drop rate even at high loads (faster allocation).

    Uses SS_PRIMARY_ARRIVAL_RATES for uniform method coverage.
    """
    dfc = _ss_clean_df(df)
    if dfc.empty or "tasks_drop_rate" not in dfc.columns:
        print("  [!] Skipping queue_drop_rate (no SS data or tasks_drop_rate missing)")
        return

    methods_here = _canonical_methods_present(dfc, CANONICAL_SS_METHODS)
    comm_ranges = sorted(dfc["comm_range"].unique())

    # Show low / mid / high CR
    cr_low  = min(comm_ranges)
    cr_high = max(comm_ranges)
    cr_mid  = comm_ranges[len(comm_ranges) // 2]
    show_crs = sorted(set([cr_low, cr_mid, cr_high]))

    fig, axes = plt.subplots(1, len(show_crs), figsize=(5 * len(show_crs), 5), sharey=False)
    if len(show_crs) == 1:
        axes = [axes]

    for ax_idx, cr in enumerate(show_crs):
        ax = axes[ax_idx]
        df_cr = dfc[dfc["comm_range"] == cr]

        for cfg in methods_here:
            sub = df_cr[df_cr["config_name"] == cfg]
            g = (sub.groupby("task_arrival_rate")["tasks_drop_rate"]
                 .agg(["mean", "std"]).reset_index())
            g["mean_pct"] = g["mean"] * 100
            g["std_pct"]  = g["std"]  * 100
            ax.errorbar(g["task_arrival_rate"], g["mean_pct"], yerr=g["std_pct"],
                        label=get_label(cfg), color=get_color(cfg),
                        marker="o", capsize=4, linewidth=2)

        conn_str = "Disconnected" if cr < CONNECTIVITY_THRESHOLD else "Well Connected"
        ax.set_xlabel("Arrival Rate (tasks/ts/station)")
        ax.set_ylabel("Queue Drop Rate (%)")
        ax.set_title(f"cr={int(cr)} ({conn_str})")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        ax.set_ylim(bottom=0)

    fig.suptitle(
        "Task Queue Drop Rate vs Arrival Rate (Steady-State)\n"
        "(Higher drop rate = allocation too slow to clear queue; GCBBA should be lowest)",
        fontsize=13, y=1.03,
    )
    fig.tight_layout()
    _savefig(fig, plot_dir, "queue_drop_rate_ss")


# -----------------------------------------------------------------
#  Plot 31: Batch failure heatmap (completion rate grid)
# -----------------------------------------------------------------

def plot_batch_failure_heatmap(df: pd.DataFrame, plot_dir: str) -> None:
    """
    2D heatmap: comm_range (Y) × tasks_per_induct (X) → mean completion rate.
    One subplot per canonical batch method.

    Red cells = method failed to complete all tasks at that (cr, tpi) combination.
    Shows GCBBA's robustness across the load × connectivity parameter space.
    """
    dfc = _batch_df(df)
    if dfc.empty or "all_tasks_completed" not in dfc.columns:
        print("  [!] Skipping batch_failure_heatmap (no batch runs)")
        return

    methods_here = _canonical_methods_present(dfc, CANONICAL_BATCH_METHODS)
    task_counts   = sorted(dfc["tasks_per_induct"].unique())
    comm_ranges   = sorted(dfc["comm_range"].unique(), reverse=True)  # high CR at top

    n_methods = len(methods_here)
    fig, axes = plt.subplots(1, n_methods, figsize=(4 * n_methods, 5), sharey=True)
    if n_methods == 1:
        axes = [axes]

    import matplotlib.colors as mcolors
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "completion", ["#d62728", "#f5f5f5", "#2ca02c"], N=256
    )

    for ax_idx, cfg in enumerate(methods_here):
        ax = axes[ax_idx]
        sub = dfc[dfc["config_name"] == cfg]

        # Build 2D grid
        grid = np.full((len(comm_ranges), len(task_counts)), np.nan)
        for ci, cr in enumerate(comm_ranges):
            for ti, tpi in enumerate(task_counts):
                vals = sub[
                    (sub["comm_range"] == cr) & (sub["tasks_per_induct"] == tpi)
                ]["all_tasks_completed"]
                if not vals.empty:
                    grid[ci, ti] = float(vals.mean())

        im = ax.imshow(grid, cmap=cmap, vmin=0, vmax=1,
                       aspect="auto", interpolation="nearest")

        # Annotate cells
        for ci in range(len(comm_ranges)):
            for ti in range(len(task_counts)):
                val = grid[ci, ti]
                if not np.isnan(val):
                    ax.text(ti, ci, f"{val:.0%}",
                            ha="center", va="center", fontsize=9,
                            color="black" if 0.3 < val < 0.85 else "white")

        ax.set_xticks(range(len(task_counts)))
        ax.set_xticklabels([str(t) for t in task_counts])
        ax.set_yticks(range(len(comm_ranges)))
        ax.set_yticklabels([str(int(cr)) for cr in comm_ranges])
        ax.set_xlabel("Tasks per Induct Station")
        if ax_idx == 0:
            ax.set_ylabel("Communication Range")
        short = cfg.replace("_batch", "")
        ax.set_title(f"{get_label(cfg).split('(')[0].strip()}\n({short})")

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label("Completion Rate")

    fig.suptitle(
        "Batch Task Completion Rate: Load × Connectivity Heatmap\n"
        "(Green=100% success; Red=failure; GCBBA should stay green across more cells)",
        fontsize=13, y=1.03,
    )
    fig.tight_layout()
    _savefig(fig, plot_dir, "batch_failure_heatmap")


# -----------------------------------------------------------------
#  Plot 32: Throughput ramp-up curve (from tasks_completed_over_time)
# -----------------------------------------------------------------

def plot_throughput_rampup(exp_dir: str, df: pd.DataFrame, plot_dir: str) -> None:
    """
    Reads tasks_completed_over_time from individual metrics.json files and plots
    the rolling instantaneous throughput (derivative) over time.

    Shows how quickly each method reaches steady-state throughput after t=0.
    Methods that ramp up faster are better for dynamic workloads.

    Averages across seeds at the median arrival rate and highest CR (best case).
    Uses a rolling window to smooth the derivative.
    """
    dfc = _ss_clean_df(df)
    if dfc.empty:
        print("  [!] Skipping throughput_rampup (no SS data)")
        return

    ar_vals = sorted(dfc["task_arrival_rate"].unique())
    ar_focus = ar_vals[len(ar_vals) // 2]  # median arrival rate
    cr_max   = int(dfc["comm_range"].max())
    methods_here = _canonical_methods_present(dfc, CANONICAL_SS_METHODS)

    WINDOW = 20  # rolling window for smoothed derivative (timesteps)

    fig, ax = plt.subplots(figsize=(10, 5))
    any_plotted = False

    for cfg in methods_here:
        subset = dfc[
            (dfc["config_name"] == cfg)
            & (dfc["task_arrival_rate"] == ar_focus)
            & (dfc["comm_range"] == cr_max)
        ]
        if subset.empty:
            continue

        all_curves = []
        max_len = 0
        for _, row in subset.iterrows():
            run_id = row.get("run_id", "")
            # Walk the experiment dir for this run's metrics.json
            json_path = None
            for root, _, files in os.walk(exp_dir):
                if "metrics.json" in files and os.path.basename(root) == run_id:
                    json_path = os.path.join(root, "metrics.json")
                    break
            if json_path is None:
                continue
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    m = json.load(f)
                cot = m.get("tasks_completed_over_time", [])
                if len(cot) < 2:
                    continue
                cot_arr = np.array(cot, dtype=float)
                all_curves.append(cot_arr)
                max_len = max(max_len, len(cot_arr))
            except Exception:
                continue

        if not all_curves:
            continue

        # Pad shorter curves with their final value, then average
        padded = np.array([
            np.pad(c, (0, max_len - len(c)), mode="edge") for c in all_curves
        ])
        mean_cot = padded.mean(axis=0)

        # Rolling derivative → instantaneous throughput
        t = np.arange(len(mean_cot))
        half_w = WINDOW // 2
        inst_tp = np.zeros(len(mean_cot))
        for i in range(len(mean_cot)):
            lo = max(0, i - half_w)
            hi = min(len(mean_cot) - 1, i + half_w)
            if hi > lo:
                inst_tp[i] = (mean_cot[hi] - mean_cot[lo]) / (hi - lo)

        ax.plot(t, inst_tp, label=get_label(cfg), color=get_color(cfg), linewidth=2)
        any_plotted = True

    if not any_plotted:
        print("  [!] Skipping throughput_rampup (no metrics.json found)")
        plt.close(fig)
        return

    # Theoretical maximum throughput for reference
    n_stations = 8
    ax.axhline(ar_focus * n_stations, color="black", linestyle="--",
               linewidth=1.2, alpha=0.6, label=f"Max possible ({n_stations} stations)")

    ax.set_xlabel("Timestep")
    ax.set_ylabel(f"Instantaneous Throughput (tasks/ts, {WINDOW}-step rolling avg)")
    ax.set_title(
        f"Throughput Ramp-Up Curve (SS, ar={ar_focus}, cr={cr_max})\n"
        f"(Faster rise = faster convergence to steady state; {len(methods_here)} methods shown)"
    )
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    _savefig(fig, plot_dir, "throughput_rampup_ss")


# -----------------------------------------------------------------
#  Plot 33: Allocation time distribution (violin/box from metrics.json)
# -----------------------------------------------------------------

def plot_allocation_time_distribution(exp_dir: str, df: pd.DataFrame, plot_dir: str) -> None:
    """
    Reads gcbba_run_durations_ms from individual metrics.json files to build
    the full empirical distribution of per-call allocation times per method.

    A violin/box plot shows not just mean/max but the full spread:
    - Heavy right tail = occasional expensive calls (bad for real-time).
    - GCBBA should have a tighter distribution than CBBA/SGA.

    Uses fully connected CR and primary arrival rates for SS.
    Also generates a batch version using fully connected CR.
    """
    for label_prefix, dfc, methods, filter_col, filter_val in [
        ("Steady-State", _ss_clean_df(df),  CANONICAL_SS_METHODS,    "task_arrival_rate",
         sorted(_ss_clean_df(df)["task_arrival_rate"].unique())[len(sorted(_ss_clean_df(df)["task_arrival_rate"].unique()))//2]
         if not _ss_clean_df(df).empty else None),
        ("Batch",        _batch_df(df),      CANONICAL_BATCH_METHODS,  "tasks_per_induct",
         sorted(_batch_df(df)["tasks_per_induct"].unique())[-1]
         if not _batch_df(df).empty else None),
    ]:
        if dfc.empty or filter_val is None:
            continue

        cr_max = int(dfc["comm_range"].max())
        methods_here = _canonical_methods_present(dfc, methods)

        distributions = {cfg: [] for cfg in methods_here}

        for cfg in methods_here:
            subset = dfc[
                (dfc["config_name"] == cfg)
                & (dfc[filter_col] == filter_val)
                & (dfc["comm_range"] == cr_max)
            ]
            for _, row in subset.iterrows():
                run_id = row.get("run_id", "")
                json_path = None
                for root, dirs, files in os.walk(exp_dir):
                    if "metrics.json" in files and os.path.basename(root) == run_id:
                        json_path = os.path.join(root, "metrics.json")
                        break
                if json_path is None:
                    continue
                try:
                    with open(json_path, "r", encoding="utf-8") as f:
                        m = json.load(f)
                    durations = m.get("gcbba_run_durations_ms", [])
                    distributions[cfg].extend(durations)
                except Exception:
                    continue

        # Keep only methods with data
        valid = {cfg: v for cfg, v in distributions.items() if len(v) > 0}
        if not valid:
            print(f"  [!] Skipping allocation_time_distribution_{label_prefix.lower().replace(' ','')} (no json data)")
            continue

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        # Left: violin plot
        ax = axes[0]
        positions = list(range(1, len(valid) + 1))
        parts = ax.violinplot(
            [valid[cfg] for cfg in valid],
            positions=positions,
            showmedians=True, showextrema=True,
        )
        for body, cfg in zip(parts["bodies"], valid.keys()):
            body.set_facecolor(get_color(cfg))
            body.set_alpha(0.7)
        ax.axhline(1000, color="red", linestyle="--", linewidth=1, alpha=0.7,
                   label="1 s threshold")
        ax.set_xticks(positions)
        ax.set_xticklabels(
            [get_label(cfg).split("(")[0].strip() for cfg in valid],
            rotation=15, ha="right", fontsize=9
        )
        ax.set_ylabel("Allocation Call Duration (ms)")
        ax.set_title(f"Distribution of Allocation Durations ({label_prefix})\n(violin; median line shown)")
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylim(bottom=0)

        # Right: log-scale box plot (shows outliers / tail better)
        ax = axes[1]
        bp = ax.boxplot(
            [valid[cfg] for cfg in valid],
            positions=positions,
            patch_artist=True,
            medianprops={"color": "black", "linewidth": 2},
            showfliers=True,
            flierprops={"marker": ".", "markersize": 3, "alpha": 0.4},
        )
        for patch, cfg in zip(bp["boxes"], valid.keys()):
            patch.set_facecolor(get_color(cfg))
            patch.set_alpha(0.7)
        ax.axhline(1000, color="red", linestyle="--", linewidth=1, alpha=0.7,
                   label="1 s threshold")
        ax.set_yscale("log")
        ax.set_xticks(positions)
        ax.set_xticklabels(
            [get_label(cfg).split("(")[0].strip() for cfg in valid],
            rotation=15, ha="right", fontsize=9
        )
        ax.set_ylabel("Allocation Call Duration (ms, log scale)")
        ax.set_title(f"Allocation Duration Tail Behaviour ({label_prefix})\n(log scale; outliers shown)")
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3, which="both")

        load_str = f"{filter_col}={filter_val}, cr={cr_max}"
        fig.suptitle(
            f"Allocation Time Distribution — {label_prefix} ({load_str})\n"
            "(Tighter distribution + lower tail = more predictable real-time behaviour)",
            fontsize=12, y=1.03,
        )
        fig.tight_layout()
        suffix = "ss" if label_prefix == "Steady-State" else "batch"
        _savefig(fig, plot_dir, f"allocation_time_distribution_{suffix}")


# -----------------------------------------------------------------
#  Plot 35: Computational budget breakdown (allocation vs path planning)
# -----------------------------------------------------------------

def plot_compute_budget_breakdown(df: pd.DataFrame, plot_dir: str) -> None:
    """
    Stacked bar chart: total_gcbba_time_ms (allocation) vs total_path_plan_time_ms
    (A* / BFS path planning) per method, grouped by load.

    Shows where wall time actually goes — crucial for the real-time feasibility
    argument.  If GCBBA's allocation dominates, optimising A* won't help much,
    and vice versa.  GCBBA should show a different allocation/planning ratio than
    CBBA because it reallocates more often but with fewer tasks each time.

    Secondary panel: avg_path_plan_time_ms vs avg_gcbba_time_ms as a scatter,
    one point per (method, load) combination — reveals per-call cost differences.
    """
    required = {"total_gcbba_time_ms", "total_path_plan_time_ms"}
    if not required.issubset(df.columns):
        print("  [!] Skipping compute_budget_breakdown (path plan columns not in data — re-run experiments)")
        return

    for label_prefix, dfc, methods, x_col, x_label in [
        ("Steady-State", _ss_clean_df(df), CANONICAL_SS_METHODS,    "task_arrival_rate", "Arrival Rate (tasks/ts/station)"),
        ("Batch",        _batch_df(df),    CANONICAL_BATCH_METHODS,  "tasks_per_induct",  "Initial Tasks per Induct Station"),
    ]:
        if dfc.empty:
            continue
        methods_here = _canonical_methods_present(dfc, methods)
        x_vals = sorted(dfc[x_col].unique())
        x = np.arange(len(x_vals))
        width = 0.8 / max(len(methods_here), 1)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # ── Left: stacked bar (total ms per run) ─────────────────────
        ax = axes[0]
        for i, cfg in enumerate(methods_here):
            sub = dfc[dfc["config_name"] == cfg]
            alloc_means, plan_means = [], []
            for xv in x_vals:
                rows = sub[sub[x_col] == xv]
                alloc_means.append(rows["total_gcbba_time_ms"].mean() if not rows.empty else 0)
                plan_means.append(rows["total_path_plan_time_ms"].mean() if not rows.empty else 0)

            offset = (i - (len(methods_here) - 1) / 2) * width
            color = get_color(cfg)
            ax.bar(x + offset, alloc_means, width,
                   label=f"{get_label(cfg)} — alloc",
                   color=color, alpha=0.9, edgecolor="black", linewidth=0.4)
            ax.bar(x + offset, plan_means, width,
                   bottom=alloc_means,
                   label=f"{get_label(cfg)} — A*",
                   color=color, alpha=0.45, edgecolor="black", linewidth=0.4,
                   hatch="///")

        ax.set_xlabel(x_label)
        ax.set_ylabel("Total Time (ms, mean per run)")
        ax.set_title(f"Allocation + Path Planning Budget ({label_prefix})\n"
                     "(solid = allocation; hatched = A* path planning)")
        ax.set_xticks(x)
        ax.set_xticklabels([str(xv) for xv in x_vals])
        ax.legend(fontsize=7, ncol=2)
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylim(bottom=0)

        # ── Right: per-call cost scatter ──────────────────────────────
        ax = axes[1]
        markers = ["o", "s", "^", "D"]
        for i, cfg in enumerate(methods_here):
            sub = dfc[dfc["config_name"] == cfg]
            g = (sub.groupby(x_col)[["avg_gcbba_time_ms", "avg_path_plan_time_ms"]]
                 .mean().reset_index())
            if g.empty or g["avg_gcbba_time_ms"].isna().all():
                continue
            ax.scatter(
                g["avg_gcbba_time_ms"], g["avg_path_plan_time_ms"],
                color=get_color(cfg), marker=markers[i % len(markers)],
                s=80, zorder=5, edgecolors="black", linewidth=0.5,
                label=get_label(cfg),
            )
            # Annotate with load value
            for _, row in g.iterrows():
                ax.annotate(
                    str(row[x_col]),
                    (row["avg_gcbba_time_ms"], row["avg_path_plan_time_ms"]),
                    fontsize=7, ha="left", va="bottom",
                    xytext=(3, 3), textcoords="offset points", color=get_color(cfg),
                )

        # Diagonal: equal allocation and planning time
        lim_max = max(ax.get_xlim()[1], ax.get_ylim()[1]) if ax.get_xlim()[1] > 0 else 1000
        diag = [0, lim_max]
        ax.plot(diag, diag, "k--", linewidth=1, alpha=0.4, label="allocation = planning")
        ax.set_xlabel("Avg Allocation Time per Call (ms)")
        ax.set_ylabel("Avg Path Planning Time per Event (ms)")
        ax.set_title(f"Per-Call Cost: Allocation vs Path Planning ({label_prefix})\n"
                     "(points annotated with load value; dashed = equal cost)")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)

        fig.suptitle(
            f"Computational Budget: Allocation vs A* Path Planning ({label_prefix})\n"
            "(Understanding where wall time goes informs real-time deployment decisions)",
            fontsize=12, y=1.03,
        )
        fig.tight_layout()
        suffix = "ss" if label_prefix == "Steady-State" else "batch"
        _savefig(fig, plot_dir, f"compute_budget_breakdown_{suffix}")


# -----------------------------------------------------------------
#  Plot 34: Per-timestep wall time (real-time feasibility)
# -----------------------------------------------------------------

def plot_step_wall_time(df: pd.DataFrame, plot_dir: str) -> None:
    """
    Plots avg_step_time_ms and max_step_time_ms per method vs load axis.

    avg_step_time = mean cost of one simulation tick (allocation + path planning).
    max_step_time = worst-case tick latency.

    The 1000 ms reference line marks the real-time feasibility boundary:
    if avg < 1 s/tick, the system can sustain a 1 Hz control loop in deployment.
    GCBBA should stay well below this line; CBBA/SGA should breach it at high load.

    Two rows × two columns:
      Row 1: Steady-state (x = arrival rate)
      Row 2: Batch        (x = tasks per induct)
      Col 1: Average step time (with ±1 std error bars)
      Col 2: Maximum step time (worst-case latency)
    """
    if "avg_step_time_ms" not in df.columns:
        print("  [!] Skipping step_wall_time (avg_step_time_ms not in data — re-run experiments)")
        return

    has_ss    = not _ss_clean_df(df).empty
    has_batch = not _batch_df(df).empty
    if not has_ss and not has_batch:
        return

    rows = []
    if has_ss:
        rows.append(("Steady-State", _ss_clean_df(df), CANONICAL_SS_METHODS,
                     "task_arrival_rate", "Arrival Rate (tasks/ts/station)"))
    if has_batch:
        rows.append(("Batch", _batch_df(df), CANONICAL_BATCH_METHODS,
                     "tasks_per_induct", "Initial Tasks per Induct Station"))

    fig, axes = plt.subplots(len(rows), 2, figsize=(13, 5 * len(rows)))
    if len(rows) == 1:
        axes = [axes]  # ensure 2D indexing

    for row_idx, (label_prefix, dfc, methods, x_col, x_label) in enumerate(rows):
        methods_here = _canonical_methods_present(dfc, methods)

        for col_idx, (metric, ylabel, title_suffix) in enumerate([
            ("avg_step_time_ms", "Avg Step Time (ms)", "Average (mean ± std)"),
            ("max_step_time_ms", "Max Step Time (ms)", "Worst-Case (peak)"),
        ]):
            ax = axes[row_idx][col_idx]

            for cfg in methods_here:
                sub = dfc[dfc["config_name"] == cfg]
                g = (sub.groupby(x_col)[metric]
                     .agg(["mean", "std"])
                     .reset_index())
                yerr = g["std"] if col_idx == 0 else None
                ax.errorbar(
                    g[x_col], g["mean"], yerr=yerr,
                    label=get_label(cfg), color=get_color(cfg),
                    marker="o" if col_idx == 0 else "s",
                    capsize=4, linewidth=2,
                    linestyle="-" if col_idx == 0 else "--",
                )

            ax.axhline(1000, color="red", linestyle="--", linewidth=1.5, alpha=0.8,
                       label="1 s/tick (real-time threshold)")
            ax.set_xlabel(x_label)
            ax.set_ylabel(ylabel)
            ax.set_title(f"{label_prefix} — {title_suffix}")
            ax.legend(fontsize=9)
            ax.grid(alpha=0.3)
            ax.set_ylim(bottom=0)

    fig.suptitle(
        "Per-Timestep Wall Time: Real-Time Feasibility\n"
        "(Below 1 s/tick = feasible for 1 Hz deployment; GCBBA should stay under threshold)",
        fontsize=13, y=1.02,
    )
    fig.tight_layout()
    _savefig(fig, plot_dir, "step_wall_time")


# -----------------------------------------------------------------
#  Plot 36: Energy safety margin (D3)
# -----------------------------------------------------------------

def plot_energy_safety_margin(df: pd.DataFrame, plot_dir: str) -> None:
    """
    Plot 36 — Energy safety margin.

    Three energy indicators per method (both SS and batch):
      • avg_final_energy  — mean across agents at end of run
      • min_final_energy  — worst-case agent at end of run
      • min_energy_ever   — lowest point any agent reached mid-run (A5)

    The gap between min_final_energy and min_energy_ever reveals brownout events
    that aren't visible in the end-state alone.  A large gap means the method
    pushed agents into a low-energy state mid-run that self-recovered by the end.

    One panel per mode (SS / batch), bars grouped by method.
    """
    SS_METHODS    = CANONICAL_SS_METHODS
    BATCH_METHODS = CANONICAL_BATCH_METHODS

    panels = []
    dfc_ss    = _ss_clean_df(df)
    dfc_batch = _batch_df(df)
    if not dfc_ss.empty:
        panels.append(("Steady-State", dfc_ss,    SS_METHODS,    "task_arrival_rate"))
    if not dfc_batch.empty:
        panels.append(("Batch",        dfc_batch, BATCH_METHODS, "initial_tasks"))

    if not panels:
        print("  [!] Skipping energy_safety_margin (no data)")
        return

    n_panels = len(panels)
    fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 5), squeeze=False)

    for col_idx, (title, dfc, methods, _) in enumerate(panels):
        ax = axes[0, col_idx]
        methods_here = _canonical_methods_present(dfc, methods)
        x = np.arange(len(methods_here))
        width = 0.25

        avg_vals, min_final_vals, min_ever_vals = [], [], []
        for cfg in methods_here:
            sub = dfc[dfc["config_name"] == cfg]
            avg_vals.append(sub["avg_final_energy"].mean() if "avg_final_energy" in sub else 0)
            min_final_vals.append(sub["min_final_energy"].mean() if "min_final_energy" in sub else 0)
            min_ever_vals.append(sub["min_energy_ever"].mean() if "min_energy_ever" in sub else 0)

        ax.bar(x - width, avg_vals,       width, label="Avg final energy",    color="#4878D0", alpha=0.85)
        ax.bar(x,         min_final_vals,  width, label="Min final energy",    color="#EE854A", alpha=0.85)
        ax.bar(x + width, min_ever_vals,   width, label="Min energy ever",     color="#D65F5F", alpha=0.85)

        # Annotate the brownout gap (min_final - min_ever) for each method
        for i, (mf, me) in enumerate(zip(min_final_vals, min_ever_vals)):
            gap = mf - me
            if gap > 0.5:
                ax.annotate(
                    f"Δ{gap:.0f}",
                    xy=(x[i] + width, me),
                    xytext=(x[i] + width, me + gap / 2),
                    ha="center", va="center", fontsize=9, color="#8B0000",
                    fontweight="bold",
                )

        ax.set_xticks(x)
        ax.set_xticklabels([get_label(m) for m in methods_here], rotation=15, ha="right")
        ax.set_ylabel("Energy (% of max)")
        ax.set_title(title)
        ax.set_ylim(0, 110)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.legend(loc="upper right", fontsize=10)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(
        "Energy Safety Margin: Final State vs Mid-Run Minimum\n"
        r"$\Delta$ = brownout gap (min_final – min_ever); large gap → mid-run energy stress",
        fontsize=13, y=1.04,
    )
    fig.tight_layout()
    _savefig(fig, plot_dir, "energy_safety_margin")


# -----------------------------------------------------------------
#  Plot 37: Queue depth time series (D4)
# -----------------------------------------------------------------

def plot_queue_depth_timeseries(exp_dir: str, df: pd.DataFrame, plot_dir: str) -> None:
    """
    Plot 37 — Queue depth over time (reads queue_depth_over_time from metrics.json).

    Shows the rolling-average mean queue depth across induct stations over the
    course of the simulation for each allocation method.

    Useful for diagnosing whether a method clears the queue fast enough, or
    whether queue depth grows without bound (indicating overload).

    Plots at the median arrival rate and highest comm_range (best-case connectivity).
    Averaged across seeds.  Vertical line at warmup boundary.
    """
    dfc = _ss_clean_df(df)
    if dfc.empty:
        print("  [!] Skipping queue_depth_timeseries (no SS data)")
        return

    ar_vals   = sorted(dfc["task_arrival_rate"].unique())
    ar_focus  = ar_vals[len(ar_vals) // 2]   # median arrival rate
    cr_max    = int(dfc["comm_range"].max())
    methods_here = _canonical_methods_present(dfc, CANONICAL_SS_METHODS)

    WINDOW = 25  # rolling average window (timesteps)

    fig, ax = plt.subplots(figsize=(11, 5))
    any_plotted = False

    for cfg in methods_here:
        subset = dfc[
            (dfc["config_name"] == cfg)
            & (dfc["task_arrival_rate"] == ar_focus)
            & (dfc["comm_range"] == cr_max)
        ]
        if subset.empty:
            continue

        all_curves = []
        max_len = 0
        for _, row in subset.iterrows():
            run_id = row.get("run_id", "")
            json_path = None
            for root, _, files in os.walk(exp_dir):
                if "metrics.json" in files and os.path.basename(root) == run_id:
                    json_path = os.path.join(root, "metrics.json")
                    break
            if json_path is None:
                continue
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    m = json.load(f)
                qdot = m.get("queue_depth_over_time", [])
                if len(qdot) < 2:
                    continue
                all_curves.append(np.array(qdot, dtype=float))
                max_len = max(max_len, len(qdot))
            except Exception:
                continue

        if not all_curves:
            continue

        padded = np.array([
            np.pad(c, (0, max_len - len(c)), mode="edge") for c in all_curves
        ])
        mean_qd = padded.mean(axis=0)
        std_qd  = padded.std(axis=0)

        # Rolling average
        kernel = np.ones(WINDOW) / WINDOW
        smooth = np.convolve(mean_qd, kernel, mode="same")
        t = np.arange(len(smooth))

        color = get_color(cfg)
        ls    = "-"
        label = get_label(cfg)

        ax.plot(t, smooth, color=color, linestyle=ls, linewidth=2, label=label)
        ax.fill_between(
            t,
            np.maximum(0, smooth - std_qd),
            smooth + std_qd,
            alpha=0.12, color=color,
        )
        any_plotted = True

    if not any_plotted:
        plt.close(fig)
        print("  [!] Skipping queue_depth_timeseries (no JSON data found)")
        return

    # Warmup boundary
    warmup_ts = int(dfc["warmup_timesteps"].iloc[0]) if "warmup_timesteps" in dfc.columns else 200
    ax.axvline(warmup_ts, color="black", linestyle="--", linewidth=1.2, label=f"Warmup end (t={warmup_ts})")

    ax.set_xlabel("Timestep")
    ax.set_ylabel(f"Mean queue depth (rolling {WINDOW}-ts avg)")
    ax.set_title(
        f"Queue Depth Over Time — ar={ar_focus}, CR={cr_max}\n"
        "Shaded band = ±1 std across seeds"
    )
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _savefig(fig, plot_dir, "queue_depth_timeseries")


# -----------------------------------------------------------------
#  Orchestrator
# -----------------------------------------------------------------

def _generate_plots_for(exp_dir: str, plot_dir: str) -> None:
    os.makedirs(plot_dir, exist_ok=True)
    print(f"\nGenerating plots from: {exp_dir}")
    print(f"Output to:            {plot_dir}\n")

    df = load_data(exp_dir)

    has_batch = not _batch_df(df).empty
    has_ss    = not _ss_df(df).empty

    print(f"  Batch runs present:        {has_batch}")
    print(f"  Steady-state runs present: {has_ss}\n")

    # ── Batch-mode plots ──────────────────────────────────────────
    plot_makespan_bar(df, plot_dir)
    plot_makespan_vs_comm_range(df, plot_dir)
    plot_improvement_ratio(df, plot_dir)
    plot_completion_rate(df, plot_dir)
    plot_completion_rate_by_tpi(df, plot_dir)

    # ── Shared plots (batch + SS) ─────────────────────────────────
    plot_gcbba_timing(df, plot_dir)
    plot_agent_utilization(df, plot_dir)
    plot_throughput_curves(exp_dir, df, plot_dir)
    plot_collision_deadlock(df, plot_dir)
    plot_graph_connectivity(df, plot_dir)
    plot_optimality_ratio(df, plot_dir)
    plot_rerun_interval_sweep(df, plot_dir)
    plot_trigger_breakdown(df, plot_dir)
    plot_charging_overhead(df, plot_dir)
    plot_energy_performance(df, plot_dir)

    # ── Steady-state-specific plots ───────────────────────────────
    plot_throughput_vs_arrival_rate(df, plot_dir)
    plot_wait_time(df, plot_dir)
    plot_queue_metrics(df, plot_dir)
    plot_throughput_vs_comm_range_ss(df, plot_dir)
    plot_allocation_scalability(df, plot_dir)        # SS: all data incl. ar=0.2
    plot_batch_allocation_scalability(df, plot_dir)  # batch: log scale, SGA vs GCBBA

    # ── New metric plots (both modes) ─────────────────────────────
    plot_task_execution_quality(df, plot_dir)    # distance/task + abort rate
    plot_agent_time_breakdown(df, plot_dir)      # working / idle / charging stacked bar
    plot_scaling_exponent(df, plot_dir)          # power-law fit, annotated k values
    plot_decentralization_penalty(df, plot_dir)  # (sga - method) / sga vs comm_range
    plot_peak_vs_avg_allocation_time(df, plot_dir)       # Plot 29: peak vs avg timing
    plot_queue_drop_rate(df, plot_dir)                   # Plot 30: queue drop rate (SS)
    plot_batch_failure_heatmap(df, plot_dir)             # Plot 31: cr×tpi heatmap
    plot_throughput_rampup(exp_dir, df, plot_dir)        # Plot 32: ramp-up curve (json)
    plot_allocation_time_distribution(exp_dir, df, plot_dir)  # Plot 33: violin/box (json)
    plot_compute_budget_breakdown(df, plot_dir)          # Plot 35: allocation vs path plan
    plot_step_wall_time(df, plot_dir)                    # Plot 34: per-step wall time
    plot_energy_safety_margin(df, plot_dir)              # Plot 36: brownout gap (D3)
    plot_queue_depth_timeseries(exp_dir, df, plot_dir)   # Plot 37: queue depth over time (D4)

    # ── LaTeX tables ──────────────────────────────────────────────
    generate_latex_table(df, plot_dir)
    generate_latex_table_ss(df, plot_dir)

    print(f"\n{'='*50}")
    print(f"All plots saved to: {plot_dir}")
    print(f"{'='*50}")


def _find_all_experiment_dirs() -> list:
    project_root = Path(__file__).resolve().parent.parent
    base = project_root / "results" / "experiments"
    if not base.is_dir():
        return []
    return [
        str(entry)
        for entry in sorted(base.iterdir())
        if entry.is_dir() and (entry / "summary.csv").exists()
    ]


def main():
    parser = argparse.ArgumentParser(
        description="Generate plots from GCBBA experiment results"
    )
    parser.add_argument(
        "exp_dir",
        nargs="?",
        default=None,
        help="Path to a specific experiment directory. "
             "If omitted, all experiments under results/experiments/ are plotted.",
    )
    parser.add_argument(
        "--plot-dir",
        default=None,
        help="Override output directory for plots (only used with a specific exp_dir)",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent

    if args.exp_dir:
        exp_dir = args.exp_dir
        if args.plot_dir:
            plot_dir = args.plot_dir
        else:
            exp_folder_name = os.path.basename(os.path.normpath(exp_dir))
            plot_dir = str(
                project_root / "results" / "plots" / exp_folder_name
            )
        _generate_plots_for(exp_dir, plot_dir)
    else:
        all_dirs = _find_all_experiment_dirs()
        if not all_dirs:
            print(
                "ERROR: No experiment folders with summary.csv found "
                "under results/experiments/"
            )
            sys.exit(1)

        print(f"Found {len(all_dirs)} experiment(s) to plot.")
        for exp_dir in all_dirs:
            exp_folder_name = os.path.basename(os.path.normpath(exp_dir))
            plot_dir = str(
                project_root / "results" / "plots" / exp_folder_name
            )
            _generate_plots_for(exp_dir, plot_dir)

        print(f"\n{'='*50}")
        print(f"Done. Generated plots for {len(all_dirs)} experiment(s).")
        print(f"{'='*50}")


if __name__ == "__main__":
    main()
