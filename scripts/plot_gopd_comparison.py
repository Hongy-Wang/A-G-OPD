#!/usr/bin/env python3
"""Create comparison figures and summary tables from extracted G-OPD CSV files.

This script expects the wide CSV produced by ``extract_gopd_logs.py``. It:

- compares lambda values on the same axes;
- keeps validation points at their actual evaluation steps;
- automatically aggregates multiple seeds/runs for the same lambda;
- draws mean ± standard deviation when multiple runs are available;
- writes run-level and lambda-level summary CSV files;
- optionally creates token-position profile plots from the G-OPD diagnostics.

Examples
--------
Core comparison:

    python plot_gopd_comparison.py parsed_gopd/gopd_metrics_wide.csv \
        --output-dir gopd_lambda_analysis

Comprehensive comparison including position profiles:

    python plot_gopd_comparison.py parsed_gopd/gopd_metrics_wide.csv \
        --output-dir gopd_lambda_analysis \
        --preset full \
        --include-position \
        --formats png pdf

Plot only selected metrics:

    python scripts/plot_gopd_comparison.py metrics/gopd_metrics_wide.csv \
        --metrics \
          val-core/AIME2024/reward/mean@1 \
          gopd/conflict_ratio \
          gopd/grad/opd_extra_cosine
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DERIVED_VAL_MEAN = "derived/val_AIME24_AIME25_mean@1"
AIME24_METRIC = "val-core/AIME2024/reward/mean@1"
AIME25_METRIC = "val-core/AIME2025/reward/mean@1"

ID_COLUMNS = {"run", "lambda", "seed", "source_log", "step", "_run_id"}

PERFORMANCE_METRICS = [
    DERIVED_VAL_MEAN,
    AIME24_METRIC,
    AIME25_METRIC,
    "critic/score/mean",
    "critic/rewards/mean",
]

BEHAVIOR_METRICS = [
    "response_length/mean",
    "response_length/clip_ratio",
    "response_length/max",
    "actor/entropy",
]

SIGNAL_METRICS = [
    "gopd/adv_abs_mean",
    "gopd/opd_adv_abs_mean",
    "gopd/extra_adv_abs_mean",
    "gopd/extra_term_abs_mean",
    "gopd/reinforce_ratio",
    "gopd/conflict_ratio",
    "gopd/conflict_mass_ratio",
    "gopd/flip_ratio",
    "gopd/extra_contribution_mean",
    "gopd/extreme_position_mean",
]

GEOMETRY_METRICS = [
    "gopd/grad/opd_norm",
    "gopd/grad/extra_norm",
    "gopd/grad/total_norm",
    "gopd/grad/extra_opd_norm_ratio",
    "gopd/grad/opd_extra_cosine",
    "gopd/grad/parallel_coef",
    "gopd/grad/orthogonal_ratio",
    "gopd/grad/opd_total_cosine",
    "actor/grad_norm",
]

CORE_METRICS = [
    DERIVED_VAL_MEAN,
    AIME24_METRIC,
    AIME25_METRIC,
    "critic/score/mean",
    "response_length/mean",
    "response_length/clip_ratio",
    "gopd/opd_adv_abs_mean",
    "gopd/extra_adv_abs_mean",
    "gopd/conflict_ratio",
    "gopd/conflict_mass_ratio",
    "gopd/flip_ratio",
    "gopd/extra_contribution_mean",
    "gopd/grad/extra_opd_norm_ratio",
    "gopd/grad/opd_extra_cosine",
    "gopd/grad/opd_total_cosine",
    "actor/grad_norm",
]

POSITION_BINS = [
    ("0_20", 10.0, "0–20%"),
    ("20_40", 30.0, "20–40%"),
    ("40_60", 50.0, "40–60%"),
    ("60_80", 70.0, "60–80%"),
    ("80_100", 90.0, "80–100%"),
]

DEFAULT_POSITION_SUFFIXES = [
    "extra_term_abs_mean",
    "extra_contribution_mean",
    "conflict_ratio",
    "flip_ratio",
    "extreme_enrichment",
]

DISPLAY_NAMES = {
    DERIVED_VAL_MEAN: "AIME 2024/2025 Mean Accuracy",
    AIME24_METRIC: "AIME 2024 Accuracy",
    AIME25_METRIC: "AIME 2025 Accuracy",
    "critic/score/mean": "Training Accuracy / Score",
    "critic/rewards/mean": "Training Reward",
    "response_length/mean": "Mean Response Length",
    "response_length/max": "Maximum Response Length",
    "response_length/clip_ratio": "Response Truncation Ratio",
    "actor/entropy": "Policy Entropy",
    "gopd/adv_abs_mean": "Total Advantage |A| Mean",
    "gopd/opd_adv_abs_mean": "OPD Advantage |A_OPD| Mean",
    "gopd/extra_adv_abs_mean": "Raw Extrapolation |A_extra| Mean",
    "gopd/extra_term_abs_mean": "Weighted Extrapolation Term |(λ−1)A_extra| Mean",
    "gopd/reinforce_ratio": "Token Reinforcement Ratio",
    "gopd/conflict_ratio": "Token Conflict Ratio",
    "gopd/conflict_mass_ratio": "Conflict Mass Ratio",
    "gopd/flip_ratio": "Token Sign-Flip Ratio",
    "gopd/extra_contribution_mean": "Extrapolation Contribution Mean",
    "gopd/extreme_position_mean": "Mean Relative Position of Extreme Tokens",
    "gopd/grad/opd_norm": "OPD Gradient Norm",
    "gopd/grad/extra_norm": "Extrapolation Gradient Norm",
    "gopd/grad/total_norm": "Total Gradient Norm",
    "gopd/grad/extra_opd_norm_ratio": "Gradient Norm Ratio: Extra / OPD",
    "gopd/grad/opd_extra_cosine": "Gradient Cosine: OPD vs Extra",
    "gopd/grad/parallel_coef": "Extra Gradient Parallel Coefficient",
    "gopd/grad/orthogonal_ratio": "Extra Gradient Orthogonal Ratio",
    "gopd/grad/opd_total_cosine": "Gradient Cosine: OPD vs Total",
    "actor/grad_norm": "Actor Gradient Norm",
}

POSITION_DISPLAY_NAMES = {
    "extra_term_abs_mean": "Weighted Extrapolation Magnitude by Token Position",
    "extra_contribution_mean": "Extrapolation Contribution by Token Position",
    "conflict_ratio": "Conflict Ratio by Token Position",
    "flip_ratio": "Sign-Flip Ratio by Token Position",
    "extreme_enrichment": "Extreme-Token Enrichment by Token Position",
    "extreme_token_share": "Extreme-Token Share by Token Position",
    "position_token_share": "Token Share by Token Position",
    "extra_term_abs_p95": "Weighted Extrapolation P95 by Token Position",
}


def unique_preserving_order(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def sanitize_filename(name: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_.")
    return sanitized or "metric"


def format_lambda(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(number):
        return "unknown"
    return f"{number:g}"


def display_name(metric: str) -> str:
    if metric in DISPLAY_NAMES:
        return DISPLAY_NAMES[metric]
    # Reasonable fallback for metrics not in the curated preset.
    return metric.replace("_", " ").replace("/", " / ")


def load_wide_csvs(paths: list[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in paths:
        frame = pd.read_csv(path)
        if "run" not in frame.columns:
            frame["run"] = path.stem
        if "source_log" not in frame.columns:
            frame["source_log"] = str(path.resolve())
        frames.append(frame)

    df = pd.concat(frames, ignore_index=True, sort=False)
    required = {"run", "step"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if "lambda" not in df.columns:
        df["lambda"] = np.nan
    if "seed" not in df.columns:
        df["seed"] = np.nan

    df["step"] = pd.to_numeric(df["step"], errors="coerce")
    df["lambda"] = pd.to_numeric(df["lambda"], errors="coerce")
    df["seed"] = pd.to_numeric(df["seed"], errors="coerce")
    df = df.dropna(subset=["step"]).copy()
    df["step"] = df["step"].astype(int)

    # Convert all non-string identifier columns in one operation.  Bulk
    # conversion avoids fragmenting very wide metric DataFrames.
    numeric_columns = [
        column for column in df.columns if column not in {"run", "source_log"}
    ]
    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors="coerce")
    df = df.copy()

    source_component = df["source_log"].fillna("").astype(str)
    seed_component = df["seed"].map(lambda x: "nan" if pd.isna(x) else f"{x:g}")
    df["_run_id"] = (
        df["run"].astype(str)
        + "|seed="
        + seed_component
        + "|source="
        + source_component
    )

    # Avoid duplicate rows if the same CSV is accidentally supplied twice.
    df = df.drop_duplicates(subset=["_run_id", "step"], keep="last")
    df = df.sort_values(["lambda", "run", "seed", "step"], na_position="last")
    return df.reset_index(drop=True)


def add_derived_metrics(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    if AIME24_METRIC in result.columns and AIME25_METRIC in result.columns:
        # Require both values at a validation step; do not silently average one benchmark.
        result[DERIVED_VAL_MEAN] = result[[AIME24_METRIC, AIME25_METRIC]].mean(
            axis=1, skipna=False
        )
    return result


def select_metrics(df: pd.DataFrame, preset: str, explicit: list[str] | None) -> list[str]:
    if explicit:
        requested = explicit
    elif preset == "core":
        requested = CORE_METRICS
    elif preset == "performance":
        requested = PERFORMANCE_METRICS
    elif preset == "behavior":
        requested = BEHAVIOR_METRICS
    elif preset == "signal":
        requested = SIGNAL_METRICS
    elif preset == "geometry":
        requested = GEOMETRY_METRICS
    elif preset == "full":
        requested = unique_preserving_order(
            PERFORMANCE_METRICS + BEHAVIOR_METRICS + SIGNAL_METRICS + GEOMETRY_METRICS
        )
    elif preset == "all":
        requested = [
            column
            for column in df.columns
            if column not in ID_COLUMNS
            and pd.api.types.is_numeric_dtype(df[column])
            and not column.startswith("gopd/position/")
            and column not in {"training/global_step"}
        ]
    else:
        raise ValueError(f"Unknown preset: {preset}")

    missing = [metric for metric in requested if metric not in df.columns]
    if missing:
        print("[WARN] Metrics not present and skipped:")
        for metric in missing:
            print(f"       {metric}")

    selected = [
        metric
        for metric in requested
        if metric in df.columns and df[metric].notna().any()
    ]
    if not selected:
        raise ValueError("None of the requested metrics are present in the CSV.")
    return unique_preserving_order(selected)


def smooth_nonmissing_points(sub: pd.DataFrame, metric: str, rolling: int) -> pd.DataFrame:
    if rolling <= 1:
        return sub
    result_parts: list[pd.DataFrame] = []
    for _, run_frame in sub.groupby("_run_id", sort=False):
        run_frame = run_frame.sort_values("step").copy()
        run_frame[metric] = run_frame[metric].rolling(
            window=rolling, min_periods=1
        ).mean()
        result_parts.append(run_frame)
    return pd.concat(result_parts, ignore_index=True)


def grouping_mode(df: pd.DataFrame) -> str:
    return "lambda" if df["lambda"].notna().any() else "run"


def sorted_group_values(sub: pd.DataFrame, mode: str) -> list[Any]:
    if mode == "lambda":
        values = sub["lambda"].dropna().unique().tolist()
        return sorted(values, key=float)
    return sorted(sub["run"].dropna().astype(str).unique().tolist())


def group_mask(sub: pd.DataFrame, mode: str, value: Any) -> pd.Series:
    if mode == "lambda":
        return sub["lambda"] == value
    return sub["run"].astype(str) == str(value)


def group_label(value: Any, mode: str, n_runs: int) -> str:
    if mode == "lambda":
        label = f"λ={format_lambda(value)}"
    else:
        label = str(value)
    if n_runs > 1:
        label += f" (n={n_runs})"
    return label


def plot_metric(
    df: pd.DataFrame,
    metric: str,
    output_dir: Path,
    formats: list[str],
    dpi: int,
    rolling: int,
) -> list[Path]:
    sub = df[["_run_id", "run", "lambda", "step", metric]].dropna(subset=[metric]).copy()
    if sub.empty:
        return []
    sub = smooth_nonmissing_points(sub, metric, rolling)

    mode = grouping_mode(sub)
    fig, ax = plt.subplots(figsize=(8.2, 5.2))

    for value in sorted_group_values(sub, mode):
        group = sub[group_mask(sub, mode, value)].copy()
        if group.empty:
            continue

        # First average accidental duplicate values within a run/step, then aggregate seeds.
        per_run = (
            group.groupby(["_run_id", "step"], as_index=False)[metric]
            .mean()
            .sort_values("step")
        )
        aggregate = per_run.groupby("step")[metric].agg(["mean", "std", "count"]).reset_index()
        n_runs = per_run["_run_id"].nunique()
        marker = "o" if len(aggregate) <= 15 else None
        (line,) = ax.plot(
            aggregate["step"],
            aggregate["mean"],
            linewidth=1.8,
            marker=marker,
            markersize=4.2,
            label=group_label(value, mode, n_runs),
        )

        valid_std = aggregate["std"].notna() & (aggregate["count"] > 1)
        if valid_std.any():
            x = aggregate.loc[valid_std, "step"].to_numpy()
            mean = aggregate.loc[valid_std, "mean"].to_numpy()
            std = aggregate.loc[valid_std, "std"].to_numpy()
            ax.fill_between(
                x,
                mean - std,
                mean + std,
                alpha=0.18,
                color=line.get_color(),
                linewidth=0,
            )

    title = display_name(metric)
    if rolling > 1:
        title += f" (rolling={rolling})"
    ax.set_title(title)
    ax.set_xlabel("Training Step")
    ax.set_ylabel(display_name(metric))
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()

    metric_dir = output_dir / "time_series"
    metric_dir.mkdir(parents=True, exist_ok=True)
    base = sanitize_filename(metric)
    outputs: list[Path] = []
    for extension in formats:
        path = metric_dir / f"{base}.{extension}"
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        outputs.append(path)
    plt.close(fig)
    return outputs


def summarize_runs(df: pd.DataFrame, metrics: list[str], last_k: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for run_key, run_frame in df.groupby("_run_id", sort=False):
        run_frame = run_frame.sort_values("step")
        first_row = run_frame.iloc[0]
        identity = {
            "run_id": run_key,
            "run": first_row.get("run"),
            "lambda": first_row.get("lambda"),
            "seed": first_row.get("seed"),
            "source_log": first_row.get("source_log"),
        }
        for metric in metrics:
            series = run_frame[["step", metric]].dropna(subset=[metric])
            if series.empty:
                continue
            tail = series.tail(last_k)
            rows.append(
                {
                    **identity,
                    "metric": metric,
                    "n_points": len(series),
                    "first_step": int(series.iloc[0]["step"]),
                    "first_value": float(series.iloc[0][metric]),
                    "last_step": int(series.iloc[-1]["step"]),
                    "last_value": float(series.iloc[-1][metric]),
                    "minimum": float(series[metric].min()),
                    "maximum": float(series[metric].max()),
                    "mean_all_points": float(series[metric].mean()),
                    "last_k": min(last_k, len(series)),
                    "last_k_mean": float(tail[metric].mean()),
                    "last_k_std": float(tail[metric].std(ddof=1)) if len(tail) > 1 else math.nan,
                }
            )

    columns = [
        "run_id",
        "run",
        "lambda",
        "seed",
        "source_log",
        "metric",
        "n_points",
        "first_step",
        "first_value",
        "last_step",
        "last_value",
        "minimum",
        "maximum",
        "mean_all_points",
        "last_k",
        "last_k_mean",
        "last_k_std",
    ]
    return pd.DataFrame(rows, columns=columns)


def summarize_lambdas(run_summary: pd.DataFrame) -> pd.DataFrame:
    if run_summary.empty:
        return pd.DataFrame()

    # When lambda is unavailable, summarize by run name rather than dropping everything.
    if run_summary["lambda"].notna().any():
        group_columns = ["lambda", "metric"]
    else:
        group_columns = ["run", "metric"]

    grouped = run_summary.groupby(group_columns, dropna=False)
    summary = grouped.agg(
        num_runs=("run_id", "nunique"),
        last_step_min=("last_step", "min"),
        last_step_max=("last_step", "max"),
        last_value_mean=("last_value", "mean"),
        last_value_std=("last_value", "std"),
        last_k_mean_across_runs=("last_k_mean", "mean"),
        last_k_mean_std_across_runs=("last_k_mean", "std"),
        run_maximum_mean=("maximum", "mean"),
        run_minimum_mean=("minimum", "mean"),
    ).reset_index()
    return summary


def write_validation_points(df: pd.DataFrame, output_dir: Path) -> Path | None:
    metrics = [m for m in [AIME24_METRIC, AIME25_METRIC, DERIVED_VAL_MEAN] if m in df.columns]
    if not metrics:
        return None
    columns = ["run", "lambda", "seed", "source_log", "step"] + metrics
    validation = df[columns].dropna(subset=metrics, how="all").copy()
    if validation.empty:
        return None
    validation = validation.sort_values(["lambda", "run", "seed", "step"], na_position="last")
    path = output_dir / "validation_points.csv"
    validation.to_csv(path, index=False)
    return path


def discover_position_suffixes(df: pd.DataFrame) -> list[str]:
    suffixes: set[str] = set()
    pattern = re.compile(r"^gopd/position/(?:0_20|20_40|40_60|60_80|80_100)/(.+)$")
    for column in df.columns:
        match = pattern.match(column)
        if match:
            suffixes.add(match.group(1))
    ordered = [suffix for suffix in DEFAULT_POSITION_SUFFIXES if suffix in suffixes]
    ordered += sorted(suffixes - set(ordered))
    return ordered


def build_position_summary(
    df: pd.DataFrame,
    suffixes: list[str],
    last_k: int,
) -> pd.DataFrame:
    run_rows: list[dict[str, Any]] = []

    for run_id, run_frame in df.groupby("_run_id", sort=False):
        run_frame = run_frame.sort_values("step")
        identity = run_frame.iloc[0]
        for suffix in suffixes:
            for bin_name, midpoint, label in POSITION_BINS:
                column = f"gopd/position/{bin_name}/{suffix}"
                if column not in run_frame.columns:
                    continue
                values = run_frame[["step", column]].dropna(subset=[column])
                if values.empty:
                    continue
                tail = values.tail(last_k)
                run_rows.append(
                    {
                        "run_id": run_id,
                        "run": identity.get("run"),
                        "lambda": identity.get("lambda"),
                        "seed": identity.get("seed"),
                        "suffix": suffix,
                        "position_bin": bin_name,
                        "position_midpoint": midpoint,
                        "position_label": label,
                        "last_k": min(last_k, len(values)),
                        "run_tail_mean": float(tail[column].mean()),
                    }
                )

    run_summary = pd.DataFrame(run_rows)
    if run_summary.empty:
        return run_summary

    if run_summary["lambda"].notna().any():
        group_columns = [
            "lambda",
            "suffix",
            "position_bin",
            "position_midpoint",
            "position_label",
        ]
    else:
        group_columns = [
            "run",
            "suffix",
            "position_bin",
            "position_midpoint",
            "position_label",
        ]

    aggregate = (
        run_summary.groupby(group_columns, dropna=False)["run_tail_mean"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(
            columns={
                "mean": "mean_across_runs",
                "std": "std_across_runs",
                "count": "num_runs",
            }
        )
    )
    return aggregate


def plot_position_profiles(
    position_summary: pd.DataFrame,
    output_dir: Path,
    formats: list[str],
    dpi: int,
) -> list[tuple[str, Path]]:
    if position_summary.empty:
        return []

    mode = "lambda" if "lambda" in position_summary.columns else "run"
    outputs: list[tuple[str, Path]] = []
    profile_dir = output_dir / "position_profiles"
    profile_dir.mkdir(parents=True, exist_ok=True)

    for suffix, suffix_frame in position_summary.groupby("suffix", sort=False):
        fig, ax = plt.subplots(figsize=(8.2, 5.2))
        values = (
            sorted(suffix_frame["lambda"].dropna().unique(), key=float)
            if mode == "lambda"
            else sorted(suffix_frame["run"].dropna().astype(str).unique())
        )

        for value in values:
            if mode == "lambda":
                group = suffix_frame[suffix_frame["lambda"] == value].sort_values(
                    "position_midpoint"
                )
            else:
                group = suffix_frame[suffix_frame["run"].astype(str) == str(value)].sort_values(
                    "position_midpoint"
                )
            if group.empty:
                continue

            n_runs = int(group["num_runs"].max())
            (line,) = ax.plot(
                group["position_midpoint"],
                group["mean_across_runs"],
                linewidth=1.8,
                marker="o",
                markersize=4.5,
                label=group_label(value, mode, n_runs),
            )
            valid_std = group["std_across_runs"].notna() & (group["num_runs"] > 1)
            if valid_std.any():
                x = group.loc[valid_std, "position_midpoint"].to_numpy()
                mean = group.loc[valid_std, "mean_across_runs"].to_numpy()
                std = group.loc[valid_std, "std_across_runs"].to_numpy()
                ax.fill_between(
                    x,
                    mean - std,
                    mean + std,
                    alpha=0.18,
                    color=line.get_color(),
                    linewidth=0,
                )

        labels = [label for _, _, label in POSITION_BINS]
        midpoints = [midpoint for _, midpoint, _ in POSITION_BINS]
        ax.set_xticks(midpoints, labels)
        ax.set_xlabel("Relative Token Position")
        ax.set_ylabel(POSITION_DISPLAY_NAMES.get(suffix, suffix.replace("_", " ")))
        ax.set_title(POSITION_DISPLAY_NAMES.get(suffix, suffix.replace("_", " ")))
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False)
        fig.tight_layout()

        base = f"position_{sanitize_filename(suffix)}"
        for extension in formats:
            path = profile_dir / f"{base}.{extension}"
            fig.savefig(path, dpi=dpi, bbox_inches="tight")
            outputs.append((f"position/{suffix}", path))
        plt.close(fig)

    return outputs


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot lambda comparisons from G-OPD wide CSV files."
    )
    parser.add_argument(
        "csv_files",
        nargs="+",
        type=Path,
        help="One or more gopd_metrics_wide.csv files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("figures_gopd"),
        help="Output directory (default: figures_gopd).",
    )
    parser.add_argument(
        "--preset",
        choices=["core", "performance", "behavior", "signal", "geometry", "full", "all"],
        default="core",
        help="Metric preset used when --metrics is not supplied (default: core).",
    )
    parser.add_argument(
        "--metrics",
        nargs="*",
        default=None,
        help="Explicit metric names. Overrides --preset.",
    )
    parser.add_argument(
        "--rolling",
        type=int,
        default=1,
        help="Optional rolling mean over non-missing points within each run (default: 1/no smoothing).",
    )
    parser.add_argument(
        "--last-k",
        type=int,
        default=10,
        help="Number of latest available points used in summary tables and position profiles.",
    )
    parser.add_argument(
        "--include-position",
        action="store_true",
        help="Generate final-window token-position profile plots.",
    )
    parser.add_argument(
        "--position-metrics",
        nargs="*",
        default=None,
        help="Position metric suffixes; defaults to all suffixes found in the CSV.",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=["png", "pdf", "svg"],
        default=["png"],
        help="Figure formats (default: png).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="Raster output DPI (default: 200).",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.rolling < 1:
        raise SystemExit("--rolling must be at least 1")
    if args.last_k < 1:
        raise SystemExit("--last-k must be at least 1")

    df = add_derived_metrics(load_wide_csvs(args.csv_files))
    metrics = select_metrics(df, args.preset, args.metrics)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, str]] = []
    for metric in metrics:
        outputs = plot_metric(
            df=df,
            metric=metric,
            output_dir=args.output_dir,
            formats=args.formats,
            dpi=args.dpi,
            rolling=args.rolling,
        )
        for path in outputs:
            manifest_rows.append({"kind": "time_series", "metric": metric, "path": str(path)})
        print(f"[PLOT] {metric}")

    run_summary = summarize_runs(df, metrics, args.last_k)
    lambda_summary = summarize_lambdas(run_summary)
    run_summary_path = args.output_dir / "run_metric_summary.csv"
    lambda_summary_path = args.output_dir / "lambda_metric_summary.csv"
    run_summary.to_csv(run_summary_path, index=False)
    lambda_summary.to_csv(lambda_summary_path, index=False)
    print(f"[WRITE] {run_summary_path}")
    print(f"[WRITE] {lambda_summary_path}")

    validation_path = write_validation_points(df, args.output_dir)
    if validation_path is not None:
        print(f"[WRITE] {validation_path}")

    if args.include_position:
        available_suffixes = discover_position_suffixes(df)
        if args.position_metrics:
            requested = args.position_metrics
            missing = [suffix for suffix in requested if suffix not in available_suffixes]
            for suffix in missing:
                print(f"[WARN] Position metric not found and skipped: {suffix}")
            suffixes = [suffix for suffix in requested if suffix in available_suffixes]
        else:
            suffixes = available_suffixes

        if suffixes:
            position_summary = build_position_summary(df, suffixes, args.last_k)
            position_summary_path = args.output_dir / "position_profile_summary.csv"
            position_summary.to_csv(position_summary_path, index=False)
            print(f"[WRITE] {position_summary_path}")
            for metric, path in plot_position_profiles(
                position_summary,
                output_dir=args.output_dir,
                formats=args.formats,
                dpi=args.dpi,
            ):
                manifest_rows.append({"kind": "position_profile", "metric": metric, "path": str(path)})
                print(f"[PLOT] {metric}")
        else:
            print("[WARN] No position-level metrics were found.")

    manifest_path = args.output_dir / "plot_manifest.csv"
    pd.DataFrame(manifest_rows, columns=["kind", "metric", "path"]).to_csv(
        manifest_path, index=False
    )
    print(f"[WRITE] {manifest_path}")
    print(f"[DONE] Generated {len(manifest_rows)} figure files in {args.output_dir}")


if __name__ == "__main__":
    main()
