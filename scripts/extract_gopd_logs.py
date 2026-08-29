#!/usr/bin/env python3
"""Extract VERL / G-OPD console logs into analysis-friendly CSV files.

The parser is intentionally metric-agnostic: every numeric ``key:value`` pair
found on a VERL line of the form

    step:12 - metric/a:1.23 - metric/b:-4.5e-3

is retained automatically. This makes it robust to future additions to the
G-OPD diagnostics without requiring changes to the parser.

Outputs
-------
1. gopd_metrics_wide.csv
   One row per (run, step), with one column per metric.
2. gopd_metrics_long.csv
   Tidy format: one row per (run, step, metric).
3. gopd_run_metadata.csv
   Run-level configuration recovered from the shell trace at the top of each log.

Example
-------
python scripts/extract_gopd_logs.py logs/ --output-dir metrics

python scripts/extract_gopd_logs.py \
    gopd_qwen3_4b_lam0.5.log \
    gopd_qwen3_4b_lam1.25.log \
    gopd_qwen3_4b_lam2.log \
    --output-dir metrics
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
RAY_PREFIX_RE = re.compile(r"^\s*\([^)]*(?:pid=\d+|raylet)[^)]*\)\s*")

# Supports integers, decimals, scientific notation, inf, and nan.
FLOAT_RE = re.compile(
    r"^[+-]?(?:"
    r"(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
    r"|inf|nan"
    r")$",
    flags=re.IGNORECASE,
)

STEP_RE = re.compile(r"\bstep\s*:\s*(\d+)\s+-\s+(.*)$")
LAMBDA_IN_NAME_RE = re.compile(
    r"(?:^|[_\-.])lam(?:bda)?(?:[_=]?)"
    r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)",
    flags=re.IGNORECASE,
)
SEED_IN_NAME_RE = re.compile(r"(?:^|[_\-.])seed[_=]?(\d+)", flags=re.IGNORECASE)

ID_COLUMNS = ["run", "lambda", "seed", "source_log", "step"]

SHELL_TO_METADATA = {
    "GOPD_LAM": "lambda",
    "PYTHONHASHSEED": "seed",
    "EXPERIMENT_NAME": "experiment_name",
    "PROJECT_NAME": "project_name",
    "STUDENT_MODEL": "student_model",
    "TEACHER_MODEL": "teacher_model",
    "GOPD_REF_MODEL": "gopd_ref_model",
    "TRAIN_FILE": "train_file",
    "VAL_FILES": "val_files",
    "TRAINING_STEPS": "training_steps",
    "TRAIN_BATCH_SIZE": "train_batch_size",
    "PPO_MINI_BATCH_SIZE": "ppo_mini_batch_size",
    "PPO_MICRO_BATCH_SIZE_PER_GPU": "ppo_micro_batch_size_per_gpu",
    "MAX_PROMPT_LENGTH": "max_prompt_length",
    "MAX_RESPONSE_LENGTH": "max_response_length",
    "ACTOR_LR": "actor_lr",
    "ROLLOUT_N": "rollout_n",
    "ROLLOUT_TEMPERATURE": "rollout_temperature",
    "ROLLOUT_TOP_P": "rollout_top_p",
    "TEST_FREQ": "test_freq",
    "SAVE_FREQ": "save_freq",
    "NUM_GPUS": "num_gpus",
    "ROLLOUT_TP_SIZE": "rollout_tp_size",
}

NUMERIC_METADATA = {
    "lambda",
    "seed",
    "training_steps",
    "train_batch_size",
    "ppo_mini_batch_size",
    "ppo_micro_batch_size_per_gpu",
    "max_prompt_length",
    "max_response_length",
    "actor_lr",
    "rollout_n",
    "rollout_temperature",
    "rollout_top_p",
    "test_freq",
    "save_freq",
    "num_gpus",
    "rollout_tp_size",
}


def strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text)


def normalize_log_text(text: str) -> str:
    """Remove ANSI color codes and Ray worker prefixes, preserving line breaks."""
    normalized_lines: list[str] = []
    for raw_line in strip_ansi(text).splitlines():
        line = RAY_PREFIX_RE.sub("", raw_line)
        normalized_lines.append(line)
    return "\n".join(normalized_lines)


def parse_float(value: str) -> float | None:
    value = value.strip()
    if not FLOAT_RE.fullmatch(value):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def clean_shell_value(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        value = value[1:-1]
    return value.strip()


def collect_shell_assignments(text: str) -> dict[str, str]:
    """Collect Bash ``set -x`` assignments such as ``+ GOPD_LAM=1.25``."""
    assignments: dict[str, str] = {}
    pattern = re.compile(r"^\+\s+([A-Z][A-Z0-9_]*)=(.*)$", flags=re.MULTILINE)
    for match in pattern.finditer(text):
        key, value = match.groups()
        assignments[key] = clean_shell_value(value)
    return assignments


def first_regex_group(text: str, patterns: Iterable[str]) -> str | None:
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.MULTILINE)
        if match:
            return match.group(1).strip()
    return None


def infer_metadata(text: str, log_path: Path) -> dict[str, Any]:
    normalized = normalize_log_text(text)
    shell = collect_shell_assignments(normalized)

    metadata: dict[str, Any] = {
        "source_log": str(log_path.resolve()),
        "source_file": log_path.name,
    }

    for shell_key, output_key in SHELL_TO_METADATA.items():
        if shell_key in shell:
            metadata[output_key] = shell[shell_key]

    # Fallbacks for logs that do not include the shell trace.
    if "lambda" not in metadata:
        raw = first_regex_group(
            normalized,
            [
                r"algorithm\.gopd\.lam=([^\s]+)",
                r"['\"]lam['\"]\s*:\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))",
            ],
        )
        if raw is None:
            name_match = LAMBDA_IN_NAME_RE.search(log_path.stem)
            raw = name_match.group(1) if name_match else None
        if raw is not None:
            metadata["lambda"] = raw

    if "seed" not in metadata:
        raw = first_regex_group(normalized, [r"(?:\+)?data\.seed=([^\s]+)"])
        if raw is None:
            name_match = SEED_IN_NAME_RE.search(log_path.stem)
            raw = name_match.group(1) if name_match else None
        if raw is not None:
            metadata["seed"] = raw

    if "experiment_name" not in metadata:
        raw = first_regex_group(
            normalized,
            [
                r"trainer\.experiment_name=([^\s]+)",
                r"['\"]experiment_name['\"]\s*:\s*['\"]([^'\"]+)['\"]",
            ],
        )
        if raw:
            metadata["experiment_name"] = raw

    if "training_steps" not in metadata:
        raw = first_regex_group(
            normalized,
            [
                r"trainer\.total_training_steps=([^\s]+)",
                r"Total training steps:\s*(\d+)",
            ],
        )
        if raw is not None:
            metadata["training_steps"] = raw

    if "max_response_length" not in metadata:
        raw = first_regex_group(normalized, [r"data\.max_response_length=([^\s]+)"])
        if raw is not None:
            metadata["max_response_length"] = raw

    # Convert known numeric fields while retaining missing values as NaN later.
    for key in NUMERIC_METADATA:
        if key not in metadata:
            continue
        parsed = parse_float(str(metadata[key]))
        if parsed is not None:
            if key in {
                "seed",
                "training_steps",
                "train_batch_size",
                "ppo_mini_batch_size",
                "ppo_micro_batch_size_per_gpu",
                "max_prompt_length",
                "max_response_length",
                "rollout_n",
                "test_freq",
                "save_freq",
                "num_gpus",
                "rollout_tp_size",
            } and math.isfinite(parsed):
                metadata[key] = int(parsed)
            else:
                metadata[key] = parsed

    run_name = metadata.get("experiment_name") or log_path.stem
    metadata["run"] = str(run_name)
    return metadata


def parse_metric_segments(body: str) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for segment in re.split(r"\s+-\s+", body.strip()):
        if ":" not in segment:
            continue
        key, raw_value = segment.rsplit(":", 1)
        key = key.strip()
        value = parse_float(raw_value)
        if key and value is not None:
            metrics[key] = value
    return metrics


def parse_exact_validation_metrics(text: str, label: str) -> dict[str, float]:
    """Recover full-precision metrics from Initial/Final validation dictionaries.

    Console ``step:`` lines are formatted to three decimals. VERL also prints an
    initial and final Python dictionary with full precision, split across quoted
    lines. This helper recovers those exact endpoint values when available.
    """
    normalized = normalize_log_text(text)
    marker = f"{label} validation metrics:"
    start = normalized.find(marker)
    if start < 0:
        return {}

    # The dictionary is short; limiting the window avoids accidentally matching
    # unrelated quoted configuration values later in the log.
    segment = normalized[start : start + 3000]
    segment = segment.replace('"', " ").replace("\n", " ")

    pair_re = re.compile(
        r"'([^']+)'\s*:\s*"
        r"([+-]?(?:(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?|inf|nan))",
        flags=re.IGNORECASE,
    )
    result: dict[str, float] = {}
    for key, raw_value in pair_re.findall(segment):
        value = parse_float(raw_value)
        if value is not None:
            result[key] = value
    return result


def parse_log(log_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any], set[str]]:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    metadata = infer_metadata(text, log_path)
    normalized = normalize_log_text(text)

    records_by_step: dict[int, dict[str, float]] = {}
    duplicate_step_lines = 0

    for line in normalized.splitlines():
        match = STEP_RE.search(line)
        if not match:
            continue
        step = int(match.group(1))
        metrics = parse_metric_segments(match.group(2))
        if not metrics:
            continue
        if step in records_by_step:
            duplicate_step_lines += 1
        records_by_step.setdefault(step, {}).update(metrics)

    if not records_by_step:
        raise ValueError(f"No VERL 'step:<n> - key:value' metric lines found in {log_path}")

    # Replace rounded endpoint validation values with full-precision dictionary values.
    min_step = min(records_by_step)
    max_step = max(records_by_step)
    initial_metrics = parse_exact_validation_metrics(text, "Initial")
    final_metrics = parse_exact_validation_metrics(text, "Final")
    if initial_metrics:
        records_by_step.setdefault(min_step, {}).update(initial_metrics)
    if final_metrics:
        records_by_step.setdefault(max_step, {}).update(final_metrics)

    metric_names: set[str] = set()
    rows: list[dict[str, Any]] = []
    for step in sorted(records_by_step):
        metrics = records_by_step[step]
        metric_names.update(metrics)
        row: dict[str, Any] = {
            "run": metadata["run"],
            "lambda": metadata.get("lambda", math.nan),
            "seed": metadata.get("seed", math.nan),
            "source_log": metadata["source_log"],
            "step": step,
        }
        row.update(metrics)
        rows.append(row)

    metadata.update(
        {
            "parsed_steps": len(records_by_step),
            "min_step": min_step,
            "max_step": max_step,
            "num_metrics": len(metric_names),
            "duplicate_step_lines_merged": duplicate_step_lines,
            "has_exact_initial_validation": bool(initial_metrics),
            "has_exact_final_validation": bool(final_metrics),
        }
    )
    return rows, metadata, metric_names


def metric_sort_key(metric: str) -> tuple[int, str]:
    prefix_order = [
        "val-core/",
        "critic/score/",
        "critic/rewards/",
        "response_length/",
        "prompt_length/",
        "gopd/adv_",
        "gopd/opd_adv_",
        "gopd/extra_adv_",
        "gopd/extra_term_",
        "gopd/reinforce_ratio",
        "gopd/conflict_",
        "gopd/flip_ratio",
        "gopd/extra_contribution_",
        "gopd/grad/",
        "gopd/position/",
        "gopd/",
        "actor/",
        "training/",
        "global_seqlen/",
        "perf/",
        "timing_s/",
        "timing_per_token_ms/",
        "critic/",
    ]
    for index, prefix in enumerate(prefix_order):
        if metric.startswith(prefix):
            return index, metric
    return len(prefix_order), metric


def expand_inputs(inputs: list[Path], pattern: str, recursive: bool) -> list[Path]:
    paths: list[Path] = []
    for item in inputs:
        if item.is_file():
            paths.append(item)
        elif item.is_dir():
            iterator = item.rglob(pattern) if recursive else item.glob(pattern)
            paths.extend(path for path in iterator if path.is_file())
        else:
            print(f"[WARN] Input does not exist and will be skipped: {item}", file=sys.stderr)

    # Resolve and deduplicate while keeping deterministic order.
    unique = sorted({path.resolve() for path in paths}, key=lambda p: str(p))
    return unique


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract one or more VERL G-OPD training logs into wide and long CSV files."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="Log files and/or directories containing log files.",
    )
    parser.add_argument(
        "--glob",
        default="*.log",
        help="Filename pattern used for directory inputs (default: *.log).",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search directory inputs recursively.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("parsed_gopd"),
        help="Directory for generated CSV files (default: parsed_gopd).",
    )
    parser.add_argument(
        "--wide-name",
        default="gopd_metrics_wide.csv",
        help="Wide CSV filename.",
    )
    parser.add_argument(
        "--long-name",
        default="gopd_metrics_long.csv",
        help="Long/tidy CSV filename.",
    )
    parser.add_argument(
        "--metadata-name",
        default="gopd_run_metadata.csv",
        help="Run metadata CSV filename.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    log_paths = expand_inputs(args.inputs, args.glob, args.recursive)
    if not log_paths:
        raise SystemExit("No log files found.")

    all_rows: list[dict[str, Any]] = []
    metadata_rows: list[dict[str, Any]] = []
    all_metric_names: set[str] = set()

    for log_path in log_paths:
        try:
            rows, metadata, metric_names = parse_log(log_path)
        except Exception as exc:  # noqa: BLE001 - include filename in batch parser diagnostics
            print(f"[ERROR] Failed to parse {log_path}: {exc}", file=sys.stderr)
            continue

        all_rows.extend(rows)
        metadata_rows.append(metadata)
        all_metric_names.update(metric_names)
        print(
            f"[OK] {log_path.name}: "
            f"lambda={metadata.get('lambda', 'unknown')}, "
            f"steps={metadata['min_step']}..{metadata['max_step']} "
            f"({metadata['parsed_steps']} rows), metrics={metadata['num_metrics']}"
        )

    if not all_rows:
        raise SystemExit("No logs were parsed successfully.")

    metric_columns = sorted(all_metric_names, key=metric_sort_key)
    wide_df = pd.DataFrame(all_rows)
    for column in ID_COLUMNS + metric_columns:
        if column not in wide_df.columns:
            wide_df[column] = math.nan
    wide_df = wide_df[ID_COLUMNS + metric_columns]
    wide_df["lambda"] = pd.to_numeric(wide_df["lambda"], errors="coerce")
    wide_df["seed"] = pd.to_numeric(wide_df["seed"], errors="coerce")
    wide_df["step"] = pd.to_numeric(wide_df["step"], errors="raise").astype(int)
    wide_df = wide_df.sort_values(
        ["lambda", "run", "seed", "step"], na_position="last"
    ).reset_index(drop=True)

    long_df = wide_df.melt(
        id_vars=ID_COLUMNS,
        value_vars=metric_columns,
        var_name="metric",
        value_name="value",
    ).dropna(subset=["value"])
    long_df = long_df.sort_values(
        ["lambda", "run", "seed", "metric", "step"], na_position="last"
    ).reset_index(drop=True)

    metadata_df = pd.DataFrame(metadata_rows)
    preferred_metadata_order = [
        "run",
        "lambda",
        "seed",
        "experiment_name",
        "project_name",
        "student_model",
        "teacher_model",
        "gopd_ref_model",
        "train_file",
        "val_files",
        "training_steps",
        "train_batch_size",
        "ppo_mini_batch_size",
        "ppo_micro_batch_size_per_gpu",
        "max_prompt_length",
        "max_response_length",
        "actor_lr",
        "rollout_n",
        "rollout_temperature",
        "rollout_top_p",
        "test_freq",
        "save_freq",
        "num_gpus",
        "rollout_tp_size",
        "parsed_steps",
        "min_step",
        "max_step",
        "num_metrics",
        "duplicate_step_lines_merged",
        "has_exact_initial_validation",
        "has_exact_final_validation",
        "source_file",
        "source_log",
    ]
    ordered_metadata = [c for c in preferred_metadata_order if c in metadata_df.columns]
    ordered_metadata += [c for c in metadata_df.columns if c not in ordered_metadata]
    metadata_df = metadata_df[ordered_metadata]
    if "lambda" in metadata_df.columns:
        metadata_df["lambda"] = pd.to_numeric(metadata_df["lambda"], errors="coerce")
    metadata_df = metadata_df.sort_values(
        [c for c in ["lambda", "run", "seed"] if c in metadata_df.columns],
        na_position="last",
    ).reset_index(drop=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    wide_path = args.output_dir / args.wide_name
    long_path = args.output_dir / args.long_name
    metadata_path = args.output_dir / args.metadata_name

    wide_df.to_csv(wide_path, index=False)
    long_df.to_csv(long_path, index=False)
    metadata_df.to_csv(metadata_path, index=False)

    print(f"[WRITE] {wide_path}  ({len(wide_df)} rows, {len(wide_df.columns)} columns)")
    print(f"[WRITE] {long_path}  ({len(long_df)} rows)")
    print(f"[WRITE] {metadata_path}  ({len(metadata_df)} runs)")


if __name__ == "__main__":
    main()
