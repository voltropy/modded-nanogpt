#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import re
import statistics
import subprocess
import sys
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = ROOT / "train_gpt.py"
VAL_RE = re.compile(r"step:(\d+)/(\d+) val_loss:([0-9.]+) train_time:(\d+)ms")


@dataclass
class Variant:
    name: str
    extension_steps: int
    late_layers: int
    groups: set[str] = field(default_factory=set)


def build_variants(step_exts: list[int], layer_counts: list[int], sweeps: str) -> list[Variant]:
    variants: OrderedDict[tuple[int, int], Variant] = OrderedDict()

    def add(extension_steps: int, late_layers: int, group: str):
        key = (extension_steps, late_layers)
        if key not in variants:
            variants[key] = Variant(
                name=f"alpha_l{late_layers}_ext{extension_steps}",
                extension_steps=extension_steps,
                late_layers=late_layers,
            )
        variants[key].groups.add(group)

    if sweeps in ("all", "step"):
        for extension_steps in step_exts:
            add(extension_steps, 3, "step_retune")
    if sweeps in ("all", "layers"):
        for late_layers in layer_counts:
            add(40, late_layers, "layer_scope")
    return list(variants.values())


def parse_final_metrics(log_path: Path) -> tuple[float, int]:
    matches = VAL_RE.findall(log_path.read_text())
    if not matches:
        raise RuntimeError(f"could not find final val_loss/train_time in {log_path}")
    _, _, val_loss, train_time_ms = matches[-1]
    return float(val_loss), int(train_time_ms)


def stream_run(cmd: list[str], env: dict[str, str], stdout_path: Path) -> None:
    with stdout_path.open("w") as stdout_file:
        proc = subprocess.Popen(
            cmd,
            cwd=ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            stdout_file.write(line)
        rc = proc.wait()
    if rc != 0:
        raise subprocess.CalledProcessError(rc, cmd)


def format_ms_delta(delta_ms: int | None) -> str:
    if delta_ms is None:
        return "-"
    sign = "+" if delta_ms >= 0 else ""
    return f"{sign}{delta_ms:,}ms"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run alpha-only late-attnres step and layer sweeps.")
    parser.add_argument("--reps", type=int, default=2, help="Repetitions per unique config.")
    parser.add_argument("--nproc-per-node", type=int, default=8, help="torchrun --nproc_per_node value.")
    parser.add_argument("--torchrun-bin", default="torchrun", help="Path to torchrun binary.")
    parser.add_argument(
        "--step-exts",
        type=int,
        nargs="+",
        default=[40, 32, 30, 28, 25],
        help="Extension-step sweep for alpha_only with 3 late layers.",
    )
    parser.add_argument(
        "--layer-counts",
        type=int,
        nargs="+",
        default=[1, 2, 3],
        help="Late-layer-count sweep for alpha_only at 40 extension steps.",
    )
    parser.add_argument(
        "--sweeps",
        choices=("all", "step", "layers"),
        default="all",
        help="Which sweep groups to run.",
    )
    parser.add_argument(
        "--baseline-ms",
        type=int,
        default=None,
        help="Optional baseline train_time in ms for delta reporting.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned runs without executing them.",
    )
    args = parser.parse_args()

    if args.reps < 1:
        raise SystemExit("--reps must be >= 1")
    if any(ext <= 0 for ext in args.step_exts):
        raise SystemExit("--step-exts values must be > 0")
    if any(layer not in (1, 2, 3) for layer in args.layer_counts):
        raise SystemExit("--layer-counts values must be in {1,2,3}")

    variants = build_variants(args.step_exts, args.layer_counts, args.sweeps)
    sweep_id = time.strftime("alpha_sweeps_%Y%m%d_%H%M%S")
    sweep_dir = ROOT / "logs" / "sweeps" / sweep_id
    sweep_dir.mkdir(parents=True, exist_ok=True)

    print(f"sweep_dir: {sweep_dir}")
    print(f"variants: {len(variants)} unique configs, reps={args.reps}")
    for variant in variants:
        groups = ",".join(sorted(variant.groups))
        print(f"  {variant.name}: groups={groups} layers={variant.late_layers} ext={variant.extension_steps}")
    if args.dry_run:
        return 0

    all_rows: list[dict[str, object]] = []
    for variant in variants:
        for rep in range(1, args.reps + 1):
            run_id = f"{sweep_id}_{variant.name}_r{rep}"
            stdout_path = sweep_dir / f"{run_id}.stdout.txt"
            print(f"\n=== {variant.name} rep {rep}/{args.reps} ===")
            env = os.environ.copy()
            env.update(
                {
                    "RUN_ID": run_id,
                    "LATE_ATTNRES_MODE": "alpha_only",
                    "LATE_ATTNRES_LAYERS": str(variant.late_layers),
                    "NUM_EXTENSION_ITERATIONS": str(variant.extension_steps),
                }
            )
            cmd = [
                args.torchrun_bin,
                "--standalone",
                f"--nproc_per_node={args.nproc_per_node}",
                str(TRAIN_SCRIPT),
            ]
            print("cmd:", " ".join(cmd))
            stream_run(cmd, env, stdout_path)

            log_path = ROOT / "logs" / f"{run_id}.txt"
            val_loss, train_time_ms = parse_final_metrics(log_path)
            print(
                f"result: variant={variant.name} rep={rep} "
                f"val_loss={val_loss:.4f} train_time={train_time_ms:,}ms"
            )
            all_rows.append(
                {
                    "variant": variant.name,
                    "groups": ",".join(sorted(variant.groups)),
                    "rep": rep,
                    "late_layers": variant.late_layers,
                    "extension_steps": variant.extension_steps,
                    "val_loss": val_loss,
                    "train_time_ms": train_time_ms,
                    "run_id": run_id,
                    "log_path": str(log_path),
                    "stdout_path": str(stdout_path),
                }
            )

    results_csv = sweep_dir / "results.csv"
    with results_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)

    summary_rows = []
    for variant in variants:
        rows = [row for row in all_rows if row["variant"] == variant.name]
        loss_mean = statistics.mean(float(row["val_loss"]) for row in rows)
        time_mean = round(statistics.mean(int(row["train_time_ms"]) for row in rows))
        summary_rows.append(
            {
                "variant": variant.name,
                "groups": ",".join(sorted(variant.groups)),
                "reps": len(rows),
                "late_layers": variant.late_layers,
                "extension_steps": variant.extension_steps,
                "loss_mean": loss_mean,
                "time_mean_ms": time_mean,
                "vs_baseline_ms": None if args.baseline_ms is None else time_mean - args.baseline_ms,
            }
        )

    summary_rows.sort(key=lambda row: (row["groups"], row["time_mean_ms"], row["loss_mean"]))

    summary_csv = sweep_dir / "summary.csv"
    with summary_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    headers = ["variant", "groups", "reps", "layers", "ext", "loss_mean", "time_mean", "vs_baseline"]
    table = [headers]
    for row in summary_rows:
        table.append(
            [
                row["variant"],
                row["groups"],
                str(row["reps"]),
                str(row["late_layers"]),
                str(row["extension_steps"]),
                f"{row['loss_mean']:.4f}",
                f"{row['time_mean_ms']:,}ms",
                format_ms_delta(row["vs_baseline_ms"]),
            ]
        )
    widths = [max(len(r[i]) for r in table) for i in range(len(headers))]

    print("\nSummary")
    for idx, row in enumerate(table):
        line = "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row))
        print(line)
        if idx == 0:
            print("  ".join("-" * width for width in widths))

    print(f"\nresults_csv: {results_csv}")
    print(f"summary_csv: {summary_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
