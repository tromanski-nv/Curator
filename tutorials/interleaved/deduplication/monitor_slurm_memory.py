# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Log Slurm RSS samples and render a live terminal graph."""

import argparse
import csv
import re
import shutil
import subprocess
import sys
import time
from collections import deque
from datetime import datetime
from functools import cache
from pathlib import Path

MEMORY_RE = re.compile(r"^([0-9]+(?:\.[0-9]+)?)([KMGTPE]?)B?$", re.IGNORECASE)
MIB_PER_GIB = 1024
EXPECTED_SSTAT_FIELDS = 4
MAX_MISSING_SAMPLES = 3
UNIT_TO_MIB = {
    "": 1 / (1024 * 1024),
    "K": 1 / 1024,
    "M": 1,
    "G": 1024,
    "T": 1024**2,
    "P": 1024**3,
    "E": 1024**4,
}
SPARK_LEVELS = "▁▂▃▄▅▆▇█"


@cache
def command_path(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        msg = f"Required command {name!r} was not found on PATH"
        raise RuntimeError(msg)
    return path


def parse_memory_mib(value: str) -> float:
    value = value.strip()
    if not value or value == "0":
        return 0.0
    match = MEMORY_RE.fullmatch(value)
    if match is None:
        msg = f"Unsupported Slurm memory value: {value!r}"
        raise ValueError(msg)
    magnitude, unit = match.groups()
    return float(magnitude) * UNIT_TO_MIB[unit.upper()]


def format_memory(mib: float) -> str:
    if mib >= MIB_PER_GIB**2:
        return f"{mib / (MIB_PER_GIB**2):.2f} TiB"
    if mib >= MIB_PER_GIB:
        return f"{mib / MIB_PER_GIB:.2f} GiB"
    if mib >= 1:
        return f"{mib:.1f} MiB"
    return f"{mib * 1024:.1f} KiB"


def query_memory(job_id: str) -> dict[str, str] | None:
    result = subprocess.run(  # noqa: S603
        [
            command_path("sstat"),
            "-j",
            job_id,
            "--allsteps",
            "--noheader",
            "--parsable2",
            "--format=JobID,AveCPU,MaxRSS,AveRSS",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "sstat failed")

    rows = []
    for line in result.stdout.splitlines():
        fields = line.strip().split("|")
        if len(fields) < EXPECTED_SSTAT_FIELDS:
            continue
        step_id, ave_cpu, max_rss, ave_rss = fields[:4]
        if step_id.endswith(".extern"):
            continue
        rows.append(
            {
                "step_id": step_id,
                "ave_cpu": ave_cpu,
                "max_rss": max_rss,
                "ave_rss": ave_rss,
            },
        )
    if not rows:
        return None

    preferred_step = f"{job_id}.0"
    return next(
        (row for row in rows if row["step_id"] == preferred_step),
        max(rows, key=lambda row: parse_memory_mib(row["max_rss"])),
    )


def query_state(job_id: str) -> str:
    result = subprocess.run(  # noqa: S603
        [command_path("squeue"), "-h", "-j", job_id, "-o", "%T"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() or "NOT_IN_QUEUE"


def sparkline(values: deque[float]) -> str:
    if not values:
        return ""
    low = min(values)
    high = max(values)
    if high == low:
        return SPARK_LEVELS[len(SPARK_LEVELS) // 2] * len(values)
    return "".join(
        SPARK_LEVELS[min(len(SPARK_LEVELS) - 1, int((value - low) / (high - low) * len(SPARK_LEVELS)))]
        for value in values
    )


def usage_bar(current_mib: float, limit_mib: float | None, width: int = 50) -> tuple[str, str]:
    if limit_mib is None or limit_mib <= 0:
        return "", ""
    fraction = max(0.0, current_mib / limit_mib)
    filled = min(width, round(fraction * width))
    bar = "█" * filled + "░" * (width - filled)
    return bar, f"{fraction * 100:.1f}% of {format_memory(limit_mib)}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job_id", help="Slurm job ID, for example 433198")
    parser.add_argument("--interval", type=float, default=5.0, help="Sampling interval in seconds")
    parser.add_argument("--history", type=int, default=60, help="Number of samples shown in the graph")
    parser.add_argument("--limit-gb", type=float, help="Allocated memory in GiB, used for the utilization bar")
    parser.add_argument("--log", help="CSV output path; defaults to slurm-memory-<job-id>.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.interval <= 0 or args.history <= 0:
        msg = "--interval and --history must be positive"
        raise ValueError(msg)

    log_path = Path(args.log or f"slurm-memory-{args.job_id}.csv")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    limit_mib = args.limit_gb * MIB_PER_GIB if args.limit_gb is not None else None
    history: deque[float] = deque(maxlen=args.history)
    write_header = not log_path.exists() or log_path.stat().st_size == 0
    missing_samples = 0

    with log_path.open("a", newline="", buffering=1) as log_file:
        writer = csv.writer(log_file)
        if write_header:
            writer.writerow(
                [
                    "timestamp",
                    "job_id",
                    "step_id",
                    "state",
                    "ave_cpu",
                    "ave_rss_mib",
                    "max_rss_mib",
                    "allocation_mib",
                    "allocation_percent",
                ],
            )

        try:
            while True:
                sample_time = datetime.now().astimezone().isoformat(timespec="seconds")
                sample = query_memory(args.job_id)
                state = query_state(args.job_id)
                if sample is None:
                    missing_samples += 1
                    print(f"\rWaiting for sstat data for job {args.job_id} ({state})...", end="", flush=True)
                    if missing_samples >= MAX_MISSING_SAMPLES and state == "NOT_IN_QUEUE":
                        print("\nJob is no longer queued and no accounting samples remain.")
                        break
                    time.sleep(args.interval)
                    continue

                missing_samples = 0
                ave_rss_mib = parse_memory_mib(sample["ave_rss"])
                max_rss_mib = parse_memory_mib(sample["max_rss"])
                history.append(ave_rss_mib)
                allocation_percent = ave_rss_mib / limit_mib * 100 if limit_mib is not None and limit_mib > 0 else None
                writer.writerow(
                    [
                        sample_time,
                        args.job_id,
                        sample["step_id"],
                        state,
                        sample["ave_cpu"],
                        f"{ave_rss_mib:.3f}",
                        f"{max_rss_mib:.3f}",
                        f"{limit_mib:.3f}" if limit_mib is not None else "",
                        f"{allocation_percent:.3f}" if allocation_percent is not None else "",
                    ],
                )

                bar, bar_label = usage_bar(ave_rss_mib, limit_mib)
                if sys.stdout.isatty():
                    print("\033[2J\033[H", end="")
                print(f"Slurm job {args.job_id}  step {sample['step_id']}  state {state}")
                print(f"Sampled: {sample_time}  interval: {args.interval:g}s  AveCPU: {sample['ave_cpu']}")
                print(f"Current RSS: {format_memory(ave_rss_mib)}  High-water RSS: {format_memory(max_rss_mib)}")
                if bar:
                    print(f"[{bar}] {bar_label}")
                print(
                    f"History ({len(history)} samples; {format_memory(min(history))} - {format_memory(max(history))}):",
                )
                print(sparkline(history))
                print(f"CSV log: {log_path.resolve()}")
                print("Press Ctrl-C to stop monitoring; the Slurm job will continue.")
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\nMonitoring stopped; the Slurm job was not interrupted.")


if __name__ == "__main__":
    main()
