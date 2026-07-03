#!/usr/bin/env python3
"""Summarize per-task success rates from LIBERO evaluation logs."""

from __future__ import annotations

import csv
import re
from collections import OrderedDict
from pathlib import Path


ROOT = Path(__file__).resolve().parent

GROUP_LOGS = {
    "origin": {
        "10": ROOT / "origin" / "10.log",
        "goal": ROOT / "origin" / "goal.log",
        "object": ROOT / "origin" / "object.log",
        "spatial": ROOT / "origin" / "spatial.log",
    },
    "sam": {
        "10": ROOT / "sam" / "10_sam.log",
        "goal": ROOT / "sam" / "goal_sam.log",
        "object": ROOT / "sam" / "object_sam.log",
        "spatial": ROOT / "sam" / "spatial_sam.log",
    },
    "sam_lora": {
        "10": ROOT / "sam_lora" / "10_3.log",
        "goal": ROOT / "sam_lora" / "goal_2.log",
        "object": ROOT / "sam_lora" / "object_2.log",
        "spatial": ROOT / "sam_lora" / "spatial.log",
    },
}

DATASETS = ("10", "goal", "object", "spatial")
GROUPS = ("origin", "sam", "sam_lora")
ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")


def clean_line(line: str) -> str:
    return ANSI_RE.sub("", line).strip()


def parse_log(path: Path) -> OrderedDict[str, dict[str, int]]:
    tasks: OrderedDict[str, dict[str, int]] = OrderedDict()
    current_task: str | None = None

    with path.open("r", encoding="utf-8", errors="replace") as f:
        for raw_line in f:
            line = clean_line(raw_line)
            if "Task:" in line:
                current_task = line.split("Task:", 1)[1].strip()
                tasks.setdefault(current_task, {"successes": 0, "episodes": 0})
            elif "Success:" in line and current_task is not None:
                success_text = line.split("Success:", 1)[1].strip().split()[0]
                tasks.setdefault(current_task, {"successes": 0, "episodes": 0})
                tasks[current_task]["episodes"] += 1
                if success_text == "True":
                    tasks[current_task]["successes"] += 1

    return tasks


def rate_cell(stats: dict[str, int] | None) -> str:
    if not stats or stats["episodes"] == 0:
        return ""
    rate = stats["successes"] / stats["episodes"] * 100
    return f"{rate:.1f}% ({stats['successes']}/{stats['episodes']})"


def collect_rows() -> list[dict[str, str]]:
    parsed = {
        group: {dataset: parse_log(path) for dataset, path in logs.items()}
        for group, logs in GROUP_LOGS.items()
    }

    rows: list[dict[str, str]] = []
    for dataset in DATASETS:
        descriptions: list[str] = []
        seen = set()
        for group in GROUPS:
            for description in parsed[group][dataset]:
                if description not in seen:
                    seen.add(description)
                    descriptions.append(description)

        for task_id, description in enumerate(descriptions):
            row = {
                "dataset": dataset,
                "task_id": str(task_id),
                "task_description": description,
            }
            for group in GROUPS:
                row[group] = rate_cell(parsed[group][dataset].get(description))
            rows.append(row)

    return rows


def write_csv(rows: list[dict[str, str]], path: Path) -> None:
    fieldnames = ["dataset", "task_id", "task_description", *GROUPS]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows: list[dict[str, str]], path: Path) -> None:
    lines = [
        "# Task-level Success Rates",
        "",
        "Rates are formatted as `success_rate (successes/episodes)`.",
        "",
    ]

    for dataset in DATASETS:
        lines.extend(
            [
                f"## {dataset}",
                "",
                "| task_id | task_description | origin | sam | sam_lora |",
                "|---:|---|---:|---:|---:|",
            ]
        )
        for row in rows:
            if row["dataset"] != dataset:
                continue
            description = row["task_description"].replace("|", "\\|")
            lines.append(
                f"| {row['task_id']} | {description} | {row['origin']} | {row['sam']} | {row['sam_lora']} |"
            )
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    rows = collect_rows()
    write_csv(rows, ROOT / "task_success_rates.csv")
    write_markdown(rows, ROOT / "task_success_rates.md")
    print(f"Wrote {len(rows)} task rows")
    print(ROOT / "task_success_rates.md")
    print(ROOT / "task_success_rates.csv")


if __name__ == "__main__":
    main()
