#!/usr/bin/env python3
"""Compute harness reliability metrics from episode_results.json.

Metrics computed per model × condition:
  - pass@1         fraction of episodes that succeeded
  - pass^k (k=5)  unbiased estimator: 1 - C(n-c,k)/C(n,k)
                  computed only for (model, task_id) groups with n >= k trials

Aggregate per model:
  - fault-tolerance score = 0.2 * pass(clean) + 0.3 * pass(light) + 0.5 * pass(heavy)

Outputs:
  - Console report card (Markdown table)
  - metrics/harness_metrics.json

Usage:
    python compute_harness_metrics.py \
        --input  metrics/episode_results.json \
        --output metrics/harness_metrics.json \
        --k 5
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from math import comb
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Core estimators
# ---------------------------------------------------------------------------

def pass_at_k(n: int, c: int, k: int) -> float | None:
    """Unbiased pass@k estimator from tau-bench (Yao et al. 2024).

    Returns None when n < k (estimator is undefined).
    """
    if n < k:
        return None
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def pass_at_1(successes: int, trials: int) -> float:
    if trials == 0:
        return float("nan")
    return successes / trials


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_episodes(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "episodes" in data:
        return data["episodes"]
    if isinstance(data, list):
        return data
    raise ValueError(f"Unsupported input structure in {path}")


def is_success(ep: dict[str, Any]) -> bool:
    label = ep.get("final_label")
    if label is not None:
        return str(label).strip().lower() in {"success", "succeeded", "pass", "passed", "true", "1"}
    reward = ep.get("final_reward")
    if reward is not None:
        return float(reward) > 0.0
    raise ValueError(f"Cannot determine success for episode: {ep.keys()}")


# ---------------------------------------------------------------------------
# Metric aggregation
# ---------------------------------------------------------------------------

def aggregate(episodes: list[dict[str, Any]], k: int) -> dict[str, Any]:
    """Return nested aggregation: model -> condition -> {pass@1, pass@k, n, c}."""
    # Group by (model, condition, task_id)
    groups: dict[tuple[str, str, Any], list[bool]] = defaultdict(list)
    for ep in episodes:
        model = ep.get("model") or "unknown"
        condition = ep.get("condition") or "unknown"
        task_id = ep.get("task_id")
        groups[(model, condition, task_id)].append(is_success(ep))

    # Aggregate per (model, condition)
    mc_stats: dict[tuple[str, str], dict[str, Any]] = defaultdict(
        lambda: {"successes_by_task": [], "n_episodes": 0, "n_success": 0}
    )
    for (model, condition, _task_id), outcomes in groups.items():
        n = len(outcomes)
        c = sum(outcomes)
        s = mc_stats[(model, condition)]
        s["n_episodes"] += n
        s["n_success"] += c
        pk = pass_at_k(n, c, k)
        s["successes_by_task"].append({"n": n, "c": c, "pass_at_k": pk})

    result: dict[str, dict[str, Any]] = {}
    for (model, condition), s in mc_stats.items():
        p1 = pass_at_1(s["n_success"], s["n_episodes"])
        # Average task-level pass@k (excluding tasks where estimator is undefined)
        pk_values = [t["pass_at_k"] for t in s["successes_by_task"] if t["pass_at_k"] is not None]
        pk_mean = sum(pk_values) / len(pk_values) if pk_values else None

        if model not in result:
            result[model] = {}
        result[model][condition] = {
            "n_episodes": s["n_episodes"],
            "n_success": s["n_success"],
            "pass_at_1": round(p1, 4),
            f"pass_at_{k}": round(pk_mean, 4) if pk_mean is not None else None,
            f"tasks_with_pass_at_{k}": len(pk_values),
            "tasks_total": len(s["successes_by_task"]),
        }

    return result


def fault_tolerance_score(
    model_stats: dict[str, Any],
    w_clean: float = 0.2,
    w_light: float = 0.3,
    w_heavy: float = 0.5,
) -> float | None:
    """Weighted composite: 0.2*pass(clean) + 0.3*pass(light) + 0.5*pass(heavy)."""
    parts = []
    weights_used = 0.0
    weight_map = {"clean": w_clean, "light": w_light, "heavy": w_heavy}
    for condition, weight in weight_map.items():
        if condition in model_stats:
            p1 = model_stats[condition].get("pass_at_1")
            if p1 is not None and p1 == p1:  # not NaN
                parts.append(weight * p1)
                weights_used += weight

    if not parts:
        return None
    # Renormalise if not all conditions are present
    if weights_used < 1.0:
        return sum(parts) / weights_used
    return sum(parts)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def fmt(v: float | None, pct: bool = True) -> str:
    if v is None:
        return "  -  "
    if pct:
        return f"{100 * v:5.1f}%"
    return f"{v:.4f}"


def print_report_card(stats: dict[str, Any], k: int) -> None:
    conditions = ["clean", "light", "heavy"]
    header = (
        f"{'Model':<35} "
        + "  ".join(f"{'pass@1 '+c:>10}  {'pass@'+str(k)+' '+c:>11}" for c in conditions)
        + f"  {'fault-tol':>10}"
    )
    print("\n" + "=" * len(header))
    print("HARNESS RELIABILITY REPORT CARD")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for model in sorted(stats):
        mc = stats[model]
        row = f"{model:<35}"
        for cond in conditions:
            if cond in mc:
                p1 = mc[cond]["pass_at_1"]
                pk = mc[cond].get(f"pass_at_{k}")
                row += f"  {fmt(p1):>10}  {fmt(pk):>11}"
            else:
                row += f"  {'  -  ':>10}  {'  -  ':>11}"
        ft = fault_tolerance_score(mc)
        row += f"  {fmt(ft):>10}"
        print(row)

    print("=" * len(header))
    print(f"\npass@{k}: unbiased estimator (Yao et al. 2024), averaged over tasks with n >= {k} trials.")
    print("fault-tolerance: 0.2×clean + 0.3×light + 0.5×heavy  (renormalised if condition missing).\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("metrics/episode_results.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/harness_metrics.json"),
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="k for pass@k estimator (default: 5)",
    )
    args = parser.parse_args()

    print(f"Loading {args.input} ...")
    episodes = load_episodes(args.input)
    print(f"  {len(episodes)} episodes loaded.")

    stats = aggregate(episodes, k=args.k)

    # Build full output with fault-tolerance scores
    out: dict[str, Any] = {}
    for model, mc in stats.items():
        out[model] = {
            "conditions": mc,
            "fault_tolerance_score": fault_tolerance_score(mc),
        }

    # Console report
    print_report_card(stats, k=args.k)

    # Episode count summary
    print("Episode counts by model × condition:")
    for model in sorted(stats):
        for cond, s in sorted(stats[model].items()):
            n = s["n_episodes"]
            c = s["n_success"]
            print(f"  {model} / {cond}: {n} episodes, {c} successes ({100*c/max(1,n):.1f}%)")

    # Save JSON
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved harness metrics to {args.output}")


if __name__ == "__main__":
    main()
