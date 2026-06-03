#!/usr/bin/env python3
"""Evaluate whether an early failure predictor can save rollout compute.

This script simulates an offline early-stop policy:

  1. Use the early-prefix predictor input, for example data/predictor_early_3.json.
  2. Use the full episode data to measure original trajectory length.
  3. Choose a failure-probability threshold on the validation split.
  4. Report compute saved on the held-out test split.

Labels follow the existing training convention:
  - success = 1
  - failure = 0

The stop policy uses P(failure), i.e. class-0 probability.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any


SUCCESS_LABELS = {"success", "succeeded", "pass", "passed", "true", "1"}
FAILURE_LABELS = {"failure", "failed", "fail", "false", "0"}


def label_to_int(ep: dict[str, Any]) -> int:
    label = ep.get("final_label")
    if isinstance(label, bool):
        return int(label)
    if label is not None:
        normalized = str(label).strip().lower()
        if normalized in SUCCESS_LABELS:
            return 1
        if normalized in FAILURE_LABELS:
            return 0
    reward = ep.get("final_reward")
    if reward is not None:
        return int(float(reward) > 0.0)
    raise ValueError(f"Could not infer binary label: {ep.keys()}")


def load_episodes(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "episodes" in data:
        return data["episodes"]
    if isinstance(data, list):
        return data
    raise ValueError(f"Unsupported input structure in {path}")


def compact_json(value: Any, max_chars: int) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        text = value
    else:
        try:
            text = json.dumps(value, ensure_ascii=False, sort_keys=True)
        except TypeError:
            text = str(value)
    text = " ".join(text.split())
    return text[:max_chars]


def stringify_tool_calls(tool_calls: Any) -> str:
    if not tool_calls:
        return ""
    try:
        return json.dumps(tool_calls, ensure_ascii=False, sort_keys=True)
    except TypeError:
        return str(tool_calls)


def episode_to_text(ep: dict[str, Any], max_tool_chars: int = 1200) -> str:
    parts = [
        "Predict whether this tau-bench episode ends in success.",
        f"model: {ep.get('model', 'unknown')}",
        f"condition: {ep.get('condition', 'unknown')}",
        f"task_id: {ep.get('task_id', 'unknown')}",
    ]
    trajectory = ep.get("full_trajectory") or ep.get("trajectory") or []
    if trajectory:
        parts.append("transcript:")
        for turn in trajectory:
            role = str(turn.get("role") or "unknown")
            content = compact_json(turn.get("content"), max_tool_chars)
            if content:
                parts.append(f"{role}: {content}")
            tool_calls = stringify_tool_calls(turn.get("tool_calls"))
            if tool_calls:
                parts.append(f"{role}_tool_calls: {tool_calls[:max_tool_chars]}")
    return "\n".join(parts)


def count_tool_call_steps(ep: dict[str, Any]) -> int:
    trajectory = ep.get("full_trajectory") or ep.get("trajectory") or []
    return sum(
        1
        for turn in trajectory
        if turn.get("role") == "assistant" and turn.get("tool_calls")
    )


def prefix_turn_count(ep: dict[str, Any], prefix_tool_calls: int) -> int:
    """Turn count retained by the first N assistant tool-call rounds."""
    trajectory = ep.get("full_trajectory") or ep.get("trajectory") or []
    if not trajectory:
        return 0

    tool_rounds = 0
    i = 0
    kept = 0
    while i < len(trajectory):
        turn = trajectory[i]
        kept += 1
        if turn.get("role") == "assistant" and turn.get("tool_calls"):
            tool_rounds += 1
            i += 1
            while i < len(trajectory) and trajectory[i].get("role") == "tool":
                kept += 1
                i += 1
            if tool_rounds >= prefix_tool_calls:
                return kept
        else:
            i += 1
    return kept


def stratified_split(
    records: list[dict[str, Any]],
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(seed)
    by_label: dict[int, list[dict[str, Any]]] = {0: [], 1: []}
    for record in records:
        by_label[int(record["label"])].append(record)

    train: list[dict[str, Any]] = []
    val: list[dict[str, Any]] = []
    test: list[dict[str, Any]] = []
    for rows in by_label.values():
        rng.shuffle(rows)
        n = len(rows)
        n_test = max(1, int(round(n * test_ratio))) if n > 2 else 0
        n_val = max(1, int(round(n * val_ratio))) if n > 1 else 0
        test.extend(rows[:n_test])
        val.extend(rows[n_test:n_test + n_val])
        train.extend(rows[n_test + n_val:])

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


def run_inference(
    model_dir: Path,
    texts: list[str],
    max_length: int,
    batch_size: int,
) -> tuple[list[int], list[float], list[float]]:
    """Return predictions, P(success), and P(failure)."""
    import numpy as np
    import torch
    from peft import PeftModel
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    adapter_config = json.loads((model_dir / "adapter_config.json").read_text())
    base_model_name = adapter_config["base_model_name_or_path"]
    print(f"Loading base model: {base_model_name}")

    base = AutoModelForSequenceClassification.from_pretrained(
        base_model_name,
        num_labels=2,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    base.config.pad_token_id = tokenizer.pad_token_id
    model = PeftModel.from_pretrained(base, str(model_dir))
    model.eval()

    preds: list[int] = []
    p_success: list[float] = []
    p_failure: list[float] = []

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            enc = tokenizer(
                texts[i:i + batch_size],
                truncation=True,
                max_length=max_length,
                padding=True,
                return_tensors="pt",
            )
            enc = {k: v.to(model.device) for k, v in enc.items()}
            logits = model(**enc).logits.float().cpu().numpy()
            exp_logits = np.exp(logits - logits.max(axis=-1, keepdims=True))
            probs = exp_logits / exp_logits.sum(axis=-1, keepdims=True)
            preds.extend(np.argmax(logits, axis=-1).tolist())
            p_failure.extend(probs[:, 0].tolist())
            p_success.extend(probs[:, 1].tolist())

    return preds, p_success, p_failure


def evaluate_threshold(records: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    stopped = [r["p_failure"] >= threshold for r in records]
    labels = [int(r["label"]) for r in records]

    success_count = sum(1 for y in labels if y == 1)
    failure_count = sum(1 for y in labels if y == 0)
    stopped_success = sum(1 for stop, y in zip(stopped, labels) if stop and y == 1)
    stopped_failure = sum(1 for stop, y in zip(stopped, labels) if stop and y == 0)
    stopped_total = sum(1 for stop in stopped if stop)

    saved_tool_calls = 0
    saved_turns = 0
    full_tool_calls = 0
    full_turns = 0
    saved_tool_calls_on_failures = 0
    saved_turns_on_failures = 0

    for rec, stop in zip(records, stopped):
        full_tool_calls += rec["full_tool_calls"]
        full_turns += rec["full_turns"]
        if not stop:
            continue
        tc_saved = max(0, rec["full_tool_calls"] - rec["prefix_tool_calls"])
        turn_saved = max(0, rec["full_turns"] - rec["prefix_turns"])
        saved_tool_calls += tc_saved
        saved_turns += turn_saved
        if int(rec["label"]) == 0:
            saved_tool_calls_on_failures += tc_saved
            saved_turns_on_failures += turn_saved

    original_pass_rate = success_count / max(1, len(records))
    pass_rate_after_stop = (success_count - stopped_success) / max(1, len(records))

    return {
        "threshold": threshold,
        "n": len(records),
        "stopped_total": stopped_total,
        "stopped_success": stopped_success,
        "stopped_failure": stopped_failure,
        "stop_rate": stopped_total / max(1, len(records)),
        "false_stop_rate_success": stopped_success / max(1, success_count),
        "failure_stop_rate": stopped_failure / max(1, failure_count),
        "stop_precision_failure": stopped_failure / max(1, stopped_total),
        "original_pass_rate": original_pass_rate,
        "pass_rate_after_stop": pass_rate_after_stop,
        "pass_rate_drop": original_pass_rate - pass_rate_after_stop,
        "tool_calls_saved": saved_tool_calls,
        "turns_saved": saved_turns,
        "tool_call_compute_saved_pct": saved_tool_calls / max(1, full_tool_calls),
        "turn_compute_saved_pct": saved_turns / max(1, full_turns),
        "tool_calls_saved_on_failures": saved_tool_calls_on_failures,
        "turns_saved_on_failures": saved_turns_on_failures,
    }


def threshold_grid(records: list[dict[str, Any]]) -> list[float]:
    probs = sorted({round(float(r["p_failure"]), 6) for r in records})
    grid = {0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0}
    grid.update(probs)
    return sorted(grid)


def choose_threshold(
    val_records: list[dict[str, Any]],
    max_false_stop_rate: float,
) -> tuple[float, list[dict[str, Any]]]:
    sweep = [evaluate_threshold(val_records, t) for t in threshold_grid(val_records)]
    feasible = [
        row
        for row in sweep
        if row["false_stop_rate_success"] <= max_false_stop_rate
    ]
    if not feasible:
        return 1.0, sweep
    best = max(
        feasible,
        key=lambda row: (
            row["tool_call_compute_saved_pct"],
            row["failure_stop_rate"],
            -row["false_stop_rate_success"],
        ),
    )
    return float(best["threshold"]), sweep


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(value, f, indent=2)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def attach_predictions(records: list[dict[str, Any]], model_dir: Path, max_length: int, batch_size: int) -> None:
    _, p_success, p_failure = run_inference(
        model_dir,
        [r["text"] for r in records],
        max_length,
        batch_size,
    )
    for rec, ps, pf in zip(records, p_success, p_failure):
        rec["p_success"] = ps
        rec["p_failure"] = pf


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-input", type=Path, default=Path("metrics/episode_results.json"))
    parser.add_argument("--early-input", type=Path, default=Path("data/predictor_early_3.json"))
    parser.add_argument("--model-dir", type=Path, default=Path("results/predictor-early-3"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/predictor-early-3/compute_saved_eval"))
    parser.add_argument("--prefix-tool-calls", type=int, default=3)
    parser.add_argument("--max-false-stop-rate", type=float, default=0.05)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    full_episodes = load_episodes(args.full_input)
    early_episodes = load_episodes(args.early_input)
    if len(full_episodes) != len(early_episodes):
        raise SystemExit(
            f"Episode count mismatch: full={len(full_episodes)} early={len(early_episodes)}"
        )

    records: list[dict[str, Any]] = []
    for idx, (full_ep, early_ep) in enumerate(zip(full_episodes, early_episodes)):
        label = label_to_int(full_ep)
        records.append(
            {
                "index": idx,
                "episode_id": full_ep.get("episode_id") or full_ep.get("id") or idx,
                "model": full_ep.get("model") or "unknown",
                "condition": full_ep.get("condition") or "unknown",
                "task_id": full_ep.get("task_id") or "unknown",
                "label": label,
                "text": episode_to_text(early_ep),
                "full_tool_calls": count_tool_call_steps(full_ep),
                "prefix_tool_calls": min(args.prefix_tool_calls, count_tool_call_steps(full_ep)),
                "full_turns": len(full_ep.get("full_trajectory") or full_ep.get("trajectory") or []),
                "prefix_turns": prefix_turn_count(full_ep, args.prefix_tool_calls),
            }
        )

    _, val_records, test_records = stratified_split(
        records,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    print(f"Validation records: {len(val_records)}")
    print(f"Test records      : {len(test_records)}")

    print("Running validation inference ...")
    attach_predictions(val_records, args.model_dir, args.max_length, args.batch_size)
    chosen_threshold, val_sweep = choose_threshold(
        val_records,
        max_false_stop_rate=args.max_false_stop_rate,
    )

    print(f"Chosen threshold: {chosen_threshold}")
    print("Running test inference ...")
    attach_predictions(test_records, args.model_dir, args.max_length, args.batch_size)
    test_metrics = evaluate_threshold(test_records, chosen_threshold)

    output = {
        "chosen_threshold": chosen_threshold,
        "selection_rule": {
            "max_false_stop_rate": args.max_false_stop_rate,
            "selected_on": "validation",
        },
        "test_metrics": test_metrics,
        "config": {
            "full_input": str(args.full_input),
            "early_input": str(args.early_input),
            "model_dir": str(args.model_dir),
            "prefix_tool_calls": args.prefix_tool_calls,
            "seed": args.seed,
            "val_ratio": args.val_ratio,
            "test_ratio": args.test_ratio,
            "max_length": args.max_length,
        },
    }

    write_json(args.output_dir / "compute_saved_metrics.json", output)
    write_csv(args.output_dir / "validation_threshold_sweep.csv", val_sweep)
    write_json(
        args.output_dir / "test_predictions.json",
        [
            {
                k: rec[k]
                for k in [
                    "episode_id", "model", "condition", "task_id", "label",
                    "p_success", "p_failure", "full_tool_calls",
                    "prefix_tool_calls", "full_turns", "prefix_turns",
                ]
            }
            for rec in test_records
        ],
    )

    print(json.dumps(output, indent=2))
    print(f"Saved outputs under {args.output_dir}")


if __name__ == "__main__":
    main()
