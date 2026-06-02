#!/usr/bin/env python3
"""Compute predictor evaluation metrics with per-model breakdown and compute-saved %.

Loads the trained QLoRA binary predictor from --model-dir and runs inference on
the held-out test split (same 15% split used in training, seed=42).

Metrics reported:
  - Aggregate: AUROC, F1, precision, recall, balanced accuracy, accuracy
  - Per-model: AUROC, F1
  - Compute-saved %: among true-positive predictions (correctly flagged failing
    runs), fraction of remaining tool-call steps avoided by early termination.

Outputs:
  - Console report
  - results/predictor_metrics_full.json

Usage (GPU required):
    python compute_predictor_metrics.py \
        --input    metrics/episode_results.json \
        --model-dir results/predictor-binary \
        --output   results/predictor_metrics_full.json \
        --prefix-tool-calls 3 \
        --seed 42
"""
from __future__ import annotations

import argparse
import json
import random
from math import comb
from pathlib import Path
from typing import Any


SUCCESS_LABELS = {"success", "succeeded", "pass", "passed", "true", "1"}
FAILURE_LABELS = {"failure", "failed", "fail", "false", "0"}


# ---------------------------------------------------------------------------
# Data helpers (mirror train_predictor_binary.py)
# ---------------------------------------------------------------------------

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
    """Mirror of train_predictor_binary.episode_to_text (must stay in sync)."""
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
    else:
        for key in [
            "db_match", "read_action_count", "write_action_count",
            "action_success_count", "action_failure_count", "action_total_count",
            "partial_action_reward_percent", "termination_reason",
        ]:
            if key in ep:
                parts.append(f"{key}: {ep.get(key)}")
    return "\n".join(parts)


def count_tool_call_steps(ep: dict[str, Any]) -> int:
    """Count assistant turns with non-empty tool_calls in the full trajectory."""
    traj = ep.get("full_trajectory") or ep.get("trajectory") or []
    return sum(
        1 for t in traj
        if t.get("role") == "assistant" and t.get("tool_calls")
    )


def stratified_split(
    records: list[dict[str, Any]],
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Identical to train_predictor_binary.stratified_split."""
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
        n_val  = max(1, int(round(n * val_ratio)))  if n > 1 else 0
        test.extend(rows[:n_test])
        val.extend(rows[n_test:n_test + n_val])
        train.extend(rows[n_test + n_val:])

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_inference(
    model_dir: Path,
    texts: list[str],
    max_length: int,
    batch_size: int,
) -> tuple[list[int], list[float]]:
    """Run the saved QLoRA predictor and return (preds, pos_probs)."""
    import numpy as np
    import torch
    from peft import PeftModel
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Load base model + adapter
    base_model_name = json.loads(
        (model_dir / "adapter_config.json").read_text()
    )["base_model_name_or_path"]

    print(f"  Loading base model: {base_model_name}")
    from transformers import BitsAndBytesConfig
    quant_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    base = AutoModelForSequenceClassification.from_pretrained(
        base_model_name,
        num_labels=2,
        quantization_config=quant_cfg,
        device_map="auto",
    )
    base.config.pad_token_id = tokenizer.pad_token_id
    model = PeftModel.from_pretrained(base, str(model_dir))
    model.eval()

    all_preds: list[int] = []
    all_probs: list[float] = []

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            enc = tokenizer(
                batch_texts,
                truncation=True,
                max_length=max_length,
                padding=True,
                return_tensors="pt",
            )
            enc = {k: v.to(model.device) for k, v in enc.items()}
            logits = model(**enc).logits.float().cpu().numpy()
            exp_l = np.exp(logits - logits.max(axis=-1, keepdims=True))
            probs = exp_l / exp_l.sum(axis=-1, keepdims=True)
            all_preds.extend(np.argmax(logits, axis=-1).tolist())
            all_probs.extend(probs[:, 1].tolist())

    return all_preds, all_probs


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(
    labels: list[int],
    preds: list[int],
    probs: list[float],
) -> dict[str, float]:
    import numpy as np
    from sklearn.metrics import roc_auc_score

    labels_a = np.array(labels)
    preds_a = np.array(preds)

    tp = int(((preds_a == 1) & (labels_a == 1)).sum())
    tn = int(((preds_a == 0) & (labels_a == 0)).sum())
    fp = int(((preds_a == 1) & (labels_a == 0)).sum())
    fn = int(((preds_a == 0) & (labels_a == 1)).sum())
    total = max(1, len(labels_a))

    precision = tp / max(1, tp + fp)
    recall    = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    balanced_acc = 0.5 * (tp / max(1, tp + fn) + tn / max(1, tn + fp))

    try:
        auroc = float(roc_auc_score(labels_a, probs))
    except ValueError:
        auroc = float("nan")

    return {
        "n": total,
        "auroc": round(auroc, 4),
        "accuracy": round((tp + tn) / total, 4),
        "balanced_accuracy": round(balanced_acc, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


def compute_saved_pct(
    test_records: list[dict[str, Any]],
    preds: list[int],
    labels: list[int],
    prefix_tool_calls: int,
) -> dict[str, float]:
    """Compute compute-saved % among true positives (correctly flagged failures).

    For each TP (pred=0=failure, label=0=failure):
        steps_in_episode = total tool-call turns in the episode
        steps_saved      = max(0, steps_in_episode - prefix_tool_calls)

    compute_saved_pct = sum(steps_saved) / sum(steps_in_episode) across TPs.
    """
    total_steps = 0
    saved_steps = 0
    tp_count = 0

    for rec, pred, label in zip(test_records, preds, labels):
        if pred == 0 and label == 0:  # True positive (correctly flagged failure)
            tp_count += 1
            ep = rec.get("episode", {})
            steps = count_tool_call_steps(ep)
            total_steps += steps
            saved_steps += max(0, steps - prefix_tool_calls)

    if total_steps == 0:
        return {"compute_saved_pct": float("nan"), "tp_count": tp_count}

    return {
        "compute_saved_pct": round(saved_steps / total_steps, 4),
        "tp_count": tp_count,
        "total_steps_in_tp_runs": total_steps,
        "steps_saved": saved_steps,
    }


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
        "--model-dir",
        type=Path,
        default=Path("results/predictor-binary"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/predictor_metrics_full.json"),
    )
    parser.add_argument("--val-ratio",  type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--max-length", type=int,   default=2048)
    parser.add_argument("--batch-size", type=int,   default=4)
    parser.add_argument(
        "--prefix-tool-calls",
        type=int,
        default=3,
        help="Prefix length used at training time (for compute-saved % calculation)",
    )
    args = parser.parse_args()

    # Load episodes
    print(f"Loading episodes from {args.input} ...")
    with args.input.open(encoding="utf-8") as f:
        raw = json.load(f)
    episodes: list[dict[str, Any]] = raw["episodes"] if isinstance(raw, dict) else raw
    print(f"  {len(episodes)} episodes loaded.")

    # Build records with metadata preserved
    records_with_meta = [
        {
            "text": episode_to_text(ep),
            "label": label_to_int(ep),
            "model": ep.get("model") or "unknown",
            "condition": ep.get("condition") or "unknown",
            "episode": ep,  # keep for compute-saved %
        }
        for ep in episodes
    ]

    # Reproduce the same train/val/test split
    # (stratified_split only looks at 'label', so extra keys are ignored)
    _, _, test_rows = stratified_split(
        records_with_meta, args.val_ratio, args.test_ratio, args.seed
    )
    print(f"  Test split: {len(test_rows)} episodes")

    texts  = [r["text"]  for r in test_rows]
    labels = [r["label"] for r in test_rows]
    models = [r["model"] for r in test_rows]

    # Run inference
    print("Running predictor inference ...")
    preds, probs = run_inference(
        args.model_dir, texts, args.max_length, args.batch_size
    )

    # Aggregate metrics
    agg = compute_metrics(labels, preds, probs)
    print(f"\nAggregate test metrics (n={agg['n']}):")
    for k, v in agg.items():
        print(f"  {k}: {v}")

    # Per-model metrics
    per_model: dict[str, Any] = {}
    unique_models = sorted(set(models))
    print("\nPer-model metrics:")
    for m in unique_models:
        idx = [i for i, mod in enumerate(models) if mod == m]
        m_labels = [labels[i] for i in idx]
        m_preds  = [preds[i]  for i in idx]
        m_probs  = [probs[i]  for i in idx]
        mets = compute_metrics(m_labels, m_preds, m_probs)
        per_model[m] = mets
        print(f"  {m}: AUROC={mets['auroc']:.4f}  F1={mets['f1']:.4f}  n={mets['n']}")

    # Compute-saved %
    cs = compute_saved_pct(test_rows, preds, labels, args.prefix_tool_calls)
    print(f"\nCompute-saved %: {100*cs.get('compute_saved_pct', float('nan')):.1f}%"
          f"  (TPs: {cs.get('tp_count', 0)})")

    # Save results
    output: dict[str, Any] = {
        "aggregate": agg,
        "per_model": per_model,
        "compute_saved": cs,
        "config": {
            "model_dir": str(args.model_dir),
            "prefix_tool_calls": args.prefix_tool_calls,
            "seed": args.seed,
            "test_ratio": args.test_ratio,
            "val_ratio": args.val_ratio,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved full predictor metrics to {args.output}")


if __name__ == "__main__":
    main()
