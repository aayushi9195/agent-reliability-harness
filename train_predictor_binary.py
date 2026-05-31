#!/usr/bin/env python3
"""Train a QLoRA binary predictor on tau-bench processed episode data.

Predicts final success/failure from an episode trajectory.
Splits: 70% train / 15% val (checkpoint selection) / 15% test (held-out report).
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SUCCESS_LABELS = {"success", "succeeded", "pass", "passed", "true", "1"}
FAILURE_LABELS = {"failure", "failed", "fail", "false", "0"}


def require_training_deps() -> None:
    missing = []
    for module in ("torch", "transformers", "peft", "bitsandbytes", "accelerate", "sklearn"):
        try:
            __import__(module)
        except Exception:
            missing.append(module)
    if missing:
        joined = ", ".join(missing)
        raise SystemExit(
            f"Missing training dependencies: {joined}\n"
            "Install them with:\n"
            "  python3 -m pip install -r requirements-tbench-qlora.txt"
        )


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

    raise ValueError(f"Could not infer binary label for episode: {ep.keys()}")


def stringify_tool_calls(tool_calls: Any) -> str:
    if not tool_calls:
        return ""
    try:
        return json.dumps(tool_calls, ensure_ascii=False, sort_keys=True)
    except TypeError:
        return str(tool_calls)


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


def episode_to_text(ep: dict[str, Any], max_tool_chars: int = 1200) -> str:
    """Build model input text from a full trajectory, with metadata fallback."""
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
        metric_keys = [
            "db_match",
            "read_action_count",
            "write_action_count",
            "action_success_count",
            "action_failure_count",
            "action_total_count",
            "partial_action_reward_percent",
            "termination_reason",
        ]
        parts.append("episode_metadata:")
        for key in metric_keys:
            if key in ep:
                parts.append(f"{key}: {ep.get(key)}")

    return "\n".join(parts)


def load_episodes(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "episodes" in data:
        return data["episodes"]
    if isinstance(data, list):
        return data
    raise ValueError(f"Unsupported input structure in {path}")


def filter_episodes(
    episodes: list[dict[str, Any]],
    models: set[str] | None,
    conditions: set[str] | None,
    max_samples: int | None,
    seed: int,
) -> list[dict[str, Any]]:
    selected = []
    for ep in episodes:
        if models and str(ep.get("model")) not in models:
            continue
        if conditions and str(ep.get("condition")) not in conditions:
            continue
        selected.append(ep)

    if max_samples and len(selected) > max_samples:
        rng = random.Random(seed)
        selected = rng.sample(selected, max_samples)

    return selected


def stratified_split(
    records: list[dict[str, Any]],
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Return (train, val, test) with label-stratified 70-15-15 splits."""
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


@dataclass
class EpisodeDataset:
    rows: list[dict[str, Any]]
    tokenizer: Any
    max_length: int

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        import torch

        row = self.rows[idx]
        encoded = self.tokenizer(
            row["text"],
            truncation=True,
            max_length=self.max_length,
            padding=False,
        )
        encoded["labels"] = torch.tensor(row["label"], dtype=torch.long)
        return encoded


def compute_binary_metrics(eval_pred: Any) -> dict[str, float]:
    import numpy as np
    from sklearn.metrics import roc_auc_score

    logits, labels = eval_pred
    # Softmax over logits to get class probabilities; column 1 = P(success)
    exp_logits = np.exp(logits - logits.max(axis=-1, keepdims=True))
    probs = exp_logits / exp_logits.sum(axis=-1, keepdims=True)
    pos_probs = probs[:, 1]

    preds = np.argmax(logits, axis=-1)
    labels = np.asarray(labels)

    tp = int(((preds == 1) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    total = max(1, len(labels))
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    balanced_accuracy = 0.5 * (
        tp / max(1, tp + fn) + tn / max(1, tn + fp)
    )

    # AUROC requires both classes present in the batch
    try:
        auroc = float(roc_auc_score(labels, pos_probs))
    except ValueError:
        auroc = float("nan")

    return {
        "auroc": auroc,
        "accuracy": (tp + tn) / total,
        "balanced_accuracy": balanced_accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def parse_csv_set(value: str | None) -> set[str] | None:
    if not value or value.lower() == "all":
        return None
    return {part.strip() for part in value.split(",") if part.strip()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("/gpfs/projects/imt526a/group2/agent-reliability/metrics/episode_results.json"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/tbench-qlora-binary"))
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--models", help="Comma-separated model names to include, or all.")
    parser.add_argument("--conditions", default="all", help="Comma-separated conditions to include, or all.")
    parser.add_argument("--max-samples", type=int, help="Optional cap for quick experiments.")
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    require_training_deps()

    import torch
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        BitsAndBytesConfig,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    set_seed(args.seed)
    episodes = load_episodes(args.input)
    episodes = filter_episodes(
        episodes,
        models=parse_csv_set(args.models),
        conditions=parse_csv_set(args.conditions),
        max_samples=args.max_samples,
        seed=args.seed,
    )
    records = [{"text": episode_to_text(ep), "label": label_to_int(ep)} for ep in episodes]
    if len(records) < 6:
        raise SystemExit("Need at least six episodes after filtering.")

    positives = sum(row["label"] for row in records)
    negatives = len(records) - positives
    print(f"Loaded {len(records)} episodes: {positives} success, {negatives} failure")

    train_rows, val_rows, test_rows = stratified_split(
        records, args.val_ratio, args.test_ratio, args.seed
    )
    print(
        f"Split (70/15/15): train={len(train_rows)} | val={len(val_rows)} | test={len(test_rows)}"
    )
    print(
        f"  train success={sum(r['label'] for r in train_rows)} | "
        f"val success={sum(r['label'] for r in val_rows)} | "
        f"test success={sum(r['label'] for r in test_rows)}"
    )

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model,
        num_labels=2,
        id2label={0: "failure", 1: "success"},
        label2id={"failure": 0, "success": 1},
        quantization_config=quant_config,
        device_map="auto",
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    train_dataset = EpisodeDataset(train_rows, tokenizer, args.max_length)
    val_dataset   = EpisodeDataset(val_rows,   tokenizer, args.max_length)
    test_dataset  = EpisodeDataset(test_rows,  tokenizer, args.max_length)
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    train_args = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_strategy="steps",
        eval_steps=args.save_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="auroc",       # checkpoint selection by AUROC
        greater_is_better=True,
        bf16=use_bf16,
        fp16=torch.cuda.is_available() and not use_bf16,
        optim="paged_adamw_8bit",
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        data_collator=collator,
        compute_metrics=compute_binary_metrics,
    )
    trainer.train()

    # Val metrics (best checkpoint)
    val_metrics = trainer.evaluate(val_dataset)
    print("\nVal metrics:", json.dumps(val_metrics, indent=2))

    # Test metrics (held-out — run once, reported in final paper)
    test_metrics = trainer.evaluate(test_dataset, metric_key_prefix="test")
    print("\nTest metrics:", json.dumps(test_metrics, indent=2))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    all_metrics = {"val": val_metrics, "test": test_metrics}
    with (args.output_dir / "eval_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2, sort_keys=True)

    print(f"\nSaved QLoRA binary predictor to {args.output_dir}")


if __name__ == "__main__":
    main()
