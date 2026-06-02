#!/usr/bin/env python3
"""Prepare early-trajectory data for predictor fine-tuning.

Reads episode_results.json and truncates each episode's full_trajectory to
the first --prefix-tool-calls assistant turns that have tool_calls, plus the
immediately following tool-response turns.

The output file has the same JSON schema as episode_results.json and can be
passed directly to train_predictor_binary.py via --input.

Usage:
    python prepare_predictor_early_data.py \
        --input  metrics/episode_results.json \
        --output data/predictor_early_3.json \
        --prefix-tool-calls 3
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path


def truncate_trajectory(trajectory: list[dict], prefix_tool_calls: int) -> list[dict]:
    """Return trajectory truncated to the first prefix_tool_calls tool-call rounds.

    A "tool-call round" is one assistant turn whose tool_calls list is non-empty,
    followed by all immediately subsequent turns whose role is "tool".
    All turns before the first tool call (system prompt, initial user message, etc.)
    are preserved.
    """
    kept: list[dict] = []
    tool_call_rounds = 0
    i = 0

    while i < len(trajectory):
        turn = trajectory[i]

        if turn.get("role") == "assistant" and turn.get("tool_calls"):
            tool_call_rounds += 1
            kept.append(turn)
            i += 1
            # Consume the tool-response turns that follow
            while i < len(trajectory) and trajectory[i].get("role") == "tool":
                kept.append(trajectory[i])
                i += 1
            if tool_call_rounds >= prefix_tool_calls:
                break  # Stop after the requested number of rounds
        else:
            kept.append(turn)
            i += 1

    return kept


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("metrics/episode_results.json"),
        help="Source episode_results.json (default: metrics/episode_results.json)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/predictor_early_3.json"),
        help="Output path (default: data/predictor_early_3.json)",
    )
    parser.add_argument(
        "--prefix-tool-calls",
        type=int,
        default=3,
        metavar="N",
        help="Number of tool-call rounds to keep (default: 3)",
    )
    args = parser.parse_args()

    print(f"Reading {args.input} ...")
    with args.input.open(encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and "episodes" in data:
        episodes: list[dict] = data["episodes"]
        schema = data.get("schema", {})
        metadata = dict(data.get("metadata", {}))
    elif isinstance(data, list):
        episodes = data
        schema = {}
        metadata = {}
    else:
        raise ValueError(f"Unsupported input structure in {args.input}")

    truncated: list[dict] = []
    total_original_turns = 0
    total_truncated_turns = 0

    for ep in episodes:
        ep2 = copy.deepcopy(ep)
        traj = ep2.get("full_trajectory") or ep2.get("trajectory") or []
        if traj:
            total_original_turns += len(traj)
            ep2["full_trajectory"] = truncate_trajectory(traj, args.prefix_tool_calls)
            total_truncated_turns += len(ep2["full_trajectory"])
        truncated.append(ep2)

    metadata.update(
        {
            "prefix_tool_calls": args.prefix_tool_calls,
            "source_file": str(args.input),
            "episode_count": len(truncated),
        }
    )

    out = {"schema": schema, "metadata": metadata, "episodes": truncated}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing {len(truncated)} episodes to {args.output} ...")
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(out, f)

    kept_pct = 100.0 * total_truncated_turns / max(1, total_original_turns)
    print(
        f"Done.  Trajectory turns kept: {total_truncated_turns}/{total_original_turns} "
        f"({kept_pct:.1f}%) after truncating to {args.prefix_tool_calls} tool-call round(s)."
    )


if __name__ == "__main__":
    main()
