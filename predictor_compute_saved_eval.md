# Predictor Compute-Saved Evaluation

This note summarizes the offline compute-saving evaluation for the early failure predictor. The goal was to test whether the trained predictor could reduce rollout cost by stopping likely failing episodes before full completion.

## Evaluation Setup

The evaluation used an offline early-stop simulation rather than a live benchmark intervention.

Inputs:

- full episode data: `metrics/episode_results.json`
- early-prefix predictor data: `data/predictor_early_3.json`
- trained early predictor: `results/predictor-early-3`

The predictor observes only the trajectory prefix through the first 3 tool-call rounds. For each held-out episode, it estimates the probability that the episode will fail. If the predicted failure probability is above a selected threshold, the episode is treated as stopped at that prefix.

The threshold was selected on the validation split, not the test split. The selection rule constrained the false-stop rate on successful validation episodes to at most 5%.

Configuration:

| Field | Value |
| --- | --- |
| Prefix length | first 3 tool-call rounds |
| Validation split | 15% |
| Test split | 15% |
| Random seed | 42 |
| Threshold selection | validation only |
| Max validation false-stop rate | 5% |
| Selected threshold | `0.985496` |

## Output Files

The evaluation artifacts are stored in:

- `results/compute_saved_eval/compute_saved_metrics.json`
- `results/compute_saved_eval/validation_threshold_sweep.csv`
- `results/compute_saved_eval/test_predictions.json`

File purposes:

| File | Purpose |
| --- | --- |
| `compute_saved_metrics.json` | Main summary metrics for the selected threshold on the held-out test split. |
| `validation_threshold_sweep.csv` | Validation-set threshold sweep used to select the early-stop threshold. |
| `test_predictions.json` | Per-episode test predictions and trajectory length metadata. |

## Test-Set Results

At the validation-selected threshold of `0.985496`, the early-stop policy produced the following held-out test results:

| Metric | Value |
| --- | ---: |
| Test episodes | 461 |
| Episodes stopped early | 163 |
| Stop rate | 35.36% |
| Correctly stopped failures | 159 |
| Incorrectly stopped successes | 4 |
| Stop precision for failures | 97.55% |
| False-stop rate on successful episodes | 3.20% |
| Failure stop rate | 47.32% |
| Original pass rate | 27.11% |
| Pass rate after early-stop policy | 26.25% |
| Pass-rate drop | 0.87 percentage points |
| Tool calls saved | 195 |
| Turns saved | 1,420 |
| Tool-call compute saved | 9.91% |
| Turn compute saved | 8.29% |
| Tool calls saved on failing episodes | 191 |
| Turns saved on failing episodes | 1,404 |

## Interpretation

The early-3 predictor produced a useful compute-saving signal in offline replay. It stopped 35.36% of held-out test episodes early, and 97.55% of those stopped episodes were true failures. This indicates that the predictor was usually stopping runs that would not have succeeded anyway.

The policy saved 9.91% of tool-call steps and 8.29% of trajectory turns on the held-out test split. Most of the saved work came from failing episodes: 191 of 195 saved tool calls and 1,404 of 1,420 saved turns were from episodes that ultimately failed.

The cost of early stopping was low but nonzero. Four successful episodes were stopped early, corresponding to a 3.20% false-stop rate among successful episodes and a 0.87 percentage-point pass-rate drop.

## Report Claim

A defensible report statement is:

> In an offline early-stop simulation on the held-out test split, the early-3 predictor stopped 35.36% of episodes early. Of the stopped episodes, 97.55% were true failures. This saved 9.91% of tool-call steps and 8.29% of trajectory turns, with a 3.20% false-stop rate on successful episodes and a 0.87 percentage-point pass-rate drop.

## Caveat

This is an offline replay evaluation, not a live intervention during rollout execution. The result supports the claim that the predictor could have reduced rollout compute under a simulated early-stop policy. It should not be described as measured live GPU-time savings.
