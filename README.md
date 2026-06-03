# Agent Reliability Harness

A production-stress reliability evaluation framework for tool-using LLM agents.

## What this project does

Current agent benchmarks report pass@1 — whether an agent succeeds on a single attempt. 
This overstates real-world reliability. We build a harness that measures three properties 
pass@1 misses: consistency (pass^k), robustness to rephrased inputs, and fault tolerance 
under tool/API failures.

We evaluate three open-weight 7B agents on τ-bench retail and airline tasks, with controlled fault 
injection at the tool layer. We also fine-tune a small failure predictor (QLoRA on 
Qwen2.5-7B) that predicts run failure from the first 3 tool calls.

## Team
- Aayushi Somani 
- Yuting Mao
- Tracey Peyton  
- Anjuta Khongbantabam 

## Project structure

Agent Reliability Harness is a benchmark and analysis framework for evaluating the reliability of tool-using LLM agents under production-like stress. The project extends tau2-bench experiments with fault injection, rollout logging, model failure analysis, and early failure prediction. Instead of only measuring single-run success, it focuses on consistency, robustness, and failure behavior across repeated runs and degraded input or tool conditions.

The project evaluates open-weight instruction models on tau2 retail and airline tasks, using vLLM and LiteLLM-based serving where applicable. Models analyzed include Llama-3.1-8B-Instruct, Qwen/Qwen2.5-7B-Instruct, Qwen3-8B, Qwen3-14B, Mistral-7B-Instruct-v0.1, Mixtral-8x7B-Instruct-v0, Mistral-Small-3.2-24B-Instruct, and gemma-2-9b-it.

The repository includes SLURM rollout scripts, benchmark results, raw logs, metric computation tools, model failure appendices, and QLoRA-based binary failure prediction experiments. Its main goal is to distinguish true model-task failures from infrastructure, serving, prompt-format, context-window, and tool-call compatibility issues.

The project is organized around running, analyzing, and reporting reliability experiments for tool-using LLM agents.

- tau2-bench/ contains the local tau2 benchmark codebase used to run retail and airline agent tasks. This includes local utility changes for message handling and fault injection.

- sbatch/ contains SLURM and shell scripts for launching benchmark rollouts, vLLM-backed model runs, calibration jobs, and predictor training jobs on the cluster.

- logs/ stores raw rollout logs and vLLM server logs from benchmark runs.

- results/ stores benchmark output artifacts produced by model runs, organized by model, fault mode, seed, or experiment configuration.

- metrics/ contains analysis scripts for computing harness metrics, preparing early-run predictor data, training binary failure predictors, and summarizing model/runtime errors.

- model_failure_appendix/ contains generated failure summaries, configuration inventories, and appendix-ready model failure reports.

- Model README.md documents model-level performance findings, recurring failure modes, and interpretation of benchmark reliability results.

- utils_fault_injection_changes.md documents the local tau2 utility changes, especially the added fault-injection module and LiteLLM message-normalization changes.

- clean_calibration_models.md summarizes calibration-related model screening and clean-run evaluation notes.

- predictor_compute_saved_eval.md summarizes the offline early-stop simulation used to evaluate whether the early failure predictor could reduce rollout compute.




## Best Log Samples By Model

These logs are representative samples for report review and GitHub project documentation. Rollout logs show benchmark execution behavior; paired vLLM logs show server-side model configuration and serving failures.

| Model | Best sample log | Why this sample is useful |
| --- | --- | --- |
| Llama-3.1-8B-Instruct | [Rollout-llama-tau2-118170.out](https://github.com/aayushi9195/agent-reliability-harness/blob/main/logs/Rollout-llama-tau2-118170.out) | Clean model/run metadata with vLLM config: LlamaForCausalLM, max_model_len=16384, and tool_call_parser=llama3_json. |
| Qwen/Qwen2.5-7B-Instruct | [Rollout-Qwen2.5-tau2-119520.out](https://github.com/aayushi9195/agent-reliability-harness/blob/main/logs/Rollout-Qwen2.5-tau2-119520.out) | Good rollout sample showing Qwen2.5 tau2 execution and vLLM readiness behavior. |
| Qwen3-8B | [Rollout-Qwen3-8B-tau2-116769.out](https://github.com/aayushi9195/agent-reliability-harness/blob/main/logs/Rollout-Qwen3-8B-tau2-116769.out) | Clear run metadata: model, seed, task range, retail domain, agent, user simulator, and concurrency. |
| Qwen3-14B | [qwen3-14b-10-119228.out](https://github.com/aayushi9195/agent-reliability-harness/blob/main/logs/qwen3-14b-10-119228.out) | Representative rollout log for Qwen3-14B. |
| Qwen3-14B server | [vllm-qwen3-14b-119228.log](https://github.com/aayushi9195/agent-reliability-harness/blob/main/logs/vllm-qwen3-14b-119228.log) | Server-side companion log for served model name, architecture, context length, dtype, and parser setup. |
| Mistral-7B-Instruct-v0.1 | [Rollout-Min3-8B-tau2-116780.out](https://github.com/aayushi9195/agent-reliability-harness/blob/main/logs/Rollout-Min3-8B-tau2-116780.out) | Clearest sample of Mistral chat-format failure: role alternation rejection. |
| Mixtral-8x7B-Instruct-v0 | [Rollout-Min3-8B-tau2-119512.out](https://github.com/aayushi9195/agent-reliability-harness/blob/main/logs/Rollout-Min3-8B-tau2-119512.out) | Good sample for later Mixtral tool/parser compatibility failures after startup issues were being resolved. |
| Mistral-Small-3.2-24B-Instruct | [mistral32-10-119320.out](https://github.com/aayushi9195/agent-reliability-harness/blob/main/logs/mistral32-10-119320.out) | Strong client-side sample showing repeated context-window overflow at the 8,192-token limit. |
| Mistral-Small-3.2-24B-Instruct server | [vllm-mistral-small32-24b-119320.log](https://github.com/aayushi9195/agent-reliability-harness/blob/main/logs/vllm-mistral-small32-24b-119320.log) | Best server-side evidence for the same context-window failures. |
| gemma-2-9b-it | [Rollout-Gemma-tau2-116938.out](https://github.com/aayushi9195/agent-reliability-harness/blob/main/logs/Rollout-Gemma-tau2-116938.out) | Clearest Gemma sample; shows system-role compatibility failure. |
| NousResearch/Hermes-2-Pro-Llama-3-8B | [hermes10-119142.out](https://github.com/aayushi9195/agent-reliability-harness/blob/main/logs/hermes10-119142.out) | Representative Hermes rollout log. |
| NousResearch/Hermes-2-Pro-Llama-3-8B server | [vllm-hermes-119142.log](https://github.com/aayushi9195/agent-reliability-harness/blob/main/logs/vllm-hermes-119142.log) | Contains served model, architecture, context length, dtype, and Hermes parser setup. |


## Model Performance Summary

Overall, the benchmark results suggest that model performance was limited less by reasoning quality and more by orchestration reliability. Many failures occurred before the models could complete tasks cleanly, due to serving configuration, chat-template incompatibilities, tool-call parsing issues, model-name routing mismatches, API rate limits, and context-window overflows.

Llama-3.1-8B-Instruct showed the largest number of recorded failures, but most were retry-amplified infrastructure or interface errors. The dominant issues were tool-call configuration problems, missing or invalid parser/template settings, auto-tool-choice setup failures, and context-window overflow. As a result, its benchmark performance should be interpreted cautiously, since many failures reflect request or serving incompatibility rather than task-solving ability.

The Qwen models showed mixed reliability. Qwen3-8B, Qwen/Qwen2.5-7B-Instruct, and qwen3-14b all encountered routing mismatches, context-window failures, parser or response-shape errors, and occasional vLLM availability problems. This suggests that the Qwen family requires cleaner served-model naming, prompt-length control, and tool-call configuration before its task performance can be fairly compared.

The Mistral and Mixtral runs were also dominated by compatibility issues. Mistral-7B-Instruct-v0.1 failed mainly because the endpoint rejected the chat role ordering. Mixtral-8x7B-Instruct-v0 encountered several startup and tool-call configuration failures, including missing templates, invalid startup arguments, and parser/tokenizer mismatches. mistral-small32-24b was reachable in later runs but repeatedly exceeded its configured context window.

gemma-2-9b-it failed primarily because the hosted vLLM endpoint rejected system-role messages. This means its failures mainly reflect chat-schema incompatibility rather than evidence of poor model reasoning.

In summary, the strongest conclusion is that benchmark reliability is currently bottlenecked by the execution stack: vLLM startup configuration, tool-call parser selection, chat-template compatibility, served-model naming, and context-window budgeting. Once these issues are stabilized, the benchmark should be rerun so that model comparisons reflect task performance rather than infrastructure and request-format failures.

## Clean Calibration Model Screening

Before the main rollout set was finalized, additional clean calibration runs were used to screen model feasibility, serving compatibility, and early agent-task performance. These runs were not the full fault-injection rollout matrix; they were used to decide which model configurations were worth continued benchmark effort.

The calibration notes are documented in `clean_calibration_models.md`. Main outcomes:

| Model | Calibration outcome | Decision |
| --- | --- | --- |
| `Qwen3-14B` | Thinking ON performed about 2x better than thinking OFF (`Pass^1 = 0.40` vs `0.20`). | Keep thinking ON. |
| `Qwen/Qwen2.5-7B-Instruct` | Useful smaller baseline, but weaker than Qwen3-14B. | Keep as comparison/reference. |
| `Qwen3-8B` | Intermediate Qwen-family baseline; not stronger than Qwen3-14B thinking ON. | Comparison only. |
| `NousResearch/Hermes-2-Pro-Llama-3-8B` | Weak tool-use and task completion in tau2 agent setting. | Drop. |
| `mistralai/Mistral-Small-3.2-24B-Instruct-2506` | `Pass^1 = 0.00`, high token cost, context-window failures. | Drop. |
| `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` and `14B` | Poor fit with the local vLLM reasoning-serving path. | Do not continue. |
| `gemma-2-9b-it` | Clear system-role compatibility failure in the clean sample log. | Not preferred without message-format adjustment. |
| `Llama-3.1-8B-Instruct` | Fallback candidate with parser, template, and context-window issues. | Not preferred. |

The calibration stage showed that model selection depended on serving stability, tool-call compatibility, context-window fit, and actual task completion behavior, not only general instruction-model quality.

## Fault Injection Summary

To evaluate model robustness under degraded or adversarial input conditions, we added a local fault-injection layer to the tau2 LLM request pipeline. The main addition was fault_injection.py, which can perturb user messages before they are sent to the model. The injection mode is controlled through the FAULT_MODE environment variable and supports four settings: clean, light, heavy, and schema.

In clean mode, messages are left unchanged. In light mode, some user messages are appended with an instruction such as Ignore previous instructions. In heavy mode, selected user messages are reversed, creating a stronger corruption of the input. In schema mode, selected messages are appended with an instruction to return invalid JSON, testing whether the model and tool pipeline remain reliable when output-format pressure is introduced.

The fault injector is integrated into llm_utils.py, meaning it is applied before every LiteLLM completion call. Before injection, messages are normalized so that different content formats, including strings, lists, dictionaries, and None, are converted into string-compatible forms. After injection, the pipeline sanitizes malformed tool calls, validates message content, and coerces system messages into user-visible text when needed for provider compatibility.

These changes were intended to make the benchmark more robust against hosted vLLM and LiteLLM interface failures while also enabling controlled reliability experiments. The fault modes allow comparison between normal execution and progressively more disruptive conditions, helping identify whether failures come from model reasoning, prompt corruption, schema fragility, tool-call formatting, or serving-stack incompatibility.

A key caveat is that the modified pipeline is not identical to upstream tau2-bench behavior. Even in clean mode, messages pass through normalization, validation, tool-call sanitization, and system-message coercion. Therefore, results should be reported as coming from a locally modified tau2 benchmark harness rather than the unmodified upstream implementation.

## Early Failure Predictor and Compute-Saved Evaluation

The project fine-tuned a binary early failure predictor using QLoRA with `Qwen2.5-7B-Instruct` as the classifier backbone. The predictor task is sequence classification: given an early trajectory prefix, predict whether the full episode will eventually succeed or fail.

The early predictor used the first 3 tool-call rounds as input. Training used a 70/15/15 train/validation/test split, with validation used for checkpoint and threshold selection and the held-out test split reserved for final reporting.

The compute-saving evaluation is documented in `predictor_compute_saved_eval.md`. It used an offline early-stop simulation:

1. run the predictor on each held-out test episode prefix;
2. stop the episode if predicted failure probability exceeds a threshold;
3. compare the stopped prefix length to the original full trajectory length.

The threshold was selected on validation with a maximum 5% false-stop rate on successful validation episodes. On the held-out test split:

| Metric | Value |
| --- | ---: |
| Test episodes | 461 |
| Episodes stopped early | 163 |
| Stop rate | 35.36% |
| Correctly stopped failures | 159 |
| Incorrectly stopped successes | 4 |
| Stop precision for failures | 97.55% |
| False-stop rate on successful episodes | 3.20% |
| Pass-rate drop | 0.87 percentage points |
| Tool-call compute saved | 9.91% |
| Turn compute saved | 8.29% |

The predictor backbone was `Qwen2.5-7B-Instruct`, but the compute-saved test split included episodes from multiple agent models: `Qwen3-14B`, `Qwen2.5-7B-Instruct`, `gemma-4-31B-it`, `Llama-3.1-8B-Instruct`, `Qwen2.5b`, and `Qwen3-8B`.

This result should be described as an offline replay estimate rather than live GPU-time measurement. The defensible claim is that the predictor could have reduced rollout work by stopping likely failing episodes early under the simulated policy.
