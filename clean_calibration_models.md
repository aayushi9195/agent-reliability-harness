# Clean Calibration Model Screening Notes

This note documents additional model screening work conducted during the clean calibration stage on Tillicum. The purpose of these runs was to check model feasibility, vLLM serving compatibility, and baseline agent-task performance before the main rollout set was finalized.

## Environment and Setup

The calibration runs were executed on the Tillicum cluster using the shared project directory under:

- `/gpfs/projects/imt526a/group2/agent-reliability`

The tau2 benchmark codebase was used from:

- `/gpfs/projects/imt526a/group2/agent-reliability/repos/tau-bench`

Typical run components included:

- Slurm job scripts in the project `slurm/` directory
- local vLLM serving on compute nodes
- tau2 retail-domain evaluation
- `user_simulator` as the benchmark user side
- clean calibration configuration only



## Scope

This note covers:

- clean calibration runs only
- exploratory model screening
- early candidate selection decisions
- serving and benchmark feasibility observations



## Qwen3-14B calibration

`Qwen3-14B` was evaluated in both thinking-enabled and thinking-disabled modes.

### Result
Thinking ON performed approximately **2x better** than thinking OFF in clean calibration.

### Observed calibration results
- thinking ON:
  - `Pass^1 = 0.40`
- thinking OFF:
  - `Pass^1 = 0.20`

### Interpretation
The thinking-enabled configuration showed stronger multi-step tool-use behavior and better task completion.

### Decision
`Qwen3-14B` with thinking ON was retained as the stronger Qwen-family configuration.

---

## Qwen2.5-7B-Instruct calibration

`Qwen/Qwen2.5-7B-Instruct` was screened as a smaller baseline candidate.

### Result
It served as a useful smaller comparison point, but it was not as strong or as stable as the final `Qwen3-14B` configuration.

### Observations
- smaller and easier to deploy than larger candidates
- weaker than `Qwen3-14B` on agent-style task completion
- broader run history included setup and runtime issues such as served-model mismatch and context-window-related failures

### Decision
`Qwen2.5-7B-Instruct` remained useful as a calibration reference model, but it was not the strongest final candidate.

---

## Qwen3-8B calibration

`Qwen3-8B` was screened as another smaller Qwen-family candidate. The clearest clean-calibration references were:

- `Rollout-Qwen3-8B-tau2-116756.out`
- `Rollout-Qwen3-8B-tau2-116769.out`

### Result
`Qwen3-8B` was useful as an intermediate-size Qwen baseline, but it did not become the preferred final Qwen configuration.

### Observations
- smaller than `Qwen3-14B`, making it more practical as a lightweight comparison point
- broader run history showed model-routing, parser, and context-window-related issues in some Qwen3-8B attempts
- did not provide enough advantage over the stronger `Qwen3-14B` thinking-enabled setup

### Decision
`Qwen3-8B` was treated as a calibration/reference candidate rather than the preferred final model.

---

## Hermes-2-Pro-Llama-3-8B calibration

`NousResearch/Hermes-2-Pro-Llama-3-8B` was also screened.

### Result
Performance was weak in the tool-using tau2 agent setting.

### Observations
- weak tool-use behavior
- poor task completion relative to Qwen-based models
- not competitive as a main benchmark candidate

### Decision
Hermes2 was dropped from later consideration.

---

## Mistral-Small-3.2-24B-Instruct calibration

`mistralai/Mistral-Small-3.2-24B-Instruct-2506` was tested as a larger candidate model.

### Result
- `Pass^1 = 0.00`

### Observations
- repeated context-window failures under the tested configuration
- high token usage and high cost
- poor return relative to compute cost

### Decision
Mistral-Small-3.2-24B-Instruct was dropped because it was expensive and ineffective in clean calibration.

---

## DeepSeek-R1-Distill-Qwen-7B and 14B screening

The following models were explored:

- `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B`
- `deepseek-ai/DeepSeek-R1-Distill-Qwen-14B`

### Result
Neither model was advanced beyond clean calibration.

### Observations
- the installed `vLLM` version on the project environment did not support the intended reasoning-mode serving path
- without reasoning-aware serving, the practical advantage of the reasoning-distilled checkpoints was reduced
- this made both models a poor fit for the tau2 tool-use setup

### Decision
The DeepSeek-R1-Distill-Qwen models were not continued in the main benchmark path.

---

## Gemma-2-9B-IT calibration

`gemma-2-9b-it` was screened during clean calibration using the clearest available Gemma sample log:

- `Rollout-Gemma-tau2-116938.out`

### Result
The run exposed a clear chat-schema compatibility problem rather than a clean task-performance result.

### Observations
- the hosted vLLM endpoint rejected system-role messages
- retries repeated the same request-shape failure
- the run is most useful as evidence of system-role compatibility issues for Gemma in this tau2 setup

### Decision
`gemma-2-9b-it` was not treated as a strong clean-calibration candidate without additional chat-template or message-format adjustment.

---

## Early Llama-3.1-8B-Instruct fallback attempt

`Llama-3.1-8B-Instruct` was considered as an early fallback candidate.

### Result
It did not emerge as a preferred clean-calibration candidate.

### Observations
The broader run history showed several configuration-level issues, including:
- tool-call and parser setup mismatches
- auto-tool-choice configuration errors
- chat-template or message-format issues
- context-window failures in some runs

### Decision
Although Llama remained relevant as a fallback option at one stage, it was not the strongest clean-calibration choice relative to the `Qwen3-14B` setup.

---

## Calibration takeaway

The clean calibration stage showed that model selection for this project depended on more than general instruction quality. The most important factors were:

- stable vLLM serving on Tillicum
- compatibility with tool-calling and parser configuration
- context-window fit
- actual task completion performance in tau2 tool-using agent tasks

The calibration stage was therefore used to narrow the candidate set before the main rollout work.

---

## Summary of calibration-stage screening

| Model | What was tested | Outcome | Decision |
| --- | --- | --- | --- |
| `Qwen3-14B` | thinking ON vs OFF | ON performed about 2x better than OFF | keep thinking ON |
| `Qwen/Qwen2.5-7B-Instruct` | clean calibration | useful smaller baseline, but weaker than Qwen3-14B | comparison only |
| `Qwen3-8B` | clean calibration via `Rollout-Qwen3-8B-tau2-116756.out` and `Rollout-Qwen3-8B-tau2-116769.out` | intermediate Qwen baseline; not stronger than Qwen3-14B thinking ON | comparison only |
| `NousResearch/Hermes-2-Pro-Llama-3-8B` | clean calibration | weak agent-task performance | drop |
| `mistralai/Mistral-Small-3.2-24B-Instruct-2506` | clean calibration | `Pass^1 = 0.00`, high token cost, context issues | drop |
| `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` | clean screening | poor fit with local serving setup | do not continue |
| `deepseek-ai/DeepSeek-R1-Distill-Qwen-14B` | clean screening | poor fit with local serving setup | do not continue |
| `gemma-2-9b-it` | clean calibration via `Rollout-Gemma-tau2-116938.out` | clearest Gemma sample; system-role compatibility failure | not preferred without message-format adjustment |
| `Llama-3.1-8B-Instruct` | early fallback screening | unstable due to configuration/runtime issues | not preferred |
