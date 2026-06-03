# Best Log Samples By Model

Use these as the clearest representative rollout logs for the report. Every log file listed below contains a `Simulation Overview` section.

| Model | Best sample log | Why this is a good sample |
| --- | --- | --- |
| `Llama-3.1-8B-Instruct` | [llama31-8b-10-119321.out](https://github.com/aayushi9195/agent-reliability-harness/blob/main/logs/llama31-8b-10-119321.out) | Contains repeated `Simulation Overview` sections and clean Llama 3.1 8B rollout evidence. |
| `Qwen/Qwen2.5-7B-Instruct` | [Rollout-Qwen2.5-tau2-119153.out](https://github.com/aayushi9195/agent-reliability-harness/blob/main/logs/Rollout-Qwen2.5-tau2-119153.out) | Contains `Simulation Overview` sections and Qwen2.5 hosted-vLLM rollout behavior. |
| `Qwen3-8B` | [Rollout-Qwen3-8B-tau2-116769.out](https://github.com/aayushi9195/agent-reliability-harness/blob/main/logs/Rollout-Qwen3-8B-tau2-116769.out) | Clear run metadata with `Simulation Overview` sections for Qwen3-8B retail tasks. |
| `Qwen3-14B` | [qwen3-14b-10-119228.out](https://github.com/aayushi9195/agent-reliability-harness/blob/main/logs/qwen3-14b-10-119228.out) | Representative Qwen3-14B rollout log with multiple `Simulation Overview` sections. |
| `Mistral-Small-3.2-24B-Instruct` | [mistral32-10-119320.out](https://github.com/aayushi9195/agent-reliability-harness/blob/main/logs/mistral32-10-119320.out) | Contains `Simulation Overview` sections and client-side context-window overflow evidence. |
| `NousResearch/Hermes-2-Pro-Llama-3-8B` | [hermes10-119142.out](https://github.com/aayushi9195/agent-reliability-harness/blob/main/logs/hermes10-119142.out) | Representative Hermes rollout log with `Simulation Overview` sections. |

Server-side `vllm-*.log` files are excluded from this list because they do not contain `Simulation Overview`; use them only when discussing serving configuration or server-side failures.

## Tool-Call Error Log Samples

Use these logs when discussing tool-call, parser, chat-template, and function-call JSON failures. This list includes rollout logs and server-side `vllm-*.log` files because some parser failures are only visible on the serving side.

| Model or family | Tool-call issue | Evidence log | Why this is useful |
| --- | --- | --- | --- |
| `Llama-3.1-8B-Instruct` | Single-tool-call constraint | [test-llama-tau2-116750.out](https://github.com/aayushi9195/agent-reliability-harness/blob/main/logs/test-llama-tau2-116750.out) | Shows `This model only supports single tool-calls at once!`, indicating a model/provider constraint on parallel tool calls. |
| `Llama-3.1-8B-Instruct` | Auto tool-choice configuration missing | [Rollout-llama-tau2-117271.out](https://github.com/aayushi9195/agent-reliability-harness/blob/main/logs/Rollout-llama-tau2-117271.out) | Shows repeated failures where `"auto" tool choice requires --enable-auto-tool-choice and --tool-call-parser`. |
| `Llama-3.1-8B-Instruct` | Missing tool chat template | [Rollout-llama-tau2-117978.out](https://github.com/aayushi9195/agent-reliability-harness/blob/main/logs/Rollout-llama-tau2-117978.out) | Shows the missing `examples/tool_chat_template_llama3.1_json.jinja` template path error. |
| `Llama-3.1-8B-Instruct` | Invalid tool-call parser | [Rollout-llama-tau2-118169.out](https://github.com/aayushi9195/agent-reliability-harness/blob/main/logs/Rollout-llama-tau2-118169.out) | Shows `invalid tool call parser: llama3`, where the valid parser should be `llama3_json`. |
| `Qwen3-8B` | Invalid tool-call parser | [Rollout-Qwen2.5-tau2-118543.out](https://github.com/aayushi9195/agent-reliability-harness/blob/main/logs/Rollout-Qwen2.5-tau2-118543.out) | Shows `invalid tool call parser: qwen25`, useful for parser-selection discussion. |
| `Qwen/Qwen2.5-7B-Instruct` | Tool-call JSON decode error | [vllm-qwen-115800.log](https://github.com/aayushi9195/agent-reliability-harness/blob/main/logs/vllm-qwen-115800.log) | Shows repeated `json.decoder.JSONDecodeError: Extra data`, consistent with malformed tool-call JSON extraction. |
| `Qwen/Qwen2.5-7B-Instruct` | Tool-call JSON decode error | [vllm-qwen-115814.log](https://github.com/aayushi9195/agent-reliability-harness/blob/main/logs/vllm-qwen-115814.log) | Shows `json.decoder.JSONDecodeError: Unterminated string`, another malformed tool-call JSON case. |
| `Mixtral-8x7B-Instruct-v0` | Auto tool-choice configuration missing | [Rollout-Min3-8B-tau2-119512.out](https://github.com/aayushi9195/agent-reliability-harness/blob/main/logs/Rollout-Min3-8B-tau2-119512.out) | Shows hosted vLLM rejecting auto tool choice without the required parser setup. |
| `Mixtral-8x7B-Instruct-v0` | Tool parser/tokenizer mismatch | [Rollout-Min3-8B-tau2-119513.out](https://github.com/aayushi9195/agent-reliability-harness/blob/main/logs/Rollout-Min3-8B-tau2-119513.out) | Shows `Mistral Tool Parser could not locate the tool call token in the tokenizer!`. |
| `Mixtral-8x7B-Instruct-v0` | Startup missing tool parser | [Rollout-Min3-8B-tau2-119219.out](https://github.com/aayushi9195/agent-reliability-harness/blob/main/logs/Rollout-Min3-8B-tau2-119219.out) | Shows startup validation failure: `--enable-auto-tool-choice requires --tool-call-parser`. |
| `Mixtral-8x7B-Instruct-v0` | Missing tool-use chat template | [Rollout-Min3-8B-tau2-119430.out](https://github.com/aayushi9195/agent-reliability-harness/blob/main/logs/Rollout-Min3-8B-tau2-119430.out) | Shows the configured `mixtral_tool_use.jinja` template path did not exist. |
