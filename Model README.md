# Model Failure Appendix

This directory contains derived summaries of model failures from tau2 benchmark runs. Raw rollout and vLLM logs are stored in [logs](C:/jh/LOGS/logs), and the appendix files were generated with [summarize_errors.py](C:/jh/LOGS/summarize_errors.py).

## Contents

| Section | Description |
| --- | --- |
| [Summary Counts](#summary-counts) | Overall deduplicated failure counts by model label. |
| [Main Failure Themes](#main-failure-themes) | Cross-model failure categories. |
| [Mistral/Mixtral Failure Summaries](#mistralmixtral-failure-summaries) | Focused summaries for the Mistral-family models. |
| [Llama Failure Summary](#llama-failure-summary) | Focused summary for `Llama-3.1-8B-Instruct`. |
| [Qwen Failure Summary](#qwen-failure-summary) | Focused summary for Qwen-family model labels. |
| [Gemma Failure Summary](#gemma-failure-summary) | Focused summary for `gemma-2-9b-it`. |
| [vLLM Startup Issues](#vllm-startup-issues) | Startup failures excluded from model-failure counts. |

## Scope

The failure inventory scans `*.out`, `*.log`, and `*.txt` files in [logs](C:/jh/LOGS/logs). It extracts timestamped error records, groups them by model, deduplicates repeated error messages, and attaches available run configuration from the top of each log.

The following recurring LiteLLM cost-accounting message is intentionally excluded because it does not indicate a run failure:

```text
tau2.utils.llm_utils:get_response_cost:129 - This model isn't mapped yet...
```

Filesystem and job-environment issues are also excluded from the model-failure inventory. This includes disk quota errors, permission errors, missing git metadata, scheduler cancellations, time limits, and vLLM startup failures caused by the surrounding environment.

## Generated Files

| File | Purpose |
| --- | --- |
| [errors_by_model.md](C:/jh/LOGS/errors_by_model.md) | Human-readable deduplicated failure log grouped by model and ordered by first observed time. |
| [errors_by_model.csv](C:/jh/LOGS/errors_by_model.csv) | Spreadsheet-friendly version of the deduplicated failure log. |
| [errors_by_model.json](C:/jh/LOGS/errors_by_model.json) | Structured failure log, including per-error source locations and configs. |
| [model_configurations.csv](C:/jh/LOGS/model_configurations.csv) | One configuration record per source log. |
| [model_configurations.json](C:/jh/LOGS/model_configurations.json) | Structured configuration inventory. |

## Extraction Method

The parser records lines containing error-like signals such as `ERROR`, `Exception`, and `failed`. It then filters out benchmark report prose, stack-frame noise, and excluded environment/job issues so the output focuses on actionable model and runtime failure summaries rather than every line in a traceback.

For each source log, the parser also extracts available configuration metadata from the header or vLLM startup block, including:

| Configuration field | Examples |
| --- | --- |
| Job metadata | job start time, node, Python executable/version |
| Benchmark metadata | model, fault, seed, task range, output/result directory |
| Simulation metadata | domain, task set, agent, user, concurrency, max steps |
| vLLM metadata | served model name, model path, dtype, architecture, max model length, tool parser, eager mode |

## Summary Counts

The current scan produced:

| Metric | Count |
| --- | ---: |
| Source log files scanned | 115 |
| Unique deduplicated error records | 237 |

`Occurrences` below counts repeated appearances of a deduplicated error type, including retry cascades. `Unique errors` counts distinct normalized error messages for that model.

| Model | Unique errors | Occurrences |
| --- | ---: | ---: |
| Llama-3.1-8B-Instruct | 15 | 3,099 |
| Qwen3-8B | 21 | 738 |
| Mixtral-8x7B-Instruct-v0 | 19 | 312 |
| Qwen/Qwen2.5-7B-Instruct | 25 | 253 |
| mistral-small32-24b | 26 | 208 |
| qwen3-14b | 21 | 274 |
| mistral32-10 | 41 | 167 |
| gemma-2-9b-it | 6 | 99 |
| Mistral-7B-Instruct-v0.1 | 6 | 99 |

## Main Failure Themes

The most common failure classes observed in the deduplicated records are:

| Theme | Examples |
| --- | --- |
| vLLM/server unavailable | `Connection refused`, `InternalServerError`. |
| Model routing/configuration mismatch | `NotFoundError`, model name does not exist, served model name mismatch. |
| Chat/tool configuration mismatch | `"auto" tool choice requires --enable-auto-tool-choice and --tool-call-parser`, invalid tool parser, unsupported system role. |
| Context window exceeded | Prompt length exceeded max context length, commonly 16,384, 32,768, or 40,960 token limits depending on configuration. |
| External API pressure | OpenAI `RateLimitError` and `AuthenticationError` in runs using API-backed user/model calls. |
| Tool-call parsing failures | JSON decode errors when extracting tool calls from model responses. |

## Mistral/Mixtral Failure Summaries

These subsections summarize the requested Mistral-family entries from the filtered failure inventory. Counts include repeated retry/permanent-failure records where they are part of the same run behavior.

### Mistral-7B-Instruct-v0.1

Configuration observed:

| Field | Value |
| --- | --- |
| Source log | `Rollout-Min3-8B-tau2-116780.out` |
| Model label | `Mistral-7B-Instruct-v0.1` |
| Fault / seed / range | `clean` / `2` / `0 -> 10` |
| Domain / agent / user | `retail` / `llm_agent` / `user_simulator` |

Chronological errors:

| First observed | Count | Error summary |
| --- | ---: | --- |
| 2026-05-15 10:14:15.876 | 44 | `litellm.BadRequestError`: after the optional system message, roles must alternate `user/assistant/user/assistant/...`. |
| 2026-05-15 10:14:15.876 | 11 | Retry 1/3 for the same `BadRequestError`. |
| 2026-05-15 10:14:20.275 | 11 | Retry 2/3 for the same `BadRequestError`. |
| 2026-05-15 10:14:25.922 | 11 | Retry 3/3 for the same `BadRequestError`. |
| 2026-05-15 10:14:34.293 | 22 | Tasks failed permanently after 4 attempts due to the same role-ordering error. |

Interpretation: this run failed on chat message formatting rather than context length or server availability.

### mistral32-10

Configuration observed:

| Field | Value |
| --- | --- |
| Source logs | `mistral32-10-119306.out`; `mistral32-10-119320.out` |
| Simulation | `retail`, `llm_agent`, `user_simulator`, concurrency `3` |
| Paired vLLM logs | `vllm-mistral-small32-24b-119306.log`; `vllm-mistral-small32-24b-119320.log` |
| Served model | `mistralai/Mistral-Small-3.2-24B-Instruct-2506` |
| vLLM config | `dtype=bfloat16`, `tool_call_parser=mistral`, `architecture=PixtralForConditionalGeneration`, `max_model_len=8192`, `enforce_eager=True` |

Chronological errors:

| First observed | Count | Error summary |
| --- | ---: | --- |
| 2026-05-19 15:45:18.859 | 40 | `litellm.NotFoundError`: OpenAI reported model `gpt-4o-mini` does not exist. |
| 2026-05-19 15:45:18.865-15:45:21.306 | 30 | Retries 1-3 for the `gpt-4o-mini` `NotFoundError`. |
| 2026-05-19 15:45:22.530 | 30 | Permanent task failures after 4 attempts from the same `gpt-4o-mini` `NotFoundError`. |
| 2026-05-19 16:00:38.776-16:03:03.417 | 67 | Context-window failures against an 8,192-token limit; observed prompts ranged from 8,213 to 9,698 input tokens. |

Interpretation: this label shows two distinct failure modes: an upstream OpenAI model-name/API issue in the earlier run, then context-window overflow in the later run against the Mistral Small 3.2 vLLM configuration.

### mistral-small32-24b

Configuration observed:

| Field | Value |
| --- | --- |
| Source log | `vllm-mistral-small32-24b-119320.log` |
| Served model | `mistralai/Mistral-Small-3.2-24B-Instruct-2506` |
| vLLM config | `dtype=bfloat16`, `tool_call_parser=mistral`, `architecture=PixtralForConditionalGeneration`, `max_model_len=8192`, `enforce_eager=True` |

Chronological errors:

| First observed | Count | Error summary |
| --- | ---: | --- |
| 2026-05-19 16:00:32 | 104 | `ERROR: Exception in ASGI application` wrapper records emitted by the vLLM server. |
| 2026-05-19 16:00:32-16:03:03 | 104 | `vllm.exceptions.VLLMValidationError`: prompt exceeded the 8,192-token context limit. Observed input-token values included 8,213, 8,690, 8,886, 9,280, 9,413, 9,556, and 9,698. |

Interpretation: these vLLM-side records match the later `mistral32-10` client-side context-window failures. The server was reachable, but rejected over-length prompts.

### Mixtral-8x7B-Instruct-v0

Configuration observed:

| Field | Value |
| --- | --- |
| Source logs | `Rollout-Min3-8B-tau2-119219.out`; `Rollout-Min3-8B-tau2-119308.out`; `Rollout-Min3-8B-tau2-119327.out`; `Rollout-Min3-8B-tau2-119423.out`; `Rollout-Min3-8B-tau2-119430.out`; `Rollout-Min3-8B-tau2-119512.out`; `Rollout-Min3-8B-tau2-119513.out` |
| Model label | `Mixtral-8x7B-Instruct-v0` |
| Fault / seed / range | `clean` / `2` / `0 -> 10` |
| Runtime simulation | `retail`, `llm_agent`, `user_simulator` where present |
| vLLM config seen in startup log | local model path `/gpfs/projects/imt526a/group2/agent-reliability/models/Mixtral-8x7B-Instruct-v0`, `dtype=bfloat16`, `tool_call_parser=hermes`, `architecture=MixtralForCausalLM`, `max_model_len=16384` |

Chronological errors:

| First observed | Count | Error summary |
| --- | ---: | --- |
| no timestamp | 1 | Startup validation: `--enable-auto-tool-choice requires --tool-call-parser`. |
| no timestamp | 1 | Startup CLI mismatch: `vllm: error: unrecognized arguments: --disable-log-requests`. |
| no timestamp | 1 | Startup template path error: configured `mixtral_tool_use.jinja` did not exist. |
| 2026-05-19 15:37:29 | 1 | Startup validation: tensor-parallel world size 2 exceeded the 1 available GPU. |
| 2026-05-19 16:37:47.205 | 44 | `litellm.BadRequestError`: after the optional system message, roles must alternate `user/assistant/user/assistant/...`. |
| 2026-05-19 16:37:47.205-16:38:01.433 | 99 | Retries and permanent failures from the same role-ordering error. |
| 2026-05-20 07:12:41.186 | 44 | `litellm.BadRequestError`: `"auto" tool choice requires --enable-auto-tool-choice and --tool-call-parser to be set`. |
| 2026-05-20 07:12:56.167 | 11 | Permanent failures from the same auto-tool-choice configuration error. |
| 2026-05-20 07:15:24.443 | 44 | `litellm.InternalServerError`: Mistral tool parser could not locate the tool-call token in the tokenizer. |
| 2026-05-20 07:15:24.443-07:15:39.920 | 66 | Retries and permanent failures from the same Mistral tool-parser/tokenizer error. |

Interpretation: Mixtral failures were dominated by tool/chat configuration problems. Several runs never reached benchmark execution because vLLM startup arguments or local templates were invalid; later runs reached generation but failed on role alternation, auto-tool-choice requirements, or parser/tokenizer incompatibility.

## Llama Failure Summary

This section is limited to the `Llama-3.1-8B-Instruct` label in the filtered failure inventory.

Configuration observed:

| Field | Value |
| --- | --- |
| Model label | `Llama-3.1-8B-Instruct` |
| Common tau2 setup | `clean` fault, retail domain where captured, `llm_agent`, `user_simulator`, often concurrency `3` |
| vLLM configs | local model path `/gpfs/projects/imt526a/group2/agent-reliability/models/Llama-3.1-8B-Instruct`; `architecture=LlamaForCausalLM`; observed `max_model_len` values `16384` and `131072`; `dtype=bfloat16` where captured; parsers included `llama3_json` and one invalid `llama3` |

Inventory counts:

| Model label | Unique errors | Occurrences |
| --- | ---: | ---: |
| `Llama-3.1-8B-Instruct` | 15 | 3,099 |

Chronological errors:

| First observed | Label | Count | Error summary |
| --- | --- | ---: | --- |
| 2026-05-15 08:29:39.402 | `Llama-3.1-8B-Instruct` | 1 | `EOFError: EOF when reading a line`. |
| 2026-05-15 09:19:43.747 | `Llama-3.1-8B-Instruct` | 346 | Single-tool-call constraint plus retry records. |
| 2026-05-15 09:29:21.581 | `Llama-3.1-8B-Instruct` | 9 | Context-window overflow at a 16,384-token limit. |
| 2026-05-15 09:29:43.274 | `Llama-3.1-8B-Instruct` | 2 | vLLM internal server error: `Already borrowed`, plus retry. |
| 2026-05-15 21:15:21.866 | `Llama-3.1-8B-Instruct` | 1,368 | Auto-tool-choice configuration error: requires `--enable-auto-tool-choice` and `--tool-call-parser` to be set. |
| 2026-05-15 21:15:26.891-21:15:35.140 | `Llama-3.1-8B-Instruct` | 1,368 | Retries and permanent failures from the same auto-tool-choice configuration error. |
| 2026-05-16 20:43:58 | `Llama-3.1-8B-Instruct` | 1 | Missing chat-template file: `examples/tool_chat_template_llama3.1_json.jinja`. |
| 2026-05-17 06:49:03 | `Llama-3.1-8B-Instruct` | 1 | Invalid tool-call parser: `llama3`; valid parser list includes `llama3_json`. |
| 2026-05-17 07:10:59.987 | `Llama-3.1-8B-Instruct` | 3 | Context-window overflow where prompt plus requested output exceeded 16,384 tokens. |

Interpretation: `Llama-3.1-8B-Instruct` failures were dominated by tool/chat configuration problems: single-tool-call constraints, auto-tool-choice setup errors, missing or invalid tool-chat parser/template configuration, and context-window overflow. A small number of records show vLLM internal handling errors such as `Already borrowed`.

## Qwen Failure Summary

This section groups the Qwen-family labels present in the filtered failure inventory: `Qwen/Qwen2.5-7B-Instruct`, `Qwen3-8B`, and `qwen3-14b`. `Qwen/Qwen3-8B` and `Qwen3-8B` are treated as the same model and reported here as `Qwen3-8B`; `qwen2.5-7b-instruct` and `Qwen/Qwen2.5-7B-Instruct` are treated as the same model and reported here as `Qwen/Qwen2.5-7B-Instruct`; `qwen`, `qwen3-14b-10`, `qwen-tau2-50`, and `qwen-calib-50` are treated as the same model and reported here as `qwen3-14b`.

Configuration observed:

| Field | Value |
| --- | --- |
| Qwen 2.5 vLLM configs | local model path `/gpfs/projects/imt526a/group2/agent-reliability/models/qwen2.5-7b-instruct`; `served_model_name=Qwen/Qwen2.5-7B-Instruct`; `architecture=Qwen2ForCausalLM`; `max_model_len=32768`; `dtype=bfloat16`; `tool_call_parser=hermes`; `enforce_eager=True` |
| Qwen 3 14B vLLM configs | `served_model_name=Qwen/Qwen3-14B`; `architecture=Qwen3ForCausalLM`; `max_model_len=40960`; `dtype=bfloat16`; `tool_call_parser=hermes`; `enforce_eager=True` |
| Qwen 3 8B rollout configs | `clean` fault, retail domain where captured, `llm_agent`, `user_simulator`, often concurrency `3`; one startup config used invalid `tool_call_parser=qwen25` |
| Calibration configs | `qwen-calib-50` and `qwen-tau2-50` are included under canonical `qwen3-14b`; these runs used retail calibration-style result directories |

Inventory counts:

| Model label | Unique errors | Occurrences |
| --- | ---: | ---: |
| `Qwen3-8B` | 21 | 738 |
| `Qwen/Qwen2.5-7B-Instruct` | 25 | 253 |
| `qwen3-14b` | 21 | 274 |

Chronological errors:

| First observed | Label | Count | Error summary |
| --- | --- | ---: | --- |
| 2026-05-08 16:04:21 | `qwen3-14b` | 38 | vLLM ASGI wrapper errors and context-window overflow at a 32,768-token limit. |
| 2026-05-13 16:43:18 | `qwen3-14b` | 4 | Engine core failed to initialize; follow-up records report failed core process state. |
| 2026-05-14 12:29:42.867-12:29:59.527 | `qwen3-14b` | 4 | OpenAI `RateLimitError` for `gpt-4o-mini`, including retry records. |
| 2026-05-14 12:30:43-12:59:40 | `qwen3-14b` | 15 | Tool-call JSON parsing failures: `Extra data` and `Unterminated string` decode errors. |
| 2026-05-14 12:35:44.961 | `qwen3-14b` | 2 | Context-window overflow at a 32,768-token limit. |
| 2026-05-14 12:52:49.569 | `qwen3-14b` | 6 | Context-window overflow at a 32,768-token limit. |
| 2026-05-14 12:55:51.701-12:56:04.174 | `qwen3-14b` | 4 | OpenAI `RateLimitError` for `gpt-4o-mini`, including retry records. |
| no timestamp | `Qwen/Qwen2.5-7B-Instruct` | 1 | CLI mismatch: `run.py` did not recognize `--fault-mode clean`. |
| 2026-05-14 15:12:49.563 | `Qwen/Qwen2.5-7B-Instruct` | 4 | Context-window overflow at a 32,768-token limit. |
| 2026-05-14 15:13:56.119-15:14:13.509 | `Qwen3-8B` | 593 | Served-model mismatch: requests failed because `Qwen/Qwen2.5-7B-Instruct` did not exist, followed by retries and permanent failures. |
| 2026-05-14 15:22:08.955 | `Qwen3-8B` | 6 | Served-model mismatch: `Qwen3-8B` did not exist. |
| 2026-05-14 15:22:11.535 | `Qwen/Qwen2.5-7B-Instruct` | 2 | Served-model mismatch: `Qwen/Qwen2.5-7B-Instruct` did not exist. |
| 2026-05-14 15:28:54.299-15:29:12.015 | `Qwen3-8B` | 20 | Served-model mismatch: `Qwen3-8B` did not exist, followed by retries and permanent failures. |
| 2026-05-14 15:36:22.389 | `Qwen3-8B` | 28 | Context-window overflow at a 40,960-token limit. |
| 2026-05-15 09:25:08.316 | `Qwen3-8B` | 19 | Context-window overflow at a 16,384-token limit. |
| 2026-05-15 09:40:43.064 | `Qwen3-8B` | 1 | CLI mismatch: `tau2` did not recognize `--fault-mode clean`. |
| 2026-05-18 09:40:52 | `Qwen3-8B` | 1 | Invalid tool-call parser: `qwen25`; valid parser list did not include it. |
| 2026-05-18 10:09:33.769 | `Qwen3-8B` | 4 | Permanent task failure from context-window overflow at a 16,384-token limit. |
| 2026-05-18 11:37:31.186 | `Qwen3-8B` | 1 | CLI mismatch: `tau2` did not recognize `--concurrency 1`. |
| 2026-05-19 09:54:42.580-09:54:55.671 | `Qwen/Qwen2.5-7B-Instruct` | 121 | `BadRequestError`: `can only concatenate str (not "list") to str`, followed by retries and permanent failures. |
| 2026-05-19 10:00:03.405 | `Qwen/Qwen2.5-7B-Instruct` | 14 | Context-window overflow at a 16,384-token limit. |
| 2026-05-19 10:02:22.907-10:02:39.922 | `Qwen/Qwen2.5-7B-Instruct` | 100 | vLLM connection refused, followed by retries and permanent failures. |
| 2026-05-19 10:12:04.979-10:12:11.824 | `qwen3-14b` | 200 | vLLM connection refused, followed by retries and permanent failures. |
| 2026-05-19 10:13:56.872-10:14:42.724 | `Qwen/Qwen2.5-7B-Instruct` | 20 | Context-window overflow at an 8,192-token limit, including permanent task failures. |
| 2026-05-19 10:18:12.034 | `Qwen/Qwen2.5-7B-Instruct` | 1 | CLI mismatch: `tau2` did not recognize `--concurrency 1`. |
| 2026-05-19 09:59:37 | `qwen3-14b` | 1 | Startup/socket issue: address already in use. |

Interpretation: Qwen failures split into four main groups: served-model naming/routing mismatches, context-window overflows at several configured limits, tool-call/parser or response-shape problems, and vLLM availability issues. The calibration runs also show external API rate limiting for the user/model side of the benchmark.

## Gemma Failure Summary

This section covers the `gemma-2-9b-it` label in the filtered failure inventory.

Configuration observed:

| Field | Value |
| --- | --- |
| Source log | `Rollout-Gemma-tau2-116938.out` |
| Source model | `Gemma-tau2` |
| Model label | `gemma-2-9b-it` |
| Fault / seed / range | `clean` / `1` / `0 -> 10` |
| Domain / agent / user | `retail` / `llm_agent` / `user_simulator` |
| Concurrency | `3` |
| Output path | `/gpfs/projects/imt526a/group2/agent-reliability/results/gemma-2-9b-it/clean/seed_1` |

Inventory counts:

| Model label | Unique errors | Occurrences |
| --- | ---: | ---: |
| `gemma-2-9b-it` | 6 | 99 |

Chronological errors:

| First observed | Count | Error summary |
| --- | ---: | --- |
| 2026-05-15 15:01:00.344 | 44 | `litellm.BadRequestError`: hosted vLLM rejected the request because `System role not supported`. |
| 2026-05-15 15:01:00.344 | 11 | Retry 1/3 for the same `System role not supported` bad request. |
| 2026-05-15 15:01:05.394 | 11 | Retry 2/3 for the same bad request. |
| 2026-05-15 15:01:10.480 | 11 | Retry 3/3 for the same bad request. |
| 2026-05-15 15:01:17.172 | 22 | Permanent task and runner-progress failures after 4 attempts due to the same unsupported system-role error. |

Interpretation: Gemma failed on chat message schema compatibility. The hosted vLLM endpoint did not accept a system-role message, so retries repeated the same client-side request-shape failure rather than exposing a context-window or server-availability problem.

## vLLM Startup Issues

The model-failure counts above exclude vLLM startup and surrounding job-environment issues, but the raw logs contain 6 explicit `vLLM failed to start` markers after applying the report scope. These are useful for appendix context because they explain runs that did not reach benchmark execution cleanly.

| Category | Count | Affected logs | Evidence |
| --- | ---: | --- | --- |
| Readiness-check false negative or delayed readiness | 1 | `Rollout-Qwen2.5-tau2-119520.out` | The log prints `vLLM failed to start`, but later shows successful `GET /v1/models` responses. |
| Invalid tool-call parser | 1 | `Rollout-Qwen2.5-tau2-118543.out` | `KeyError: invalid tool call parser: qwen25`. |
| Auto tool choice missing parser | 1 | `Rollout-Min3-8B-tau2-119219.out` | `--enable-auto-tool-choice requires --tool-call-parser`. |
| Tensor parallelism exceeds available GPUs | 1 | `Rollout-Min3-8B-tau2-119308.out` | World size 2 was requested with only 1 available GPU. |
| vLLM CLI argument mismatch | 1 | `Rollout-Min3-8B-tau2-119423.out` | `vllm: error: unrecognized arguments: --disable-log-requests`. |
| Missing chat-template file | 1 | `Rollout-Min3-8B-tau2-119430.out` | The configured `mixtral_tool_use.jinja` path did not exist. |

Interpretation: most true vLLM startup failures were configuration mismatches rather than model behavior during a completed benchmark run. One entry appears to be a readiness polling artifact because the server becomes available after the failure marker.

## Notes for Report Use

The summary is intended as an appendix-level failure inventory rather than a causal analysis. Several occurrences are retry-amplified: one underlying configuration problem can produce many retry and permanent-failure records.

When citing specific failures, prefer [errors_by_model.md](C:/jh/LOGS/errors_by_model.md) for readable evidence and use the `Source:` line to locate the original log and line number. Use [model_configurations.csv](C:/jh/LOGS/model_configurations.csv) to connect each failure back to the run configuration.

To regenerate after adding logs:

```powershell
.\.venv\Scripts\python.exe summarize_errors.py
```
