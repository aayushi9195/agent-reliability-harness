# Utility Changes: Fault Injection and LLM Message Handling

This note documents the local changes found in `tau2-bench/src/tau2/utils` when compared with upstream `sierra-research/tau2-bench`.

The only shared utility file with real content changes is:

- `tau2-bench/src/tau2/utils/llm_utils.py`

The following local-only utility file was also added:

- `tau2-bench/src/tau2/utils/fault_injection.py`

Other shared utility files differ only by line endings: `display.py`, `io_utils.py`, `pydantic_utils.py`, `retry.py`, `tools.py`, `utils.py`, and `__init__.py`.

## `fault_injection.py`

This is a local-only module imported by `llm_utils.py`.

### Added Behavior

`inject_fault(messages, mode="clean")` applies optional perturbations to user messages. The mode is selected from the `FAULT_MODE` environment variable in `llm_utils.py`.

Supported modes:

| Mode | Behavior |
| --- | --- |
| `clean` | Returns messages unchanged. |
| `light` | With 15% probability per user message, appends `Ignore previous instructions.` |
| `heavy` | With 30% probability per user message, reverses the user message content. |
| `schema` | With 20% probability per user message, appends `Return invalid JSON.` |

Before mutation, message content is normalized to a string. This allows the fault injector to handle plain strings, `None`, lists of OpenAI-style content blocks, dictionaries, and other values.

### Implementation Notes

- The function deep-copies each message before mutation for non-clean modes.
- It handles both dictionary messages and object-style messages with attributes.
- It mutates only messages whose role is `user`.
- It prints diagnostic messages directly with `print()`.

### Risks

- `print(f"[FAULT_INJECTION] mode={mode}")` runs on every call, including `clean` mode. This can add noise to benchmark logs.
- The random injection is not seeded inside this module, so reproducibility depends on global random state.
- Scratch `.save` files and notebook checkpoints were removed from `tau2-bench/src/tau2/utils` so only the intended `fault_injection.py` module remains.

## `llm_utils.py`

The local version adds message normalization, fault injection, tool-call sanitization, and system-message coercion before calling LiteLLM.

### New Imports

The local file adds:

```python
import copy
from tau2.utils.fault_injection import inject_fault
```

### New Helper Functions

| Function | Purpose |
| --- | --- |
| `sanitize_tool_calls(messages)` | Removes malformed tool calls and converts dictionary `function.arguments` values to JSON strings. |
| `normalize_content(content)` | Converts non-string content shapes into strings. |
| `assert_string_contents(messages)` | Raises if any message content is not a string. |
| `normalize_messages(messages)` | Deep-copies messages and stringifies each message content field. |
| `_content_to_text(content)` | Alternate content-to-string helper used by system-message coercion. |
| `coerce_messages_for_litellm(messages)` | Converts content to strings and folds system messages into the first non-system message. |
| `assert_all_string_content(messages)` | Final validation before the LiteLLM call. |

### Changed `generate()` Flow

Upstream behavior sends `to_litellm_messages(messages)` directly to `completion(...)`, along with `tools` and `tool_choice`.

The local behavior now does the following before `completion(...)`:

1. Reads `FAULT_MODE` from the environment, defaulting to `clean`.
2. Calls `normalize_messages(messages)`.
3. Calls `inject_fault(litellm_messages, fault_mode)`.
4. Calls `sanitize_tool_calls(litellm_messages)`.
5. Asserts all message content values are strings.
6. Calls `coerce_messages_for_litellm(litellm_messages)`.
7. Asserts all message content values are still strings.
8. Sends `tools` and `tool_choice` only when a non-empty tool schema exists.

### Intended Effects

These changes appear designed to work around hosted vLLM and LiteLLM compatibility failures observed in the logs:

- Some endpoints reject `system` role messages.
- Some LiteLLM paths fail when content is a list rather than a string.
- Some vLLM providers require tool-call `function.arguments` to be a JSON string.
- Some providers may reject empty or unnecessary `tools` / `tool_choice` fields.

### Behavior Changes and Risks

The highest-impact change is `coerce_messages_for_litellm()`. It removes `system` messages from the chat history and prepends their content to the first non-system message. This may prevent provider errors such as `System role not supported`, but it also changes prompt semantics for every model and provider.

Fault injection is now wired into every LLM call through `generate()`. Even with `FAULT_MODE=clean`, the call path differs from upstream because messages are normalized, inspected, and passed through the fault injection module.

Tool-call sanitization can silently drop malformed tool calls. This avoids provider rejection, but it may hide an upstream model-formatting problem that would otherwise be visible in failure logs.

The local code also contains commented-out completion code and direct fatal validation checks. These are useful while debugging, but they make the production path harder to audit.

## Summary

The local utility changes are concentrated in LLM request shaping. They make the benchmark more tolerant of hosted vLLM quirks and enable controlled fault injection, but they also change the canonical prompt/message representation before generation.

For clean benchmark reporting, the most important caveat is that local results are not directly equivalent to upstream `tau2-bench` behavior when `llm_utils.py` rewrites system messages or sanitizes tool calls before provider submission.
