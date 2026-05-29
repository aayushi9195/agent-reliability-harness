from __future__ import annotations

import csv
import ast
import json
import re
from collections import OrderedDict
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent
LOG_DIR = ROOT / "logs"
OUT_MD = ROOT / "errors_by_model.md"
OUT_CSV = ROOT / "errors_by_model.csv"
OUT_JSON = ROOT / "errors_by_model.json"
OUT_CONFIG_CSV = ROOT / "model_configurations.csv"
OUT_CONFIG_JSON = ROOT / "model_configurations.json"

LOG_GLOBS = ("*.out", "*.log", "*.txt")
ERROR_WORDS = (
    " error",
    "error:",
    "| ERROR",
    "exception",
    "failed",
    "cancelled",
    "killed",
    "time limit",
)
IGNORE_SUBSTRINGS = (
    "This model isn't mapped yet.",
    "model_prices_and_context_window.json",
    "Max Errors:",
    "Disk quota exceeded",
    "Permission denied",
    "Failed to get git hash",
    "not a git repository",
    "Stopping at filesystem boundary",
    "CANCELLED AT",
    "DUE TO TIME LIMIT",
    "DUE to SIGNAL Terminated",
    "DUE TO SIGNAL Terminated",
    "vLLM failed to start",
)

TS_PATTERNS = (
    re.compile(r"(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(?:\.\d+)?)"),
    re.compile(r"\[(?P<ts>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?)\]"),
)
INNER_TS = re.compile(r"\b(?P<md>\d{2}-\d{2}) (?P<hms>\d{2}:\d{2}:\d{2})\b")
MODEL_LINE = re.compile(r"^Model:\s*(?P<model>.+?)\s*$")
MODEL_ARG = re.compile(r"\bmodel=(?P<model>[^,\s]+)")
JOB_ID = re.compile(r"-(?P<job>\d{5,})(?:\.|$)")
SIMPLE_CONFIG_LINE = re.compile(r"^(?P<key>[A-Za-z][A-Za-z0-9 /_-]*):\s*(?P<value>.+?)\s*$")
NON_DEFAULT_ARGS = re.compile(r"non-default args:\s*(?P<args>\{.+\})")
VLLM_BANNER = re.compile(r"\bversion\s+(?P<version>\S+)\s+model\s+(?P<model>\S+)")
ARCHITECTURE = re.compile(r"Resolved architecture:\s*(?P<architecture>\S+)")
MAX_MODEL_LEN = re.compile(r"Using max model len\s+(?P<max_model_len>\d+)")
EXCEPTION_SUMMARY = re.compile(r"\b(?:[\w.]+)?(?:Error|Exception):\s+.+")
LOG_PREFIX = re.compile(r"^(?:\([^)]*\)\s*)?(?:ERROR|CRITICAL)\s+\d{2}-\d{2} \d{2}:\d{2}:\d{2}\s+\[[^]]+\]\s*")
LOGURU_PREFIX = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(?:\.\d+)? \| \w+\s+\| ")
MODEL_ALIASES = {
    "Qwen/Qwen3-8B": "Qwen3-8B",
    "qwen2.5-7b-instruct": "Qwen/Qwen2.5-7B-Instruct",
    "qwen": "qwen3-14b",
    "qwen3-14b-10": "qwen3-14b",
    "qwen-tau2-50": "qwen3-14b",
    "qwen-calib-50": "qwen3-14b",
}
REPORT_GROUPS = (
    (
        "Mistral/Mixtral Failure Summaries",
        "Focused summaries for the Mistral-family models.",
        (
            "Mistral-7B-Instruct-v0.1",
            "mistral32-10",
            "mistral-small32-24b",
            "Mixtral-8x7B-Instruct-v0",
        ),
    ),
    (
        "Llama Failure Summary",
        "Focused summary for Llama-3.1-8B-Instruct.",
        ("Llama-3.1-8B-Instruct",),
    ),
    (
        "Qwen Failure Summary",
        "Focused summary for Qwen-family model labels.",
        (
            "Qwen/Qwen2.5-7B-Instruct",
            "Qwen3-8B",
            "qwen3-14b",
        ),
    ),
    (
        "Gemma Failure Summary",
        "Focused summary for gemma-2-9b-it.",
        ("gemma-2-9b-it",),
    ),
)
SKIP_STACK_SNIPPETS = (
    "Traceback (most recent call last):",
    'File "',
    "  File ",
    "await ",
    "return ",
    "raise ",
    "^^^^",
    "self.",
    "super().",
    "raw_response =",
    "response =",
    "result =",
    "content =",
)


def parse_time(line: str) -> datetime | None:
    for pattern in TS_PATTERNS:
        match = pattern.search(line)
        if not match:
            continue
        raw = match.group("ts").replace("T", " ")
        try:
            return datetime.fromisoformat(raw)
        except ValueError:
            return None
    return None


def parse_inner_time(line: str, year: int = 2026) -> datetime | None:
    match = INNER_TS.search(line)
    if not match:
        return None
    try:
        return datetime.fromisoformat(f"{year}-{match.group('md')} {match.group('hms')}")
    except ValueError:
        return None


def filename_model(path: Path) -> str:
    name = path.stem
    name = re.sub(r"^vllm-", "", name, flags=re.IGNORECASE)
    name = re.sub(r"^(rollout|test)-", "", name, flags=re.IGNORECASE)
    name = re.sub(r"-?\d{5,}$", "", name)
    return name or "unknown"


def clean_config_value(value: object) -> str:
    if isinstance(value, list):
        return ", ".join(str(item) for item in value)
    return str(value).strip()


def canonical_model(model: str) -> str:
    return MODEL_ALIASES.get(model, model)


def strip_log_prefix(line: str) -> str:
    line = re.sub(r"^\([^)]*\)\s*", "", line)
    line = re.sub(r"^(?:INFO|WARNING|ERROR|CRITICAL)\s+\d{2}-\d{2} \d{2}:\d{2}:\d{2}\s+\[[^]]+\]\s*", "", line)
    return line.strip()


def config_summary(config: dict) -> str:
    priority = (
        "job_started",
        "node",
        "model",
        "fault",
        "seed",
        "range",
        "domain",
        "agent",
        "user",
        "concurrency",
        "served_model_name",
        "model_dir",
        "vllm_model",
        "architecture",
        "max_model_len",
        "dtype",
        "tool_call_parser",
        "enforce_eager",
        "output",
        "result_dir",
    )
    parts = []
    for key in priority:
        value = config.get(key)
        if value not in (None, ""):
            parts.append(f"{key}={clean_config_value(value)}")
    return "; ".join(parts) if parts else "none found"


def parse_simulation_line(line: str, config: dict) -> None:
    if "Domain:" in line:
        match = re.search(r"Domain:\s*(.*?)\s+Task Set:\s*(.*?)\s+Tasks:\s*(.*?)(?:\s{2,}|[â│]|$)", line)
        if match:
            config["domain"] = match.group(1).strip()
            config["task_set"] = match.group(2).strip()
            config["tasks"] = match.group(3).strip()
    if "Trials:" in line:
        match = re.search(r"Trials:\s*(\S+)\s+Max Steps:\s*(\S+)\s+Max Errors:\s*(\S+)", line)
        if match:
            config["trials"] = match.group(1)
            config["max_steps"] = match.group(2)
            config["max_errors"] = match.group(3)
    if "Agent:" in line:
        match = re.search(r"Agent:\s*(.*?)\s*(?:â†’|->|→)\s*(.*?)(?:\s{2,}|[â│]|$)", line)
        if match:
            config["agent"] = match.group(1).strip()
            config["agent_model"] = match.group(2).strip()
    if "User:" in line:
        match = re.search(r"User:\s*(.*?)\s*(?:â†’|->|→)\s*(.*?)(?:\s{2,}|[â│]|$)", line)
        if match:
            config["user"] = match.group(1).strip()
            config["user_model"] = match.group(2).strip()
    if "Save:" in line:
        match = re.search(r"Save:\s*(.*?)\s+Concurrency:\s*(\S+)\s+Verbose:\s*(\S+)", line)
        if match:
            config["save"] = match.group(1).strip()
            config["concurrency"] = match.group(2)
            config["verbose"] = match.group(3)


def parse_top_config(path: Path, lines: list[str]) -> dict:
    config: dict[str, object] = {"source_file": path.name, "source_model": filename_model(path)}
    for line in lines[:240]:
        stripped = strip_log_prefix(line)

        simple = SIMPLE_CONFIG_LINE.match(stripped)
        if simple:
            key = simple.group("key").strip().lower().replace(" ", "_")
            value = simple.group("value").strip()
            if key in {
                "job_started",
                "node",
                "python",
                "project_dir",
                "tau2_dir",
                "model_dir",
                "result_dir",
                "vllm_log",
                "model",
                "fault",
                "seed",
                "range",
                "output",
            }:
                config[key] = value
            elif key == "python" and "python_executable" not in config:
                config["python_executable"] = value

        if re.fullmatch(r"Python \d+\.\d+\.\d+", stripped):
            config["python_version"] = stripped

        parse_simulation_line(stripped, config)

        banner = VLLM_BANNER.search(stripped)
        if banner:
            config["vllm_version"] = banner.group("version")
            config["vllm_model"] = banner.group("model")

        args_match = NON_DEFAULT_ARGS.search(stripped)
        if args_match:
            try:
                args = ast.literal_eval(args_match.group("args"))
            except (SyntaxError, ValueError):
                args = {}
            for key in (
                "model_tag",
                "model",
                "dtype",
                "enforce_eager",
                "tool_call_parser",
                "served_model_name",
                "generation_config",
                "enable_auto_tool_choice",
                "host",
            ):
                if key in args:
                    config[key if key != "model" else "vllm_model"] = args[key]

        architecture = ARCHITECTURE.search(stripped)
        if architecture:
            config["architecture"] = architecture.group("architecture")

        max_len = MAX_MODEL_LEN.search(stripped)
        if max_len:
            config["max_model_len"] = max_len.group("max_model_len")

    return config


def normalize_message(line: str) -> str:
    line = re.sub(r"^\[?\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:\.\d+)?\]?\s*", "", line)
    line = LOGURU_PREFIX.sub("", line)
    line = LOG_PREFIX.sub("", line)
    line = re.sub(r"^\([^)]*\)\s*", "", line)
    line = re.sub(r"\btask \d+\b", "task <id>", line, flags=re.IGNORECASE)
    line = re.sub(r"\bJOB \d+\b", "JOB <id>", line)
    line = re.sub(r"\bON g\d+\b", "ON <node>", line)
    line = re.sub(r"2026-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}", "<timestamp>", line)
    line = re.sub(r"\s+", " ", line).strip()
    return line


def is_error_line(line: str) -> bool:
    lower = f" {line.lower()}"
    return any(word.lower() in lower for word in ERROR_WORDS)


def is_summary_line(line: str) -> bool:
    cleaned = normalize_message(line)
    if cleaned.startswith(("│", "â”‚")):
        return False
    if any(snippet in cleaned for snippet in SKIP_STACK_SNIPPETS):
        return False
    if EXCEPTION_SUMMARY.search(cleaned):
        return True
    if re.search(r"\berror:\s+.+", cleaned, flags=re.IGNORECASE):
        return True
    if re.search(r"\b(cancelled|failed|killed|time limit)\b", cleaned, flags=re.IGNORECASE):
        return True
    return False


def ignored(line: str) -> bool:
    return any(part in line for part in IGNORE_SUBSTRINGS)


def source_key(path: Path) -> tuple[int, str]:
    match = JOB_ID.search(path.name)
    job = int(match.group("job")) if match else 0
    return job, path.name.lower()


def read_lines(path: Path) -> list[str]:
    data = path.read_bytes()
    for encoding in ("utf-8", "cp1252", "latin-1"):
        try:
            return data.decode(encoding).splitlines()
        except UnicodeDecodeError:
            pass
    return data.decode("utf-8", errors="replace").splitlines()


def collect_records() -> list[dict]:
    records: list[dict] = []
    paths: list[Path] = []
    for glob in LOG_GLOBS:
        paths.extend(LOG_DIR.glob(glob))

    for path in sorted(set(paths), key=source_key):
        if path.name in {OUT_MD.name, OUT_CSV.name, OUT_JSON.name}:
            continue

        lines = read_lines(path)
        source_config = parse_top_config(path, lines)
        model = canonical_model(clean_config_value(source_config.get("model", filename_model(path))))
        last_ts: datetime | None = None

        for line_no, line in enumerate(lines, start=1):
            found_ts = parse_time(line) or parse_inner_time(line)
            if found_ts:
                last_ts = found_ts

            model_match = MODEL_LINE.match(line.strip())
            if model_match:
                model = canonical_model(model_match.group("model"))

            model_arg = MODEL_ARG.search(line)
            line_model = canonical_model(model_arg.group("model") if model_arg else model)

            if not is_error_line(line) or ignored(line) or not is_summary_line(line):
                continue

            records.append(
                {
                    "time": found_ts or last_ts,
                    "model": line_model,
                    "message": normalize_message(line),
                    "file": path.relative_to(ROOT).as_posix(),
                    "line": line_no,
                    "config": source_config,
                }
            )

    return records


def dedupe(records: list[dict]) -> list[dict]:
    def sort_key(record: dict) -> tuple[datetime, str, int]:
        return record["time"] or datetime.max, record["file"], record["line"]

    seen: OrderedDict[tuple[str, str], dict] = OrderedDict()
    for record in sorted(records, key=sort_key):
        key = (record["model"], record["message"])
        if key not in seen:
            seen[key] = {
                **record,
                "count": 0,
                "sources": [],
                "configs": OrderedDict(),
            }
        seen[key]["count"] += 1
        seen[key]["sources"].append(f"{record['file']}:{record['line']}")
        config_key = json.dumps(
            {key: value for key, value in record["config"].items() if key not in {"source_file"}},
            sort_keys=True,
            default=str,
        )
        seen[key]["configs"][config_key] = record["config"]
    for record in seen.values():
        record["configs"] = list(record["configs"].values())
    return list(seen.values())


def write_outputs(records: list[dict]) -> None:
    by_model: OrderedDict[str, list[dict]] = OrderedDict()
    for record in records:
        by_model.setdefault(record["model"], []).append(record)

    grouped_record_count = sum(
        len(by_model.get(model, []))
        for _, _, models in REPORT_GROUPS
        for model in models
    )

    with OUT_MD.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("# Errors by Model\n\n")
        handle.write("Ignored: LiteLLM model-price/context-window mapping errors.\n\n")
        handle.write(
            "This human-readable appendix is grouped to match the report sections in README.md. "
            "The CSV and JSON files retain the full parsed inventory.\n\n"
        )
        handle.write(f"Unique error records in these report groups: {grouped_record_count}\n\n")
        handle.write("## Contents\n\n")
        for group_title, group_description, _ in REPORT_GROUPS:
            anchor = group_title.lower().replace("/", "").replace(" ", "-")
            handle.write(f"- [{group_title}](#{anchor}): {group_description}\n")
        handle.write("\n")

        for group_title, group_description, models in REPORT_GROUPS:
            handle.write(f"## {group_title}\n\n")
            handle.write(f"{group_description}\n\n")
            for model in models:
                model_records = by_model.get(model, [])
                if not model_records:
                    continue
                handle.write(f"### {model}\n\n")
                for record in model_records:
                    ts = record["time"].isoformat(sep=" ") if record["time"] else "no timestamp"
                    sources = ", ".join(record["sources"][:5])
                    more = "" if len(record["sources"]) <= 5 else f", ... +{len(record['sources']) - 5} more"
                    configs = record["configs"][:3]
                    config_more = "" if len(record["configs"]) <= 3 else f"; ... +{len(record['configs']) - 3} more configs"
                    handle.write(f"- {ts} | count={record['count']} | {record['message']}\n")
                    handle.write(f"  Source: {sources}{more}\n")
                    handle.write(f"  Config: {' | '.join(config_summary(config) for config in configs)}{config_more}\n")
                handle.write("\n")

    with OUT_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("time", "model", "count", "message", "sources", "configs_json"))
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "time": record["time"].isoformat(sep=" ") if record["time"] else "",
                    "model": record["model"],
                    "count": record["count"],
                    "message": record["message"],
                    "sources": "; ".join(record["sources"]),
                    "configs_json": json.dumps(record["configs"], sort_keys=True, default=str),
                }
            )

    with OUT_JSON.open("w", encoding="utf-8") as handle:
        json.dump(
            [
                {
                    "time": record["time"].isoformat(sep=" ") if record["time"] else None,
                    "model": record["model"],
                    "count": record["count"],
                    "message": record["message"],
                    "sources": record["sources"],
                    "configs": record["configs"],
                }
                for record in records
            ],
            handle,
            indent=2,
        )


def collect_configs() -> list[dict]:
    configs = []
    paths: list[Path] = []
    for glob in LOG_GLOBS:
        paths.extend(LOG_DIR.glob(glob))
    for path in sorted(set(paths), key=source_key):
        if path.name in {OUT_MD.name, OUT_CSV.name, OUT_JSON.name, OUT_CONFIG_CSV.name, OUT_CONFIG_JSON.name}:
            continue
        config = parse_top_config(path, read_lines(path))
        config["summary"] = config_summary(config)
        configs.append(config)
    return configs


def write_config_outputs(configs: list[dict]) -> None:
    keys = []
    for config in configs:
        for key in config:
            if key not in keys:
                keys.append(key)

    with OUT_CONFIG_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for config in configs:
            writer.writerow({key: clean_config_value(config.get(key, "")) for key in keys})

    with OUT_CONFIG_JSON.open("w", encoding="utf-8") as handle:
        json.dump(configs, handle, indent=2, default=str)


def main() -> None:
    records = dedupe(collect_records())
    configs = collect_configs()
    write_outputs(records)
    write_config_outputs(configs)
    print(f"Wrote {len(records)} unique error records")
    print(f"Wrote {len(configs)} source configuration records")
    print(OUT_MD.name)
    print(OUT_CSV.name)
    print(OUT_JSON.name)
    print(OUT_CONFIG_CSV.name)
    print(OUT_CONFIG_JSON.name)


if __name__ == "__main__":
    main()
