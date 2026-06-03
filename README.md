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

  tau2-bench/ contains the local tau2 benchmark codebase used to run retail and airline agent tasks. This includes local utility changes for message handling and fault injection.

  sbatch/ contains SLURM and shell scripts for launching benchmark rollouts, vLLM-backed model runs, calibration jobs, and predictor training jobs on the cluster.

  logs/ stores raw rollout logs and vLLM server logs from benchmark runs.

  results/ stores benchmark output artifacts produced by model runs, organized by model, fault mode, seed, or experiment configuration.

  metrics/ contains analysis scripts for computing harness metrics, preparing early-run predictor data, training binary failure predictors, and summarizing model/runtime errors.

  model_failure_appendix/ contains generated failure summaries, configuration inventories, and appendix-ready model failure reports.

  Model README.md documents model-level performance findings, recurring failure modes, and interpretation of benchmark reliability results.

  utils_fault_injection_changes.md documents the local tau2 utility changes, especially the added fault-injection module and LiteLLM message-normalization changes.

  clean_calibration_models.md summarizes calibration-related model screening and clean-run evaluation notes.
