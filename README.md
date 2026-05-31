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

