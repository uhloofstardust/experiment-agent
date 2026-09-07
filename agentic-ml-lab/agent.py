"""
agent.py
--------
The agent loop. Two modes:

  - real mode: uses an OpenAI-compatible chat.completions API with function
    (tool) calling. The model decides which tool to call each step.
  - mock mode: a small deterministic policy that calls the same tools in a
    sensible order, so the whole system runs end-to-end with no API key.

Both modes print the same kind of trace and stop once the harness's
experiment budget is exhausted or the agent decides it's done.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List

from tools import BudgetExceededError, Harness

SYSTEM_PROMPT = """You are an autonomous ML experimentation agent.
Goal: maximize validation F1 score on the breast cancer dataset.

You have four tools: inspect_dataset, search_knowledge, get_experiment_history,
and run_experiment. You may only interact with the world through these tools
(no arbitrary code execution). You have a small, limited experiment budget.

Work step by step:
1. Inspect the dataset.
2. Search the knowledge base for relevant ML advice.
3. Propose and run an experiment.
4. Look at the result and decide what to try next (you may search knowledge
   or check history again).
5. Once your budget is used up, or you believe further experiments are
   unlikely to help, stop and give a final one or two sentence conclusion
   naming the best config and its F1 score.

Be concise. Always explain briefly why you're proposing a given experiment
before calling run_experiment.
"""


def _print_tool_call(name: str, arguments: Dict[str, Any]) -> None:
    if name == "run_experiment":
        print(f"[AGENT] Proposed experiment: {arguments}")
    elif name == "search_knowledge":
        print(f"[AGENT] Searching knowledge: {arguments.get('query')!r}")
    elif name == "inspect_dataset":
        print("[AGENT] Inspecting dataset...")
    elif name == "get_experiment_history":
        print("[AGENT] Checking experiment history...")


def _print_tool_result(name: str, result: Any) -> None:
    if name == "run_experiment":
        m = result["metrics"]
        print(
            f"[HARNESS] Configuration validated -> [EXPERIMENT] "
            f"F1={m['f1']:.4f} acc={m['accuracy']:.4f} precision={m['precision']:.4f} recall={m['recall']:.4f}"
        )


# ---------------------------------------------------------------------------
# Real LLM agent (OpenAI-compatible, tool calling)
# ---------------------------------------------------------------------------

def run_llm_agent(harness: Harness, task: str, model: str, max_steps: int = 12) -> str:
    from openai import OpenAI

    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
        base_url=os.environ.get("OPENAI_BASE_URL") or None,
    )

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task},
    ]

    for step in range(max_steps):
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=harness.tool_specs(),
        )
        msg = response.choices[0].message
        messages.append(msg.model_dump(exclude_unset=True))

        if not msg.tool_calls:
            print(f"[AGENT] {msg.content}")
            return msg.content or ""

        for call in msg.tool_calls:
            name = call.function.name
            try:
                arguments = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError:
                arguments = {}

            _print_tool_call(name, arguments)
            try:
                result = harness.dispatch(name, arguments)
                _print_tool_result(name, result)
                content = json.dumps(result)
            except BudgetExceededError as e:
                content = json.dumps({"error": str(e)})
                print(f"[HARNESS] {e}")
            except Exception as e:  # validation errors etc.
                content = json.dumps({"error": str(e)})
                print(f"[HARNESS] Rejected: {e}")

            messages.append(
                {"role": "tool", "tool_call_id": call.id, "content": content}
            )

    return "[AGENT] Reached max steps without a final answer."


# ---------------------------------------------------------------------------
# Mock agent (no API key needed) — deterministic policy over the same tools
# ---------------------------------------------------------------------------

def run_mock_agent(harness: Harness, task: str) -> str:
    print(f"[AGENT] Task: {task}")

    print("[AGENT] Inspecting dataset...")
    info = harness.inspect_dataset()
    print(f"[HARNESS] {info['train_size']} train / {info['val_size']} val samples, {info['n_features']} features")

    plan = [
        ("baseline logistic regression", "logistic regression regularization C", {"model_type": "logistic_regression", "hyperparams": {"C": 1.0, "penalty": "l2"}}),
        ("random forest", "random forest hyperparameters overfitting", {"model_type": "random_forest", "hyperparams": {"n_estimators": 100, "max_depth": 10}}),
        ("gradient boosting", "gradient boosting learning rate depth", {"model_type": "gradient_boosting", "hyperparams": {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 3}}),
    ]

    for label, query, config in plan:
        print(f"[AGENT] Searching knowledge: {query!r}")
        hits = harness.search_knowledge(query)
        if hits:
            print(f"[AGENT] Knowledge hint ({hits[0]['source']}): {hits[0]['text'][:100]}...")

        print(f"[AGENT] Proposed experiment: {label} -> {config}")
        try:
            result = harness.run_experiment(config)
            _print_tool_result("run_experiment", result)
        except BudgetExceededError as e:
            print(f"[HARNESS] {e}")
            break

    # One refinement step around the best model found so far, if budget allows
    history = harness.get_experiment_history()
    best = max(history, key=lambda r: r["metrics"]["f1"]) if history else None
    if best and harness.experiments_run < harness.max_experiments:
        tuned = dict(best["config"])
        if tuned["model_type"] == "random_forest":
            tuned["hyperparams"] = {"n_estimators": 200, "max_depth": None}
        elif tuned["model_type"] == "gradient_boosting":
            tuned["hyperparams"] = {**tuned["hyperparams"], "n_estimators": 200}
        else:
            tuned["hyperparams"] = {"C": 10.0, "penalty": "l2"}
        print(f"[AGENT] Best so far is {best['config']['model_type']} (F1={best['metrics']['f1']:.4f}). Trying a tuned variant.")
        print(f"[AGENT] Proposed experiment: tuned {tuned['model_type']} -> {tuned}")
        try:
            result = harness.run_experiment(tuned)
            _print_tool_result("run_experiment", result)
        except BudgetExceededError as e:
            print(f"[HARNESS] {e}")

    history = harness.get_experiment_history()
    best = max(history, key=lambda r: r["metrics"]["f1"])
    conclusion = (
        f"Best configuration: {best['config']['model_type']} with {best['config']['hyperparams']} "
        f"reaching validation F1={best['metrics']['f1']:.4f}."
    )
    print(f"[AGENT] {conclusion}")
    return conclusion
