"""
main.py
-------
Entry point. Wires up memory + harness, runs the agent (real LLM or mock),
then prints a final summary comparing to a fixed baseline.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

import experiment
from agent import run_llm_agent, run_mock_agent
from memory import Memory
from tools import Harness

TASK = "Maximize validation F1 score on the breast cancer dataset."
MAX_EXPERIMENTS = 5


def main() -> None:
    load_dotenv()

    mock_mode = os.environ.get("MOCK_MODE", "").lower() == "true" or not os.environ.get("OPENAI_API_KEY")
    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

    memory = Memory()
    memory.reset()  # fresh run each time, keep the demo simple
    harness = Harness(memory=memory, max_experiments=MAX_EXPERIMENTS)

    print("=" * 70)
    print(f"Task: {TASK}")
    print(f"Mode: {'MOCK (no LLM API)' if mock_mode else f'LLM ({model})'}")
    print(f"Experiment budget: {MAX_EXPERIMENTS}")
    print("=" * 70)

    baseline = experiment.baseline_score()
    print(f"[HARNESS] Baseline (logistic regression, default) F1 = {baseline['metrics']['f1']:.4f}\n")

    if mock_mode:
        conclusion = run_mock_agent(harness, TASK)
    else:
        conclusion = run_llm_agent(harness, TASK, model=model)

    history = memory.get_history()
    best = memory.best("f1")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Baseline validation F1:       {baseline['metrics']['f1']:.4f}")
    print(f"Experiments performed:        {len(history)}")
    for rec in history:
        print(f"  #{rec['id']} {rec['config']['model_type']:<20} {rec['config']['hyperparams']} -> F1={rec['metrics']['f1']:.4f}")
    if best:
        print(f"Best configuration:           {best['config']['model_type']} {best['config']['hyperparams']}")
        print(f"Best validation F1:           {best['metrics']['f1']:.4f}")
        test_result = experiment.evaluate_on_test(best["config"])
        print(f"(Held-out test F1 for best config, for reference only: {test_result['metrics']['f1']:.4f})")
    print(f"Final agent conclusion:       {conclusion}")
    print("=" * 70)


if __name__ == "__main__":
    main()
