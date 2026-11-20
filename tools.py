"""
tools.py
--------
The harness: the ONLY interface the agent has to the outside world. Four
tools, each validated / budget-checked here. No arbitrary code execution is
ever exposed to the agent.
"""

from __future__ import annotations

from typing import Any, Dict, List

import experiment
import rag
from memory import Memory


class BudgetExceededError(Exception):
    pass


class Harness:
    """Wraps the four tools and enforces the experiment budget."""

    def __init__(self, memory: Memory, max_experiments: int = 5):
        self.memory = memory
        self.max_experiments = max_experiments
        self.experiments_run = 0

    # ---- tool implementations -------------------------------------------------

    def inspect_dataset(self) -> Dict[str, Any]:
        return experiment.inspect_dataset()

    def search_knowledge(self, query: str) -> List[Dict[str, Any]]:
        return rag.search_knowledge(query)

    def get_experiment_history(self) -> List[Dict[str, Any]]:
        return self.memory.get_history()

    def run_experiment(self, config: Dict[str, Any]) -> Dict[str, Any]:
        if self.experiments_run >= self.max_experiments:
            raise BudgetExceededError(
                f"Experiment budget exhausted ({self.max_experiments} runs used)."
            )
        # Validate before spending budget
        experiment.validate_config(config)
        result = experiment.run_experiment(config)
        self.experiments_run += 1
        self.memory.add_experiment(result["config"], result["metrics"])
        return result

    # ---- OpenAI-style tool schemas ---------------------------------------------

    def tool_specs(self) -> List[Dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "inspect_dataset",
                    "description": "Inspect metadata about the dataset and train/validation split (no test data).",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "search_knowledge",
                    "description": "Search a small local knowledge base of ML concepts. Returns relevant text chunks.",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_experiment_history",
                    "description": "Get the list of all experiments run so far, with configs and metrics.",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "run_experiment",
                    "description": (
                        "Run one experiment on the training set and evaluate on the validation set. "
                        "model_type must be one of: logistic_regression, random_forest, gradient_boosting. "
                        f"Allowed hyperparams: {experiment.ALLOWED_CONFIGS}"
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "model_type": {"type": "string"},
                            "hyperparams": {"type": "object"},
                        },
                        "required": ["model_type"],
                    },
                },
            },
        ]

    def dispatch(self, name: str, arguments: Dict[str, Any]) -> Any:
        if name == "inspect_dataset":
            return self.inspect_dataset()
        if name == "search_knowledge":
            return self.search_knowledge(arguments.get("query", ""))
        if name == "get_experiment_history":
            return self.get_experiment_history()
        if name == "run_experiment":
            config = {
                "model_type": arguments.get("model_type"),
                "hyperparams": arguments.get("hyperparams", {}),
            }
            return self.run_experiment(config)
        raise ValueError(f"Unknown tool: {name}")
