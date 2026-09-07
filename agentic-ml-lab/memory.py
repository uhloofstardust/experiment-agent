"""
memory.py
---------
Simple JSON-file-backed memory of past experiments (config, metrics, a short
conclusion) plus retrieval of experiment history for the agent.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List

MEMORY_PATH = os.path.join(os.path.dirname(__file__), "experiments.json")


class Memory:
    def __init__(self, path: str = MEMORY_PATH):
        self.path = path
        self._records: List[Dict[str, Any]] = []
        self._load()

    def _load(self) -> None:
        if os.path.exists(self.path):
            with open(self.path, "r", encoding="utf-8") as f:
                try:
                    self._records = json.load(f)
                except json.JSONDecodeError:
                    self._records = []
        else:
            self._records = []

    def _save(self) -> None:
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self._records, f, indent=2)

    def reset(self) -> None:
        self._records = []
        self._save()

    def add_experiment(self, config: Dict[str, Any], metrics: Dict[str, Any], conclusion: str = "") -> Dict[str, Any]:
        record = {
            "id": len(self._records) + 1,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "config": config,
            "metrics": metrics,
            "conclusion": conclusion,
        }
        self._records.append(record)
        self._save()
        return record

    def get_history(self) -> List[Dict[str, Any]]:
        return list(self._records)

    def best(self, metric: str = "f1") -> Dict[str, Any] | None:
        if not self._records:
            return None
        return max(self._records, key=lambda r: r["metrics"].get(metric, float("-inf")))
