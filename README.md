# Agentic ML Experimentation Lab (MVP)

A minimal demo of an LLM agent that autonomously proposes and runs ML
experiments to maximize validation F1 score on the sklearn breast cancer
dataset, through a small set of controlled tools.

## Architecture

```
experiment-agent/
├── main.py        # entry point: wires everything up, prints trace + summary
├── agent.py        # the agent loop (real LLM tool-calling, or mock policy)
├── tools.py         # the harness: validates configs, enforces budget, exposes 4 tools
├── experiment.py    # dataset split, model training/eval, config validation
├── rag.py           # TF-IDF chunking + retrieval over knowledge/*.md
├── memory.py         # JSON-file experiment log (config, metrics, conclusion)
├── knowledge/         # 8 short markdown docs on ML concepts
└── experiments.json   # written at runtime (experiment log)
```

**Flow:** `main.py` creates a `Memory` + `Harness`, then hands control to the
agent. The agent can only touch the world through four tools exposed by the
harness: `inspect_dataset`, `search_knowledge`, `get_experiment_history`,
`run_experiment`. The harness validates every config against an allowed
hyperparameter space and stops the agent once a fixed experiment budget (5
runs) is used up. No arbitrary code execution is ever exposed to the model.

Data is split once, deterministically, into train (60%) / validation (20%)
/ test (20%). The agent only ever sees train/validation results. The test
set is evaluated exactly once, by the harness itself, on the best config —
after the agent's run — purely for a reference number in the final summary.

## Tools

| Tool | Purpose |
|---|---|
| `inspect_dataset()` | dataset size, features, class balance (train/val only) |
| `search_knowledge(query)` | TF-IDF similarity search over `knowledge/*.md` |
| `get_experiment_history()` | past configs + metrics from `memory.py` |
| `run_experiment(config)` | validates config, trains on train, scores on val |

Allowed models: `logistic_regression`, `random_forest`, `gradient_boosting`,
each with a small fixed hyperparameter grid (see `experiment.ALLOWED_CONFIGS`).

## RAG

`knowledge/` has 8 short markdown docs (logistic regression, random forest,
gradient boosting, overfitting, class imbalance, cross-validation, feature
scaling, hyperparameter search strategy). `rag.py` splits each doc into
paragraph chunks, embeds them with `TfidfVectorizer`, and retrieves the
top-k most similar chunks by cosine similarity for a given query — no
external embedding API required.

## Running it

```bash
pip install -r requirements.txt
cp .env.example .env   # optional, only needed for real LLM mode
python main.py
```

### Mock mode (default, no API key needed)

If `OPENAI_API_KEY` is unset (or `MOCK_MODE=true` in `.env`), the system
runs a deterministic mock agent that drives the exact same tools in a
sensible order (baseline → random forest → gradient boosting → tuned
best-of). This lets the whole pipeline run end-to-end with zero external
dependencies, and was used to verify everything works.

### Real LLM mode

Set `OPENAI_API_KEY` (and optionally `OPENAI_MODEL`, `OPENAI_BASE_URL` for
any OpenAI-compatible endpoint) in `.env`. The agent then drives an actual
tool-calling loop against `chat.completions`, deciding for itself which
experiments to run and when to stop, using the same four tools and budget.

## Sample output

```
Task: Maximize validation F1 score on the breast cancer dataset.
Mode: MOCK (no LLM API)
Experiment budget: 5
[HARNESS] Baseline (logistic regression, default) F1 = 0.9930
[AGENT] Inspecting dataset...
[HARNESS] 341 train / 114 val samples, 30 features
[AGENT] Searching knowledge: 'logistic regression regularization C'
[AGENT] Proposed experiment: baseline logistic regression -> {...}
[HARNESS] Configuration validated -> [EXPERIMENT] F1=0.9930 ...
...
SUMMARY
Baseline validation F1:       0.9930
Experiments performed:        4
Best configuration:           logistic_regression {'C': 1.0, 'penalty': 'l2'}
Best validation F1:           0.9930
Final agent conclusion:       Best configuration: logistic_regression ...
```

## Notes / scope

This is intentionally minimal: no frontend, no LangChain, no vector DB
(TF-IDF is enough for 8 short docs), no persistent cross-run memory beyond
one JSON file (reset at the start of each run for a clean demo). It's meant
to demonstrate the loop — inspect → retrieve → propose → run → observe →
decide-next → stop — clearly, not to be a production experimentation
platform.
