#!/usr/bin/env python3
"""Generate synthetic CPS-like microdata with ablation over prompt conditions.

Three conditions (ablation over prompt knowledge):
  minimal   - field specs only, no economic knowledge
  structural - + logical constraints (employed->hours>0, lifecycle)
  priors    - + explicit target values (income by education, education dist)

Three new (2026), cheap, fast paid models via OpenRouter (all under $0.10 total):
  inclusionai/ling-3.0-flash
  ~deepseek/deepseek-v4-flash-latest
  upstage/solar-pro4

Generation is PARALLEL across (model, condition, seed) tasks using a thread
pool, which cuts wall-clock time by ~4x (OpenRouter handles concurrent calls
without rate-limiting at this volume).

Default: 3 models x 3 conditions x 2 seeds x 400 records = 7,200 records,
~25-35 min with 4 workers, cost ~$0.04. Use --seeds 42 2024 7 --records 600
for the full 16,200-record run (~1.5-2h, ~$0.08).

Usage:
  python generate_data.py                          # default (fast)
  python generate_data.py --seeds 42 2024 7 --records 600   # full
  python generate_data.py --models mistralai/mistral-nemo
  python generate_data.py --conditions minimal structural
  python generate_data.py --workers 8
"""
import json
import os
import random
import sys
import time
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from llm_client import LLMClient

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
os.makedirs(DATA_DIR, exist_ok=True)

BATCH_SIZE = 20
N_RECORDS = 400        # default per (model, condition, seed)
SEEDS = [42, 2024]     # default 2 seeds -> 800 per (model, condition)
DEFAULT_WORKERS = 4

# ---------------------------------------------------------------------------
# Prompt conditions (the ablation)
# ---------------------------------------------------------------------------

FIELD_SPEC = (
    '{{"age": <int 18-85>, "gender": "<male|female>", '
    '"education": "<high_school|some_college|bachelors|masters|phd>", '
    '"income": <int annual USD>, "employed": <true|false>, '
    '"hours_worked": <int 0-80>, '
    '"marital_status": "<single|married|divorced|widowed>", '
    '"children": <int 0-6>, "state": "<2-letter US state>"}}'
)

PROMPTS = {
    "minimal": (
        "Generate {n} synthetic US household records as a JSON array. "
        "Each element must be a JSON object with EXACTLY these fields:\n"
        f"{FIELD_SPEC}\n\n"
        "Make the records look like a realistic US population survey. "
        "Return the array only."
    ),
    "structural": (
        "Generate {n} synthetic US household records as a JSON array. "
        "Each element must be a JSON object with EXACTLY these fields:\n"
        f"{FIELD_SPEC}\n\n"
        "Make the population realistic and internally consistent:\n"
        "- Employed people work positive hours (typically 20-80); "
        "unemployed/retired people work 0 hours.\n"
        "- Income is 0 for people who are not employed.\n"
        "- Age and income follow a lifecycle: income rises into middle age "
        "then declines.\n"
        "- Gender split roughly 50/50. Married people are more common "
        "in middle age.\n"
        "- Use a realistic spread of US states.\n\n"
        "Return the array only."
    ),
    "priors": (
        "Generate {n} synthetic US household records as a JSON array. "
        "Each element must be a JSON object with EXACTLY these fields:\n"
        f"{FIELD_SPEC}\n\n"
        "Make the population realistic and internally consistent:\n"
        "- Higher education correlates with higher income (median annual "
        "income by education: high_school ~$48k, some_college ~$53k, "
        "bachelors ~$80k, masters ~$96k, phd ~$118k).\n"
        "- Employed people work positive hours (typically 20-80); "
        "unemployed/retired people work 0 hours.\n"
        "- Income is 0 for people who are not employed.\n"
        "- Age and income follow a lifecycle: income rises into middle age "
        "then declines.\n"
        "- Education distribution roughly: high_school ~28%, "
        "some_college ~25%, bachelors ~24%, masters ~11%, phd ~4%, "
        "rest high_school.\n"
        "- Gender split roughly 50/50. Married people are more common "
        "in middle age.\n"
        "- Use a realistic spread of US states.\n\n"
        "Return the array only."
    ),
}

SYSTEM_PROMPT = (
    "You are a data generator for a US household survey (like the Current "
    "Population Survey). You produce realistic, internally consistent "
    "synthetic microdata. Return ONLY valid JSON. No markdown, no code "
    "fences, no explanation."
)

MODELS = [
    # New (2026), cheap, fast paid models on OpenRouter (Aug 2026).
    # Full run (4 models x 3 conditions x 2 seeds x 400 records = 9,600)
    # costs ~$0.06 total, well under the $0.10 budget.
    ("inclusionai/ling-3.0-flash", True),        # 2026-07, $0.021/M in, $0.063/M out
    ("~deepseek/deepseek-v4-flash-latest", True),# 2026-08, $0.030/M in, $0.160/M out
    ("upstage/solar-pro4", True),                # 2026-08, $0.030/M in, $0.120/M out
    ("qwen/qwen3.7-flash", True),               # 2026-08, $0.025/M in, $0.080/M out
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _filename(model_name, condition, seed):
    safe = model_name.replace("/", "_")
    return os.path.join(DATA_DIR, f"{safe}_{condition}_seed{seed}.json")


def parse_records(content):
    """Extract a JSON array from model output, tolerating code fences."""
    if not content:
        return None
    text = content.strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        if text.startswith("json"):
            text = text[4:]
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        data = json.loads(text[start:end + 1])
    except Exception:
        return None
    return data if isinstance(data, list) else None


def sanitize(records):
    """Coerce types and drop invalid records."""
    valid = []
    for r in records:
        if not isinstance(r, dict):
            continue
        try:
            rec = {
                "age": int(r.get("age")),
                "gender": str(r.get("gender", "")).strip().lower(),
                "education": str(r.get("education", "")).strip().lower(),
                "income": int(r.get("income")),
                "employed": bool(r.get("employed")),
                "hours_worked": int(r.get("hours_worked")),
                "marital_status": str(r.get("marital_status", "")).strip().lower(),
                "children": int(r.get("children")),
                "state": str(r.get("state", "")).strip().upper(),
            }
            if not (18 <= rec["age"] <= 85):
                continue
            if rec["gender"] not in ("male", "female"):
                continue
            if rec["education"] not in (
                "high_school", "some_college", "bachelors", "masters", "phd"
            ):
                continue
            if not (0 <= rec["income"] <= 5_000_000):
                continue
            if not (0 <= rec["hours_worked"] <= 80):
                continue
            if rec["marital_status"] not in (
                "single", "married", "divorced", "widowed"
            ):
                continue
            if not (0 <= rec["children"] <= 6):
                continue
            if len(rec["state"]) != 2 or not rec["state"].isalpha():
                continue
            valid.append(rec)
        except Exception:
            continue
    return valid


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate_one(client, model_name, condition, seed, n_records=N_RECORDS):
    """Generate n_records for one (model, condition, seed).

    Returns the record count (may be < n_records if the model struggles).
    """
    tag = f"{model_name} / {condition} / seed={seed}"
    out = _filename(model_name, condition, seed)

    # Resume: skip if file already has enough records
    if os.path.exists(out):
        with open(out) as f:
            existing = json.load(f)
        if len(existing) >= n_records:
            print(f"  [SKIP] {tag}: already has {len(existing)} records")
            return len(existing)

    rng = random.Random(seed)
    all_records = []
    attempts = 0
    max_attempts = 200

    while len(all_records) < n_records and attempts < max_attempts:
        attempts += 1
        n = min(BATCH_SIZE, n_records - len(all_records))
        prompt = PROMPTS[condition].format(n=n)
        content = client.complete([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ])
        if content is None:
            time.sleep(2)
            continue
        records = parse_records(content)
        if records is None:
            time.sleep(1)
            continue
        valid = sanitize(records)
        if not valid:
            time.sleep(1)
            continue
        all_records.extend(valid)
        print(f"  [{tag}] +{len(valid)} valid ({len(all_records)}/{n_records})")
        time.sleep(rng.uniform(0.3, 1.0))

    with open(out, "w") as f:
        json.dump(all_records, f, indent=2)
    print(f"  [SAVE] {tag}: {len(all_records)} records -> {out}")
    return len(all_records)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic microdata with prompt ablation."
    )
    parser.add_argument(
        "--models", nargs="*",
        help="Subset of model names to run (e.g. qwen/qwen3.7-flash)."
    )
    parser.add_argument(
        "--conditions", nargs="*",
        choices=list(PROMPTS.keys()),
        help="Subset of conditions (minimal, structural, priors)."
    )
    parser.add_argument(
        "--seeds", nargs="*", type=int,
        help="Subset of random seeds."
    )
    parser.add_argument(
        "--records", type=int, default=N_RECORDS,
        help=f"Records per (model, condition, seed). Default: {N_RECORDS}."
    )
    parser.add_argument(
        "--workers", type=int, default=DEFAULT_WORKERS,
        help=f"Number of parallel generation workers. Default: {DEFAULT_WORKERS}."
    )
    args = parser.parse_args()

    models = MODELS
    if args.models:
        models = [(m, r) for m, r in MODELS if m in args.models]

    conditions = list(PROMPTS.keys())
    if args.conditions:
        conditions = [c for c in conditions if c in args.conditions]

    seeds = SEEDS
    if args.seeds:
        seeds = [s for s in seeds if s in args.seeds]

    n_records = args.records
    workers = args.workers

    # Build the full task list: (model, condition, seed)
    tasks = [(m, c, s) for m, _ in models for c in conditions for s in seeds]
    print(f"Tasks: {len(tasks)} (models={len(models)}, "
          f"conditions={len(conditions)}, seeds={len(seeds)})")
    print(f"Records per task: {n_records}, workers: {workers}")
    print(f"Estimated total records: {len(tasks) * n_records}")

    # Each task gets its own LLMClient (thread-safe per task).
    def run_task(task):
        model_name, condition, seed = task
        reasoning_off = dict(models)[model_name]
        client = LLMClient(model_name, temperature=0.8, reasoning_off=reasoning_off)
        n = generate_one(client, model_name, condition, seed, n_records)
        return model_name, condition, seed, n, client.stats()

    grand_total = 0
    grand_cost = 0.0
    per_model_cost = {}

    t_start = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(run_task, t): t for t in tasks}
        for fut in as_completed(futures):
            model_name, condition, seed, n, stats = fut.result()
            grand_total += n
            grand_cost += stats["cost"]
            per_model_cost[model_name] = per_model_cost.get(model_name, 0.0) + stats["cost"]
            print(f"  [DONE] {model_name} / {condition} / seed={seed}: "
                  f"{n} records, ${stats['cost']:.5f}")

    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"  TOTAL: {grand_total} records, ${grand_cost:.4f} API cost")
    print(f"  Elapsed: {elapsed/60:.1f} min")
    for m, c in per_model_cost.items():
        print(f"    {m}: ${c:.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
