"""
compute_per_instance_accuracy.py

Computes per-instance (per-excerpt) accuracy for each model configuration on the
causal-pairs dataset. This metric complements the graph-level metrics by answering
the relation-extraction question: for each excerpt, did the model predict the
exact ground-truth edge?

Input (JSONLs in ./data/):
    llama3_1_8b.jsonl
    gemma4_e4b.jsonl
    qwen3_5_9b_instruct.jsonl
    deepseek-r1_8b_original.jsonl  + deepseek-r1_8b_rerun.jsonl  (merged)
    qwen3_5_9b_original.jsonl      + qwen3_5_9b_rerun.jsonl      (merged)

Output:
    ./per_instance_accuracy.csv
    ./per_instance_breakdown.csv  (status-by-status breakdown per model)

Methodology:
    - Per-instance accuracy = fraction of 39 excerpts for which the model predicted
      the exact ground-truth edge (both cause and effect match).
    - A prediction is "correct" iff status == "ok" AND predicted_cause matches
      ground_truth_cause AND predicted_effect matches ground_truth_effect.
    - Malformed, none, not_in_ontology, and error records all count as incorrect,
      since they fail to produce a matching prediction.
    - For two-pass models (DeepSeek, Qwen-thinking), rerun records supersede
      originals for the same excerpt_id.
"""

import json
import csv
from pathlib import Path
from collections import Counter

DATA_DIR = Path("/home/claude/data")
OUT_DIR = Path(__file__).parent


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def merge_records(original, rerun):
    rerun_by_id = {r["excerpt_id"]: r for r in rerun}
    return [rerun_by_id.get(r["excerpt_id"], r) for r in original]


def compute_per_instance(records):
    total = len(records)
    correct = sum(1 for r in records if r.get("correct") is True)
    status_counts = Counter(r.get("status", "error") for r in records)
    # error records have no "status" field
    error_count = sum(1 for r in records if "error" in r and "status" not in r)
    return {
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total > 0 else 0.0,
        "ok": status_counts.get("ok", 0),
        "malformed": status_counts.get("malformed", 0),
        "none": status_counts.get("none", 0),
        "not_in_ontology": status_counts.get("not_in_ontology", 0),
        "error": error_count,
    }


def main():
    configs = {}
    configs["LLaMA 3.1 8B"] = load_jsonl(DATA_DIR / "llama3_1_8b.jsonl")
    configs["Gemma 4 E4B"] = load_jsonl(DATA_DIR / "gemma4_e4b.jsonl")
    configs["Qwen 3.5 9B (instruct)"] = load_jsonl(DATA_DIR / "qwen3_5_9b_instruct.jsonl")
    configs["DeepSeek-R1 8B"] = merge_records(
        load_jsonl(DATA_DIR / "deepseek-r1_8b_original.jsonl"),
        load_jsonl(DATA_DIR / "deepseek-r1_8b_rerun.jsonl"),
    )
    configs["Qwen 3.5 9B (thinking)"] = merge_records(
        load_jsonl(DATA_DIR / "qwen3_5_9b_original.jsonl"),
        load_jsonl(DATA_DIR / "qwen3_5_9b_rerun.jsonl"),
    )

    results = {name: compute_per_instance(recs) for name, recs in configs.items()}

    # Print table
    print("=" * 70)
    print("PER-INSTANCE ACCURACY")
    print("=" * 70)
    print(f"{'Model':<28} {'Correct':>9} {'Total':>6} {'Accuracy':>10}")
    print("-" * 60)
    for name, r in results.items():
        print(f"{name:<28} {r['correct']:>9} {r['total']:>6} {r['accuracy']:>9.3f}")

    print()
    print("=" * 70)
    print("STATUS BREAKDOWN PER MODEL")
    print("=" * 70)
    print(f"{'Model':<28} {'ok':>4} {'mal':>4} {'none':>5} {'nio':>4} {'err':>4}")
    print("-" * 55)
    for name, r in results.items():
        print(f"{name:<28} {r['ok']:>4} {r['malformed']:>4} "
              f"{r['none']:>5} {r['not_in_ontology']:>4} {r['error']:>4}")

    # Write CSV
    with open(OUT_DIR / "per_instance_accuracy.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Model", "Correct", "Total", "Per-instance Accuracy"])
        for name, r in results.items():
            w.writerow([name, r["correct"], r["total"], f"{r['accuracy']:.4f}"])

    # Breakdown CSV
    with open(OUT_DIR / "per_instance_breakdown.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Model", "Total", "Correct", "Accuracy",
                    "ok", "malformed", "none", "not_in_ontology", "error"])
        for name, r in results.items():
            w.writerow([name, r["total"], r["correct"], f"{r['accuracy']:.4f}",
                        r["ok"], r["malformed"], r["none"],
                        r["not_in_ontology"], r["error"]])

    print()
    print("Written:")
    print(f"  {OUT_DIR / 'per_instance_accuracy.csv'}")
    print(f"  {OUT_DIR / 'per_instance_breakdown.csv'}")

    return results


if __name__ == "__main__":
    main()
