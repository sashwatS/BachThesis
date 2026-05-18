"""
build_adjacency_matrices.py

Constructs adjacency matrices for the ground-truth DAG and for each of the five
model configurations evaluated in the thesis.

Inputs (in ./data/):
  - ontology.txt:                     13 concepts, one per line (canonical order)
  - llama3_1_8b.jsonl:                LLaMA 3.1 8B (instruct), 39 records
  - gemma4_e4b.jsonl:                 Gemma 4 E4B, 39 records
  - qwen3_5_9b_instruct.jsonl:        Qwen 3.5 9B (instruct mode), 39 records
  - deepseek-r1_8b_original.jsonl:    DeepSeek-R1 8B original run, 39 records
  - deepseek-r1_8b_rerun.jsonl:       DeepSeek-R1 8B rerun (supersedes malformed)
  - qwen3_5_9b_original.jsonl:        Qwen 3.5 9B (thinking) original run, 39 records
  - qwen3_5_9b_rerun.jsonl:           Qwen 3.5 9B (thinking) rerun (supersedes malformed)

Design choices (match thesis methodology section):
  1. Aggregation: UNION. An edge appears in the predicted DAG if it was predicted
     in at least one excerpt. Duplicate predictions across excerpts collapse to
     a single matrix entry.
  2. Matrix layout: rows = cause, columns = effect. Index order = ontology.txt.
  3. Missing predictions (status in {malformed, none, not_in_ontology, error})
     contribute NO edge to the predicted adjacency matrix. They are treated as
     the model abstaining rather than as an assertion of edge absence.
  4. Merge rule for two-pass models (DeepSeek, Qwen-thinking):
     For each excerpt, the rerun record supersedes the original if one exists.
     In practice, only the originally-malformed excerpts were rerun, so the
     merge preserves the original OK records and replaces the original
     malformed records with whatever the rerun produced.

Outputs (in ./output/):
  - matrices/*.csv:       Six 13x13 adjacency matrices with concept headers
  - summary_counts.csv:   Per-configuration status breakdown (ok/mal/none/etc.)
  - edge_summary.csv:     Side-by-side view of all six matrices in long form
"""

import json
import csv
from pathlib import Path
from collections import Counter

DATA_DIR = Path("/home/claude/data")
OUTPUT_DIR = Path("/home/claude/output")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
(OUTPUT_DIR / "matrices").mkdir(exist_ok=True, parents=True)


# ============================================================
# STEP 1: Load ontology (canonical concept ordering)
# ============================================================

def load_ontology():
    with open(DATA_DIR / "ontology.txt") as f:
        return [line.strip() for line in f if line.strip()]

ONTOLOGY = load_ontology()
N = len(ONTOLOGY)
INDEX = {c: i for i, c in enumerate(ONTOLOGY)}
assert N == 13, f"Expected 13 concepts, got {N}"


# ============================================================
# STEP 2: Load + merge per-model records
# ============================================================

def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]

def merge_records(original, rerun):
    """
    Merge two runs: for each excerpt_id, the rerun record supersedes the
    original if present. Returns a list of 39 records.
    """
    rerun_by_id = {r["excerpt_id"]: r for r in rerun}
    merged = []
    for r in original:
        if r["excerpt_id"] in rerun_by_id:
            merged.append(rerun_by_id[r["excerpt_id"]])
        else:
            merged.append(r)
    return merged

# Load each configuration's final records
CONFIGS = {}

CONFIGS["LLaMA 3.1 8B"] = load_jsonl(DATA_DIR / "llama3_1_8b.jsonl")

CONFIGS["Gemma 4 E4B"] = load_jsonl(DATA_DIR / "gemma4_e4b.jsonl")

CONFIGS["Qwen 3.5 9B (instruct)"] = load_jsonl(DATA_DIR / "qwen3_5_9b_instruct.jsonl")

CONFIGS["DeepSeek-R1 8B"] = merge_records(
    load_jsonl(DATA_DIR / "deepseek-r1_8b_original.jsonl"),
    load_jsonl(DATA_DIR / "deepseek-r1_8b_rerun.jsonl"),
)

CONFIGS["Qwen 3.5 9B (thinking)"] = merge_records(
    load_jsonl(DATA_DIR / "qwen3_5_9b_original.jsonl"),
    load_jsonl(DATA_DIR / "qwen3_5_9b_rerun.jsonl"),
)

# Verify each configuration has exactly 39 records
for name, recs in CONFIGS.items():
    assert len(recs) == 39, f"{name}: expected 39 records, got {len(recs)}"


# ============================================================
# STEP 3: Build ground-truth matrix
# ============================================================
# Ground truth is identical across all files (same excerpts, same labels),
# so we can pull it from any configuration.

def build_ground_truth_matrix(records):
    """Build 13x13 binary adjacency matrix from ground-truth labels (union)."""
    A = [[0] * N for _ in range(N)]
    for r in records:
        c = r["ground_truth_cause"]
        e = r["ground_truth_effect"]
        if c in INDEX and e in INDEX:
            A[INDEX[c]][INDEX[e]] = 1
        else:
            raise ValueError(f"GT concept not in ontology: {c} or {e}")
    return A

GT_MATRIX = build_ground_truth_matrix(CONFIGS["LLaMA 3.1 8B"])


# ============================================================
# STEP 4: Build predicted matrix per configuration
# ============================================================

def build_predicted_matrix(records):
    """
    Build 13x13 binary adjacency matrix from predicted edges (union).
    Only records with status=="ok" contribute; all others (malformed, none,
    not_in_ontology, error) abstain and add nothing.
    """
    A = [[0] * N for _ in range(N)]
    contributing = 0
    for r in records:
        if r.get("status") != "ok":
            continue
        c = r.get("predicted_cause")
        e = r.get("predicted_effect")
        if c in INDEX and e in INDEX:
            A[INDEX[c]][INDEX[e]] = 1
            contributing += 1
        # status=="ok" records should always have ontology-aligned terms by
        # construction (parse_edge upstream normalizes them), so no else needed.
    return A, contributing


# ============================================================
# STEP 5: Write matrix CSVs with concept headers
# ============================================================

def write_matrix_csv(A, path, label=""):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        # Header row: blank corner + all concepts as column headers (effects)
        w.writerow([f"cause \\ effect ({label})" if label else "cause \\ effect"] + ONTOLOGY)
        for i, row in enumerate(A):
            w.writerow([ONTOLOGY[i]] + row)

# Ground truth
write_matrix_csv(GT_MATRIX, OUTPUT_DIR / "matrices" / "ground_truth.csv", "GT")

# Per-configuration predicted matrices
contribution_counts = {}
pred_matrices = {}
for name, records in CONFIGS.items():
    A, n_contrib = build_predicted_matrix(records)
    pred_matrices[name] = A
    contribution_counts[name] = n_contrib
    # Make filename-safe
    safe = name.replace(" ", "_").replace(".", "").replace("(", "").replace(")", "")
    write_matrix_csv(A, OUTPUT_DIR / "matrices" / f"{safe}.csv", name)


# ============================================================
# STEP 6: Write status-counts summary
# ============================================================

with open(OUTPUT_DIR / "summary_counts.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow([
        "Configuration",
        "Total records",
        "ok",
        "malformed",
        "none",
        "not_in_ontology",
        "error",
        "Contributing to matrix",
        "Unique edges in matrix",
    ])
    for name, records in CONFIGS.items():
        statuses = Counter(r.get("status", "ERROR") for r in records)
        error_count = sum(1 for r in records if "error" in r)
        # 'ERROR' label in Counter is for records with no status field (error records)
        # to avoid double-counting, use the actual key presence
        ok = statuses.get("ok", 0)
        mal = statuses.get("malformed", 0)
        none_ = statuses.get("none", 0)
        nio = statuses.get("not_in_ontology", 0)
        err = error_count
        unique_edges = sum(sum(row) for row in pred_matrices[name])
        w.writerow([
            name, len(records), ok, mal, none_, nio, err,
            contribution_counts[name], unique_edges,
        ])

# Ground-truth summary row
gt_edges = sum(sum(row) for row in GT_MATRIX)
with open(OUTPUT_DIR / "summary_counts.csv", "a", newline="") as f:
    w = csv.writer(f)
    w.writerow([])
    w.writerow(["Ground truth", 39, "—", "—", "—", "—", "—", 39, gt_edges])


# ============================================================
# STEP 7: Long-form edge summary
# ============================================================
# For every (cause, effect) pair where ANY matrix has a 1, emit one row.

all_positions = set()
for row_i in range(N):
    for col_j in range(N):
        if GT_MATRIX[row_i][col_j] == 1:
            all_positions.add((row_i, col_j))
for A in pred_matrices.values():
    for row_i in range(N):
        for col_j in range(N):
            if A[row_i][col_j] == 1:
                all_positions.add((row_i, col_j))

with open(OUTPUT_DIR / "edge_summary.csv", "w", newline="") as f:
    w = csv.writer(f)
    header = ["Cause", "Effect", "Ground truth"] + list(CONFIGS.keys())
    w.writerow(header)
    for i, j in sorted(all_positions):
        row = [ONTOLOGY[i], ONTOLOGY[j], GT_MATRIX[i][j]]
        for name in CONFIGS.keys():
            row.append(pred_matrices[name][i][j])
        w.writerow(row)


# ============================================================
# STEP 8: Print human-readable summary
# ============================================================

print("=" * 72)
print("ADJACENCY MATRIX CONSTRUCTION - SUMMARY")
print("=" * 72)
print()
print(f"Ontology size: {N} concepts")
print(f"Corpus size: 39 excerpts per configuration")
print(f"Matrix dimensions: {N}x{N} = {N*N} cells each")
print()
print(f"Ground-truth DAG:")
print(f"  Unique edges: {gt_edges}")
print(f"  Density: {gt_edges}/{N*N} = {100*gt_edges/(N*N):.1f}%")
print()
print("Predicted DAGs:")
print(f"{'Configuration':<28} {'Unique edges':>13} {'OK records':>12}")
print("-" * 55)
for name in CONFIGS.keys():
    edges = sum(sum(row) for row in pred_matrices[name])
    print(f"{name:<28} {edges:>13} {contribution_counts[name]:>12}")

print()
print("Outputs written to:")
print(f"  {OUTPUT_DIR / 'matrices' / 'ground_truth.csv'}")
for name in CONFIGS.keys():
    safe = name.replace(" ", "_").replace(".", "").replace("(", "").replace(")", "")
    print(f"  {OUTPUT_DIR / 'matrices' / (safe + '.csv')}")
print(f"  {OUTPUT_DIR / 'summary_counts.csv'}")
print(f"  {OUTPUT_DIR / 'edge_summary.csv'}")
