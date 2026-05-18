"""
compute_all_metrics.py

Combined evaluation: computes both graph-level metrics (from adjacency matrices)
and per-instance accuracy (from raw JSONLs), merging them into a single
comparison table.

Inputs:
    ./ground_truth.csv                — 13x13 GT adjacency matrix
    ./<Model>.csv                     — 13x13 predicted adjacency matrix per model
    /home/claude/data/*.jsonl         — raw per-excerpt prediction records

Outputs:
    ./metrics_full.csv                — all metrics, wide format
    ./metrics_summary.csv             — streamlined key metrics for reporting
    ./confusion_counts.csv            — TP/FP/FN/TN per model
    ./per_instance_accuracy.csv       — per-excerpt correctness per model
    ./per_instance_breakdown.csv      — status breakdown per model
    ./metrics_combined.csv            — graph-level + per-instance in one table

Methodology notes (same as compute_pairs_metrics.py):
    - GT = 10 literature-supported directed pairs aggregated into a 13x13 matrix
    - Self-loops excluded (denominator = N*(N-1) = 156)
    - SID omitted: cyclic predictions, pairs-collage GT not suitable for do-calculus
    - SHD follows Tsamardinos et al. (2006)
    - Per-instance accuracy complements graph-level metrics and reflects the
      relation-extraction nature of the underlying task
"""

import json
import csv
import math
from pathlib import Path
from collections import Counter

HERE = Path(__file__).parent
DATA_DIR = Path("/home/claude/data")


# ============================================================
# Load adjacency matrices
# ============================================================

def load_matrix(path):
    with open(path) as f:
        reader = csv.reader(f)
        header = next(reader)
        concepts = header[1:]
        matrix = [[int(x) for x in row[1:]] for row in reader]
    return concepts, matrix


# ============================================================
# Matrix iteration helpers
# ============================================================

def iter_cells(N):
    """Off-diagonal directed cells."""
    for i in range(N):
        for j in range(N):
            if i != j:
                yield i, j

def iter_pairs(N):
    """Unordered pairs (i < j)."""
    for i in range(N):
        for j in range(i + 1, N):
            yield i, j


# ============================================================
# Graph-level metric computations
# ============================================================

def confusion_counts(gt, pred, N):
    tp = fp = fn = tn = 0
    for i, j in iter_cells(N):
        if gt[i][j] == 1 and pred[i][j] == 1: tp += 1
        elif gt[i][j] == 0 and pred[i][j] == 1: fp += 1
        elif gt[i][j] == 1 and pred[i][j] == 0: fn += 1
        else: tn += 1
    return tp, fp, fn, tn

def undirected(A, N):
    U = [[0] * N for _ in range(N)]
    for i, j in iter_pairs(N):
        if A[i][j] == 1 or A[j][i] == 1:
            U[i][j] = U[j][i] = 1
    return U

def adjacency_confusion(gt, pred, N):
    U_gt, U_pred = undirected(gt, N), undirected(pred, N)
    tp = fp = fn = tn = 0
    for i, j in iter_pairs(N):
        if U_gt[i][j] == 1 and U_pred[i][j] == 1: tp += 1
        elif U_gt[i][j] == 0 and U_pred[i][j] == 1: fp += 1
        elif U_gt[i][j] == 1 and U_pred[i][j] == 0: fn += 1
        else: tn += 1
    return tp, fp, fn, tn

def orientation_confusion(gt, pred, N):
    tp = fp = fn = 0
    for i, j in iter_pairs(N):
        for (a, b) in [(i, j), (j, i)]:
            if gt[a][b] == 1:
                if pred[a][b] == 1: tp += 1
                else: fn += 1
            elif pred[a][b] == 1:
                fp += 1
    return tp, fp, fn

def safe_div(a, b):
    return a / b if b != 0 else 0.0

def precision(tp, fp): return safe_div(tp, tp + fp)
def recall(tp, fn):    return safe_div(tp, tp + fn)
def f1(p, r):          return safe_div(2 * p * r, p + r)

def mcc(tp, tn, fp, fn):
    num = tp * tn - fp * fn
    denom_sq = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    if denom_sq <= 0:
        return 0.0
    return num / math.sqrt(denom_sq)

def shd(gt, pred, N):
    """Tsamardinos et al. (2006) SHD: additions + deletions + reversals."""
    additions = deletions = reversals = 0
    for i, j in iter_pairs(N):
        gt_ij, gt_ji = gt[i][j], gt[j][i]
        pr_ij, pr_ji = pred[i][j], pred[j][i]
        gt_any = gt_ij or gt_ji
        pr_any = pr_ij or pr_ji
        if not gt_any and pr_any:
            additions += pr_ij + pr_ji
        elif gt_any and not pr_any:
            deletions += gt_ij + gt_ji
        elif gt_any and pr_any:
            if gt_ij == pr_ij and gt_ji == pr_ji:
                pass  # exact match on this pair
            elif gt_ij != pr_ij and gt_ji != pr_ji and gt_ij == pr_ji and gt_ji == pr_ij:
                reversals += 1
            else:
                if gt_ij and not pr_ij: deletions += 1
                if not gt_ij and pr_ij: additions += 1
                if gt_ji and not pr_ji: deletions += 1
                if not gt_ji and pr_ji: additions += 1
    return additions + deletions + reversals, additions, deletions, reversals

def skeleton_shd(gt, pred, N):
    U_gt, U_pred = undirected(gt, N), undirected(pred, N)
    return sum(1 for i, j in iter_pairs(N) if U_gt[i][j] != U_pred[i][j])

def exact_match(gt, pred, N):
    return 1 if all(gt[i][j] == pred[i][j] for i, j in iter_cells(N)) else 0


# ============================================================
# Per-instance metrics (from raw JSONLs)
# ============================================================

def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]

def merge_records(original, rerun):
    rerun_by_id = {r["excerpt_id"]: r for r in rerun}
    return [rerun_by_id.get(r["excerpt_id"], r) for r in original]

def per_instance_stats(records):
    total = len(records)
    correct = sum(1 for r in records if r.get("correct") is True)
    statuses = Counter(r.get("status", "_error") for r in records)
    error_count = sum(1 for r in records if "error" in r and "status" not in r)
    return {
        "total": total,
        "correct": correct,
        "per_instance_accuracy": correct / total if total > 0 else 0.0,
        "ok": statuses.get("ok", 0),
        "malformed": statuses.get("malformed", 0),
        "none": statuses.get("none", 0),
        "not_in_ontology": statuses.get("not_in_ontology", 0),
        "error": error_count,
    }


# ============================================================
# Per-model evaluation
# ============================================================

def evaluate_model(name, gt, pred, N, instance_stats):
    total_possible = N * (N - 1)
    d_tp, d_fp, d_fn, d_tn = confusion_counts(gt, pred, N)
    a_tp, a_fp, a_fn, a_tn = adjacency_confusion(gt, pred, N)
    o_tp, o_fp, o_fn = orientation_confusion(gt, pred, N)

    shd_total, additions, deletions, reversals = shd(gt, pred, N)
    skel_shd = skeleton_shd(gt, pred, N)
    n_gt_edges = sum(sum(row) for row in gt)

    return {
        "Model": name,
        # Per-instance
        "Per-instance Correct":       instance_stats["correct"],
        "Per-instance Total":         instance_stats["total"],
        "Per-instance Accuracy":      instance_stats["per_instance_accuracy"],
        # Status breakdown
        "Status: ok":                 instance_stats["ok"],
        "Status: malformed":          instance_stats["malformed"],
        "Status: none":               instance_stats["none"],
        "Status: not_in_ontology":    instance_stats["not_in_ontology"],
        "Status: error":              instance_stats["error"],
        # Directed confusion
        "TP": d_tp, "FP": d_fp, "FN": d_fn, "TN": d_tn,
        # Adjacency (undirected skeleton) metrics
        "Adjacency Precision":        precision(a_tp, a_fp),
        "Adjacency Recall":           recall(a_tp, a_fn),
        "Adjacency F1":               f1(precision(a_tp, a_fp), recall(a_tp, a_fn)),
        # Orientation metrics (directed)
        "Orientation Precision":      precision(o_tp, o_fp),
        "Orientation Recall":         recall(o_tp, o_fn),
        "Orientation F1":             f1(precision(o_tp, o_fp), recall(o_tp, o_fn)),
        # SHD breakdown
        "SHD":                        shd_total,
        "  SHD additions":            additions,
        "  SHD deletions":            deletions,
        "  SHD reversals":            reversals,
        "Skeleton SHD":               skel_shd,
        "Normalized SHD (by possible)": safe_div(shd_total, total_possible),
        "Normalized SHD (by GT edges)": safe_div(shd_total, n_gt_edges),
        # Classification metrics (directed cells)
        "Accuracy":                   safe_div(d_tp + d_tn, d_tp + d_tn + d_fp + d_fn),
        "Balanced Accuracy":          (safe_div(d_tp, d_tp + d_fn) + safe_div(d_tn, d_tn + d_fp)) / 2,
        "MCC":                        mcc(d_tp, d_tn, d_fp, d_fn),
        "FDR":                        safe_div(d_fp, d_fp + d_tp),
        "Specificity (TNR)":          safe_div(d_tn, d_tn + d_fp),
        "FPR":                        safe_div(d_fp, d_fp + d_tn),
        "FNR":                        safe_div(d_fn, d_fn + d_tp),
        "Exact Match":                exact_match(gt, pred, N),
    }


# ============================================================
# Main
# ============================================================

def main():
    # Load GT and predicted matrices
    concepts, gt = load_matrix(HERE / "ground_truth.csv")
    N = len(concepts)
    n_gt_edges = sum(sum(row) for row in gt)
    total_possible = N * (N - 1)

    print(f"Ground truth: {N} concepts, {n_gt_edges} edges, "
          f"{total_possible} possible directed off-diagonal slots")
    print()

    # Load per-instance records
    instance_records = {}
    instance_records["LLaMA 3.1 8B"] = load_jsonl(DATA_DIR / "llama3_1_8b.jsonl")
    instance_records["Gemma 4 E4B"] = load_jsonl(DATA_DIR / "gemma4_e4b.jsonl")
    instance_records["Qwen 3.5 9B (instruct)"] = load_jsonl(DATA_DIR / "qwen3_5_9b_instruct.jsonl")
    instance_records["DeepSeek-R1 8B"] = merge_records(
        load_jsonl(DATA_DIR / "deepseek-r1_8b_original.jsonl"),
        load_jsonl(DATA_DIR / "deepseek-r1_8b_rerun.jsonl"),
    )
    instance_records["Qwen 3.5 9B (thinking)"] = merge_records(
        load_jsonl(DATA_DIR / "qwen3_5_9b_original.jsonl"),
        load_jsonl(DATA_DIR / "qwen3_5_9b_rerun.jsonl"),
    )

    models = [
        ("LLaMA 3.1 8B",           "LLaMA_31_8B.csv"),
        ("Gemma 4 E4B",            "Gemma_4_E4B.csv"),
        ("Qwen 3.5 9B (instruct)", "Qwen_35_9B_instruct.csv"),
        ("DeepSeek-R1 8B",         "DeepSeek-R1_8B.csv"),
        ("Qwen 3.5 9B (thinking)", "Qwen_35_9B_thinking.csv"),
    ]

    results = []
    for name, fname in models:
        _, pred = load_matrix(HERE / fname)
        instance_stats = per_instance_stats(instance_records[name])
        results.append(evaluate_model(name, gt, pred, N, instance_stats))

    # ---- Print full metric table ----
    print("=" * 100)
    print("ALL METRICS — graph-level + per-instance")
    print("=" * 100)
    keys = list(results[0].keys())
    for key in keys:
        if key == "Model":
            continue
        vals = [r[key] for r in results]
        if isinstance(vals[0], float):
            row = f"{key:<34}" + "  ".join(f"{v:>12.3f}" for v in vals)
        else:
            row = f"{key:<34}" + "  ".join(f"{v:>12}" for v in vals)
        print(row)

    print()
    print("Column order:")
    for r in results:
        print(f"  {r['Model']}")

    # ---- Write all CSVs ----
    # metrics_full.csv — everything
    with open(HERE / "metrics_full.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Metric"] + [r["Model"] for r in results])
        for key in keys:
            if key == "Model":
                continue
            row = [key]
            for r in results:
                v = r[key]
                row.append(f"{v:.4f}" if isinstance(v, float) else v)
            w.writerow(row)

    # metrics_summary.csv — curated
    summary_keys = [
        "Per-instance Correct", "Per-instance Total", "Per-instance Accuracy",
        "TP", "FP", "FN", "TN",
        "Adjacency Precision", "Adjacency Recall", "Adjacency F1",
        "Orientation Precision", "Orientation Recall", "Orientation F1",
        "SHD", "Skeleton SHD", "Normalized SHD (by GT edges)",
        "MCC", "Balanced Accuracy", "FDR",
        "Accuracy", "Exact Match",
    ]
    with open(HERE / "metrics_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Metric"] + [r["Model"] for r in results])
        for key in summary_keys:
            row = [key]
            for r in results:
                v = r[key]
                row.append(f"{v:.4f}" if isinstance(v, float) else v)
            w.writerow(row)

    # metrics_combined.csv — headline table for thesis
    combined_keys = [
        "Per-instance Accuracy",
        "Adjacency F1", "Orientation F1", "SHD",
        "MCC", "Balanced Accuracy", "FDR",
    ]
    with open(HERE / "metrics_combined.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Metric"] + [r["Model"] for r in results])
        for key in combined_keys:
            row = [key]
            for r in results:
                v = r[key]
                row.append(f"{v:.4f}" if isinstance(v, float) else v)
            w.writerow(row)

    # confusion_counts.csv
    with open(HERE / "confusion_counts.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Model", "TP", "FP", "FN", "TN"])
        for r in results:
            w.writerow([r["Model"], r["TP"], r["FP"], r["FN"], r["TN"]])

    # per_instance_accuracy.csv
    with open(HERE / "per_instance_accuracy.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Model", "Correct", "Total", "Per-instance Accuracy"])
        for r in results:
            w.writerow([r["Model"], r["Per-instance Correct"], r["Per-instance Total"],
                        f"{r['Per-instance Accuracy']:.4f}"])

    # per_instance_breakdown.csv
    with open(HERE / "per_instance_breakdown.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Model", "Total", "Correct", "Accuracy",
                    "ok", "malformed", "none", "not_in_ontology", "error"])
        for r in results:
            w.writerow([r["Model"], r["Per-instance Total"], r["Per-instance Correct"],
                        f"{r['Per-instance Accuracy']:.4f}",
                        r["Status: ok"], r["Status: malformed"], r["Status: none"],
                        r["Status: not_in_ontology"], r["Status: error"]])

    print()
    print("Written to", HERE)
    for name in ["metrics_full.csv", "metrics_summary.csv", "metrics_combined.csv",
                 "confusion_counts.csv", "per_instance_accuracy.csv",
                 "per_instance_breakdown.csv"]:
        print(f"  {name}")


if __name__ == "__main__":
    main()
