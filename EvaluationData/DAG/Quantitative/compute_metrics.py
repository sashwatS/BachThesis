"""
Compute evaluation metrics on Chain A adjacency matrices.

Input:  JSONL results files + ontology.txt + excerpts.json
Output: /mnt/user-data/outputs/metrics/
        13nodes/metrics_per_pair.csv       one row per (model, excerpt)
        13nodes/metrics_per_model.csv      one row per model (macro mean across excerpts)
        12nodes/metrics_per_pair.csv
        12nodes/metrics_per_model.csv

Metrics reported (all values rounded to 4 decimals where float):

Core (per Hendro's "at minimum" list):
  adj_precision, adj_recall, adj_f1
  ori_precision, ori_recall, ori_f1          -- Convention A: conditional on shared skeleton
  ori_precision_full, ori_recall_full, ori_f1_full  -- Convention B: unconditional
  shd_strict                                  -- reversals cost 2
  shd_reversal                                -- reversals cost 1

Descriptive counts:
  tp, fp, fn, tn
  tp_skeleton, fp_skeleton, fn_skeleton, tn_skeleton

Derived rates:
  specificity, fdr, fpr, fnr
  accuracy, balanced_accuracy, mcc

Normalisations & exact match:
  shd_strict_norm_edges     shd_strict / max(1, num GT edges)
  shd_strict_norm_possible  shd_strict / (n*(n-1))
  shd_reversal_norm_edges
  shd_reversal_norm_possible
  exact_match               1 if pred == GT else 0

Causal-specific:
  sid                       NaN if predicted graph has cycles; otherwise the
                            Peters & Bühlmann 2015 intervention distance.

Skeleton-only:
  skeleton_shd              number of undirected edge disagreements
  skeleton_precision, skeleton_recall, skeleton_f1

Plus diagnostics: status, pred_edge_count, gt_edge_count, is_dag.
"""

import csv
import json
import math
from collections import OrderedDict
from itertools import combinations
from pathlib import Path


# ============================================================
# CONFIG (mirrors build_adjacency_matrices.py)
# ============================================================

ONTOLOGY_PATH = Path("/mnt/user-data/uploads/ontology.txt")
EXCERPTS_PATH = Path("/mnt/user-data/uploads/excerpts.json")
RESULTS_DIR = Path("/mnt/user-data/uploads")
OUTPUT_ROOT = Path("/mnt/user-data/outputs/metrics")

MODEL_SPECS = [
    ("LLaMA 3.1 8B",            "LLaMA_3_1_8B",          "llama3_1_8b.jsonl"),
    ("Qwen 3.5 9B (thinking)",  "Qwen_3_5_9B_thinking",  "qwen3_5_9b.jsonl"),
    ("Qwen 3.5 9B (instruct)",  "Qwen_3_5_9B_instruct",  "qwen3_5_9b_instruct.jsonl"),
    ("Gemma 4 E4B",             "Gemma_4_E4B",           "gemma4_e4b.jsonl"),
    ("DeepSeek-R1 8B",          "DeepSeek-R1_8B",        "deepseek-r1_8b.jsonl"),
]

COMPOUND_MBV_MEMBERS_LOWER = {"pbv", "market value of equity"}
COMPOUND_MBV_LABEL = "Market-based Firm Value (compound)"


# ============================================================
# Ontology & matrix construction (same logic as builder)
# ============================================================

def build_node_list_13(ontology):
    return list(ontology), {item: item for item in ontology}


def build_node_list_12(ontology):
    labels, mapping = [], {}
    mbv_added = False
    for item in ontology:
        if item.lower() in COMPOUND_MBV_MEMBERS_LOWER:
            mapping[item] = COMPOUND_MBV_LABEL
            if not mbv_added:
                labels.append(COMPOUND_MBV_LABEL)
                mbv_added = True
        else:
            mapping[item] = item
            labels.append(item)
    return labels, mapping


def empty_matrix(n):
    return [[0] * n for _ in range(n)]


def build_gt_matrix(gt_edges, canonical_map, label_to_index, n):
    mat = empty_matrix(n)
    for cause, effect in gt_edges:
        c = canonical_map.get(cause)
        e = canonical_map.get(effect)
        if c is None or e is None:
            continue
        i = label_to_index.get(c)
        j = label_to_index.get(e)
        if i is None or j is None or i == j:
            continue
        mat[i][j] = 1
    return mat


def build_pred_matrix(predicted_edges, canonical_map, label_to_index, n):
    mat = empty_matrix(n)
    seen = set()
    for ed in (predicted_edges or []):
        if ed.get("status") == "not_in_ontology":
            continue
        cause = ed.get("cause")
        effect = ed.get("effect")
        c = canonical_map.get(cause)
        e = canonical_map.get(effect)
        if c is None or e is None:
            continue
        i = label_to_index.get(c)
        j = label_to_index.get(e)
        if i is None or j is None or i == j:
            continue
        key = (c, e)
        if key in seen:
            continue
        seen.add(key)
        mat[i][j] = 1
    return mat


# ============================================================
# Matrix helpers
# ============================================================

def directed_edges(mat, n):
    return {(i, j) for i in range(n) for j in range(n) if i != j and mat[i][j] == 1}


def skeleton_edges(mat, n):
    """Undirected skeleton: unordered pairs {i, j} where at least one direction has an edge."""
    skel = set()
    for i in range(n):
        for j in range(i + 1, n):
            if mat[i][j] == 1 or mat[j][i] == 1:
                skel.add(frozenset((i, j)))
    return skel


def has_cycle(mat, n):
    """DFS-based cycle detection on the directed graph."""
    WHITE, GRAY, BLACK = 0, 1, 2
    color = [WHITE] * n

    def visit(u):
        color[u] = GRAY
        for v in range(n):
            if mat[u][v] == 1:
                if color[v] == GRAY:
                    return True
                if color[v] == WHITE and visit(v):
                    return True
        color[u] = BLACK
        return False

    for u in range(n):
        if color[u] == WHITE:
            if visit(u):
                return True
    return False


# ============================================================
# Directed adjacency metrics
# ============================================================

def adjacency_metrics(pred_edges, gt_edges, n):
    """Adjacency (ignoring direction) precision/recall/F1 on the skeleton."""
    pred_skel = {frozenset(e) for e in pred_edges}
    gt_skel = {frozenset(e) for e in gt_edges}
    tp = len(pred_skel & gt_skel)
    fp = len(pred_skel - gt_skel)
    fn = len(gt_skel - pred_skel)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {
        "adj_precision": precision,
        "adj_recall": recall,
        "adj_f1": f1,
    }


def orientation_metrics_conditional(pred_edges_dir, gt_edges_dir):
    """Convention A: orientation metrics conditional on shared skeleton.
    Among edges present in both skeletons, how many directions agree?"""
    pred_skel = {frozenset(e) for e in pred_edges_dir}
    gt_skel = {frozenset(e) for e in gt_edges_dir}
    shared_skel = pred_skel & gt_skel

    if not shared_skel:
        return {
            "ori_precision": 0.0,
            "ori_recall": 0.0,
            "ori_f1": 0.0,
            "ori_shared_skeleton_count": 0,
            "ori_correct_on_shared": 0,
        }

    # For each shared skeleton edge, we look at directions in each graph.
    # Let pred_dir[{i,j}] = set of directed edges present in pred on this pair.
    pred_dir_for = {frozenset(e): set() for e in pred_edges_dir}
    for e in pred_edges_dir:
        pred_dir_for[frozenset(e)].add(e)
    gt_dir_for = {frozenset(e): set() for e in gt_edges_dir}
    for e in gt_edges_dir:
        gt_dir_for[frozenset(e)].add(e)

    tp = 0
    fp_orient = 0
    fn_orient = 0
    for sk in shared_skel:
        p_dirs = pred_dir_for[sk]
        g_dirs = gt_dir_for[sk]
        # A correctly oriented direction is one present in both.
        tp_here = len(p_dirs & g_dirs)
        fp_here = len(p_dirs - g_dirs)
        fn_here = len(g_dirs - p_dirs)
        tp += tp_here
        fp_orient += fp_here
        fn_orient += fn_here

    precision = tp / (tp + fp_orient) if (tp + fp_orient) > 0 else 0.0
    recall = tp / (tp + fn_orient) if (tp + fn_orient) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {
        "ori_precision": precision,
        "ori_recall": recall,
        "ori_f1": f1,
        "ori_shared_skeleton_count": len(shared_skel),
        "ori_correct_on_shared": tp,
    }


def orientation_metrics_unconditional(pred_edges_dir, gt_edges_dir):
    """Convention B: full directed-edge set comparison. Reversals penalize."""
    pred = set(pred_edges_dir)
    gt = set(gt_edges_dir)
    tp = len(pred & gt)
    fp = len(pred - gt)
    fn = len(gt - pred)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {
        "ori_precision_full": precision,
        "ori_recall_full": recall,
        "ori_f1_full": f1,
    }


# ============================================================
# SHD
# ============================================================

def shd(pred_mat, gt_mat, n, strict=True):
    """Structural Hamming Distance.
    strict=True  : reversals count as 2 operations (delete + add).
    strict=False : reversals count as 1 operation.
    """
    pred = directed_edges(pred_mat, n)
    gt = directed_edges(gt_mat, n)

    if strict:
        # count every disagreement cell
        count = 0
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if pred_mat[i][j] != gt_mat[i][j]:
                    count += 1
        return count
    else:
        # reversal-aware
        pred_skel = {frozenset(e) for e in pred}
        gt_skel = {frozenset(e) for e in gt}
        skel_only_pred = pred_skel - gt_skel  # spurious edges
        skel_only_gt = gt_skel - pred_skel    # missing edges
        shared = pred_skel & gt_skel
        reversals = 0
        for sk in shared:
            # Is direction the same in both?
            i, j = tuple(sk) if len(sk) == 2 else (list(sk)[0], list(sk)[0])
            # For each shared skeleton edge, check per-direction presence
            p_ij = pred_mat[i][j] == 1
            p_ji = pred_mat[j][i] == 1
            g_ij = gt_mat[i][j] == 1
            g_ji = gt_mat[j][i] == 1
            # A reversal: exactly one direction in each, opposite.
            if (p_ij != g_ij) or (p_ji != g_ji):
                reversals += 1
        return len(skel_only_pred) + len(skel_only_gt) + reversals


# ============================================================
# TP/FP/FN/TN on the full adjacency matrix (cell-level)
# ============================================================

def confusion_counts(pred_mat, gt_mat, n):
    """Cell-level confusion on the directed adjacency matrix, excluding diagonal."""
    tp = fp = fn = tn = 0
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if pred_mat[i][j] == 1 and gt_mat[i][j] == 1:
                tp += 1
            elif pred_mat[i][j] == 1 and gt_mat[i][j] == 0:
                fp += 1
            elif pred_mat[i][j] == 0 and gt_mat[i][j] == 1:
                fn += 1
            else:
                tn += 1
    return tp, fp, fn, tn


def derived_rates(tp, fp, fn, tn):
    def safe_div(a, b):
        return a / b if b > 0 else 0.0

    specificity = safe_div(tn, tn + fp)          # TN / (TN + FP)
    fdr = safe_div(fp, tp + fp)                  # FP / (TP + FP)
    fpr = safe_div(fp, fp + tn)                  # FP / (FP + TN)
    fnr = safe_div(fn, fn + tp)                  # FN / (FN + TP)
    accuracy = safe_div(tp + tn, tp + fp + fn + tn)
    sens = safe_div(tp, tp + fn)
    balanced_accuracy = (sens + specificity) / 2

    # MCC
    num = tp * tn - fp * fn
    denom_sq = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    mcc = num / math.sqrt(denom_sq) if denom_sq > 0 else 0.0

    return {
        "specificity": specificity,
        "fdr": fdr,
        "fpr": fpr,
        "fnr": fnr,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "mcc": mcc,
    }


# ============================================================
# Skeleton metrics
# ============================================================

def skeleton_metrics(pred_mat, gt_mat, n):
    pred_sk = skeleton_edges(pred_mat, n)
    gt_sk = skeleton_edges(gt_mat, n)
    tp = len(pred_sk & gt_sk)
    fp = len(pred_sk - gt_sk)
    fn = len(gt_sk - pred_sk)
    total_pairs = n * (n - 1) // 2
    tn = total_pairs - tp - fp - fn

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # Skeleton SHD = edges that disagree in skeleton
    skeleton_shd = fp + fn

    return {
        "skeleton_precision": precision,
        "skeleton_recall": recall,
        "skeleton_f1": f1,
        "skeleton_shd": skeleton_shd,
        "tp_skeleton": tp,
        "fp_skeleton": fp,
        "fn_skeleton": fn,
        "tn_skeleton": tn,
    }


# ============================================================
# SID (Structural Intervention Distance) — Peters & Bühlmann 2015
# ============================================================
#
# For every pair (x, y) with x != y, SID counts the pair as "wrong" if the
# predicted DAG does not give a valid adjustment set for the true causal
# effect of x on y. We use the parent adjustment criterion: for the predicted
# DAG G', the adjustment set for (x, y) is pa_{G'}(x) \ {y}. This set is
# valid if and only if it d-separates x from y in the MANIPULATED true DAG
# (where incoming edges to x are removed). See Peters & Bühlmann (2015)
# Section 3 for the formal definition.
#
# Implementation constraints: pred_mat must be a DAG; if it's cyclic,
# we return None. We implement d-separation via the ancestral-graph
# moralisation method.

def parents(mat, node, n):
    return {i for i in range(n) if mat[i][node] == 1}


def children(mat, node, n):
    return {j for j in range(n) if mat[node][j] == 1}


def ancestors_of(mat, node, n):
    """Set of all ancestors of `node` in the directed graph `mat` (inclusive)."""
    anc = set([node])
    stack = [node]
    while stack:
        v = stack.pop()
        for p in parents(mat, v, n):
            if p not in anc:
                anc.add(p)
                stack.append(p)
    return anc


def d_separated(mat, n, x, y, Z):
    """Return True iff x is d-separated from y by Z in the directed graph `mat`.

    Standard algorithm: restrict to ancestors of {x, y} U Z; moralise
    (connect unmarried parents of every node, then undirect); remove nodes
    in Z; check connectivity between x and y.
    """
    if x == y:
        return False
    # 1) Ancestral set
    anc = set()
    for node in [x, y] + list(Z):
        anc |= ancestors_of(mat, node, n)

    # 2) Build undirected moral graph restricted to `anc`
    undirected = {v: set() for v in anc}
    for v in anc:
        pa_v = parents(mat, v, n) & anc
        # edges v -> parent(v)
        for p in pa_v:
            undirected[v].add(p)
            undirected[p].add(v)
        # marry parents of v
        for p1, p2 in combinations(pa_v, 2):
            undirected[p1].add(p2)
            undirected[p2].add(p1)

    # 3) Remove nodes in Z
    Zset = set(Z)
    for z in Zset:
        if z in undirected:
            neighbors = undirected.pop(z)
            for nb in neighbors:
                if nb in undirected:
                    undirected[nb].discard(z)

    # 4) Check x-y connectivity in what remains
    if x not in undirected or y not in undirected:
        # x or y was in Z, or isolated -> d-separated by definition
        return True
    seen = {x}
    stack = [x]
    while stack:
        v = stack.pop()
        if v == y:
            return False
        for nb in undirected[v]:
            if nb not in seen:
                seen.add(nb)
                stack.append(nb)
    return True


def manipulated_graph_remove_incoming(mat, n, x):
    """Return a copy of `mat` with all incoming edges to x removed."""
    new_mat = [row[:] for row in mat]
    for i in range(n):
        if new_mat[i][x] == 1:
            new_mat[i][x] = 0
    return new_mat


def sid(pred_mat, gt_mat, n):
    """Structural Intervention Distance (Peters & Bühlmann 2015).
    Returns None if pred_mat has cycles. Otherwise returns an integer count
    of (x, y) pairs for which the predicted DAG's parent-adjustment set is
    not a valid adjustment set in the true DAG.
    """
    if has_cycle(pred_mat, n):
        return None

    mismatches = 0
    for x in range(n):
        for y in range(n):
            if x == y:
                continue
            # Adjustment set from predicted DAG: parents of x in G', minus {y}
            Z = (parents(pred_mat, x, n)) - {y}
            # Build GT graph with incoming edges to x removed (do-operation)
            manip = manipulated_graph_remove_incoming(gt_mat, n, x)
            # In the manipulated GT, is x d-separated from y given Z?
            # If the predicted DAG's adjustment set is valid, x should be
            # d-connected to y in the manipulated graph only via the direct
            # (remaining) paths, and Z should NOT block the causal paths.
            # Peters & Bühlmann 2015 criterion: adjustment set Z is valid
            # for the causal effect of x on y iff:
            #   (a) Z contains no descendants of x in GT, AND
            #   (b) Z blocks every non-causal path from x to y in GT.
            # We implement the criterion directly.
            desc_x = descendants_of(gt_mat, x, n)
            if Z & desc_x:
                mismatches += 1
                continue
            # Blocking non-causal paths: equivalent to d-sep in the
            # manipulated graph where incoming edges to x are removed.
            if not d_separated(manip, n, x, y, list(Z)):
                mismatches += 1
    return mismatches


def descendants_of(mat, node, n):
    """Set of descendants of `node` (inclusive of node)."""
    desc = set([node])
    stack = [node]
    while stack:
        v = stack.pop()
        for c in children(mat, v, n):
            if c not in desc:
                desc.add(c)
                stack.append(c)
    return desc


# ============================================================
# Aggregate per-pair into one metric row
# ============================================================

def compute_all_metrics(pred_mat, gt_mat, n):
    """Compute every metric for one (pred, gt) matrix pair."""
    pred_dir = directed_edges(pred_mat, n)
    gt_dir = directed_edges(gt_mat, n)
    pred_skel = skeleton_edges(pred_mat, n)
    gt_skel = skeleton_edges(gt_mat, n)

    out = {}

    # Adjacency (skeleton-level P/R/F1)
    out.update(adjacency_metrics(pred_dir, gt_dir, n))

    # Orientation — two conventions
    out.update(orientation_metrics_conditional(pred_dir, gt_dir))
    out.update(orientation_metrics_unconditional(pred_dir, gt_dir))

    # SHD — two variants
    out["shd_strict"] = shd(pred_mat, gt_mat, n, strict=True)
    out["shd_reversal"] = shd(pred_mat, gt_mat, n, strict=False)

    # Confusion counts on directed adjacency
    tp, fp, fn, tn = confusion_counts(pred_mat, gt_mat, n)
    out.update({"tp": tp, "fp": fp, "fn": fn, "tn": tn})
    out.update(derived_rates(tp, fp, fn, tn))

    # Normalisations
    gt_edge_count = len(gt_dir)
    possible_edges = n * (n - 1)
    out["shd_strict_norm_edges"] = out["shd_strict"] / max(1, gt_edge_count)
    out["shd_strict_norm_possible"] = out["shd_strict"] / possible_edges
    out["shd_reversal_norm_edges"] = out["shd_reversal"] / max(1, gt_edge_count)
    out["shd_reversal_norm_possible"] = out["shd_reversal"] / possible_edges

    # Exact match
    out["exact_match"] = int(pred_dir == gt_dir)

    # Skeleton metrics
    out.update(skeleton_metrics(pred_mat, gt_mat, n))

    # SID (may return None if predicted graph has cycles)
    sid_val = sid(pred_mat, gt_mat, n)
    out["sid"] = sid_val
    out["is_dag"] = int(not has_cycle(pred_mat, n))

    # Diagnostics
    out["pred_edge_count"] = len(pred_dir)
    out["gt_edge_count"] = gt_edge_count

    return out


# ============================================================
# Aggregation (macro mean across excerpts, for a given model)
# ============================================================

def macro_aggregate(per_excerpt_rows):
    """Given a list of metric dicts (one per excerpt, all same model), return
    a dict of means. Skips None values for SID. Integer metrics get float means."""
    if not per_excerpt_rows:
        return {}
    keys = list(per_excerpt_rows[0].keys())
    agg = {}
    for k in keys:
        vals = [r[k] for r in per_excerpt_rows if r[k] is not None]
        if not vals:
            agg[k + "_mean"] = None
            continue
        if isinstance(vals[0], (int, float)):
            agg[k + "_mean"] = sum(vals) / len(vals)
        else:
            agg[k + "_mean"] = None
    # Also report the fraction of excerpts where SID was computable
    sid_vals = [r.get("sid") for r in per_excerpt_rows]
    agg["sid_computable_fraction"] = sum(1 for v in sid_vals if v is not None) / len(sid_vals)
    return agg


# ============================================================
# I/O
# ============================================================

def load_ontology(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def load_excerpts(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path):
    records = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            records[rec["excerpt_id"]] = rec
    return records


def format_value(v):
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.4f}"
    return v


def write_csv(path, rows, column_order):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(column_order)
        for r in rows:
            w.writerow([format_value(r.get(c)) for c in column_order])


# ============================================================
# Per-view pipeline
# ============================================================

PER_PAIR_COLUMNS = [
    "model", "excerpt_id", "firm", "status", "is_dag",
    "gt_edge_count", "pred_edge_count",
    # Core
    "adj_precision", "adj_recall", "adj_f1",
    "ori_precision", "ori_recall", "ori_f1",
    "ori_precision_full", "ori_recall_full", "ori_f1_full",
    "ori_shared_skeleton_count", "ori_correct_on_shared",
    "shd_strict", "shd_reversal",
    # Descriptive counts
    "tp", "fp", "fn", "tn",
    "tp_skeleton", "fp_skeleton", "fn_skeleton", "tn_skeleton",
    # Derived rates
    "specificity", "fdr", "fpr", "fnr",
    "accuracy", "balanced_accuracy", "mcc",
    # Normalisations
    "shd_strict_norm_edges", "shd_strict_norm_possible",
    "shd_reversal_norm_edges", "shd_reversal_norm_possible",
    "exact_match",
    # Skeleton
    "skeleton_precision", "skeleton_recall", "skeleton_f1", "skeleton_shd",
    # SID
    "sid",
]


def run_view(view_name, labels, canonical_map, excerpts, model_records_by_display):
    out_dir = OUTPUT_ROOT / view_name
    out_dir.mkdir(parents=True, exist_ok=True)

    n = len(labels)
    label_to_index = {label: i for i, label in enumerate(labels)}

    excerpt_order = [e["id"] for e in excerpts]
    firm_by_id = {e["id"]: e["firm"] for e in excerpts}
    gt_edges_by_id = {e["id"]: e["ground_truth_edges"] for e in excerpts}

    gt_matrices = {
        eid: build_gt_matrix(gt_edges_by_id[eid], canonical_map, label_to_index, n)
        for eid in excerpt_order
    }

    per_pair_rows = []
    per_model_rows = []

    for display_name, _slug, _fname in MODEL_SPECS:
        records = model_records_by_display.get(display_name, {})
        model_excerpt_metrics = []

        for eid in excerpt_order:
            rec = records.get(eid)
            gt_mat = gt_matrices[eid]
            status = (rec or {}).get("status") if rec else "missing"

            if rec is None:
                pred_mat = empty_matrix(n)
            else:
                pred_mat = build_pred_matrix(
                    rec.get("predicted_edges"), canonical_map, label_to_index, n
                )

            metrics = compute_all_metrics(pred_mat, gt_mat, n)

            row = {
                "model": display_name,
                "excerpt_id": eid,
                "firm": firm_by_id[eid],
                "status": status,
                **metrics,
            }
            per_pair_rows.append(row)
            model_excerpt_metrics.append(metrics)

        # Macro aggregate for this model
        agg = macro_aggregate(model_excerpt_metrics)
        per_model_rows.append({
            "model": display_name,
            "n_excerpts": len(model_excerpt_metrics),
            **agg,
        })

    write_csv(out_dir / "metrics_per_pair.csv", per_pair_rows, PER_PAIR_COLUMNS)

    per_model_columns = ["model", "n_excerpts"] + [
        k + "_mean" for k in PER_PAIR_COLUMNS
        if k not in ("model", "excerpt_id", "firm", "status")
    ] + ["sid_computable_fraction"]
    write_csv(out_dir / "metrics_per_model.csv", per_model_rows, per_model_columns)

    return per_pair_rows, per_model_rows


# ============================================================
# MAIN
# ============================================================

def main():
    ontology = load_ontology(ONTOLOGY_PATH)
    excerpts = load_excerpts(EXCERPTS_PATH)

    model_records_by_display = {}
    for display_name, _slug, fname in MODEL_SPECS:
        path = RESULTS_DIR / fname
        if not path.exists():
            print(f"WARNING: {fname} not found; {display_name} will have empty predictions")
            model_records_by_display[display_name] = {}
            continue
        model_records_by_display[display_name] = load_jsonl(path)

    # 13-node view
    labels_13, map_13 = build_node_list_13(ontology)
    print(f"=== 13-node view: {len(labels_13)} nodes ===")
    pp13, pm13 = run_view("13nodes", labels_13, map_13, excerpts, model_records_by_display)

    # 12-node view
    labels_12, map_12 = build_node_list_12(ontology)
    print(f"=== 12-node view: {len(labels_12)} nodes (compound MBV) ===")
    pp12, pm12 = run_view("12nodes", labels_12, map_12, excerpts, model_records_by_display)

    # Human-readable summary: key metrics per model (13-node view)
    print()
    print("=" * 100)
    print("Per-model macro means (13-node view):")
    print("=" * 100)
    header = f"{'model':<28s} {'adj_f1':>8s} {'ori_f1':>8s} {'shd_str':>8s} {'mcc':>8s} {'ex_match':>10s} {'sid':>8s} {'sid_ok':>8s}"
    print(header)
    for r in pm13:
        m = r["model"]
        adj = r.get("adj_f1_mean")
        ori = r.get("ori_f1_mean")
        shd_s = r.get("shd_strict_mean")
        mcc = r.get("mcc_mean")
        em = r.get("exact_match_mean")
        sid_m = r.get("sid_mean")
        sid_ok = r.get("sid_computable_fraction")
        def fmt(v, w, prec=3):
            if v is None:
                return f"{'—':>{w}}"
            if isinstance(v, float):
                return f"{v:>{w}.{prec}f}"
            return f"{v:>{w}}"
        print(f"{m:<28s} {fmt(adj, 8)} {fmt(ori, 8)} {fmt(shd_s, 8, 2)} {fmt(mcc, 8)} {fmt(em, 10)} {fmt(sid_m, 8, 1)} {fmt(sid_ok, 8, 2)}")

    print()
    print("Output tree:")
    for p in sorted(OUTPUT_ROOT.rglob("*.csv")):
        size = p.stat().st_size
        rel = p.relative_to(OUTPUT_ROOT)
        print(f"  {rel}  ({size} bytes)")


if __name__ == "__main__":
    main()
