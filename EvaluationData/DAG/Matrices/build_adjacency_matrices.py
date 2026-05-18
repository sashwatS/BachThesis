"""
Build adjacency matrices from Kaggle JSONL results and emit as CSVs.

Produces two parallel output trees:
    matrices/13nodes/   — ontology as-is (13 nodes). PBV and MVE distinct.
    matrices/12nodes/   — compound MBV canonicalisation applied
                          (PBV and Market Value of Equity merged into
                          'Market-based Firm Value (compound)').

Inside each tree:
    ground_truth.csv         Single matrix (Chain A GT is identical across
                             all 5 excerpts, so one is sufficient).
    summary_counts.csv       Per-model status counts.
    edge_summary.csv         Long-form union: every edge ever predicted or
                             in GT, one row per unique edge, one column per
                             model, cells are counts across excerpts (0-5).
    <Model>.csv              One file per model. Contains stacked per-excerpt
                             matrix blocks separated by blank rows with
                             excerpt_id / firm / status markers (Option B).

Rerun behaviour: idempotent. To fold Qwen-thinking-32K rerun results in,
either (a) replace /mnt/user-data/uploads/qwen3_5_9b.jsonl with merged
results, or (b) add a new entry to MODEL_SPECS pointing at the rerun file
and rerun this script.
"""

import csv
import json
from collections import OrderedDict
from pathlib import Path


# ============================================================
# CONFIG
# ============================================================

ONTOLOGY_PATH = Path("/mnt/user-data/uploads/ontology.txt")
EXCERPTS_PATH = Path("/mnt/user-data/uploads/excerpts.json")
RESULTS_DIR = Path("/mnt/user-data/uploads")
OUTPUT_ROOT = Path("/mnt/user-data/outputs/matrices")

# (display_name, filename_slug, jsonl_filename)
MODEL_SPECS = [
    ("LLaMA 3.1 8B",            "LLaMA_3_1_8B",          "llama3_1_8b.jsonl"),
    ("Qwen 3.5 9B (thinking)",  "Qwen_3_5_9B_thinking",  "qwen3_5_9b.jsonl"),
    ("Qwen 3.5 9B (instruct)",  "Qwen_3_5_9B_instruct",  "qwen3_5_9b_instruct.jsonl"),
    ("Gemma 4 E4B",             "Gemma_4_E4B",           "gemma4_e4b.jsonl"),
    ("DeepSeek-R1 8B",          "DeepSeek-R1_8B",        "deepseek-r1_8b.jsonl"),
]

# Compound MBV: these ontology items collapse to a single node in 12-node view.
COMPOUND_MBV_MEMBERS_LOWER = {"pbv", "market value of equity"}
COMPOUND_MBV_LABEL = "Market-based Firm Value (compound)"


# ============================================================
# ONTOLOGY CANONICALISATION
# ============================================================

def build_node_list_13(ontology):
    """13-node view: ontology as-is. Returns (labels, original_to_canonical).
    original_to_canonical is identity here."""
    labels = list(ontology)
    mapping = {item: item for item in ontology}
    return labels, mapping


def build_node_list_12(ontology):
    """12-node view: PBV and Market Value of Equity collapse to compound node."""
    labels = []
    mapping = {}
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


# ============================================================
# LOAD
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


# ============================================================
# MATRIX CONSTRUCTION
# ============================================================

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
    """Build prediction matrix + stats (not_in_ontology, self_loop, duplicate)."""
    mat = empty_matrix(n)
    stats = {
        "predicted_edges_total": len(predicted_edges or []),
        "on_matrix_edges":       0,
        "not_in_ontology_edges": 0,
        "self_loop_edges":       0,
        "duplicate_edges":       0,
    }
    seen = set()
    for ed in (predicted_edges or []):
        status = ed.get("status")
        cause = ed.get("cause")
        effect = ed.get("effect")
        if status == "not_in_ontology":
            stats["not_in_ontology_edges"] += 1
            continue
        c = canonical_map.get(cause)
        e = canonical_map.get(effect)
        if c is None or e is None:
            stats["not_in_ontology_edges"] += 1
            continue
        i = label_to_index.get(c)
        j = label_to_index.get(e)
        if i is None or j is None:
            stats["not_in_ontology_edges"] += 1
            continue
        if i == j:
            stats["self_loop_edges"] += 1
            continue
        key = (c, e)
        if key in seen:
            stats["duplicate_edges"] += 1
            continue
        seen.add(key)
        mat[i][j] = 1
        stats["on_matrix_edges"] += 1
    return mat, stats


# ============================================================
# CSV EMITTERS
# ============================================================

def write_matrix_csv(path, matrix, labels, header_label):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([header_label] + labels)
        for i, row_label in enumerate(labels):
            w.writerow([row_label] + matrix[i])


def write_model_csv_option_b(path, display_name, excerpt_order, matrices_by_excerpt,
                              metadata_by_excerpt, labels):
    """Option B: stacked per-excerpt matrix blocks, one per excerpt."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        for k, eid in enumerate(excerpt_order):
            md = metadata_by_excerpt.get(eid, {})
            w.writerow([f"excerpt_id: {eid}"])
            w.writerow([f"firm: {md.get('firm', '')}"])
            w.writerow([f"status: {md.get('status', '')}"])
            w.writerow([f"cause \\ effect ({display_name})"] + labels)
            mat = matrices_by_excerpt[eid]
            for i, row_label in enumerate(labels):
                w.writerow([row_label] + mat[i])
            if k < len(excerpt_order) - 1:
                w.writerow([])  # blank separator between blocks


def write_summary_counts_csv(path, per_model_stats, gt_record_count, gt_unique_edges):
    """per_model_stats: OrderedDict keyed by display_name, each value is
    a dict with status_counts (sub-dict), total_records, contributing, unique_edges.
    Matches the pair-run schema."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "Configuration", "Total records",
            "ok", "malformed", "none", "not_in_ontology", "error",
            "Contributing to matrix", "Unique edges in matrix",
        ])
        for display_name, stats in per_model_stats.items():
            sc = stats["status_counts"]
            w.writerow([
                display_name,
                stats["total_records"],
                sc.get("ok", 0),
                sc.get("malformed", 0),
                sc.get("none", 0),
                sc.get("not_in_ontology", 0),
                sc.get("error", 0),
                stats["contributing"],
                stats["unique_edges"],
            ])
        w.writerow([])
        w.writerow([
            "Ground truth",
            gt_record_count,
            "—", "—", "—", "—", "—",
            gt_record_count,
            gt_unique_edges,
        ])


def write_edge_summary_csv(path, model_display_names, all_edges, gt_counts, pred_counts_by_model):
    """Long-form: one row per unique edge (cause, effect). Columns: Cause,
    Effect, Ground truth count, then one column per model with the number
    of excerpts (out of 5) in which the model predicted that edge."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Cause", "Effect", "Ground truth"] + list(model_display_names))
        for (cause, effect) in all_edges:
            row = [cause, effect, gt_counts.get((cause, effect), 0)]
            for m in model_display_names:
                row.append(pred_counts_by_model.get(m, {}).get((cause, effect), 0))
            w.writerow(row)


# ============================================================
# PER-VIEW PIPELINE
# ============================================================

def run_view(view_name, labels, canonical_map, excerpts, model_records_by_display):
    """Run the full pipeline for one canonicalisation view (13nodes or 12nodes).
    Writes all CSVs into OUTPUT_ROOT/view_name/."""
    out_dir = OUTPUT_ROOT / view_name
    out_dir.mkdir(parents=True, exist_ok=True)

    n = len(labels)
    label_to_index = {label: i for i, label in enumerate(labels)}

    excerpt_order = [e["id"] for e in excerpts]
    firm_by_id = {e["id"]: e["firm"] for e in excerpts}
    gt_edges_by_id = {e["id"]: e["ground_truth_edges"] for e in excerpts}

    # --- Ground truth: single matrix (Chain A GT is uniform across excerpts) ---
    gt_matrices = {
        eid: build_gt_matrix(gt_edges_by_id[eid], canonical_map, label_to_index, n)
        for eid in excerpt_order
    }
    # Chain A GT is the same across excerpts, so just use the first one.
    gt_matrix_single = gt_matrices[excerpt_order[0]]
    write_matrix_csv(
        out_dir / "ground_truth.csv",
        gt_matrix_single, labels, "cause \\ effect (GT)"
    )

    # Sanity check: assert GT is uniform across excerpts (should be for Chain A).
    for eid in excerpt_order[1:]:
        if gt_matrices[eid] != gt_matrix_single:
            print(f"WARNING [{view_name}]: GT matrix for {eid} differs from first excerpt")

    # --- Per-model matrices + metadata + stats ---
    per_model_stats = OrderedDict()
    pred_matrices_by_model_excerpt = {}  # {display_name: {eid: matrix}}
    pred_counts_by_model_for_edge_summary = {}  # {display_name: {(cause,effect): count}}

    for display_name, slug, _fname in MODEL_SPECS:
        records = model_records_by_display.get(display_name, {})

        matrices_by_excerpt = {}
        metadata_by_excerpt = {}
        status_counts = {}
        contributing = 0
        unique_edges = set()
        edge_excerpt_count = {}  # (cause, effect) -> number of excerpts it appeared in

        for eid in excerpt_order:
            rec = records.get(eid)
            if rec is None:
                matrices_by_excerpt[eid] = empty_matrix(n)
                metadata_by_excerpt[eid] = {"firm": firm_by_id[eid], "status": "missing"}
                status_counts["missing"] = status_counts.get("missing", 0) + 1
                continue

            status = rec.get("status") or ("error" if rec.get("error") else "unknown")
            status_counts[status] = status_counts.get(status, 0) + 1

            mat, stats = build_pred_matrix(
                rec.get("predicted_edges"), canonical_map, label_to_index, n
            )
            matrices_by_excerpt[eid] = mat
            metadata_by_excerpt[eid] = {
                "firm": rec.get("firm") or firm_by_id[eid],
                "status": status,
            }

            if stats["on_matrix_edges"] > 0:
                contributing += 1
            # Collect unique edges from this matrix
            for i in range(n):
                for j in range(n):
                    if mat[i][j] == 1:
                        edge = (labels[i], labels[j])
                        unique_edges.add(edge)
                        edge_excerpt_count[edge] = edge_excerpt_count.get(edge, 0) + 1

        # Write the per-model CSV
        write_model_csv_option_b(
            out_dir / f"{slug}.csv",
            display_name, excerpt_order, matrices_by_excerpt, metadata_by_excerpt, labels
        )

        per_model_stats[display_name] = {
            "status_counts": status_counts,
            "total_records": sum(status_counts.values()),
            "contributing": contributing,
            "unique_edges": len(unique_edges),
        }
        pred_matrices_by_model_excerpt[display_name] = matrices_by_excerpt
        pred_counts_by_model_for_edge_summary[display_name] = edge_excerpt_count

    # --- summary_counts.csv ---
    # GT: one "record" per excerpt, unique edges from the single GT matrix.
    gt_unique_edges_set = set()
    for i in range(n):
        for j in range(n):
            if gt_matrix_single[i][j] == 1:
                gt_unique_edges_set.add((labels[i], labels[j]))
    write_summary_counts_csv(
        out_dir / "summary_counts.csv",
        per_model_stats,
        gt_record_count=len(excerpt_order),
        gt_unique_edges=len(gt_unique_edges_set),
    )

    # --- edge_summary.csv ---
    # GT counts: how many excerpts have this edge in GT (for Chain A uniform GT,
    # this will be 5 for every GT edge).
    gt_edge_counts = {edge: 0 for edge in gt_unique_edges_set}
    for eid in excerpt_order:
        mat = gt_matrices[eid]
        for i in range(n):
            for j in range(n):
                if mat[i][j] == 1:
                    edge = (labels[i], labels[j])
                    gt_edge_counts[edge] = gt_edge_counts.get(edge, 0) + 1

    # Universe of unique edges across GT + all model predictions
    all_edges_set = set(gt_edge_counts.keys())
    for m_edges in pred_counts_by_model_for_edge_summary.values():
        all_edges_set.update(m_edges.keys())
    # Order edges by (cause-row-index, effect-col-index) for readability
    all_edges_sorted = sorted(
        all_edges_set,
        key=lambda ce: (label_to_index[ce[0]], label_to_index[ce[1]])
    )
    write_edge_summary_csv(
        out_dir / "edge_summary.csv",
        model_display_names=[d for d, _, _ in MODEL_SPECS],
        all_edges=all_edges_sorted,
        gt_counts=gt_edge_counts,
        pred_counts_by_model=pred_counts_by_model_for_edge_summary,
    )

    return per_model_stats


# ============================================================
# MAIN
# ============================================================

def main():
    ontology = load_ontology(ONTOLOGY_PATH)
    excerpts = load_excerpts(EXCERPTS_PATH)

    # Pre-load all model JSONLs once; reused across both views.
    model_records_by_display = {}
    for display_name, _slug, fname in MODEL_SPECS:
        path = RESULTS_DIR / fname
        if not path.exists():
            print(f"WARNING: {fname} not found; {display_name} will have no records")
            model_records_by_display[display_name] = {}
            continue
        model_records_by_display[display_name] = load_jsonl(path)

    # View 1: 13-node (ontology as-is)
    labels_13, map_13 = build_node_list_13(ontology)
    print(f"=== 13-node view: {len(labels_13)} nodes ===")
    stats_13 = run_view("13nodes", labels_13, map_13, excerpts, model_records_by_display)

    # View 2: 12-node (compound MBV)
    labels_12, map_12 = build_node_list_12(ontology)
    print(f"=== 12-node view: {len(labels_12)} nodes (compound MBV) ===")
    stats_12 = run_view("12nodes", labels_12, map_12, excerpts, model_records_by_display)

    # Summary print
    print()
    print(f"Output tree: {OUTPUT_ROOT}")
    for view in ["13nodes", "12nodes"]:
        view_dir = OUTPUT_ROOT / view
        files = sorted(p.name for p in view_dir.iterdir())
        print(f"\n  {view}/")
        for fn in files:
            size = (view_dir / fn).stat().st_size
            print(f"    {fn}  ({size} bytes)")

    print("\n13-node status counts per model:")
    for m, s in stats_13.items():
        print(f"  {m:<30s}  {s['status_counts']}  unique_edges={s['unique_edges']}")


if __name__ == "__main__":
    main()
