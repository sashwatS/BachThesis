#!/usr/bin/env python3
"""Generate causal-graph and DAG figures for the thesis.

Outputs PNG @ 300 DPI (portrait-friendly):
  gt_dag_chainA.png    - 4-node ground-truth DAG (Chain A)
  gt_pairs_graph.png   - 13-node, 10-edge pairs ground-truth graph (single portrait)
  graphs_pairs.png     - 2x3 grid: GT + 5 model predictions, pairs experiment
  graphs_dag.png       - 2x3 grid: GT + 5 model predictions, DAG (compound)

Predicted-graph panels colour edges by classification against the ground truth:
  TP (in GT and predicted): solid dark green
  FP (predicted but not in GT): solid dark red
  FN (in GT but not predicted): dashed gray
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


# ---------------------------------------------------------------------------
# Ground-truth definitions
# ---------------------------------------------------------------------------

NODES_13 = [
    "Board Strategy", "Environmental Reporting", "GHG Emissions", "PBV",
    "Sustainable Finance", "Market Price of Share", "Environmental Fines",
    "Market Value of Equity", "Audit", "Corporate Governance",
    "Earnings Quality", "Employee Development", "Net Sales",
]

NODES_12 = [
    "Board Strategy", "Environmental Reporting", "GHG Emissions",
    "Market-based Firm Value (compound)", "Sustainable Finance",
    "Market Price of Share", "Environmental Fines", "Audit",
    "Corporate Governance", "Earnings Quality", "Employee Development",
    "Net Sales",
]

GT_PAIRS_EDGES = [
    ("Board Strategy", "Environmental Reporting"),
    ("Environmental Reporting", "GHG Emissions"),
    ("GHG Emissions", "PBV"),
    ("GHG Emissions", "Market Price of Share"),
    ("Sustainable Finance", "GHG Emissions"),
    ("Sustainable Finance", "Market Price of Share"),
    ("Environmental Fines", "Market Value of Equity"),
    ("Audit", "Earnings Quality"),
    ("Corporate Governance", "Earnings Quality"),
    ("Employee Development", "Net Sales"),
]

# Edge signs (derived from the prose enumerating each edge in Section 3.4
# of the thesis). "+" = positive effect (raises/improves/increases),
# "-" = negative effect (reduces/priced negatively/imposes losses).
GT_PAIRS_SIGNS = {
    ("Audit", "Earnings Quality"): "+",
    ("Corporate Governance", "Earnings Quality"): "+",
    ("Board Strategy", "Environmental Reporting"): "+",
    ("Environmental Reporting", "GHG Emissions"): "−",
    ("Sustainable Finance", "Market Price of Share"): "+",
    ("Sustainable Finance", "GHG Emissions"): "−",
    ("GHG Emissions", "PBV"): "−",
    ("GHG Emissions", "Market Price of Share"): "−",
    ("Environmental Fines", "Market Value of Equity"): "−",
    ("Employee Development", "Net Sales"): "+",
}

# Edges whose sign should be placed BELOW the arrow rather than the default
# above-the-arrow placement (used when the default position would collide
# with another edge in the layout).
GT_PAIRS_SIGN_BELOW = {
    ("Corporate Governance", "Earnings Quality"),
}

GT_DAG_EDGES = [
    ("Corporate Governance", "Environmental Reporting"),
    ("Environmental Reporting", "GHG Emissions"),
    ("GHG Emissions", "Market-based Firm Value (compound)"),
]

MODEL_ORDER = [
    "LLaMA 3.1 8B",
    "Gemma 4 E4B",
    "Qwen 3.5 9B (instruct)",
    "DeepSeek-R1 8B",
    "Qwen 3.5 9B (thinking)",
]


# ---------------------------------------------------------------------------
# Manual layouts (shared across all panels of an experiment)
# ---------------------------------------------------------------------------

POS_PAIRS = {
    "Board Strategy":          (0.0, 6.5),
    "Environmental Reporting": (3.0, 6.5),
    "GHG Emissions":           (6.0, 6.5),
    "PBV":                     (8.7, 7.4),
    "Market Price of Share":   (8.7, 5.6),
    "Sustainable Finance":     (4.0, 4.2),
    "Environmental Fines":     (11.4, 6.5),
    "Market Value of Equity":  (11.4, 4.5),
    "Audit":                   (0.0, 1.5),
    "Corporate Governance":    (2.5, 0.5),
    "Earnings Quality":        (5.0, 1.5),
    "Employee Development":    (8.0, 1.0),
    "Net Sales":                (11.0, 1.0),
}

POS_DAG = {
    "Corporate Governance":               (0.0, 5.0),
    "Environmental Reporting":            (3.5, 5.0),
    "GHG Emissions":                      (7.0, 5.0),
    "Market-based Firm Value (compound)": (10.5, 5.0),
    "Board Strategy":          (0.0, 2.5),
    "Sustainable Finance":     (2.5, 2.5),
    "Market Price of Share":   (5.0, 2.5),
    "Environmental Fines":     (7.5, 2.5),
    "Audit":                   (1.0, 0.6),
    "Earnings Quality":        (3.5, 0.6),
    "Employee Development":    (6.0, 0.6),
    "Net Sales":               (8.5, 0.6),
}


# ---------------------------------------------------------------------------
# Colours / styles
# ---------------------------------------------------------------------------

C_GT = "#222222"
C_TP = "#1b7a3a"
C_FP = "#c0322b"
C_FN = "#666666"
C_NODE_FACE = "white"
C_NODE_EDGE = "#222222"

DPI = 300


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------

def read_edge_summary(path: Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def edges_for_model(rows: list[dict[str, str]], model_col: str) -> list[tuple[str, str]]:
    out = []
    for r in rows:
        try:
            v = int(r.get(model_col, 0) or 0)
        except ValueError:
            v = 0
        if v > 0:
            out.append((r["Cause"], r["Effect"]))
    return out


# ---------------------------------------------------------------------------
# Drawing primitives
# ---------------------------------------------------------------------------

def _wrap_label(name: str) -> str:
    if name == "Market-based Firm Value (compound)":
        return "Market-based\nFirm Value"
    if name == "Market Value of Equity":
        return "Market Value\nof Equity"
    if name == "Market Price of Share":
        return "Market Price\nof Share"
    if name == "Environmental Reporting":
        return "Environmental\nReporting"
    if name == "Environmental Fines":
        return "Environmental\nFines"
    if name == "Corporate Governance":
        return "Corporate\nGovernance"
    if name == "Sustainable Finance":
        return "Sustainable\nFinance"
    if name == "Employee Development":
        return "Employee\nDevelopment"
    if name == "Earnings Quality":
        return "Earnings\nQuality"
    if name == "GHG Emissions":
        return "GHG\nEmissions"
    if name == "Board Strategy":
        return "Board\nStrategy"
    return name


def _wrap_label_compact(name: str) -> str:
    """Tighter labels used in the dense 2x3 panel grids: shortens
    'Environmental *' to 'Env. *' so the text fits inside the node boxes.
    Everything else falls through to the default wrapping."""
    if name == "Environmental Reporting":
        return "Env.\nReporting"
    if name == "Environmental Fines":
        return "Env. Fines"
    return _wrap_label(name)


# Per-node font-size overrides used in the panel grids. Labels with long
# words (e.g. "Development") need a slightly smaller font to fit cleanly.
GRID_FS_OVERRIDES = {
    "Employee Development": 5.6,
}


def _draw_node_box(ax, x, y, label, w=1.6, h=0.85, fontsize=6.5):
    box = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.0,rounding_size=0.18",
        facecolor=C_NODE_FACE, edgecolor=C_NODE_EDGE,
        linewidth=0.9, zorder=3,
    )
    ax.add_patch(box)
    ax.text(x, y, label, ha="center", va="center",
            fontsize=fontsize, zorder=4)
    return box


def _draw_edge(ax, p_from, p_to, color, linestyle="-", lw=1.4, alpha=1.0,
               rad=0.0, mutation_scale=14, patch_a=None, patch_b=None,
               zorder=2):
    arrow = FancyArrowPatch(
        p_from, p_to,
        arrowstyle="-|>", mutation_scale=mutation_scale,
        connectionstyle=f"arc3,rad={rad}",
        color=color, linestyle=linestyle, lw=lw, alpha=alpha,
        patchA=patch_a, patchB=patch_b,
        shrinkA=2, shrinkB=2,
        zorder=zorder,
    )
    ax.add_patch(arrow)


def _classify_edges(gt_edges, pred_edges):
    gt = set(gt_edges)
    pr = set(pred_edges) if pred_edges is not None else None
    if pr is None:
        return list(gt), [], []
    return list(gt & pr), list(pr - gt), list(gt - pr)


def draw_panel(ax, nodes, gt_edges, pred_edges, pos, title="",
               node_w=1.7, node_h=0.85, node_fs=6.5,
               mutation_scale=14, title_fs=10,
               label_fn=None, fs_overrides=None):
    tp, fp, fn = _classify_edges(gt_edges, pred_edges)
    label_fn = label_fn or _wrap_label
    fs_overrides = fs_overrides or {}

    patches: dict[str, FancyBboxPatch] = {}
    for node in nodes:
        if node not in pos:
            continue
        x, y = pos[node]
        fs = fs_overrides.get(node, node_fs)
        patches[node] = _draw_node_box(
            ax, x, y, label_fn(node),
            w=node_w, h=node_h, fontsize=fs,
        )

    # Draw FN first (bottom), then FP, then TP on top so the green TP edges
    # remain visible even when a red FP edge crosses over the same region
    # (e.g. Gemma's GHG \to MBV true positive being hidden under the
    # Sustainable Finance \to MBV false positive).
    all_edges = [(*e, "fn") for e in fn] + [(*e, "fp") for e in fp] + [(*e, "tp") for e in tp]
    pair_seen: dict[frozenset, int] = {}
    for u, v, _ in all_edges:
        key = frozenset({u, v})
        pair_seen[key] = pair_seen.get(key, 0) + 1

    # (colour, linestyle, lw, alpha, zorder)
    style_map = {
        "tp": (C_TP if pred_edges is not None else C_GT, "-", 1.9, 1.0, 4),
        "fp": (C_FP, "-", 1.3, 0.95, 2),
        "fn": (C_FN, (0, (4, 2)), 1.3, 0.85, 2),
    }

    # When two arrows share the same node pair (bidirectional pairs), shorten
    # the lower-priority one so both arrowheads remain visible on the same
    # line. Priority: TP (green) keeps full length; FP (red) is shorter;
    # FN (gray) is shortest. Solo edges always render at full length.
    length_fraction = {"tp": 1.0, "fp": 0.80, "fn": 0.65}

    used_dirs: dict[frozenset, list[tuple[str, str]]] = {}

    for u, v, kind in all_edges:
        if u not in patches or v not in patches:
            continue
        color, ls, lw, alpha, zo = style_map[kind]
        key = frozenset({u, v})
        used_dirs.setdefault(key, []).append((u, v))

        is_multi = pair_seen[key] > 1
        rad = 0.0
        fraction = 1.0
        if is_multi:
            # Slight curvature still helps separate the two paths visually
            rad = 0.20 if used_dirs[key][0] == (u, v) else -0.20
            fraction = length_fraction[kind]

        if fraction < 1.0:
            # Compute a manually shrunk target along the (source, target)
            # vector. Source side is still auto-clipped to the source node
            # via patchA; target side floats free (no patchB), so the
            # arrowhead lands at the shrunk endpoint rather than at the
            # target node border.
            x1, y1 = pos[u]
            x2, y2 = pos[v]
            end_point = (x1 + (x2 - x1) * fraction,
                         y1 + (y2 - y1) * fraction)
            _draw_edge(
                ax, pos[u], end_point, color=color, linestyle=ls,
                lw=lw, alpha=alpha, rad=rad,
                mutation_scale=mutation_scale,
                patch_a=patches[u], patch_b=None,
                zorder=zo,
            )
        else:
            _draw_edge(
                ax, pos[u], pos[v], color=color, linestyle=ls,
                lw=lw, alpha=alpha, rad=rad,
                mutation_scale=mutation_scale,
                patch_a=patches[u], patch_b=patches[v],
                zorder=zo,
            )

    if title:
        ax.set_title(title, fontsize=title_fs, pad=6)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_aspect("equal")

    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    ax.set_xlim(min(xs) - 1.2, max(xs) + 1.2)
    ax.set_ylim(min(ys) - 0.8, max(ys) + 0.8)


# ---------------------------------------------------------------------------
# High-level builders
# ---------------------------------------------------------------------------

def build_gt_dag_chainA(outpath: Path):
    fig, ax = plt.subplots(figsize=(8.0, 2.6))
    chain_pos = {
        "Corporate Governance":               (0.0, 0.0),
        "Environmental Reporting":            (3.2, 0.0),
        "GHG Emissions":                      (6.4, 0.0),
        "Market-based Firm Value (compound)": (9.6, 0.0),
    }
    signs = {
        ("Corporate Governance", "Environmental Reporting"): "+",
        ("Environmental Reporting", "GHG Emissions"): "−",
        ("GHG Emissions", "Market-based Firm Value (compound)"): "−",
    }
    patches: dict[str, FancyBboxPatch] = {}
    for node, (x, y) in chain_pos.items():
        patches[node] = _draw_node_box(
            ax, x, y, _wrap_label(node),
            w=2.6, h=1.05, fontsize=10,
        )
    for (u, v), sign in signs.items():
        x1, y1 = chain_pos[u]
        x2, y2 = chain_pos[v]
        _draw_edge(
            ax, (x1, y1), (x2, y2), color="black",
            lw=1.8, mutation_scale=20,
            patch_a=patches[u], patch_b=patches[v],
        )
        ax.text((x1 + x2) / 2, y1 + 0.30, sign,
                ha="center", va="bottom",
                fontsize=14, fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xlim(-1.6, 11.2)
    ax.set_ylim(-0.85, 0.85)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(outpath, format="png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def _sign_position(p1, p2, offset=0.55, below=False):
    """Compute a point offset perpendicular to the edge from its midpoint.

    By default the offset is biased toward the +y side so signs sit "above"
    the edge. Set `below=True` to flip the offset to the -y side, which
    avoids collisions when another edge runs just above this one.
    """
    x1, y1 = p1
    x2, y2 = p2
    mx, my = (x1 + x2) / 2, (y1 + y2) / 2
    dx, dy = x2 - x1, y2 - y1
    length = (dx * dx + dy * dy) ** 0.5
    if length == 0:
        return (mx, my + (offset if not below else -offset))
    nx, ny = -dy / length, dx / length
    if ny < 0:
        nx, ny = -nx, -ny  # default: above the arrow
    if below:
        nx, ny = -nx, -ny  # flip: below the arrow
    return (mx + nx * offset, my + ny * offset)


def build_gt_pairs_single(outpath: Path):
    """Single portrait-friendly PNG of the 13-node, 10-edge pairs GT graph,
    with +/- signs labelling each edge (matching the prose)."""
    fig, ax = plt.subplots(figsize=(10.0, 5.6))

    # Draw nodes first so the edges can use them as patchA/patchB.
    patches: dict[str, FancyBboxPatch] = {}
    for node in NODES_13:
        if node not in POS_PAIRS:
            continue
        x, y = POS_PAIRS[node]
        patches[node] = _draw_node_box(
            ax, x, y, _wrap_label(node),
            w=2.0, h=1.0, fontsize=10,
        )

    # Draw edges + perpendicular-offset signs.
    for u, v in GT_PAIRS_EDGES:
        if u not in patches or v not in patches:
            continue
        p1, p2 = POS_PAIRS[u], POS_PAIRS[v]
        _draw_edge(
            ax, p1, p2, color=C_GT, lw=1.7, mutation_scale=18,
            patch_a=patches[u], patch_b=patches[v],
        )
        sign = GT_PAIRS_SIGNS.get((u, v))
        if sign:
            below = (u, v) in GT_PAIRS_SIGN_BELOW
            sx, sy = _sign_position(p1, p2, offset=0.55, below=below)
            ax.text(
                sx, sy, sign,
                ha="center", va="center",
                fontsize=14, fontweight="bold", color=C_GT,
                bbox=dict(
                    boxstyle="circle,pad=0.08",
                    facecolor="white", edgecolor="none",
                ),
                zorder=5,
            )

    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_aspect("equal")
    xs = [p[0] for p in POS_PAIRS.values()]
    ys = [p[1] for p in POS_PAIRS.values()]
    ax.set_xlim(min(xs) - 1.2, max(xs) + 1.2)
    ax.set_ylim(min(ys) - 0.8, max(ys) + 0.8)
    fig.tight_layout()
    fig.savefig(outpath, format="png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def _legend_handles():
    return [
        plt.Line2D([], [], color=C_TP, lw=2,
                   label=r"True positive (GT $\cap$ predicted)"),
        plt.Line2D([], [], color=C_FP, lw=2,
                   label="False positive (predicted only)"),
        plt.Line2D([], [], color=C_FN, lw=2, ls=(0, (4, 2)),
                   label="False negative (GT only)"),
    ]


def build_panel_grid_2x3(outpath: Path, nodes, gt_edges, pos,
                          edge_summary_path: Path,
                          model_columns: list[str],
                          model_titles: list[str],
                          gt_title: str,
                          figsize=(10.0, 14.0),
                          node_w=1.95, node_h=0.95, node_fs=8.0):
    """2-column x 3-row portrait grid: GT + up-to-5 model panels.

    Layout (row-major):
      [GT       ] [model 0]
      [model 1  ] [model 2]
      [model 3  ] [model 4]
    """
    rows = read_edge_summary(edge_summary_path)
    fig, axes = plt.subplots(3, 2, figsize=figsize)
    axes_flat = axes.flatten()

    # Panel 0: ground truth
    draw_panel(axes_flat[0], nodes, gt_edges, None, pos,
               title=gt_title,
               node_w=node_w, node_h=node_h, node_fs=node_fs,
               title_fs=11,
               label_fn=_wrap_label_compact,
               fs_overrides=GRID_FS_OVERRIDES)

    # Panels 1..5: models
    for i, (col, title) in enumerate(zip(model_columns, model_titles), start=1):
        if i >= len(axes_flat):
            break
        pred = edges_for_model(rows, col)
        gt_set = set(gt_edges); pr_set = set(pred)
        tp = len(gt_set & pr_set); fp = len(pr_set - gt_set); fn = len(gt_set - pr_set)
        subtitle = f"{title}\n(TP={tp}, FP={fp}, FN={fn})"
        draw_panel(axes_flat[i], nodes, gt_edges, pred, pos,
                   title=subtitle,
                   node_w=node_w, node_h=node_h, node_fs=node_fs,
                   title_fs=11,
                   label_fn=_wrap_label_compact,
                   fs_overrides=GRID_FS_OVERRIDES)

    fig.legend(handles=_legend_handles(),
               loc="lower center", ncol=3, frameon=False,
               fontsize=10, bbox_to_anchor=(0.5, 0.005))
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    fig.savefig(outpath, format="png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    here = Path(__file__).resolve().parent
    repo_root = here.parent
    default_outdir = (
        "/home/shepard/Desktop/Class Stuff/sem6/thesis/"
        "BachThesisDrafts/draft4/SashwatSharma_BachelorThesis"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaldata", default=str(repo_root / "EvaluationData"),
                        help="Path to EvaluationData/ directory")
    parser.add_argument("--outdir", default=default_outdir,
                        help="Directory to write the PNGs into")
    args = parser.parse_args()

    eval_dir = Path(args.evaldata)
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    pairs_summary = eval_dir / "Pairs" / "Matrices" / "edge_summary.csv"
    dag_summary = eval_dir / "DAG" / "Matrices" / "Compound" / "edge_summary.csv"

    print(f"Pairs summary: {pairs_summary}")
    print(f"DAG summary:   {dag_summary}")
    print(f"Output dir:    {out}")

    build_gt_dag_chainA(out / "gt_dag_chainA.png")
    build_gt_pairs_single(out / "gt_pairs_graph.png")

    build_panel_grid_2x3(
        out / "graphs_pairs.png",
        NODES_13, GT_PAIRS_EDGES, POS_PAIRS, pairs_summary,
        model_columns=MODEL_ORDER,
        model_titles=MODEL_ORDER,
        gt_title="Ground truth (10 edges)",
        figsize=(8.0, 10.5),
        node_w=2.25, node_h=0.95, node_fs=6.8,
    )
    build_panel_grid_2x3(
        out / "graphs_dag.png",
        NODES_12, GT_DAG_EDGES, POS_DAG, dag_summary,
        model_columns=MODEL_ORDER,
        model_titles=MODEL_ORDER,
        gt_title="Ground truth (3-edge chain)",
        figsize=(8.0, 10.5),
        node_w=2.3, node_h=0.95, node_fs=6.8,
    )
    print("Done.")


if __name__ == "__main__":
    main()
