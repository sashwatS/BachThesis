#!/usr/bin/env python3
"""Within-section proximity scan: find contiguous row-windows in a single
report/section where all 4 Chain A nodes (CG, ER, GHG, MBV) are covered
*and* causal markers connect them. Reuses NODE_PATTERNS / CAUSAL_MARKERS
from _prefilter."""

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _prefilter import NODE_REGEX, CAUSAL_REGEX, BANK_DIR, UNUSED_DIR, OUT_DIR


def load_rows(path: Path):
    rows = []
    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = row.get("text", "")
            if not text or len(text) < 40:
                continue
            row["_lineno"] = lineno
            row["_nodes"] = {n for n, rx in NODE_REGEX.items() if rx.search(text)}
            row["_markers"] = [m.group(0) for m in CAUSAL_REGEX.finditer(text)]
            rows.append(row)
    return rows


def scan_windows(rows, window_rows=20, max_page_span=6, require_section=False):
    """Return windows that collectively cover all 4 nodes and keep within
    ~max_page_span pages, bridged by causal markers. Gaps allowed."""
    hits = []
    n = len(rows)
    for i in range(n):
        # Anchor: row i must itself contain at least one node (otherwise
        # the window is just starting at filler).
        if not rows[i]["_nodes"]:
            continue
        for size in range(2, window_rows + 1):
            j = i + size
            if j > n:
                break
            window = rows[i:j]
            if require_section:
                sections = {r.get("section") for r in window}
                if len(sections) > 1:
                    continue
            pages = [r.get("page") for r in window if r.get("page") is not None]
            if pages and (max(pages) - min(pages)) > max_page_span:
                continue
            nodes_union = set().union(*(r["_nodes"] for r in window))
            if nodes_union != {"CG", "ER", "GHG", "MBV"}:
                continue
            # Last row must also contain a node (tight boundary).
            if not window[-1]["_nodes"]:
                continue
            # At least two rows must carry causal markers so the chain is
            # bridged, not just co-mentioned.
            rows_with_markers = sum(1 for r in window if r["_markers"])
            if rows_with_markers < 2:
                continue
            hits.append({
                "file": window[0].get("_file", ""),
                "start_line": window[0]["_lineno"],
                "end_line": window[-1]["_lineno"],
                "n_rows": size,
                "page_range": [min(pages), max(pages)] if pages else None,
                "section": window[0].get("section"),
                "subsection_set": sorted({r.get("subsection") for r in window if r.get("subsection")}),
                "nodes_per_row": [sorted(r["_nodes"]) for r in window],
                "markers_per_row": [r["_markers"] for r in window],
                "texts": [r.get("text", "") for r in window],
                "rows_with_markers": rows_with_markers,
            })
            break  # smallest qualifying window per start index is enough
    return hits


def main():
    all_hits = []
    for directory in (BANK_DIR, UNUSED_DIR):
        for path in sorted(directory.glob("*.jsonl")):
            rows = load_rows(path)
            for r in rows:
                r["_file"] = path.name
            hits = scan_windows(rows, window_rows=20, max_page_span=6, require_section=False)
            for h in hits:
                h["_tier"] = "Bank" if directory == BANK_DIR else "Unused"
            all_hits.extend(hits)
            print(f"{path.name}: {len(hits)} 4-node windows", file=sys.stderr)

    # Rank: smaller windows first, then more causal-marker rows.
    all_hits.sort(key=lambda h: (h["n_rows"], -h["rows_with_markers"]))

    out_file = OUT_DIR / "_fullchain_windows.json"
    out_file.write_text(json.dumps(all_hits, ensure_ascii=False, indent=2))
    print(f"\nWrote {len(all_hits)} 4-node windows to {out_file}", file=sys.stderr)


if __name__ == "__main__":
    main()
