#!/usr/bin/env python3
"""MBV-anchored scan: for each row that mentions MBV, look at a +/-10 row
neighbourhood and assess whether the full Chain A can be assembled."""

import json
import sys
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


def find_mbv_neighbourhoods(rows, window=10, min_markers=3):
    hits = []
    for idx, r in enumerate(rows):
        if "MBV" not in r["_nodes"]:
            continue
        lo = max(0, idx - window)
        hi = min(len(rows), idx + window + 1)
        nbhd = rows[lo:hi]
        nodes_union = set().union(*(x["_nodes"] for x in nbhd))
        if nodes_union != {"CG", "ER", "GHG", "MBV"}:
            continue
        # Require each edge to have a marker somewhere in the nbhd.
        total_markers = sum(len(x["_markers"]) for x in nbhd)
        if total_markers < min_markers:
            continue
        # Pages should be relatively tight
        pages = [x.get("page") for x in nbhd if x.get("page") is not None]
        if pages and (max(pages) - min(pages)) > 6:
            continue
        hits.append({
            "anchor_line": r["_lineno"],
            "anchor_page": r.get("page"),
            "anchor_text": r.get("text", "")[:300],
            "window_span": [nbhd[0]["_lineno"], nbhd[-1]["_lineno"]],
            "page_range": [min(pages), max(pages)] if pages else None,
            "total_markers": total_markers,
            "nodes_in_nbhd": sorted(nodes_union),
            "section": r.get("section"),
        })
    return hits


def main():
    results = {}
    for directory in (BANK_DIR, UNUSED_DIR):
        for path in sorted(directory.glob("*.jsonl")):
            rows = load_rows(path)
            hits = find_mbv_neighbourhoods(rows, window=15, min_markers=2)
            if hits:
                results[path.name] = hits
                print(f"{path.name}: {len(hits)} MBV-anchored hits", file=sys.stderr)

    out_file = OUT_DIR / "_mbv_anchored.json"
    out_file.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    total = sum(len(v) for v in results.values())
    print(f"\nTotal MBV-anchored nbhds: {total} in {len(results)} reports", file=sys.stderr)


if __name__ == "__main__":
    main()
