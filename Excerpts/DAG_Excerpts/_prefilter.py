#!/usr/bin/env python3
"""Chain A pre-filter: finds ESG passages that contain accepted surface forms
from >=2 Chain A nodes AND an explicit causal marker. Dumps ranked candidates
to JSON for manual review under the full extraction prompt."""

import json
import re
import sys
from pathlib import Path

ROOT = Path("/home/shepard/Desktop/Class Stuff/sem6/thesis")
BANK_DIR = ROOT / "5050BankIndustryReports"
UNUSED_DIR = ROOT / "UnusedReports"
OUT_DIR = ROOT / "ChainA_Excerpts"

# Accepted surface forms per node (lowercased regex-friendly alternatives).
NODE_PATTERNS = {
    "CG": [
        r"corporate governance", r"\bgovernance mechanism", r"\bgovernance structure",
        r"board of directors", r"board composition", r"board characteristic",
        r"board independence", r"board size", r"board gender diversity",
        r"sustainability committee", r"environmental committee",
        r"climate committee", r"climate change committee", r"csr committee",
        r"governance committee", r"board oversight", r"board[- ]level (climate|esg|sustainability)",
        r"esg[- ]linked (executive )?compensation", r"sustainable compensation",
        r"pay[- ]for[- ]sustainability",
    ],
    "ER": [
        r"environmental reporting", r"environmental disclosure",
        r"carbon disclosure", r"ghg disclosure", r"climate disclosure",
        r"climate[- ]related disclosure", r"climate risk disclosure",
        r"voluntary carbon disclosure", r"carbon transparency",
        r"sustainability reporting", r"integrated reporting",
        r"\bcdp\b", r"cdp reporting", r"cdp participation", r"cdp disclosure",
        r"\btcfd\b", r"tcfd[- ]aligned", r"tcfd disclosure",
        r"esrs environmental", r"non[- ]financial report",
        r"science[- ]based target", r"\bsbti\b",
        r"climate policy disclosure", r"carbon management policy",
        r"emissions reduction target",
    ],
    "GHG": [
        r"ghg emission", r"greenhouse gas emission", r"carbon emission",
        r"co2 emission", r"co₂ emission", r"co2e emission", r"co₂e emission",
        r"carbon dioxide emission", r"scope 1", r"scope 2", r"scope 3",
        r"scope 1\s*\+\s*2", r"direct emission", r"indirect emission",
        r"carbon intensity", r"emissions intensity", r"carbon footprint",
        r"carbon performance",
    ],
    "MBV": [
        r"tobin'?s q", r"tobin q", r"\bq[- ]ratio\b", r"\bpbv\b",
        r"price[- ]to[- ]book", r"\bp/b\b", r"market[- ]to[- ]book",
        r"\bm/b\b", r"\bmbv\b",
        r"market capitali[sz]ation", r"market cap\b", r"market value",
        r"\bmve\b", r"equity market value", r"market value of equity",
        r"market value of common equity",
        r"firm value", r"firm valuation", r"corporate valuation",
        r"market valuation", r"market[- ]based firm value",
        r"valuation premium", r"valuation discount", r"valuation profile",
        r"shareholder value", r"investor confidence",
    ],
}

# Causal markers (verbs / connectives / mediators / impact framings).
CAUSAL_MARKERS = [
    r"\bcaused? by\b", r"\bcause[ds]?\b", r"\bcausing\b",
    r"\bdriv(e|es|en|ing)\b", r"\bproduce[ds]?\b", r"\bproducing\b",
    r"\benable[ds]?\b", r"\benabling\b",
    r"\bresult(s|ed|ing)? in\b", r"\blead(s|ing)? to\b", r"\bled to\b",
    r"\btranslate[ds]? into\b", r"\bcontribut(e|es|ed|ing) to\b",
    r"\breduce[ds]?\b", r"\breducing\b", r"\blower(s|ed|ing)?\b",
    r"\braise[ds]?\b", r"\braising\b", r"\bincrease[ds]?\b",
    r"\bimprove[ds]?\b", r"\bimproving\b",
    r"\bstrengthen(s|ed|ing)?\b", r"\bweaken(s|ed|ing)?\b",
    r"\bmitigate[ds]?\b", r"\bmitigating\b",
    r"\bsupport(s|ed|ing)?\b", r"\bunderpin(s|ned|ning)?\b",
    r"\bgenerate[ds]?\b", r"\byield(s|ed|ing)?\b",
    r"\bbecause\b", r"\bdue to\b", r"\bas a result of\b",
    r"\bconsequence of\b", r"\bthanks to\b", r"\bowing to\b",
    r"\bin response to\b", r"\breflected (in|by)\b",
    r"\bthrough\b", r"\bvia\b", r"\bby means of\b",
    r"\bdriven by\b", r"\bsupported by\b", r"\bunderpinned by\b",
    r"\benabled by\b",
    r"\bimpact of\b", r"\beffect of\b", r"\binfluences?\b",
    r"\bhelp(s|ed|ing)? (to )?\b",
]

NODE_REGEX = {
    node: re.compile("|".join(f"({p})" for p in patterns), re.IGNORECASE)
    for node, patterns in NODE_PATTERNS.items()
}
CAUSAL_REGEX = re.compile("|".join(f"({p})" for p in CAUSAL_MARKERS), re.IGNORECASE)


def nodes_in(text: str) -> set[str]:
    return {n for n, rx in NODE_REGEX.items() if rx.search(text)}


def causal_hits(text: str) -> list[str]:
    return list({m.group(0) for m in CAUSAL_REGEX.finditer(text)})


def scan_file(path: Path) -> list[dict]:
    candidates = []
    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
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
            nodes = nodes_in(text)
            markers = causal_hits(text)
            if len(nodes) >= 2 and markers:
                candidates.append({
                    "file": path.name,
                    "line": lineno,
                    "page": row.get("page"),
                    "section": row.get("section"),
                    "subsection": row.get("subsection"),
                    "nodes": sorted(nodes),
                    "n_nodes": len(nodes),
                    "causal_markers": markers,
                    "text": text,
                })
    return candidates


def main():
    all_candidates = []
    for directory in (BANK_DIR, UNUSED_DIR):
        for path in sorted(directory.glob("*.jsonl")):
            cands = scan_file(path)
            all_candidates.extend(cands)
            print(f"{path.name}: {len(cands)} candidates", file=sys.stderr)

    # Score: prefer more nodes covered, then passages with MBV (rarer).
    for c in all_candidates:
        c["score"] = c["n_nodes"] * 10 + (5 if "MBV" in c["nodes"] else 0)
    all_candidates.sort(key=lambda c: -c["score"])

    out_file = OUT_DIR / "_candidates.json"
    out_file.write_text(json.dumps(all_candidates, ensure_ascii=False, indent=2))
    print(f"\nWrote {len(all_candidates)} candidates to {out_file}", file=sys.stderr)

    # Quick breakdown.
    breakdown = {}
    for c in all_candidates:
        key = tuple(c["nodes"])
        breakdown[key] = breakdown.get(key, 0) + 1
    print("\nNode-combo breakdown:", file=sys.stderr)
    for key, n in sorted(breakdown.items(), key=lambda x: -x[1]):
        print(f"  {list(key)}: {n}", file=sys.stderr)


if __name__ == "__main__":
    main()
