#!/usr/bin/env python3
"""
pdfCleaner.py

ESG Report PDF -> cleaned, structured, ESG-tagged dataset (JSONL and/or CSV).

Layout-aware extraction for multi-column ESG/sustainability reports using
pdfplumber for text extraction and spaCy for sentence segmentation.

The output prioritizes READABILITY: every record should read like natural text
(subject, verb, object), be semantically meaningful, grouped by ESG topic,
and ready for analysis or scoring by humans and LLMs alike.

Install:
  pip install pdfplumber spacy
  python -m spacy download en_core_web_sm

Optional OCR support (for scanned PDFs):
  pip install pillow pytesseract
  # plus install the Tesseract binary on your system

Run:
  python pdfCleaner.py /path/to/report.pdf --out output.jsonl
  python pdfCleaner.py /path/to/report.pdf --out output.jsonl --csv output.csv
  python pdfCleaner.py /path/to/report.pdf --out output.jsonl --level paragraph
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import unicodedata
from dataclasses import dataclass
from typing import Dict, List, Optional

import pdfplumber
import spacy


# ---------------------------------------------------------------------------
# ESG section detection
# ---------------------------------------------------------------------------

# ESRS section heading patterns — only match at the start of a paragraph
ESRS_SECTION_PATTERNS = [
    (r"^(?:ESRS\s+2|General\s+[Dd]isclosures)", "ESRS2", "General", "General Disclosures"),
    (r"^(?:ESRS\s+)?E1[\s\-–].*(?:Climate|climate)", "E1", "E", "Climate Change"),
    (r"^(?:ESRS\s+)?E2[\s\-–].*(?:Pollution|pollution)", "E2", "E", "Pollution"),
    (r"^(?:ESRS\s+)?E3[\s\-–].*(?:Water|water)", "E3", "E", "Water and Marine Resources"),
    (r"^(?:ESRS\s+)?E4[\s\-–].*(?:Biodiversity|biodiversity)", "E4", "E", "Biodiversity and Ecosystems"),
    (r"^(?:ESRS\s+)?E5[\s\-–].*(?:Resource|resource|Circular|circular)", "E5", "E", "Resource Use and Circular Economy"),
    (r"^(?:ESRS\s+)?S1[\s\-–].*(?:Own\s+[Ww]orkforce|workforce)", "S1", "S", "Own Workforce"),
    (r"^(?:ESRS\s+)?S2[\s\-–].*(?:Workers|workers|Value\s+Chain|value\s+chain)", "S2", "S", "Workers in the Value Chain"),
    (r"^(?:ESRS\s+)?S3[\s\-–].*(?:Affected|affected|Communities|communities)", "S3", "S", "Affected Communities"),
    (r"^(?:ESRS\s+)?S4[\s\-–].*(?:Consumers|consumers|End[\s-]Users|end[\s-]users)", "S4", "S", "Consumers and End Users"),
    (r"^(?:ESRS\s+)?G1[\s\-–].*(?:Business|business|Conduct|conduct)", "G1", "G", "Business Conduct"),
    (r"^EU\s+Taxonomy\b", "EU-Tax", "General", "EU Taxonomy"),
]

# Sub-section heading patterns (GOV-1, SBM-1, E1-1, S1-2, IRO-1, BP-1, etc.)
ESRS_SUBSECTION_RE = re.compile(
    r"((?:GOV|SBM|IRO|BP|MDR|E[1-5]|S[1-4]|G1)-\d+)"
    r"\s*[-–—]\s*"
    r"([^.]+)",  # Capture only up to the first period = short title
    re.IGNORECASE,
)

# Map subsection prefixes to their parent ESRS section
SUBSECTION_PARENT_MAP = {
    "GOV": ("ESRS2", "General", "General Disclosures"),
    "SBM": ("ESRS2", "General", "General Disclosures"),
    "IRO": ("ESRS2", "General", "General Disclosures"),
    "BP":  ("ESRS2", "General", "General Disclosures"),
    "MDR": ("ESRS2", "General", "General Disclosures"),
    "E1": ("E1", "E", "Climate Change"),
    "E2": ("E2", "E", "Pollution"),
    "E3": ("E3", "E", "Water and Marine Resources"),
    "E4": ("E4", "E", "Biodiversity and Ecosystems"),
    "E5": ("E5", "E", "Resource Use and Circular Economy"),
    "S1": ("S1", "S", "Own Workforce"),
    "S2": ("S2", "S", "Workers in the Value Chain"),
    "S3": ("S3", "S", "Affected Communities"),
    "S4": ("S4", "S", "Consumers and End Users"),
    "G1": ("G1", "G", "Business Conduct"),
}


def detect_esrs_section(text: str) -> Optional[tuple]:
    """Return (section_code, category, section_name) if text is a section heading."""
    head = text.strip()[:100]
    for pattern, code, category, name in ESRS_SECTION_PATTERNS:
        if re.search(pattern, head, re.IGNORECASE):
            return (code, category, name)
    return None


def detect_esrs_subsection(text: str) -> Optional[tuple]:
    """Return (subsection_code, short_title) if text contains a sub-section heading."""
    m = ESRS_SUBSECTION_RE.search(text.strip()[:200])
    if m:
        code = m.group(1).upper()
        # Take only the first clause as the title (before any period or long text)
        raw_title = m.group(2).strip()
        # Truncate at 80 chars max for a clean title
        if len(raw_title) > 80:
            raw_title = raw_title[:80].rsplit(" ", 1)[0]
        return (code, raw_title)
    return None


# ---------------------------------------------------------------------------
# Content filters — aggressive cleaning for readable output
# ---------------------------------------------------------------------------

# Repeated navigation header pattern
NAV_HEADER_RE = re.compile(
    r"(?:TO\s+OUR\s+SHAREHOLDERS|GROUP\s+MANAGEMENT\s+REPORT|"
    r"FINANCIAL\s+REVIEW|SUSTAINABILITY\s+STATEMENT|"
    r"CONSOLIDATED\s+FINANCIAL\s+STATEMENTS|ADDITIONAL\s+INFORMATION)",
    re.IGNORECASE,
)

# Cross-reference patterns to strip
CROSS_REF_RE = re.compile(
    r"►\s*SEE\s+[A-Z][A-Z\s\-–&,/'.]+|"
    r"►\s*[A-Z][A-Z\-]+\.[A-Z]+[A-Z/\-\.]*|"
    r"►\s*SEE\s+(?:NOTE|ESRS)\s+\S+|"
    r"►\s+[A-Z][A-Z\s]+",
    re.IGNORECASE,
)

PAGE_NUM_RE = re.compile(r"^\s*\d{1,4}\s*$")


def is_allcaps_block(text: str) -> bool:
    """Check if text is an all-uppercase block (navigation or heading)."""
    stripped = text.strip()
    if len(stripped) < 20:
        return False
    alpha = [c for c in stripped if c.isalpha()]
    if not alpha:
        return False
    return sum(1 for c in alpha if c.isupper()) / len(alpha) > 0.80


def is_toc_or_index_page(text: str) -> bool:
    """Detect table-of-contents or ESRS Index pages."""
    lines = text.strip().splitlines()
    if not lines:
        return True

    toc_pattern = re.compile(r".{5,}\s+\d{1,4}\s*$")
    toc_count = sum(1 for ln in lines if toc_pattern.match(ln.strip()))
    if toc_count > 5 and toc_count / max(len(lines), 1) > 0.3:
        return True

    esrs_code_count = len(re.findall(r"(?:►|ESRS\s+[A-Z]\d|[ESGB][1-5][\-,])", text))
    if esrs_code_count > 10:
        return True

    if re.search(r"ESRS\s+Index", text):
        return True

    # Infographic/diagram pages: lots of short fragments, very few complete sentences
    sentences_with_period = re.findall(r"[A-Za-z]{10,}[^.]*\.\s", text)
    if len(text) > 500 and len(sentences_with_period) < 3:
        # Page has lots of text but almost no complete sentences — likely a diagram
        return True

    heading_with_num = re.compile(r"ESRS\s+[A-Z]\d.*\d{2,3}")
    if sum(1 for ln in lines if heading_with_num.search(ln.strip())) >= 5:
        return True

    return False


def looks_like_nav_header(text: str) -> bool:
    """Check if text looks like a repeated navigation header."""
    return len(NAV_HEADER_RE.findall(text)) >= 2


def is_scrambled_table_text(text: str) -> bool:
    """Detect text that was garbled by multi-column or table extraction.

    Symptoms: many short fragments without proper sentence structure,
    excessive dashes, column-like layout artifacts, or text that jumps
    between unrelated topics mid-sentence.
    """
    stripped = text.strip()
    if not stripped:
        return True

    words = stripped.split()
    if not words:
        return True

    # High ratio of very short "words" (1-2 chars) suggests table columns
    short_words = sum(1 for w in words if len(w) <= 2)
    if len(words) > 10 and short_words / len(words) > 0.4:
        return True

    # Many dashes in a short span — typical of multi-column scrambling
    dash_count = stripped.count("–") + stripped.count("—") + stripped.count("─")
    if dash_count > 5 and len(stripped) < 500:
        return True

    # Table-like: many lines with just a word or two each when re-split
    pieces = re.split(r"\s{3,}", stripped)
    if len(pieces) > 6 and sum(1 for p in pieces if len(p.split()) <= 3) > len(pieces) * 0.5:
        return True

    # Multi-column merge: text contains multiple ESRS section names jumbled together
    # e.g., "General Disclosures Climate change Own workforce Business Conduct"
    esrs_topic_count = len(re.findall(
        r"(?:General Disclosures|Climate [Cc]hange|Own [Ww]orkforce|Business [Cc]onduct|"
        r"Pollution|Water and [Mm]arine|Biodiversity|Resource [Uu]se|Affected [Cc]ommunities|"
        r"Consumers and [Ee]nd|Workers in the [Vv]alue)",
        stripped,
    ))
    if esrs_topic_count >= 3:
        return True

    # Infographic / process diagram text: many short capitalized phrases
    # with no connecting verbs
    if len(stripped) > 200:
        sentences_approx = re.split(r"[.!?]\s+", stripped)
        avg_sentence_len = sum(len(s) for s in sentences_approx) / max(len(sentences_approx), 1)
        if avg_sentence_len < 30 and len(sentences_approx) > 5:
            return True

    # Process diagram text: many action words without sentence structure
    if len(stripped) > 150:
        cap_words_mid = re.findall(r"(?<!\. )[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?(?=\s+[A-Z])", stripped)
        if len(cap_words_mid) > 8:
            return True

    # Policy/IRO overview tables: contain structured metadata columns
    # e.g., "Senior level Third-party standards/ Stakeholder Policies Content Scope"
    policy_table_markers = re.findall(
        r"\bSenior\s+level\b|\bThird-party\s+standards\b|\bStakeholder\b.*\bPolicies\b|"
        r"\bContent\s+Scope\b|\bAvailabil(?:ity|e)\b.*\bAccessible\b|"
        r"\bExecutive\s+(?:Board|Company)\b.*\bAgreement\b",
        stripped, re.IGNORECASE,
    )
    if len(policy_table_markers) >= 2:
        return True

    # IRO summary tables: "Impacts Risks — ... — ..."
    if re.match(r"Impacts,?\s+risks\s+and\s+opportunities\s+Impacts", stripped, re.IGNORECASE):
        return True

    # Text containing "Compliance management system" followed by table-like fragments
    if re.search(r"(?:Own\s+Operations|Executive\s+Company|Accessible\s+via)", stripped) and \
       re.search(r"(?:Agreement|understanding|applies\s+to\s+all)", stripped):
        return True

    return False


def is_name_or_committee_list(text: str) -> bool:
    """Detect paragraphs that are primarily lists of names/roles (non-narrative)."""
    stripped = text.strip()

    # Count bullet-prefixed name entries: "− Name – Role" pattern
    name_entries = re.findall(
        r"[−–—\-]\s+[A-Z][a-zà-ö]+(?:\s+[A-Z][a-zà-ö]+)+\s*(?:[–—\-]\s|$|\()",
        stripped,
    )
    if len(name_entries) >= 3:
        # But only reject if the name entries make up most of the text
        # (some paragraphs have a list followed by narrative)
        name_text_len = sum(len(m) for m in name_entries)
        if name_text_len > 0.5 * len(stripped):
            return True

    # Committee lists: "Committee: Name, Name, Name" repeated
    committee_entries = re.findall(r"Committee[:\s]", stripped, re.IGNORECASE)
    if len(committee_entries) >= 3:
        return True

    return False


def is_non_narrative(text: str) -> bool:
    """Check if text is non-narrative (tables, pure references, numbers, etc.)."""
    stripped = text.strip()
    if not stripped:
        return True

    # Standalone bullet fragment
    if re.fullmatch(r"[►\-–•]\s*.*", stripped) and len(stripped) < 60:
        return True

    # Mostly numbers/symbols (less than 40% alphabetic)
    alpha = sum(1 for c in stripped if c.isalpha())
    if alpha < 0.4 * len(stripped) and len(stripped) > 20:
        return True

    # Heavy on percentages/numbers — chart or KPI table, not narrative
    pct_count = len(re.findall(r"~?\d+\.?\d*%", stripped))
    num_count = len(re.findall(r"\b\d+\.?\d*\b", stripped))
    if pct_count >= 4:
        return True
    if num_count >= 8 and len(stripped) < 300:
        return True

    # Very short and no verb-like structure
    if len(stripped) < 40 and not re.search(r"\b(?:is|are|was|were|has|have|had|will|shall|can|do|does|did)\b", stripped, re.IGNORECASE):
        if not re.match(r"^[A-Z]", stripped):
            return True

    return False


def is_garbled_fragment(text: str) -> bool:
    """Detect garbled fragments from broken table extraction.

    Examples: "Impact stream", "Negative Up- stream", "n.a.", "use change",
    or longer text that reads like jumbled table cells.
    """
    stripped = text.strip()
    if not stripped:
        return True

    words = stripped.split()

    # Very short with no sentence structure
    if len(words) <= 5 and not re.search(r"[.!?]$", stripped):
        if not re.search(r"\b(?:is|are|was|were|has|have|had|will|shall|can|do|does|did|make|take|set|use|include|apply|ensure|manage|lead|work|provide|report|cover|focus|address)\b", stripped, re.IGNORECASE):
            return True

    # Starts with lowercase and is short — likely a continuation fragment
    if stripped[0].islower() and len(words) <= 8:
        return True

    # Contains table-cell markers: "n.a.", "Actual", "Negative Up-", "Impact stream"
    table_markers = re.findall(
        r"\bn\.a\.\b|"
        r"\bActual\b(?=\s+n\.a\.)|"
        r"(?:Negative|Positive)\s+(?:Up|Down)|"
        r"\bImpact\s+stream\b|"
        r"(?:Short|Medium|Long)-\s*(?:term\s+)?(?:Up|Down)?stream|"
        r"\bRisk\s+n\.a\.|"
        r"\bOpportunit(?:y|ies)\s+n\.a\.",
        stripped, re.IGNORECASE,
    )
    if len(table_markers) >= 1:
        return True

    # "Impact stream" at start — dead giveaway for table row continuation
    if re.match(r"Impact\s+stream\b", stripped, re.IGNORECASE):
        return True

    # Broken hyphenation across table cells
    if re.search(r"Up-\s+\w|Down-\s+\w|Long-\s+Up|Short-\s+Up", stripped):
        if len(stripped) < 300:
            return True

    # Text with SVP/SOP abbreviations followed by jumbled words (policy table)
    if re.search(r"\bSVP\b.*\bSOP\b|\bSOP\b.*\bSVP\b", stripped):
        return True

    # "Upfresh water-" or similar compound garble from table cells
    if re.search(r"Up(?:fresh|stream|term)|Down(?:stream|term)", stripped) and "n.a." in stripped:
        return True

    # IRO table classification columns leaking through
    if re.search(r"(?:Classifi-|cation|horizon)\s+(?:level|chain)", stripped, re.IGNORECASE):
        return True

    # "dependen-" "Negative" pattern from split table cells
    if re.search(r"dependen-\s*(?:Negative|cies)", stripped, re.IGNORECASE):
        return True

    return False


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """Normalize and clean extracted text, joining wrapped lines into paragraphs."""
    if not text:
        return ""

    # Unicode normalize
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u00A0", " ")

    # Normalize newlines
    text = text.replace("\r", "\n")

    # Remove control chars except newlines/tabs
    text = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F]", "", text)

    # De-hyphenate line breaks: sustain-\nability -> sustainability
    text = re.sub(r"(?<=\w)-\n(?=\w)", "", text)

    # Strip cross-references
    text = CROSS_REF_RE.sub("", text)

    # Clean dangling artifacts from cross-ref removal
    text = re.sub(r":\s*[:,;.]\s*", ". ", text)
    text = re.sub(r"(?:can be\s+)?found (?:here|under this link)[.:]*\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"More detailed information (?:on .{0,60})?(?:can be found|is outlined).*?[.:]\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"please refer to\s*[.:]*\s*$", "", text, flags=re.IGNORECASE | re.MULTILINE)
    # Remove orphaned possessives left after cross-ref removal ("here: 's strategy")
    text = re.sub(r"\s+'s\s+", " ", text)

    # Remove footnote superscript numbers — only after sentence-ending punctuation
    # e.g., "activities.8 Additionally" -> "activities. Additionally"
    # but NOT "2024" or year numbers
    text = re.sub(r"(?<=[.!?])(\d{1,2})(?=\s+[A-Z])", "", text)

    # Remove standalone footnote lines ("15 Source : UN https://...")
    text = re.sub(r"^\d{1,2}\s+Source\s*:.*$", "", text, flags=re.MULTILINE | re.IGNORECASE)

    # Remove URL references
    text = re.sub(r"https?://\S+", "", text)

    # Remove standalone page numbers
    lines = text.split("\n")
    lines = [ln for ln in lines if not PAGE_NUM_RE.match(ln)]

    # Remove navigation header lines
    lines = [ln for ln in lines if not looks_like_nav_header(ln)]

    # Remove all-caps header/navigation lines
    lines = [ln for ln in lines if not is_allcaps_block(ln)]

    # Remove report title lines
    lines = [ln for ln in lines if not re.fullmatch(r"\s*ANNUAL\s+REPORT\s+\d{4}\s*", ln, re.IGNORECASE)]

    # Remove navigation column number sequences ("1 2 3 4 5 6")
    lines = [ln for ln in lines if not re.fullmatch(r"\s*(?:\d\s+){2,}\d?\s*", ln)]
    lines = [ln for ln in lines if not re.fullmatch(r"[\s\d]+", ln.strip()) or len(ln.strip()) < 2]

    text = "\n".join(lines)

    # Whitespace cleanup
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" *\n *", "\n", text)

    # Smart paragraph detection:
    # pdfplumber single \n can be line wrap OR paragraph break.
    # Heuristic: if a line ends with sentence-ending punctuation
    # and the next line starts with a capital letter, treat as paragraph break.
    lines = text.split("\n")
    paragraphs = []
    current = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            if current:
                paragraphs.append(" ".join(current))
                current = []
            continue

        current.append(stripped)

        is_last = (i == len(lines) - 1)
        if not is_last:
            next_line = lines[i + 1].strip()
            ends_sentence = bool(re.search(r'[.!?:;]\s*$', stripped)) or \
                            bool(re.search(r'[.!?]\s*$', stripped))
            next_starts_para = bool(re.match(r'[A-Z"(]', next_line)) or \
                               bool(re.match(r'(?:─|−|–|—|\-)\s', next_line)) or \
                               bool(ESRS_SUBSECTION_RE.match(next_line))
            if ends_sentence and next_starts_para:
                paragraphs.append(" ".join(current))
                current = []

    if current:
        paragraphs.append(" ".join(current))

    text = "\n\n".join(paragraphs)

    # Final cleanup
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\n+", "\n\n", text)

    return text.strip()


# ---------------------------------------------------------------------------
# Extraction using pdfplumber
# ---------------------------------------------------------------------------

@dataclass
class PageText:
    page: int
    text: str
    method: str  # "text" or "ocr"


def extract_pages_pdfplumber(pdf_path: str, *, ocr: bool = False,
                              ocr_lang: str = "eng",
                              min_chars_per_page: int = 80) -> List[PageText]:
    """Extract per-page text using pdfplumber."""
    pages: List[PageText] = []

    with pdfplumber.open(pdf_path) as pdf:
        total = len(pdf.pages)
        print(f"[1/3] Extracting text from {total} pages...", flush=True)
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            method = "text"

            if ocr and len(text.strip()) < min_chars_per_page:
                try:
                    import pytesseract
                    from PIL import Image as PILImage
                    img = page.to_image(resolution=300).original
                    text = pytesseract.image_to_string(img, lang=ocr_lang)
                    method = "ocr"
                except ImportError:
                    pass

            pages.append(PageText(page=i + 1, text=text, method=method))
            if (i + 1) % 50 == 0 or (i + 1) == total:
                print(f"    ... {i + 1}/{total} pages done", flush=True)

    return pages


# ---------------------------------------------------------------------------
# Paragraph extraction and sentence splitting
# ---------------------------------------------------------------------------

@dataclass
class Paragraph:
    text: str
    page: int
    esrs_section: Optional[str] = None
    esrs_category: Optional[str] = None
    esrs_section_name: Optional[str] = None
    esrs_subsection: Optional[str] = None
    esrs_subsection_title: Optional[str] = None


def extract_paragraphs(pages: List[PageText]) -> List[Paragraph]:
    """Extract clean paragraphs from pages, filtering non-narrative content."""
    paragraphs: List[Paragraph] = []

    # Default to ESRS2/General since the first narrative pages are General Disclosures
    # (the TOC and infographic pages that contain the heading get filtered out)
    current_section = "ESRS2"
    current_category = "General"
    current_section_name = "General Disclosures"
    current_subsection = None
    current_subsection_title = None

    for p in pages:
        raw = p.text or ""

        if len(raw.strip()) < 50:
            continue

        if is_toc_or_index_page(raw):
            continue

        cleaned = clean_text(raw)
        if not cleaned:
            continue

        raw_paragraphs = re.split(r"\n\n+", cleaned)

        for para_text in raw_paragraphs:
            para_text = para_text.strip()

            if not para_text or len(para_text) < 30:
                continue

            # --- Content quality filters ---

            # Skip non-narrative fragments (mostly numbers/symbols)
            if is_non_narrative(para_text):
                continue

            # Skip scrambled multi-column / table text
            if is_scrambled_table_text(para_text):
                continue

            # Skip name/committee lists
            if is_name_or_committee_list(para_text):
                continue

            # Skip garbled short fragments from broken tables
            if is_garbled_fragment(para_text):
                continue

            # --- ESG section tracking ---

            # Check for sub-section heading first
            subsection_info = detect_esrs_subsection(para_text)
            if subsection_info:
                current_subsection, current_subsection_title = subsection_info
                prefix = re.match(r"([A-Z]+\d?)", current_subsection)
                if prefix and prefix.group(1) in SUBSECTION_PARENT_MAP:
                    current_section, current_category, current_section_name = \
                        SUBSECTION_PARENT_MAP[prefix.group(1)]

            # Check for top-level ESRS section heading
            section_info = detect_esrs_section(para_text)
            if section_info:
                current_section, current_category, current_section_name = section_info

            paragraphs.append(Paragraph(
                text=para_text,
                page=p.page,
                esrs_section=current_section,
                esrs_category=current_category,
                esrs_section_name=current_section_name,
                esrs_subsection=current_subsection,
                esrs_subsection_title=current_subsection_title,
            ))

    return paragraphs


def split_sentences_spacy(nlp, text: str) -> List[str]:
    """Use spaCy for robust sentence segmentation."""
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents if sent.text.strip()]


def is_good_sentence(sent: str, *, min_chars: int = 20, max_chars: int = 2000) -> bool:
    """Filter out bad sentences: too short, too long, or non-narrative."""
    stripped = sent.strip()

    if len(stripped) < min_chars:
        return False

    if len(stripped) > max_chars:
        return False

    # Must be at least 50% alphabetic
    alpha_count = sum(1 for c in stripped if c.isalpha())
    if alpha_count < 0.5 * len(stripped):
        return False

    # Reject pure bullets/references
    if stripped.startswith("►") or stripped.startswith("•"):
        return False

    # Reject garbled fragments
    if is_garbled_fragment(stripped):
        return False

    return True


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def _build_record(text: str, para: "Paragraph", para_id: int, *,
                   sent_id: int | None = None, compact: bool = True) -> Dict:
    """Build an output record. In compact mode, text comes first and nulls are omitted."""
    if compact:
        rec: Dict = {"text": text}
        if para.esrs_section:
            rec["section"] = para.esrs_section
        if para.esrs_category:
            rec["category"] = para.esrs_category
        if para.esrs_subsection:
            rec["subsection"] = para.esrs_subsection
        rec["page"] = para.page
        return rec
    else:
        rec = {
            "text": text,
            "page": para.page,
            "para_id": para_id,
            "esrs_section": para.esrs_section,
            "esrs_category": para.esrs_category,
            "esrs_section_name": para.esrs_section_name,
        }
        if sent_id is not None:
            rec["sent_id"] = sent_id
        if para.esrs_subsection:
            rec["esrs_subsection"] = para.esrs_subsection
            rec["esrs_subsection_title"] = para.esrs_subsection_title
        return rec


def pdf_to_records(
    pdf_path: str,
    *,
    doc_id: Optional[str] = None,
    ocr: bool = False,
    ocr_lang: str = "eng",
    min_chars_per_page: int = 80,
    min_sentence_chars: int = 20,
    output_level: str = "both",
    compact: bool = True,
) -> List[Dict]:
    """Main pipeline: PDF -> structured records."""

    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        print("Downloading spaCy model 'en_core_web_sm'...")
        spacy.cli.download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")

    nlp.max_length = 2_000_000

    base = os.path.splitext(os.path.basename(pdf_path))[0]
    doc_id = doc_id or base

    # Step 1: Extract pages
    pages = extract_pages_pdfplumber(
        pdf_path, ocr=ocr, ocr_lang=ocr_lang,
        min_chars_per_page=min_chars_per_page,
    )

    # Step 2: Extract clean paragraphs with ESG tagging
    print("[2/3] Cleaning text and detecting ESG sections...", flush=True)
    paragraphs = extract_paragraphs(pages)
    print(f"    ... {len(paragraphs)} paragraphs extracted", flush=True)

    # Step 3: Build records
    print("[3/3] Running spaCy sentence segmentation...", flush=True)
    records: List[Dict] = []
    para_id = 0
    sent_id = 0

    for para in paragraphs:
        para_id += 1

        if output_level in ("paragraph", "both"):
            rec = _build_record(para.text, para, para_id, compact=compact)
            records.append(rec)

        if output_level in ("sentence", "both"):
            sentences = split_sentences_spacy(nlp, para.text)
            for sent_text in sentences:
                if not is_good_sentence(sent_text, min_chars=min_sentence_chars):
                    continue
                sent_id += 1
                rec = _build_record(sent_text, para, para_id, sent_id=sent_id, compact=compact)
                records.append(rec)

    return records


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def write_jsonl(records: List[Dict], out_path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_csv(records: List[Dict], out_path: str) -> None:
    if not records:
        return
    # Collect all unique keys in insertion order across records
    fieldnames: List[str] = []
    seen: set = set()
    for r in records:
        for k in r:
            if k not in seen:
                fieldnames.append(k)
                seen.add(k)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in records:
            w.writerow(r)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="ESG Report PDF -> cleaned, structured, ESG-tagged dataset (JSONL/CSV)."
    )
    ap.add_argument("pdf", help="Path to input PDF")
    ap.add_argument("--out", required=True, help="Path to output JSONL file")
    ap.add_argument("--csv", default=None, help="Optional path to output CSV file")
    ap.add_argument("--doc-id", default=None, help="Optional doc_id override (default: filename stem)")

    ap.add_argument("--ocr", action="store_true",
                    help="Use OCR for pages with too little extracted text")
    ap.add_argument("--ocr-lang", default="eng", help="Tesseract language (default: eng)")
    ap.add_argument("--min-chars-per-page", type=int, default=80,
                    help="OCR pages if extracted text < this many chars")

    ap.add_argument("--min-sentence-chars", type=int, default=20,
                    help="Drop sentence fragments shorter than this (default: 20)")
    ap.add_argument("--level", choices=["sentence", "paragraph", "both"], default="both",
                    help="Output level: sentence, paragraph, or both (default: both)")
    ap.add_argument("--verbose-fields", action="store_true",
                    help="Include all metadata fields (doc_id, source_file, level, etc.). "
                         "Default is compact mode with only text, section, category, page.")

    args = ap.parse_args()

    if not os.path.exists(args.pdf):
        raise FileNotFoundError(f"Input PDF not found: {args.pdf}")

    records = pdf_to_records(
        args.pdf,
        doc_id=args.doc_id,
        ocr=args.ocr,
        ocr_lang=args.ocr_lang,
        min_chars_per_page=args.min_chars_per_page,
        min_sentence_chars=args.min_sentence_chars,
        output_level=args.level,
        compact=not args.verbose_fields,
    )

    write_jsonl(records, args.out)
    if args.csv:
        write_csv(records, args.csv)

    print(f"Done. {len(records)} records -> {args.out}")
    if args.csv:
        print(f"CSV -> {args.csv}")


if __name__ == "__main__":
    main()
