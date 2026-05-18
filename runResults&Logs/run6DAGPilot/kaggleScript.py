import json
import re
import time
import logging
from pathlib import Path
from datetime import datetime
import requests

# ============================================================
# CONFIG
# ============================================================

LIMIT = 1  # 1 for pilot, None for full run

OLLAMA_URL = "http://localhost:11434/api/generate"

# Model variants. "qwen3.5:9b" and "qwen3.5:9b_instruct" are two variants of
# the same underlying Ollama model that differ only in whether thinking is
# enabled. OLLAMA_MODEL_FOR_VARIANT maps the synthetic variant ID used for
# bookkeeping to the actual Ollama tag.
MODELS = [
    "llama3.1:8b",
    "qwen3.5:9b",
    "qwen3.5:9b_instruct",
    "gemma4:e4b",
    "deepseek-r1:8b",
]

SOFT_GATED_MODELS = {"deepseek-r1:8b", "gemma4:e4b", "qwen3.5:9b"}
INSTRUCT_MODELS = {"qwen3.5:9b_instruct"}

OLLAMA_MODEL_FOR_VARIANT = {
    "qwen3.5:9b_instruct": "qwen3.5:9b",
}

# Kaggle paths
EXCERPTS_PATH = Path("/kaggle/input/inputdocuments/excerpts.json")
ONTOLOGY_PATH = Path("/kaggle/input/inputdocuments/ontology.txt")
RESULTS_DIR = Path("/kaggle/working/results")
LOGS_DIR = Path("/kaggle/working/logs")

DEFAULT_OPTIONS = {
    "temperature": 0,
    "top_p": 1.0,
    "seed": 42,
}

NUM_PREDICT = {
    "llama3.1:8b": 200,             # answer-only, sized for multi-edge output
    "qwen3.5:9b": 20000,            # raised for thinking-mode headroom
    "qwen3.5:9b_instruct": 500,     # no thinking; direct answer
    "gemma4:e4b": 2500,
    "deepseek-r1:8b": 20000,        # raised for thinking-mode headroom
}

REQUEST_TIMEOUT = 600

# Compound MBV node members: treated as equivalent during ground-truth matching.
# PBV and Market Value of Equity both map to the single compound node, following
# the canonicalisation policy in the methodology chapter.
ONTOLOGY_MBV_MEMBERS = {"pbv", "market value of equity"}
COMPOUND_MBV_LABEL = "__MBV_COMPOUND__"

# ============================================================
# LOGGING
# ============================================================

LOGS_DIR.mkdir(exist_ok=True, parents=True)
RESULTS_DIR.mkdir(exist_ok=True, parents=True)

log_filename = LOGS_DIR / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_filename, encoding="utf-8"),
        logging.StreamHandler(),
    ],
    force=True,
)
log = logging.getLogger(__name__)

# ============================================================
# LOAD INPUTS
# ============================================================

def load_excerpts():
    with open(EXCERPTS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def load_ontology():
    with open(ONTOLOGY_PATH, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

# ============================================================
# MODE DISPATCH
# ============================================================

def mode_for(model):
    if model in INSTRUCT_MODELS:
        return "instruct"
    if model in SOFT_GATED_MODELS:
        return "soft"
    return "hard"

# ============================================================
# PROMPT BUILDING
# ============================================================

def build_prompt_hard(excerpt_text, ontology):
    ontology_block = "\n".join(f"- {kpi}" for kpi in ontology)
    return f"""You are analyzing a passage from a corporate sustainability (ESG) report to identify causal relationships between ESG concepts.

ONTOLOGY (concepts to choose from):
{ontology_block}

TASK:
Read the passage below. Identify all causal relationships (cause -> effect) between concepts from the ontology that the passage discusses.

IMPORTANT: The passage may not use the exact words from the ontology. You must match the passage's ideas to the closest ontology concepts. For example (these examples use concepts NOT in the ontology, purely to illustrate the matching approach):
- "expansion of total company holdings" relates to "Asset Growth"
- "revenue before deductions from customer purchases" relates to "Gross Sales"
- "how the firm engages with its clients" relates to "Customer Relationship"
- "efficient use of raw materials and energy" relates to "Resource Efficiency"
- "negotiations between workers and management" relates to "Collective Bargaining"
- "political advocacy activities" relates to "Lobbying"

Your cause and effect must both be chosen from the ontology list above.

OUTPUT FORMAT:
Respond with ONLY one or more lines in this exact format, one causal relationship per line:
CAUSE -> EFFECT

Do not include explanations, reasoning, preamble, numbering, or any other text. If no causal relationship from the ontology can be reasonably inferred, respond with exactly:
NONE

PASSAGE:
{excerpt_text}"""

def build_prompt_soft(excerpt_text, ontology):
    ontology_block = "\n".join(f"- {kpi}" for kpi in ontology)
    return f"""You are analyzing a passage from a corporate sustainability (ESG) report to identify causal relationships between ESG concepts.

ONTOLOGY (concepts to choose from):
{ontology_block}

TASK:
Read the passage below. Identify all causal relationships (cause -> effect) between concepts from the ontology that the passage discusses.

IMPORTANT: The passage may not use the exact words from the ontology. You must match the passage's ideas to the closest ontology concepts. For example (these examples use concepts NOT in the ontology, purely to illustrate the matching approach):
- "expansion of total company holdings" relates to "Asset Growth"
- "revenue before deductions from customer purchases" relates to "Gross Sales"
- "how the firm engages with its clients" relates to "Customer Relationship"
- "efficient use of raw materials and energy" relates to "Resource Efficiency"
- "negotiations between workers and management" relates to "Collective Bargaining"
- "political advocacy activities" relates to "Lobbying"

Your cause and effect must both be chosen from the ontology list above. Do not refuse to match just because the passage uses different wording — use your judgment to find the closest ontology concepts.

You may reason through your answer. When you have decided, end your response with one FINAL: line per causal relationship identified, in exactly this format:
FINAL: CAUSE -> EFFECT

Write a separate FINAL: line for each causal relationship. If no causal relationship from the ontology can be reasonably inferred, end with:
FINAL: NONE

PASSAGE:
{excerpt_text}"""

def build_prompt_instruct(excerpt_text, ontology):
    # SUSPENDERS: /no_think appended as prompt-level Qwen thinking toggle
    # (complements the API-level "think": False in call_ollama).
    return build_prompt_hard(excerpt_text, ontology) + "\n\n/no_think"

def build_prompt(excerpt_text, ontology, model):
    m = mode_for(model)
    if m == "instruct":
        return build_prompt_instruct(excerpt_text, ontology)
    if m == "soft":
        return build_prompt_soft(excerpt_text, ontology)
    return build_prompt_hard(excerpt_text, ontology)

# ============================================================
# OLLAMA CALL
# ============================================================

def call_ollama(model, prompt):
    options = dict(DEFAULT_OPTIONS)
    options["num_predict"] = NUM_PREDICT.get(model, 500)

    ollama_model = OLLAMA_MODEL_FOR_VARIANT.get(model, model)

    payload = {
        "model": ollama_model,
        "prompt": prompt,
        "stream": False,
        "options": options,
    }
    # BELT: API-level think=False for instruct variants.
    if model in INSTRUCT_MODELS:
        payload["think"] = False

    start = time.time()
    response = requests.post(OLLAMA_URL, json=payload, timeout=REQUEST_TIMEOUT)
    elapsed = time.time() - start
    response.raise_for_status()

    data = response.json()
    return {
        "response": data.get("response", ""),
        "thinking": data.get("thinking", ""),
        "done_reason": data.get("done_reason", ""),
    }, elapsed

# ============================================================
# OUTPUT PARSING
# ============================================================

THINK_PATTERNS = [
    re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE),
    re.compile(r"<\|think\|>.*?<\|/think\|>", re.DOTALL),
    re.compile(r"<\|channel\|>thought.*?<\|/channel\|>", re.DOTALL),
]

EDGE_PATTERN = re.compile(r"^\s*(.+?)\s*->\s*(.+?)\s*$")
FINAL_PATTERN = re.compile(r"^\s*FINAL:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)

def strip_thinking(raw_text):
    cleaned = raw_text
    for pattern in THINK_PATTERNS:
        cleaned = pattern.sub("", cleaned)
    return cleaned.strip()

def clean_endpoint(s):
    """Strip common list-bullet / numbering / trailing-punctuation artefacts."""
    s = re.sub(r"^[\*\-\u2022\d\.\)]+\s*", "", s)
    s = re.sub(r"[\.\,\;\:]+$", "", s)
    return s.strip()

def parse_edges_hard(cleaned):
    """Hard-gated parser (also used for instruct mode).
    Returns (edges, overall_status) where edges is a list of (cause_raw,
    effect_raw) tuples and overall_status is 'ok' | 'none' | 'malformed'.
    """
    lines = [ln.strip() for ln in cleaned.splitlines() if ln.strip()]
    if not lines:
        return [], "malformed"

    # Pure NONE response: NONE on its own with no edges above it.
    if lines[-1].upper() == "NONE":
        edge_lines_before = [ln for ln in lines[:-1] if EDGE_PATTERN.match(ln)]
        if not edge_lines_before:
            return [], "none"
        # else: mixed NONE + edges. Fall through and collect the edges.

    # Collect consecutive edge-matching lines from the bottom upward.
    edges = []
    for ln in reversed(lines):
        if ln.upper() == "NONE":
            continue
        m = EDGE_PATTERN.match(ln)
        if m:
            edges.append((clean_endpoint(m.group(1)), clean_endpoint(m.group(2))))
        else:
            break
    edges.reverse()

    if not edges:
        return [], "malformed"
    return edges, "ok"

def parse_edges_soft(cleaned):
    """Soft-gated parser. Extracts every FINAL: line."""
    matches = FINAL_PATTERN.findall(cleaned)
    if not matches:
        return [], "malformed"

    edges = []
    none_seen = False
    for m in matches:
        content = m.strip()
        if content.upper() == "NONE":
            none_seen = True
            continue
        em = EDGE_PATTERN.match(content)
        if em:
            edges.append((clean_endpoint(em.group(1)), clean_endpoint(em.group(2))))

    if not edges and none_seen:
        return [], "none"
    if not edges:
        return [], "malformed"
    return edges, "ok"

def normalize_to_ontology(cause_raw, effect_raw, ontology_lower):
    """Map raw endpoints to canonical ontology surface forms (case-insensitive)."""
    cause_norm = ontology_lower.get(cause_raw.lower())
    effect_norm = ontology_lower.get(effect_raw.lower())
    if cause_norm and effect_norm:
        return cause_norm, effect_norm, "ok"
    return cause_raw, effect_raw, "not_in_ontology"

def parse_edges(raw_data, ontology, model):
    """Top-level parser. Returns (list of edge dicts, overall_status)."""
    response_text = raw_data.get("response", "")
    thinking_text = raw_data.get("thinking", "")
    text_to_parse = response_text if response_text.strip() else thinking_text
    cleaned = strip_thinking(text_to_parse)

    m = mode_for(model)
    if m == "soft":
        raw_edges, status = parse_edges_soft(cleaned)
    else:
        # hard and instruct both use the hard-gated parser
        raw_edges, status = parse_edges_hard(cleaned)

    if status != "ok":
        return [], status

    ontology_lower = {k.lower(): k for k in ontology}
    edge_records = []
    for cause, effect in raw_edges:
        c, e, edge_status = normalize_to_ontology(cause, effect, ontology_lower)
        edge_records.append({"cause": c, "effect": e, "status": edge_status})
    return edge_records, "ok"

# ============================================================
# GRADING — exact match on canonicalised edge sets
# ============================================================

def canonicalize_endpoint_for_match(ep):
    """Map ontology endpoint to its compound-node canonical form for matching.
    PBV and Market Value of Equity collapse to the same compound node."""
    if ep is None:
        return None
    if ep.lower() in ONTOLOGY_MBV_MEMBERS:
        return COMPOUND_MBV_LABEL
    return ep

def edges_match_exactly(predicted_edges, ground_truth_edges):
    """Exact-match boolean: predicted edge set equals ground-truth edge set
    after compound-MBV canonicalisation. Direction matters; sign is ignored;
    duplicates and ordering are ignored (set semantics)."""
    pred_set = set(
        (canonicalize_endpoint_for_match(ed["cause"]),
         canonicalize_endpoint_for_match(ed["effect"]))
        for ed in predicted_edges
    )
    gt_set = set(
        (canonicalize_endpoint_for_match(c), canonicalize_endpoint_for_match(e))
        for c, e in ground_truth_edges
    )
    return pred_set == gt_set

# ============================================================
# RESULT WRITING
# ============================================================

def results_path(model):
    safe_name = model.replace(":", "_").replace("/", "_")
    return RESULTS_DIR / f"{safe_name}.jsonl"

def load_completed_ids(model):
    path = results_path(model)
    if not path.exists():
        return set()
    done = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
                done.add(rec["excerpt_id"])
            except (json.JSONDecodeError, KeyError):
                continue
    return done

def append_result(model, record):
    path = results_path(model)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

# ============================================================
# MAIN
# ============================================================

def run_model(model, excerpts, ontology):
    m = mode_for(model)
    log.info(f"=== Starting model: {model} (mode={m}) ===")
    completed = load_completed_ids(model)
    excerpts_to_run = excerpts[:LIMIT] if LIMIT is not None else excerpts
    already_done = len(completed & {e["id"] for e in excerpts_to_run})
    log.info(f"Running {len(excerpts_to_run)} excerpts "
             f"(already completed: {already_done})")

    for i, excerpt in enumerate(excerpts_to_run, start=1):
        excerpt_id = excerpt["id"]
        if excerpt_id in completed:
            log.info(f"[{model}] {i}/{len(excerpts_to_run)} {excerpt_id} — already done, skipping")
            continue

        prompt = build_prompt(excerpt["text"], ontology, model)

        try:
            raw, elapsed = call_ollama(model, prompt)
        except Exception as e:
            log.error(f"[{model}] {excerpt_id}: request failed — {e}")
            append_result(model, {
                "excerpt_id": excerpt_id, "model": model, "mode": m, "error": str(e),
                "ground_truth_edges": excerpt["ground_truth_edges"],
            })
            continue

        predicted_edges, overall_status = parse_edges(raw, ontology, model)
        is_correct = (
            overall_status == "ok"
            and edges_match_exactly(predicted_edges, excerpt["ground_truth_edges"])
        )

        record = {
            "excerpt_id": excerpt_id,
            "model": model,
            "mode": m,
            "firm": excerpt.get("firm"),
            "ground_truth_edges": excerpt["ground_truth_edges"],
            "predicted_edges": predicted_edges,
            "status": overall_status,
            "correct": is_correct,
            "elapsed_sec": round(elapsed, 2),
            "raw_response": raw.get("response", ""),
            "raw_thinking": raw.get("thinking", ""),
            "done_reason": raw.get("done_reason", ""),
        }
        append_result(model, record)

        marker = "✓" if is_correct else "✗"
        log.info(f"[{model}] {i}/{len(excerpts_to_run)} {excerpt_id} "
                 f"({elapsed:.1f}s) status={overall_status} {marker}")

    log.info(f"=== Finished model: {model} ===")

def main():
    mode = "PILOT" if LIMIT is not None else "FULL"
    log.info(f"=== Experiment starting in {mode} mode ===")
    if LIMIT is not None:
        log.info(f"LIMIT = {LIMIT}")

    excerpts = load_excerpts()
    ontology = load_ontology()
    log.info(f"Loaded {len(excerpts)} excerpts and {len(ontology)} ontology items")

    for model in MODELS:
        try:
            run_model(model, excerpts, ontology)
        except KeyboardInterrupt:
            log.warning("Interrupted.")
            return
        except Exception as e:
            log.error(f"Model {model} failed: {e}")
            continue

    log.info("=== All models complete ===")

main()
