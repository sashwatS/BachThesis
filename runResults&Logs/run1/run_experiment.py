"""
ESG Causal Extraction Experiment Pipeline
Runs excerpts through 4 Ollama models and saves structured results.

Usage:
    python run_experiment.py

Trial vs. full run:
    - Set LIMIT = 2 (or any small int) for a trial run
    - Set LIMIT = None for the full experiment
    - If you tweak the prompt after a trial, delete results/*.jsonl before rerunning

The script is resumable. If interrupted, rerun and it will skip completed excerpts.
"""

import json
import re
import time
import logging
from pathlib import Path
from datetime import datetime
import requests

# ============================================================
# CONFIG — edit these
# ============================================================

# Trial mode: set to an integer to run only the first N excerpts per model.
# Set to None for the full experiment.
LIMIT = 2

OLLAMA_URL = "http://localhost:11434/api/generate"

MODELS = [
    "llama3.1:8b",
    "qwen3.5:9b",
    "deepseek-r1:8b",
    "gemma4:e4b",
]

# Which models should receive the soft-gated (reasoning-friendly) prompt
# vs. the hard-gated (single-line-output) prompt.
SOFT_GATED_MODELS = {"deepseek-r1:8b", "gemma4:e4b"}

EXCERPTS_PATH = Path("excerpts.json")
ONTOLOGY_PATH = Path("ontology.txt")
RESULTS_DIR = Path("results")
LOGS_DIR = Path("logs")

# Ollama generation parameters
DEFAULT_OPTIONS = {
    "temperature": 0,
    "top_p": 1.0,
    "seed": 42,
}

# Per-model token caps. Thinking-mode models need room for their reasoning trace.
NUM_PREDICT = {
    "llama3.1:8b": 100,
    "qwen3.5:9b": 100,
    "deepseek-r1:8b": 2500,
    "gemma4:e4b": 2500,
}

# Request timeout (seconds). Thinking models can take several minutes per call on CPU.
REQUEST_TIMEOUT = 600

# ============================================================
# LOGGING SETUP
# ============================================================

LOGS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

log_filename = LOGS_DIR / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_filename, encoding="utf-8"),
        logging.StreamHandler(),
    ],
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
# PROMPT BUILDING
# ============================================================

def build_prompt_hard(excerpt_text, ontology):
    """Hard-gated prompt: model outputs only a single CAUSE -> EFFECT line."""
    ontology_block = "\n".join(f"- {kpi}" for kpi in ontology)
    return f"""You are analyzing a passage from a corporate sustainability (ESG) report to identify a single causal relationship between two ESG concepts.

ONTOLOGY (allowed concepts):
{ontology_block}

TASK:
Read the passage below. Identify exactly ONE causal relationship (cause -> effect) between two concepts from the ontology above. The cause and effect must both be from the ontology — do not invent new concepts.

OUTPUT FORMAT:
Respond with ONLY a single line in this exact format:
CAUSE -> EFFECT

Do not include explanations, reasoning, preamble, or any other text. If no causal relationship from the ontology is present, respond with exactly:
NONE

PASSAGE:
{excerpt_text}"""

def build_prompt_soft(excerpt_text, ontology):
    """Soft-gated prompt: model may reason freely, but ends with a FINAL: marker."""
    ontology_block = "\n".join(f"- {kpi}" for kpi in ontology)
    return f"""You are analyzing a passage from a corporate sustainability (ESG) report to identify a single causal relationship between two ESG concepts.

ONTOLOGY (allowed concepts):
{ontology_block}

TASK:
Read the passage below. Identify exactly ONE causal relationship (cause -> effect) between two concepts from the ontology above. Both the cause and the effect must be from the ontology — do not invent new concepts.

You may reason through your answer. When you have decided, end your response with a single final line in exactly this format:
FINAL: CAUSE -> EFFECT

If no causal relationship from the ontology is present, end with:
FINAL: NONE

PASSAGE:
{excerpt_text}"""

def build_prompt(excerpt_text, ontology, model):
    if model in SOFT_GATED_MODELS:
        return build_prompt_soft(excerpt_text, ontology)
    return build_prompt_hard(excerpt_text, ontology)

# ============================================================
# OLLAMA CALL
# ============================================================

def call_ollama(model, prompt):
    options = dict(DEFAULT_OPTIONS)
    options["num_predict"] = NUM_PREDICT.get(model, 100)

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": options,
    }

    start = time.time()
    response = requests.post(OLLAMA_URL, json=payload, timeout=REQUEST_TIMEOUT)
    elapsed = time.time() - start
    response.raise_for_status()

    data = response.json()
    return data.get("response", ""), elapsed

# ============================================================
# OUTPUT PARSING
# ============================================================

THINK_PATTERNS = [
    re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE),
    re.compile(r"<\|think\|>.*?<\|/think\|>", re.DOTALL),
    re.compile(r"<\|channel\|>thought.*?<\|/channel\|>", re.DOTALL),
]

EDGE_PATTERN = re.compile(r"^\s*(.+?)\s*->\s*(.+?)\s*$")
FINAL_PATTERN = re.compile(r"FINAL:\s*(.+?)$", re.MULTILINE | re.IGNORECASE)

def strip_thinking(raw_text):
    cleaned = raw_text
    for pattern in THINK_PATTERNS:
        cleaned = pattern.sub("", cleaned)
    return cleaned.strip()

def parse_edge(raw_text, ontology, model):
    """
    Returns (cause, effect, status).
    status ∈ {"ok", "none", "not_in_ontology", "malformed"}
    """
    cleaned = strip_thinking(raw_text)

    if model in SOFT_GATED_MODELS:
        # Find FINAL: markers; use the last one if multiple
        matches = FINAL_PATTERN.findall(cleaned)
        if not matches:
            return None, None, "malformed"
        final_content = matches[-1].strip()
    else:
        # Hard-gated: take the last non-empty line
        lines = [ln.strip() for ln in cleaned.splitlines() if ln.strip()]
        if not lines:
            return None, None, "malformed"
        final_content = lines[-1]

    if final_content.upper() == "NONE":
        return None, None, "none"

    match = EDGE_PATTERN.match(final_content)
    if not match:
        return None, None, "malformed"

    cause = match.group(1).strip()
    effect = match.group(2).strip()

    # Case-insensitive ontology matching, canonicalize to ontology casing
    ontology_lower = {k.lower(): k for k in ontology}
    cause_norm = ontology_lower.get(cause.lower())
    effect_norm = ontology_lower.get(effect.lower())

    if cause_norm and effect_norm:
        return cause_norm, effect_norm, "ok"
    return cause, effect, "not_in_ontology"

# ============================================================
# RESULT WRITING (with resume support)
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
# MAIN EXPERIMENT LOOP
# ============================================================

def run_model(model, excerpts, ontology):
    log.info(f"=== Starting model: {model} ===")
    completed = load_completed_ids(model)

    # Apply trial limit
    excerpts_to_run = excerpts[:LIMIT] if LIMIT is not None else excerpts
    log.info(f"Running {len(excerpts_to_run)} excerpts "
             f"(already completed: {len(completed & {e['id'] for e in excerpts_to_run})})")

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
                "excerpt_id": excerpt_id,
                "model": model,
                "error": str(e),
                "ground_truth_cause": excerpt["ground_truth_cause"],
                "ground_truth_effect": excerpt["ground_truth_effect"],
            })
            continue

        cause, effect, status = parse_edge(raw, ontology, model)

        is_correct = (
            status == "ok"
            and cause == excerpt["ground_truth_cause"]
            and effect == excerpt["ground_truth_effect"]
        )

        record = {
            "excerpt_id": excerpt_id,
            "model": model,
            "firm": excerpt.get("firm"),
            "ground_truth_cause": excerpt["ground_truth_cause"],
            "ground_truth_effect": excerpt["ground_truth_effect"],
            "predicted_cause": cause,
            "predicted_effect": effect,
            "status": status,
            "correct": is_correct,
            "elapsed_sec": round(elapsed, 2),
            "raw_output": raw,
        }
        append_result(model, record)

        marker = "✓" if is_correct else "✗"
        log.info(
            f"[{model}] {i}/{len(excerpts_to_run)} {excerpt_id} "
            f"({elapsed:.1f}s) status={status} {marker}"
        )

    log.info(f"=== Finished model: {model} ===")

def main():
    mode = "TRIAL" if LIMIT is not None else "FULL"
    log.info(f"=== Experiment starting in {mode} mode ===")
    if LIMIT is not None:
        log.info(f"LIMIT = {LIMIT} (only first {LIMIT} excerpts per model)")

    excerpts = load_excerpts()
    ontology = load_ontology()
    log.info(f"Loaded {len(excerpts)} excerpts and {len(ontology)} ontology items")

    for model in MODELS:
        try:
            run_model(model, excerpts, ontology)
        except KeyboardInterrupt:
            log.warning("Interrupted by user. Progress saved — rerun to resume.")
            return
        except Exception as e:
            log.error(f"Model {model} failed entirely: {e}")
            continue

    log.info("=== All models complete ===")

if __name__ == "__main__":
    main()
