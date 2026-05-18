import json
import re
import time
import logging
from pathlib import Path
from datetime import datetime
import requests

# ============================================================
# CONFIG — Qwen INSTRUCT MODE (thinking disabled) full rerun
# ============================================================

LIMIT = None  # Full 39 excerpts

OLLAMA_URL = "http://localhost:11434/api/generate"

MODELS = ["qwen3.5:9b"]

# Hard-gated only: instruct mode emits direct answer, no FINAL: marker needed
SOFT_GATED_MODELS = set()

# Kaggle paths
EXCERPTS_PATH = Path("/kaggle/input/datasets/sashwat04/inputdocuments/excerpts.json")
ONTOLOGY_PATH = Path("/kaggle/input/datasets/sashwat04/inputdocuments/ontology.txt")

RESULTS_DIR = Path("/kaggle/working/results_instruct")
LOGS_DIR = Path("/kaggle/working/logs")

DEFAULT_OPTIONS = {
    "temperature": 0,
    "top_p": 1.0,
    "seed": 42,
}

# Bumped to 500 for safety margin against verbose instruct-mode responses
NUM_PREDICT = {
    "qwen3.5:9b": 500,
}

REQUEST_TIMEOUT = 300

# ============================================================
# LOGGING
# ============================================================

LOGS_DIR.mkdir(exist_ok=True, parents=True)
RESULTS_DIR.mkdir(exist_ok=True, parents=True)

log_filename = LOGS_DIR / f"run_instruct_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
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
# PROMPT BUILDING
# ============================================================

def build_prompt_hard(excerpt_text, ontology):
    ontology_block = "\n".join(f"- {kpi}" for kpi in ontology)
    # SUSPENDERS: /no_think appended as prompt-level Qwen thinking toggle
    # (complements the API-level "think": False)
    return f"""You are analyzing a passage from a corporate sustainability (ESG) report to identify a causal relationship between two ESG concepts.

ONTOLOGY (concepts to choose from):
{ontology_block}

TASK:
Read the passage below. Identify the single causal relationship (cause -> effect) between two concepts from the ontology that best describes what the passage is discussing.

IMPORTANT: The passage may not use the exact words from the ontology. You must match the passage's ideas to the closest ontology concepts. For example (these examples use concepts NOT in the ontology, purely to illustrate the matching approach):
- "expansion of total company holdings" relates to "Asset Growth"
- "revenue before deductions from customer purchases" relates to "Gross Sales"
- "how the firm engages with its clients" relates to "Customer Relationship"
- "efficient use of raw materials and energy" relates to "Resource Efficiency"
- "negotiations between workers and management" relates to "Collective Bargaining"
- "political advocacy activities" relates to "Lobbying"

Your cause and effect must both be chosen from the ontology list above.

OUTPUT FORMAT:
Respond with ONLY a single line in this exact format:
CAUSE -> EFFECT

Do not include explanations, reasoning, preamble, or any other text. If no causal relationship from the ontology can be reasonably inferred, respond with exactly:
NONE

PASSAGE:
{excerpt_text}

/no_think"""

def build_prompt(excerpt_text, ontology, model):
    return build_prompt_hard(excerpt_text, ontology)

# ============================================================
# OLLAMA CALL — with think=false (BELT)
# ============================================================

def call_ollama(model, prompt):
    options = dict(DEFAULT_OPTIONS)
    options["num_predict"] = NUM_PREDICT.get(model, 500)

    # BELT: API-level think=False toggle
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "think": False,
        "options": options,
    }

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
# OUTPUT PARSING — hard-gated parser with trailing-punctuation cleanup
# ============================================================

THINK_PATTERNS = [
    re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE),
    re.compile(r"<\|think\|>.*?<\|/think\|>", re.DOTALL),
    re.compile(r"<\|channel\|>thought.*?<\|/channel\|>", re.DOTALL),
]

EDGE_PATTERN = re.compile(r"^\s*(.+?)\s*->\s*(.+?)\s*$")

def strip_thinking(raw_text):
    cleaned = raw_text
    for pattern in THINK_PATTERNS:
        cleaned = pattern.sub("", cleaned)
    return cleaned.strip()

def clean_term(term):
    """Strip trailing punctuation and whitespace from extracted cause/effect."""
    return term.strip().rstrip(".,;:!?").strip()

def parse_edge(raw_data, ontology, model):
    response_text = raw_data.get("response", "")
    thinking_text = raw_data.get("thinking", "")
    text_to_parse = response_text if response_text.strip() else thinking_text
    cleaned = strip_thinking(text_to_parse)

    lines = [ln.strip() for ln in cleaned.splitlines() if ln.strip()]
    if not lines:
        return None, None, "malformed"
    final_content = lines[-1]

    if clean_term(final_content).upper() == "NONE":
        return None, None, "none"

    match = EDGE_PATTERN.match(final_content)
    if not match:
        return None, None, "malformed"

    cause = clean_term(match.group(1))
    effect = clean_term(match.group(2))

    ontology_lower = {k.lower(): k for k in ontology}
    cause_norm = ontology_lower.get(cause.lower())
    effect_norm = ontology_lower.get(effect.lower())

    if cause_norm and effect_norm:
        return cause_norm, effect_norm, "ok"
    return cause, effect, "not_in_ontology"

# ============================================================
# RESULT WRITING
# ============================================================

def results_path(model):
    safe_name = model.replace(":", "_").replace("/", "_")
    return RESULTS_DIR / f"{safe_name}_instruct.jsonl"

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
    log.info(f"=== Starting model: {model} (INSTRUCT / think=False + /no_think) ===")
    completed = load_completed_ids(model)
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
                "excerpt_id": excerpt_id, "model": model, "error": str(e),
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
            "excerpt_id": excerpt_id, "model": model, "firm": excerpt.get("firm"),
            "mode": "instruct",
            "ground_truth_cause": excerpt["ground_truth_cause"],
            "ground_truth_effect": excerpt["ground_truth_effect"],
            "predicted_cause": cause, "predicted_effect": effect,
            "status": status, "correct": is_correct, "elapsed_sec": round(elapsed, 2),
            "raw_response": raw.get("response", ""),
            "raw_thinking": raw.get("thinking", ""),
            "done_reason": raw.get("done_reason", ""),
        }
        append_result(model, record)

        marker = "✓" if is_correct else "✗"
        log.info(f"[{model}] {i}/{len(excerpts_to_run)} {excerpt_id} "
                 f"({elapsed:.1f}s) status={status} {marker}")

    log.info(f"=== Finished model: {model} ===")

def main():
    log.info(f"=== Instruct-mode experiment starting ===")
    log.info(f"Models: {MODELS} (think=False + /no_think)")
    log.info(f"num_predict: {NUM_PREDICT}")

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
