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

LIMIT = None  # None = use RERUN_IDS filter if present, else run all

OLLAMA_URL = "http://localhost:11434/api/generate"

# Only rerunning thinking-mode models that had malformed outputs
MODELS = [
    "qwen3.5:9b",
    "deepseek-r1:8b",
]

SOFT_GATED_MODELS = {"deepseek-r1:8b", "gemma4:e4b", "qwen3.5:9b"}

# Kaggle paths
EXCERPTS_PATH = Path("/kaggle/input/datasets/sashwat04/inputdocuments/excerpts.json")
ONTOLOGY_PATH = Path("/kaggle/input/datasets/sashwat04/inputdocuments/ontology.txt")

# NEW: per-model rerun ID files (adjust path if your new dataset has a different slug)
RERUN_ID_PATHS = {
    "qwen3.5:9b": Path("/kaggle/input/datasets/sashwat04/inputdocuments/qwen_rerun_ids.txt"),
    "deepseek-r1:8b": Path("/kaggle/input/datasets/sashwat04/inputdocuments/deepseek_rerun_ids.txt"),
}

RESULTS_DIR = Path("/kaggle/working/results")
LOGS_DIR = Path("/kaggle/working/logs")

DEFAULT_OPTIONS = {
    "temperature": 0,
    "top_p": 1.0,
    "seed": 42,
}

# Bumped to 20000 to give thinking-mode models plenty of headroom.
# Models will stop naturally when done; they won't use all 20k unless needed.
NUM_PREDICT = {
    "llama3.1:8b": 100,
    "qwen3.5:9b": 20000,
    "deepseek-r1:8b": 20000,
    "gemma4:e4b": 2500,
}

# Extended timeout to match the larger token budget (20k tokens on T4 ≈ 8-10 min worst case)
REQUEST_TIMEOUT = 900

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

def load_rerun_ids(model):
    """Load the set of excerpt IDs to rerun for this model. Returns None if no filter."""
    path = RERUN_ID_PATHS.get(model)
    if path is None or not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        ids = {line.strip() for line in f if line.strip()}
    return ids

# ============================================================
# PROMPT BUILDING
# ============================================================

def build_prompt_hard(excerpt_text, ontology):
    ontology_block = "\n".join(f"- {kpi}" for kpi in ontology)
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
{excerpt_text}"""

def build_prompt_soft(excerpt_text, ontology):
    ontology_block = "\n".join(f"- {kpi}" for kpi in ontology)
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

Your cause and effect must both be chosen from the ontology list above. Do not refuse to match just because the passage uses different wording — use your judgment to find the closest ontology concepts.

You may reason through your answer. When you have decided, end your response with a single final line in exactly this format:
FINAL: CAUSE -> EFFECT

If no causal relationship from the ontology can be reasonably inferred, end with:
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
    return {
        "response": data.get("response", ""),
        "thinking": data.get("thinking", ""),
        "done_reason": data.get("done_reason", ""),
    }, elapsed

# ============================================================
# OUTPUT PARSING  (PATCHED)
# ============================================================

THINK_PATTERNS = [
    re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE),
    re.compile(r"<\|think\|>.*?<\|/think\|>", re.DOTALL),
    re.compile(r"<\|channel\|>thought.*?<\|/channel\|>", re.DOTALL),
]

EDGE_PATTERN = re.compile(r"^\s*(.+?)\s*->\s*(.+?)\s*$")

# PATCH: accept both "FINAL: X -> Y" and "CAUSE -> EFFECT: X -> Y" as final-answer markers.
# Also tolerates variations like "Final Answer:", "ANSWER:", etc.
FINAL_MARKER_PATTERNS = [
    re.compile(r"FINAL\s*(?:ANSWER)?\s*:\s*(.+?)$", re.MULTILINE | re.IGNORECASE),
    re.compile(r"CAUSE\s*->\s*EFFECT\s*:\s*(.+?)$", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^\s*ANSWER\s*:\s*(.+?)$", re.MULTILINE | re.IGNORECASE),
]

def strip_thinking(raw_text):
    cleaned = raw_text
    for pattern in THINK_PATTERNS:
        cleaned = pattern.sub("", cleaned)
    return cleaned.strip()

def find_final_answer(cleaned_text):
    """Try each FINAL-marker pattern; return the last match from any pattern."""
    all_matches = []
    for pattern in FINAL_MARKER_PATTERNS:
        all_matches.extend(pattern.findall(cleaned_text))
    if not all_matches:
        return None
    return all_matches[-1].strip()

def parse_edge(raw_data, ontology, model):
    response_text = raw_data.get("response", "")
    thinking_text = raw_data.get("thinking", "")
    text_to_parse = response_text if response_text.strip() else thinking_text
    cleaned = strip_thinking(text_to_parse)

    if model in SOFT_GATED_MODELS:
        final_content = find_final_answer(cleaned)
        if final_content is None:
            return None, None, "malformed"
    else:
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
    # Rerun results go in a subfolder to keep them separate from the original run
    out_dir = RESULTS_DIR / "rerun"
    out_dir.mkdir(exist_ok=True, parents=True)
    return out_dir / f"{safe_name}.jsonl"

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
    log.info(f"=== Starting model: {model} ===")
    completed = load_completed_ids(model)

    # Apply rerun filter if present
    rerun_ids = load_rerun_ids(model)
    if rerun_ids is not None:
        excerpts_to_run = [e for e in excerpts if e["id"] in rerun_ids]
        log.info(f"Rerun filter active for {model}: {len(excerpts_to_run)} excerpts "
                 f"(out of {len(rerun_ids)} IDs in rerun file)")
        # Sanity check: warn if any rerun IDs don't match any excerpt
        missing = rerun_ids - {e["id"] for e in excerpts}
        if missing:
            log.warning(f"Rerun IDs not found in excerpts: {missing}")
    else:
        excerpts_to_run = excerpts[:LIMIT] if LIMIT is not None else excerpts
        log.info(f"No rerun filter for {model}; running {len(excerpts_to_run)} excerpts")

    log.info(f"Already completed in this rerun: "
             f"{len(completed & {e['id'] for e in excerpts_to_run})}")

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
    log.info(f"=== Rerun experiment starting ===")
    log.info(f"Models: {MODELS}")
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
