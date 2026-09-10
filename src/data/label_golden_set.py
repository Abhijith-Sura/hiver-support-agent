"""
Auto-label the golden set using LLM (Groq).
"""
import json
import logging
import time
from pathlib import Path
from collections import Counter

from src.config import LLM_API_KEY, GOLDEN_SET_PATH, INTENT_LABELS, ESCALATION_INTENTS
from src.intent.taxonomy import INTENT_DEFINITIONS
from src.llm_client import LLMClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a data labeller for SpotifyCares customer support evaluation.
For each customer message, provide:
1. gold_intent: one of {intents}
2. gold_escalation: "escalate" or "auto_handle"
3. gold_escalation_reason: brief reason
4. gold_reply_quality_notes: what a good reply should contain
5. labeller_notes: any ambiguities

Intent definitions:
{taxonomy}

Respond with JSON containing all 5 fields."""


def label_golden_set(golden_set_path: Path | None = None) -> None:
    golden_set_path = golden_set_path or GOLDEN_SET_PATH
    llm = LLMClient()

    taxonomy_lines = [f"- {k}: {v['description']}" for k, v in INTENT_DEFINITIONS.items()]
    system = SYSTEM_PROMPT.format(intents=", ".join(INTENT_LABELS), taxonomy="\n".join(taxonomy_lines))

    examples = []
    with open(golden_set_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))

    logger.info("Labelling %d examples…", len(examples))
    labelled = []

    for i, ex in enumerate(examples):
        msg = ex["customer_message"]
        ctx = ex.get("thread_context", [])
        brand_reply = ex.get("brand_reply", "")

        prompt = f'Customer message: "{msg}"'
        if ctx:
            prompt = f"Thread context: {ctx[-2:]}\n{prompt}"
        if brand_reply:
            prompt += f'\nActual SpotifyCares reply: "{brand_reply}"'

        try:
            result = llm.generate_json(prompt, system, temperature=0.0)
            gold_intent = result.get("gold_intent", "other")
            if gold_intent not in INTENT_LABELS:
                gold_intent = "other"
            gold_esc = result.get("gold_escalation", "auto_handle")
            if gold_esc not in ("escalate", "auto_handle"):
                gold_esc = "auto_handle"

            ex["gold_intent"] = gold_intent
            ex["gold_escalation"] = gold_esc
            ex["gold_escalation_reason"] = result.get("gold_escalation_reason", "")
            ex["gold_reply_quality_notes"] = result.get("gold_reply_quality_notes", "")
            ex["labeller_notes"] = result.get("labeller_notes", "Auto-labelled, needs human review")
        except Exception as e:
            logger.warning("Failed to label %s: %s. Using fallback.", ex["id"], e)
            ex["gold_intent"] = ex.get("rough_intent_bucket", "other")
            ex["gold_escalation"] = "escalate" if ex["gold_intent"] in ESCALATION_INTENTS else "auto_handle"
            ex["gold_escalation_reason"] = "Fallback"
            ex["gold_reply_quality_notes"] = ""
            ex["labeller_notes"] = "FALLBACK — labelling failed"

        labelled.append(ex)
        if (i + 1) % 20 == 0:
            logger.info("Labelled %d/%d", i + 1, len(examples))
            # Save progress incrementally
            with open(golden_set_path, "w", encoding="utf-8") as f:
                for done_ex in labelled:
                    f.write(json.dumps(done_ex, ensure_ascii=False) + "\n")
        time.sleep(3)  # Groq free tier: 30 req/min limit

    with open(golden_set_path, "w", encoding="utf-8") as f:
        for ex in labelled:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    logger.info("Saved %d labelled examples.", len(labelled))
    logger.info("Intent dist: %s", dict(Counter(e["gold_intent"] for e in labelled)))
    logger.info("Escalation dist: %s", dict(Counter(e["gold_escalation"] for e in labelled)))


if __name__ == "__main__":
    label_golden_set()
