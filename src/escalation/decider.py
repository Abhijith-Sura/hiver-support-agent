"""
Escalation decision logic: hybrid rule-based + LLM for borderline cases.
"""
import json
import logging
import re
import time

from src.config import (
    LLM_API_KEY,
    ESCALATION_CONFIDENCE_THRESHOLD, AUTO_HANDLE_CONFIDENCE_THRESHOLD,
    ESCALATION_INTENTS,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ESCALATION_KEYWORDS = [
    "lawsuit", "lawyer", "sue", "legal", "attorney",
    "cancel subscription", "cancel my subscription", "refund",
    "human", "real person", "agent", "supervisor", "manager",
    "harassment", "threat", "unsafe", "suicide", "self-harm",
]

ESCALATION_PATTERN = re.compile(
    r"\b(?:" + "|".join(re.escape(kw) for kw in ESCALATION_KEYWORDS) + r")\b",
    re.IGNORECASE,
)


class EscalationDecider:
    """Decide whether to auto-handle or escalate a customer message."""

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or LLM_API_KEY
        self._llm = None

    def _get_llm(self):
        if self._llm is None and self.api_key:
            from src.llm_client import LLMClient
            self._llm = LLMClient(api_key=self.api_key)
        return self._llm

    def decide(self, customer_message: str, intent: str, intent_confidence: float,
               thread_context: list[str] | None = None) -> dict:
        customer_message = str(customer_message or "").strip()
        intent = str(intent or "other")
        try:
            intent_confidence = float(intent_confidence)
        except (ValueError, TypeError):
            intent_confidence = 0.5

        if not customer_message:
            return {
                "decision": "escalate",
                "reason": "Empty message received. Needs human triage.",
                "confidence": 0.9,
            }

        # Rule 1: Low confidence
        if intent_confidence < ESCALATION_CONFIDENCE_THRESHOLD:
            return {"decision": "escalate",
                    "reason": f"Low classification confidence ({intent_confidence:.2f}). Needs human review.",
                    "confidence": 0.9}
        # Rule 2: High-risk intents
        if intent in ESCALATION_INTENTS:
            return {"decision": "escalate",
                    "reason": f"Intent '{intent}' requires account-level access.",
                    "confidence": 0.85}
        # Rule 3: Keywords
        match = ESCALATION_PATTERN.search(customer_message)
        if match:
            return {"decision": "escalate",
                    "reason": f"Escalation trigger: '{match.group()}'.",
                    "confidence": 0.9}
        # Rule 4: Long thread
        if thread_context and len(thread_context) >= 3:
            return {"decision": "escalate",
                    "reason": f"Extended conversation ({len(thread_context)} prior messages).",
                    "confidence": 0.75}
        # Rule 5: High confidence safe intent
        if intent_confidence >= AUTO_HANDLE_CONFIDENCE_THRESHOLD:
            return {"decision": "auto_handle",
                    "reason": f"High confidence ({intent_confidence:.2f}) on safe intent '{intent}'.",
                    "confidence": 0.85}
        # Rule 6: Borderline → LLM
        llm = self._get_llm()
        if llm:
            return self._llm_decide(customer_message, intent, intent_confidence, thread_context)
        return {"decision": "escalate",
                "reason": f"Borderline confidence ({intent_confidence:.2f}). Escalating conservatively.",
                "confidence": 0.6}

    def _llm_decide(self, customer_message, intent, intent_confidence, thread_context):
        prompt = f"""Decide: AUTO_HANDLE (bot can respond) or ESCALATE (needs human).
Customer: "{customer_message}"
Intent: {intent} (confidence: {intent_confidence:.2f})
Context: {thread_context[-3:] if thread_context else "None"}
Respond JSON: {{"decision": "auto_handle" or "escalate", "reason": "...", "confidence": 0.0-1.0}}"""
        try:
            result = self._llm.generate_json(prompt, temperature=0.1)
            decision = result.get("decision", "escalate")
            if decision not in ("auto_handle", "escalate"):
                decision = "escalate"
            return {"decision": decision, "reason": result.get("reason", ""), "confidence": float(result.get("confidence", 0.5))}
        except Exception as e:
            logger.warning("LLM escalation failed: %s", e)
            return {"decision": "escalate", "reason": "LLM failed — escalating.", "confidence": 0.5}
