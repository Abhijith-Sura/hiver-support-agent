"""
Reply generation using LLM (Groq), grounded in retrieved historical conversations.
"""
import json
import logging
import time

from src.config import LLM_API_KEY

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are SpotifyCares, Spotify's official Twitter customer support agent.

Brand voice guidelines:
- Friendly, empathetic, and concise (tweet-length responses, under 280 characters when possible)
- Use emojis occasionally but not excessively (1-2 per reply max)
- Acknowledge the customer's frustration before offering solutions
- Provide actionable next steps (links, specific instructions)
- Sign off with agent initials like /AI
- Never fabricate links, features, or policies that don't exist
- If you're unsure, suggest the customer DM for account-specific help

You will be given the customer's message, conversation history, similar past conversations, and the classified intent.
Generate a helpful reply that matches how SpotifyCares historically handles similar issues.
Respond with JSON: {"reply_text": "...", "grounding_sources": ["brief description of which examples informed this reply"], "confidence": 0.0-1.0}"""


class ReplyGenerator:
    """Generate grounded replies using LLM."""

    def __init__(self, api_key: str | None = None, retriever=None):
        from src.llm_client import LLMClient
        self.llm = LLMClient(api_key=api_key)
        self.retriever = retriever

    def generate(self, customer_message: str, intent: str,
                 thread_context: list[str] | None = None,
                 retrieved_examples: list[dict] | None = None) -> dict:
        customer_message = str(customer_message or "").strip()
        if not customer_message:
            return {
                "reply_text": "Hey there! How can we help you today? Feel free to let us know what's going on! /AI",
                "grounding_sources": ["empty_message_fallback"],
                "confidence": 0.5,
            }

        if retrieved_examples is None and self.retriever is not None:
            try:
                retrieved_examples = self.retriever.retrieve(customer_message)
            except Exception as e:
                logger.warning("Retrieval failed: %s", e)
                retrieved_examples = []

        user_prompt = self._build_prompt(customer_message, intent, thread_context,
                                          retrieved_examples or [])
        try:
            result = self.llm.generate_json(user_prompt, SYSTEM_PROMPT, temperature=0.7)
            return {
                "reply_text": result.get("reply_text", ""),
                "grounding_sources": result.get("grounding_sources", []),
                "confidence": float(result.get("confidence", 0.5)),
            }
        except Exception as e:
            logger.error("Reply generation failed: %s", e)
            return {
                "reply_text": "Hey there! Sorry about that. Could you DM us with more details so we can look into this? /AI",
                "grounding_sources": ["fallback response"],
                "confidence": 0.0,
            }

    def generate_batch(self, messages: list[dict]) -> list[dict]:
        results = []
        for msg in messages:
            result = self.generate(
                customer_message=msg.get("customer_message", ""),
                intent=msg.get("intent", "other"),
                thread_context=msg.get("thread_context"),
                retrieved_examples=msg.get("retrieved_examples"),
            )
            results.append(result)
            time.sleep(0.3)
        return results

    @staticmethod
    def _build_prompt(message: str, intent: str, thread_context: list[str] | None,
                      retrieved_examples: list[dict]) -> str:
        parts = [f"Classified intent: {intent}", ""]
        if thread_context:
            parts.append("Conversation history:")
            for ctx in thread_context[-3:]:
                parts.append(f"  {ctx}")
            parts.append("")
        if retrieved_examples:
            parts.append("Similar past conversations (use as grounding):")
            for i, ex in enumerate(retrieved_examples[:3], 1):
                parts.append(f"  Example {i} (similarity: {ex.get('similarity_score', 'N/A')}):")
                parts.append(f"    Customer: {ex['customer_message']}")
                parts.append(f"    SpotifyCares reply: {ex['brand_reply']}")
            parts.append("")
        parts.append(f'Customer message to reply to: "{message}"')
        parts.append('\nGenerate reply as JSON: {"reply_text": "...", "grounding_sources": [...], "confidence": 0.0-1.0}')
        return "\n".join(parts)
