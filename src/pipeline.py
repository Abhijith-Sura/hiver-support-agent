"""
End-to-end support agent pipeline.

Usage:
    python -m src.pipeline --input "My Spotify won't play songs"
    python -m src.pipeline --batch evaluation/golden_set/golden_set.jsonl --output results.jsonl
"""
import argparse
import json
import logging
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from src.config import LLM_API_KEY, PROCESSED_CONVERSATIONS_PATH
from src.intent.classifier import LLMClassifier
from src.reply.retriever import ReplyRetriever
from src.reply.generator import ReplyGenerator
from src.escalation.decider import EscalationDecider

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


class SupportAgentPipeline:
    """Full pipeline: classify → retrieve → generate reply → decide escalation."""

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or LLM_API_KEY
        if not self.api_key:
            raise ValueError("API key required. Set LLM_API_KEY in .env or pass api_key.")

        logger.info("Initializing pipeline components…")
        self.classifier = LLMClassifier(api_key=self.api_key)

        self.retriever = ReplyRetriever()
        try:
            self.retriever.load_index()
            logger.info("Loaded retrieval index.")
        except FileNotFoundError:
            logger.warning("No retrieval index found. Building from processed conversations…")
            self._build_retrieval_index()

        self.generator = ReplyGenerator(api_key=self.api_key, retriever=self.retriever)
        self.decider = EscalationDecider(api_key=self.api_key)
        logger.info("Pipeline ready.")

    def _build_retrieval_index(self) -> None:
        """Build retrieval index from processed conversations."""
        if not PROCESSED_CONVERSATIONS_PATH.exists():
            logger.error("No processed conversations at %s. Run preprocessing first.",
                         PROCESSED_CONVERSATIONS_PATH)
            raise FileNotFoundError(f"Run data preprocessing first: {PROCESSED_CONVERSATIONS_PATH}")
        conversations = []
        with open(PROCESSED_CONVERSATIONS_PATH, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    conversations.append(json.loads(line))
        self.retriever.build_index(conversations)

    def process_message(
        self,
        customer_message: str,
        thread_context: list[str] | None = None,
    ) -> dict:
        """Process a single customer message through the full pipeline."""
        # Step 1: Classify intent
        intent_result = self.classifier.classify(customer_message, thread_context)

        # Step 2: Retrieve similar conversations
        try:
            retrieved = self.retriever.retrieve(customer_message)
        except Exception as e:
            logger.warning("Retrieval failed: %s", e)
            retrieved = []

        # Step 3: Generate grounded reply
        reply_result = self.generator.generate(
            customer_message=customer_message,
            intent=intent_result["intent"],
            thread_context=thread_context,
            retrieved_examples=retrieved,
        )

        # Step 4: Make escalation decision
        escalation_result = self.decider.decide(
            customer_message=customer_message,
            intent=intent_result["intent"],
            intent_confidence=intent_result["confidence"],
            thread_context=thread_context,
        )

        return {
            "customer_message": customer_message,
            "thread_context": thread_context or [],
            "intent": intent_result["intent"],
            "intent_confidence": intent_result["confidence"],
            "intent_reasoning": intent_result["reasoning"],
            "reply_text": reply_result["reply_text"],
            "grounding_sources": reply_result["grounding_sources"],
            "reply_confidence": reply_result["confidence"],
            "escalation_decision": escalation_result["decision"],
            "escalation_reason": escalation_result["reason"],
            "escalation_confidence": escalation_result["confidence"],
        }

    def process_batch(self, messages: list[dict], progress: bool = True) -> list[dict]:
        """Process a batch of messages."""
        from tqdm import tqdm
        results = []
        iterator = tqdm(messages, desc="Processing messages") if progress else messages
        for msg in iterator:
            result = self.process_message(
                customer_message=msg.get("customer_message", ""),
                thread_context=msg.get("thread_context"),
            )
            # Carry through any gold labels for evaluation
            for key in ("id", "tweet_id", "gold_intent", "gold_escalation",
                        "gold_escalation_reason", "brand_reply"):
                if key in msg:
                    result[key] = msg[key]
            results.append(result)
            import time
            time.sleep(1.0)
        return results


def pretty_print_result(result: dict) -> None:
    """Pretty print a pipeline result."""
    print("\n" + "=" * 60)
    print(f"[Customer]: {result['customer_message']}")
    print(f"\n[Intent]: {result['intent']} (confidence: {result['intent_confidence']:.2f})")
    print(f"  Reasoning: {result['intent_reasoning']}")
    print(f"\n[Reply]: {result['reply_text']}")
    print(f"\n[Escalation]: {result['escalation_decision'].upper()}")
    print(f"  Reason: {result['escalation_reason']}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SpotifyCares AI Support Agent")
    parser.add_argument("--input", "-i", type=str, help="Single message to process")
    parser.add_argument("--batch", "-b", type=str, help="JSONL file of messages to process")
    parser.add_argument("--output", "-o", type=str, help="Output JSONL file for batch results")
    args = parser.parse_args()

    if not args.input and not args.batch:
        parser.print_help()
        sys.exit(1)

    pipeline = SupportAgentPipeline()

    if args.input:
        result = pipeline.process_message(args.input)
        pretty_print_result(result)
    elif args.batch:
        messages = []
        with open(args.batch, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    messages.append(json.loads(line))

        results = pipeline.process_batch(messages)

        output_path = args.output or "pipeline_output.jsonl"
        with open(output_path, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"\nSaved {len(results)} results to {output_path}")
