"""
Intent classifiers: Random baseline, TF-IDF baseline, and LLM classifier (Groq).
"""
import json
import logging
import random
import time
from collections import Counter

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.config import LLM_API_KEY, INTENT_LABELS
from src.intent.taxonomy import get_taxonomy_prompt, get_few_shot_examples

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


class RandomBaseline:
    """Classify randomly, weighted by observed intent distribution."""

    def __init__(self, intent_distribution: dict[str, float] | None = None):
        if intent_distribution:
            self.intents = list(intent_distribution.keys())
            self.weights = [intent_distribution[i] for i in self.intents]
        else:
            self.intents = INTENT_LABELS
            self.weights = [1.0 / len(self.intents)] * len(self.intents)

    def fit(self, messages: list[str], labels: list[str]) -> "RandomBaseline":
        counts = Counter(labels)
        total = sum(counts.values())
        self.intents = list(counts.keys())
        self.weights = [counts[i] / total for i in self.intents]
        return self

    def classify(self, message: str) -> dict:
        chosen = random.choices(self.intents, weights=self.weights, k=1)[0]
        return {
            "intent": chosen,
            "confidence": round(self.weights[self.intents.index(chosen)], 3),
            "reasoning": "Random baseline — selected by weighted random choice.",
        }

    def classify_batch(self, messages: list[dict]) -> list[dict]:
        return [self.classify(m.get("customer_message", "")) for m in messages]


class TfidfBaseline:
    """TF-IDF + Logistic Regression classifier."""

    def __init__(self):
        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2),
                                       stop_words="english")),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced",
                                        random_state=42)),
        ])
        self._fitted = False

    def fit(self, messages: list[str], labels: list[str]) -> "TfidfBaseline":
        logger.info("Training TF-IDF baseline on %d examples…", len(messages))
        self.pipeline.fit(messages, labels)
        self._fitted = True
        return self

    def classify(self, message: str) -> dict:
        if not self._fitted:
            raise RuntimeError("TfidfBaseline must be fit() before classify().")
        proba = self.pipeline.predict_proba([message])[0]
        classes = self.pipeline.classes_
        best_idx = int(np.argmax(proba))
        return {
            "intent": classes[best_idx],
            "confidence": round(float(proba[best_idx]), 3),
            "reasoning": f"TF-IDF baseline — top features matched '{classes[best_idx]}'.",
        }

    def classify_batch(self, messages: list[dict]) -> list[dict]:
        texts = [m.get("customer_message", "") for m in messages]
        if not self._fitted:
            raise RuntimeError("TfidfBaseline must be fit() before classify_batch().")
        probas = self.pipeline.predict_proba(texts)
        classes = self.pipeline.classes_
        results = []
        for proba in probas:
            best_idx = int(np.argmax(proba))
            results.append({
                "intent": classes[best_idx],
                "confidence": round(float(proba[best_idx]), 3),
                "reasoning": f"TF-IDF baseline — matched '{classes[best_idx]}'.",
            })
        return results


class LLMClassifier:
    """LLM-based intent classifier using Groq API."""

    def __init__(self, api_key: str | None = None):
        from src.llm_client import LLMClient
        self.llm = LLMClient(api_key=api_key)
        self.system_prompt = get_taxonomy_prompt() + "\n" + get_few_shot_examples()

    def classify(self, message: str, thread_context: list[str] | None = None) -> dict:
        message = str(message or "").strip()
        if not message:
            return {
                "intent": "other",
                "confidence": 1.0,
                "reasoning": "Empty message received.",
            }
        user_prompt = self._build_user_prompt(message, thread_context)
        try:
            result = self.llm.generate_json(user_prompt, self.system_prompt, temperature=0.1)
            intent = result.get("intent", "other")
            if intent not in INTENT_LABELS:
                intent = "other"
            return {
                "intent": intent,
                "confidence": float(result.get("confidence", 0.5)),
                "reasoning": result.get("reasoning", ""),
            }
        except Exception as e:
            logger.error("Classification failed: %s", e)
            return {"intent": "other", "confidence": 0.0, "reasoning": "Classification failed."}

    def classify_batch(self, messages: list[dict]) -> list[dict]:
        results = []
        for msg in messages:
            result = self.classify(msg.get("customer_message", ""), msg.get("thread_context"))
            results.append(result)
            time.sleep(0.3)
        return results

    @staticmethod
    def _build_user_prompt(message: str, thread_context: list[str] | None = None) -> str:
        parts = []
        if thread_context:
            parts.append("Conversation history:")
            for ctx in thread_context[-3:]:
                parts.append(f"  {ctx}")
            parts.append("")
        parts.append(f'Customer message to classify: "{message}"')
        parts.append("")
        parts.append('Respond with JSON: {"intent": "...", "confidence": 0.0-1.0, "reasoning": "..."}')
        return "\n".join(parts)


# Alias for backwards compatibility
GeminiClassifier = LLMClassifier

