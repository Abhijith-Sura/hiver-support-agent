"""
LLM-as-judge for reply quality evaluation.
"""
import json
import logging
import time
import numpy as np
from scipy import stats
from src.config import LLM_API_KEY, LLM_JUDGE_DIMENSIONS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

JUDGE_SYSTEM_PROMPT = """You are an expert evaluator of customer support reply quality for SpotifyCares.
Score the generated reply on 5 dimensions (1-5 each):
1. relevance: Does it address the customer's actual issue? (1=off-topic, 5=directly addresses)
2. accuracy: Is info correct, not hallucinated? (1=fabricated, 5=all accurate)
3. tone: Matches SpotifyCares voice (friendly, empathetic)? (1=robotic, 5=perfect match)
4. helpfulness: Moves toward resolution? (1=no action, 5=clear resolution steps)
5. completeness: Covers all aspects? (1=fragment, 5=comprehensive)

Respond JSON: {"scores": {"relevance": N, "accuracy": N, "tone": N, "helpfulness": N, "completeness": N}, "justifications": {"relevance": "...", ...}, "overall_score": N.N}"""


class LLMJudge:
    def __init__(self, api_key: str | None = None):
        from src.llm_client import LLMClient
        self.llm = LLMClient(api_key=api_key)

    def judge_reply(self, customer_message: str, generated_reply: str,
                    reference_reply: str | None = None, intent: str | None = None) -> dict:
        parts = []
        if intent:
            parts.append(f"Intent: {intent}")
        parts.append(f'Customer: "{customer_message}"')
        parts.append(f'Generated reply: "{generated_reply}"')
        if reference_reply:
            parts.append(f'Reference (actual) reply: "{reference_reply}"')
        parts.append("\nScore on all 5 dimensions (1-5).")
        prompt = "\n".join(parts)

        try:
            result = self.llm.generate_json(prompt, JUDGE_SYSTEM_PROMPT, temperature=0.0)
            scores = result.get("scores", {})
            for dim in LLM_JUDGE_DIMENSIONS:
                if dim in scores:
                    scores[dim] = max(1, min(5, int(scores[dim])))
                else:
                    scores[dim] = 3
            overall = sum(scores.values()) / len(scores) if scores else 3.0
            return {"scores": scores, "justifications": result.get("justifications", {}),
                    "overall_score": round(overall, 2)}
        except Exception as e:
            logger.error("Judge failed: %s", e)
            return {"scores": {d: 3 for d in LLM_JUDGE_DIMENSIONS},
                    "justifications": {d: "failed" for d in LLM_JUDGE_DIMENSIONS},
                    "overall_score": 3.0}

    def judge_batch(self, examples: list[dict]) -> list[dict]:
        results = []
        for ex in examples:
            result = self.judge_reply(
                customer_message=ex.get("customer_message", ""),
                generated_reply=ex.get("reply_text", ex.get("generated_reply", "")),
                reference_reply=ex.get("brand_reply", ex.get("reference_reply")),
                intent=ex.get("intent"),
            )
            results.append(result)
            time.sleep(0.3)
        return results

    @staticmethod
    def compute_agreement(llm_scores: list[dict], human_scores: list[dict]) -> dict:
        agreement = {}
        for dim in LLM_JUDGE_DIMENSIONS:
            llm_vals = [s["scores"].get(dim, 3) for s in llm_scores]
            human_vals = [s["scores"].get(dim, 3) for s in human_scores]
            if len(llm_vals) < 3:
                agreement[dim] = {"error": "Too few samples"}
                continue
            try:
                corr, p_value = stats.spearmanr(llm_vals, human_vals)
                spearman = {"correlation": round(float(corr), 4), "p_value": round(float(p_value), 4)}
            except Exception:
                spearman = {"correlation": 0.0, "p_value": 1.0}
            def bin_score(s):
                if s <= 2: return "low"
                elif s <= 3: return "medium"
                else: return "high"
            try:
                from sklearn.metrics import cohen_kappa_score
                kappa = round(float(cohen_kappa_score(
                    [bin_score(v) for v in llm_vals], [bin_score(v) for v in human_vals])), 4)
            except Exception:
                kappa = 0.0
            mae = round(float(np.mean(np.abs(np.array(llm_vals) - np.array(human_vals)))), 4)
            agreement[dim] = {"spearman": spearman, "cohens_kappa": kappa, "mean_absolute_error": mae}
        return agreement
