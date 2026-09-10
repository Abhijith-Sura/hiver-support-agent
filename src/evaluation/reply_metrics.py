"""
Reply quality metrics: ROUGE-L, BLEU, response length ratio, and optional BERTScore.
"""
import logging
from collections import Counter

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _simple_bleu(reference: str, hypothesis: str, max_n: int = 4) -> float:
    """Compute a simple BLEU-like score (unigram to n-gram precision)."""
    ref_tokens = reference.lower().split()
    hyp_tokens = hypothesis.lower().split()

    if not hyp_tokens or not ref_tokens:
        return 0.0

    scores = []
    for n in range(1, min(max_n + 1, len(hyp_tokens) + 1)):
        ref_ngrams = Counter(tuple(ref_tokens[i:i + n]) for i in range(len(ref_tokens) - n + 1))
        hyp_ngrams = Counter(tuple(hyp_tokens[i:i + n]) for i in range(len(hyp_tokens) - n + 1))
        matches = sum((hyp_ngrams & ref_ngrams).values())
        total = sum(hyp_ngrams.values())
        if total == 0:
            scores.append(0.0)
        else:
            scores.append(matches / total)

    if not scores or all(s == 0 for s in scores):
        return 0.0

    # Geometric mean
    import math
    log_avg = sum(math.log(s + 1e-10) for s in scores) / len(scores)
    return min(math.exp(log_avg), 1.0)


def compute_reply_metrics(
    generated_replies: list[str],
    reference_replies: list[str],
) -> dict:
    """Compute automated reply quality metrics."""
    assert len(generated_replies) == len(reference_replies), \
        f"Length mismatch: {len(generated_replies)} vs {len(reference_replies)}"

    n = len(generated_replies)
    if n == 0:
        return {"error": "No examples to evaluate"}

    # --- ROUGE-L ---
    rouge_scores = []
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        for gen, ref in zip(generated_replies, reference_replies):
            if gen and ref:
                score = scorer.score(ref, gen)
                rouge_scores.append(score["rougeL"].fmeasure)
            else:
                rouge_scores.append(0.0)
    except ImportError:
        logger.warning("rouge_score not available. Skipping ROUGE-L.")
        rouge_scores = [0.0] * n

    # --- Simple BLEU ---
    bleu_scores = []
    for gen, ref in zip(generated_replies, reference_replies):
        if gen and ref:
            bleu_scores.append(_simple_bleu(ref, gen))
        else:
            bleu_scores.append(0.0)

    # --- Response length ratio ---
    length_ratios = []
    for gen, ref in zip(generated_replies, reference_replies):
        if ref and gen:
            length_ratios.append(len(gen.split()) / max(len(ref.split()), 1))
        else:
            length_ratios.append(0.0)

    # --- BERTScore (optional) ---
    bert_scores = None
    try:
        from bert_score import score as bert_score_fn
        P, R, F1 = bert_score_fn(
            generated_replies, reference_replies,
            lang="en", verbose=False, rescale_with_baseline=True,
        )
        bert_scores = {
            "precision": round(float(P.mean()), 4),
            "recall": round(float(R.mean()), 4),
            "f1": round(float(F1.mean()), 4),
        }
    except (ImportError, Exception) as e:
        logger.warning("BERTScore not available: %s", e)

    metrics = {
        "rouge_l": round(sum(rouge_scores) / n, 4),
        "bleu": round(sum(bleu_scores) / n, 4),
        "avg_length_ratio": round(sum(length_ratios) / n, 4),
        "avg_generated_length": round(sum(len(g.split()) for g in generated_replies) / n, 1),
        "avg_reference_length": round(sum(len(r.split()) for r in reference_replies) / n, 1),
        "n_samples": n,
    }
    if bert_scores:
        metrics["bert_score"] = bert_scores

    return metrics
