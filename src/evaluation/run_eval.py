"""
Main evaluation entry point.

Usage:
    python -m src.evaluation.run_eval
    python -m src.evaluation.run_eval --skip-judge
    python -m src.evaluation.run_eval --golden-set path/to/golden.jsonl
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

from tqdm import tqdm

from src.config import GOLDEN_SET_PATH, RESULTS_DIR, LLM_API_KEY
from src.evaluation.intent_metrics import compute_intent_metrics, plot_confusion_matrix
from src.evaluation.reply_metrics import compute_reply_metrics
from src.evaluation.escalation_metrics import compute_escalation_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_golden_set(path: Path) -> list[dict]:
    """Load golden evaluation set from JSONL."""
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                ex = json.loads(line)
                # Only include labelled examples
                if ex.get("gold_intent"):
                    examples.append(ex)
    logger.info("Loaded %d labelled golden set examples from %s", len(examples), path)
    return examples


def save_results(results: dict, filename: str, output_dir: Path) -> Path:
    """Save results as JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    logger.info("Saved results to %s", path)
    return path


def print_summary(intent_metrics: dict, reply_metrics: dict,
                   escalation_metrics: dict, judge_summary: dict | None = None) -> None:
    """Print a formatted summary of all evaluation results."""
    print("\n" + "=" * 70)
    print("                    EVALUATION RESULTS SUMMARY")
    print("=" * 70)

    print("\n[INTENT CLASSIFICATION]")
    print(f"   Accuracy:     {intent_metrics['accuracy']:.1%}")
    print(f"   Macro-F1:     {intent_metrics['macro_f1']:.4f}")
    print(f"   Weighted-F1:  {intent_metrics['weighted_f1']:.4f}")
    print(f"   Samples:      {intent_metrics['n_samples']}")

    print("\n[REPLY QUALITY]")
    print(f"   ROUGE-L:      {reply_metrics['rouge_l']:.4f}")
    print(f"   BLEU:         {reply_metrics['bleu']:.4f}")
    print(f"   Avg length ratio: {reply_metrics['avg_length_ratio']:.2f}")
    if "bert_score" in reply_metrics:
        print(f"   BERTScore F1: {reply_metrics['bert_score']['f1']:.4f}")

    print("\n[ESCALATION DECISIONS]")
    print(f"   Accuracy:     {escalation_metrics['accuracy']:.1%}")
    print(f"   Precision:    {escalation_metrics['precision']:.4f}")
    print(f"   Recall:       {escalation_metrics['recall']:.4f}")
    print(f"   F1:           {escalation_metrics['f1']:.4f}")
    print(f"   FN Rate:      {escalation_metrics['false_negative_rate']:.1%}")

    if judge_summary:
        print("\n[LLM JUDGE (Quality Scoring)]")
        print(f"   Avg Overall:  {judge_summary.get('avg_overall', 0):.2f}/5.0")
        for dim, score in judge_summary.get("avg_per_dimension", {}).items():
            print(f"   {dim:15s}: {score:.2f}/5.0")

    print("\n" + "=" * 70)


def run_evaluation(
    golden_set_path: Path | None = None,
    output_dir: Path | None = None,
    sample_size: int | None = None,
    use_cached: bool = False,
    skip_judge: bool = False,
    skip_baselines: bool = False,
) -> dict:
    """Run the full evaluation pipeline."""
    golden_set_path = golden_set_path or GOLDEN_SET_PATH
    output_dir = output_dir or RESULTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load golden set
    golden = load_golden_set(golden_set_path)
    if not golden:
        logger.error("No labelled examples in golden set. Run labelling first.")
        sys.exit(1)

    if sample_size and sample_size > 0:
        logger.info("Using sample of %d examples (out of %d)", sample_size, len(golden))
        golden = golden[:sample_size]

    predictions_cache = output_dir / "predictions.json"
    if use_cached and predictions_cache.exists():
        logger.info("Loading cached predictions from %s…", predictions_cache)
        with open(predictions_cache, "r", encoding="utf-8") as f:
            predictions = json.load(f)
        if sample_size and len(predictions) > sample_size:
            predictions = predictions[:sample_size]
        golden = golden[:len(predictions)]
    else:
        # ── Run pipeline on golden set ──────────────────────────────
        logger.info("Running pipeline on %d golden set examples…", len(golden))
        from src.pipeline import SupportAgentPipeline
        pipeline = SupportAgentPipeline()
        predictions = pipeline.process_batch(golden, progress=True)
        # Save raw predictions
        save_results(predictions, "predictions.json", output_dir)

    # ── Intent Metrics ──────────────────────────────────────────────
    y_true_intent = [ex["gold_intent"] for ex in golden]
    y_pred_intent = [pred["intent"] for pred in predictions]
    intent_metrics = compute_intent_metrics(y_true_intent, y_pred_intent)
    save_results(intent_metrics, "intent_metrics.json", output_dir)

    # Confusion matrix
    try:
        plot_confusion_matrix(y_true_intent, y_pred_intent,
                              save_path=output_dir / "confusion_matrix.png")
    except Exception as e:
        logger.warning("Failed to plot confusion matrix: %s", e)

    # ── Baselines ───────────────────────────────────────────────────
    baseline_results = {}
    if not skip_baselines:
        from src.intent.classifier import RandomBaseline, TfidfBaseline

        # Random baseline
        random_clf = RandomBaseline()
        random_clf.fit(
            [ex["customer_message"] for ex in golden],
            y_true_intent,
        )
        y_pred_random = [random_clf.classify(ex["customer_message"])["intent"] for ex in golden]
        baseline_results["random"] = compute_intent_metrics(y_true_intent, y_pred_random)

        # TF-IDF baseline (train on golden set itself — overfitting is expected and documented)
        tfidf_clf = TfidfBaseline()
        tfidf_clf.fit(
            [ex["customer_message"] for ex in golden],
            y_true_intent,
        )
        y_pred_tfidf = [tfidf_clf.classify(ex["customer_message"])["intent"] for ex in golden]
        baseline_results["tfidf"] = compute_intent_metrics(y_true_intent, y_pred_tfidf)

        save_results(baseline_results, "baseline_metrics.json", output_dir)

    # ── Reply Metrics ───────────────────────────────────────────────
    generated_replies = [pred["reply_text"] for pred in predictions]
    reference_replies = [ex.get("brand_reply", "") for ex in golden]
    reply_metrics = compute_reply_metrics(generated_replies, reference_replies)
    save_results(reply_metrics, "reply_metrics.json", output_dir)

    # ── Escalation Metrics ──────────────────────────────────────────
    y_true_esc = [ex.get("gold_escalation", "auto_handle") for ex in golden]
    y_pred_esc = [pred["escalation_decision"] for pred in predictions]
    escalation_metrics = compute_escalation_metrics(y_true_esc, y_pred_esc)
    save_results(escalation_metrics, "escalation_metrics.json", output_dir)

    # ── LLM Judge ───────────────────────────────────────────────────
    judge_summary = None
    judge_cache = output_dir / "llm_judge_summary.json"
    if not skip_judge:
        if use_cached and judge_cache.exists():
            logger.info("Loading cached LLM judge summary from %s…", judge_cache)
            with open(judge_cache, "r", encoding="utf-8") as f:
                judge_summary = json.load(f)
        else:
            logger.info("Running LLM judge on examples…")
            from src.evaluation.llm_judge import LLMJudge
            judge = LLMJudge()

            # Judge a subset (50 examples max to save API calls)
            judge_subset = predictions[:50]
            for i, pred in enumerate(judge_subset):
                pred["brand_reply"] = golden[i].get("brand_reply", "")

            judge_results = judge.judge_batch(judge_subset)
            save_results(judge_results, "llm_judge_results.json", output_dir)

            # Compute summary
            avg_overall = sum(r["overall_score"] for r in judge_results) / len(judge_results)
            avg_per_dim = {}
            from src.config import LLM_JUDGE_DIMENSIONS
            for dim in LLM_JUDGE_DIMENSIONS:
                avg_per_dim[dim] = sum(r["scores"].get(dim, 3) for r in judge_results) / len(judge_results)

            judge_summary = {
                "avg_overall": round(avg_overall, 2),
                "avg_per_dimension": {d: round(v, 2) for d, v in avg_per_dim.items()},
                "n_judged": len(judge_results),
            }
            save_results(judge_summary, "llm_judge_summary.json", output_dir)

    # ── Print Summary ───────────────────────────────────────────────
    print_summary(intent_metrics, reply_metrics, escalation_metrics, judge_summary)

    # Print baseline comparison
    if baseline_results:
        print("\n[BASELINE COMPARISON (Intent Classification)]")
        print(f"   {'Method':<20s} {'Accuracy':>10s} {'Macro-F1':>10s}")
        print(f"   {'─' * 40}")
        print(f"   {'Random':<20s} {baseline_results['random']['accuracy']:>10.1%} "
              f"{baseline_results['random']['macro_f1']:>10.4f}")
        print(f"   {'TF-IDF + LogReg':<20s} {baseline_results['tfidf']['accuracy']:>10.1%} "
              f"{baseline_results['tfidf']['macro_f1']:>10.4f}")
        print(f"   {'LLM Agent (ours)':<20s} {intent_metrics['accuracy']:>10.1%} "
              f"{intent_metrics['macro_f1']:>10.4f}")
        print()

    all_results = {
        "intent_metrics": intent_metrics,
        "reply_metrics": reply_metrics,
        "escalation_metrics": escalation_metrics,
        "baseline_metrics": baseline_results,
        "judge_summary": judge_summary,
    }
    save_results(all_results, "all_results.json", output_dir)
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run evaluation harness")
    parser.add_argument("--golden-set", type=str, default=None,
                        help="Path to golden set JSONL")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory for evaluation results")
    parser.add_argument("--sample-size", type=int, default=None,
                        help="Number of samples to evaluate (default: all)")
    parser.add_argument("--use-cached", action="store_true",
                        help="Use cached predictions if available")
    parser.add_argument("--skip-judge", action="store_true",
                        help="Skip LLM judge (saves API calls)")
    parser.add_argument("--skip-baselines", action="store_true",
                        help="Skip baseline comparisons")
    args = parser.parse_args()

    run_evaluation(
        golden_set_path=Path(args.golden_set) if args.golden_set else None,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        sample_size=args.sample_size,
        use_cached=args.use_cached,
        skip_judge=args.skip_judge,
        skip_baselines=args.skip_baselines,
    )
