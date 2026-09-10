"""
Unit tests for the support agent pipeline.
Run with: pytest tests/ -v
"""
import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock


# ── Config Tests ───────────────────────────────────────────────────────
class TestConfig:
    def test_intent_labels_exist(self):
        from src.config import INTENT_LABELS
        assert len(INTENT_LABELS) == 10
        assert "account_access" in INTENT_LABELS
        assert "other" in INTENT_LABELS

    def test_paths_are_pathlib(self):
        from src.config import PROJECT_ROOT, DATA_RAW_DIR, GOLDEN_SET_PATH
        assert isinstance(PROJECT_ROOT, Path)
        assert isinstance(DATA_RAW_DIR, Path)
        assert isinstance(GOLDEN_SET_PATH, Path)


# ── Taxonomy Tests ─────────────────────────────────────────────────────
class TestTaxonomy:
    def test_all_intents_have_definitions(self):
        from src.intent.taxonomy import INTENT_DEFINITIONS
        from src.config import INTENT_LABELS
        for label in INTENT_LABELS:
            assert label in INTENT_DEFINITIONS, f"Missing definition for {label}"

    def test_definitions_have_examples(self):
        from src.intent.taxonomy import INTENT_DEFINITIONS
        for intent, defn in INTENT_DEFINITIONS.items():
            assert "description" in defn, f"{intent} missing description"
            assert "examples" in defn, f"{intent} missing examples"
            assert len(defn["examples"]) >= 2, f"{intent} needs ≥2 examples"

    def test_taxonomy_prompt_format(self):
        from src.intent.taxonomy import get_taxonomy_prompt
        prompt = get_taxonomy_prompt()
        assert "account_access" in prompt
        assert "playback_issue" in prompt
        assert len(prompt) > 100

    def test_few_shot_examples(self):
        from src.intent.taxonomy import get_few_shot_examples
        examples = get_few_shot_examples()
        assert "Customer:" in examples
        assert "Classification:" in examples


# ── Classifier Tests ───────────────────────────────────────────────────
class TestRandomBaseline:
    def test_classify_returns_valid_format(self):
        from src.intent.classifier import RandomBaseline
        clf = RandomBaseline()
        result = clf.classify("test message")
        assert "intent" in result
        assert "confidence" in result
        assert "reasoning" in result
        assert result["intent"] in clf.intents

    def test_fit_updates_distribution(self):
        from src.intent.classifier import RandomBaseline
        clf = RandomBaseline()
        clf.fit(["a", "b", "c"], ["billing", "billing", "other"])
        assert "billing" in clf.intents
        assert clf.weights[clf.intents.index("billing")] > clf.weights[clf.intents.index("other")]

    def test_batch_classify(self):
        from src.intent.classifier import RandomBaseline
        clf = RandomBaseline()
        results = clf.classify_batch([
            {"customer_message": "test 1"},
            {"customer_message": "test 2"},
        ])
        assert len(results) == 2


class TestTfidfBaseline:
    def test_fit_and_classify(self):
        from src.intent.classifier import TfidfBaseline
        clf = TfidfBaseline()
        messages = [
            "I can't log in to my account",
            "songs won't play",
            "I was charged twice",
            "can't access my account",
            "music stops randomly",
            "refund my money",
        ]
        labels = [
            "account_access", "playback_issue", "billing",
            "account_access", "playback_issue", "billing",
        ]
        clf.fit(messages, labels)
        result = clf.classify("help me log in")
        assert result["intent"] in ["account_access", "playback_issue", "billing"]
        assert 0 <= result["confidence"] <= 1

    def test_classify_before_fit_raises(self):
        from src.intent.classifier import TfidfBaseline
        clf = TfidfBaseline()
        with pytest.raises(RuntimeError):
            clf.classify("test")


# ── Preprocess Tests ───────────────────────────────────────────────────
class TestPreprocess:
    def test_clean_text(self):
        from src.data.preprocess import clean_text
        assert clean_text("@SpotifyCares my music stopped") == "my music stopped"
        assert clean_text("@user @brand hello  world") == "hello world"
        assert clean_text("") == ""
        assert clean_text(None) == ""

    def test_clean_text_preserves_emojis(self):
        from src.data.preprocess import clean_text
        result = clean_text("@brand love spotify 🎵")
        assert "🎵" in result


# ── Sample Tests ───────────────────────────────────────────────────────
class TestSample:
    def test_rough_classify(self):
        from src.data.sample import rough_classify
        assert rough_classify("I can't log in") == "account_access"
        assert rough_classify("charged twice") == "billing"
        assert rough_classify("hello there") == "other"

    def test_rough_classify_returns_valid_intent(self):
        from src.data.sample import rough_classify, KEYWORD_BUCKETS
        result = rough_classify("my playlist is gone")
        valid = list(KEYWORD_BUCKETS.keys()) + ["other"]
        assert result in valid


# ── Escalation Tests ───────────────────────────────────────────────────
class TestEscalation:
    def test_low_confidence_escalates(self):
        from src.escalation.decider import EscalationDecider
        decider = EscalationDecider(api_key=None)
        result = decider.decide("test", "other", 0.3)
        assert result["decision"] == "escalate"

    def test_billing_escalates(self):
        from src.escalation.decider import EscalationDecider
        decider = EscalationDecider(api_key=None)
        result = decider.decide("refund please", "billing", 0.9)
        assert result["decision"] == "escalate"

    def test_high_confidence_safe_intent_auto_handles(self):
        from src.escalation.decider import EscalationDecider
        decider = EscalationDecider(api_key=None)
        result = decider.decide("song won't play", "playback_issue", 0.95)
        assert result["decision"] == "auto_handle"

    def test_keyword_triggers_escalation(self):
        from src.escalation.decider import EscalationDecider
        decider = EscalationDecider(api_key=None)
        result = decider.decide("I want to talk to a human agent", "other", 0.9)
        assert result["decision"] == "escalate"

    def test_long_thread_escalates(self):
        from src.escalation.decider import EscalationDecider
        decider = EscalationDecider(api_key=None)
        ctx = ["msg1", "msg2", "msg3"]
        result = decider.decide("still broken", "playback_issue", 0.9, thread_context=ctx)
        assert result["decision"] == "escalate"


# ── Metrics Tests ──────────────────────────────────────────────────────
class TestMetrics:
    def test_intent_metrics(self):
        from src.evaluation.intent_metrics import compute_intent_metrics
        y_true = ["billing", "billing", "other", "other"]
        y_pred = ["billing", "other", "other", "other"]
        metrics = compute_intent_metrics(y_true, y_pred, labels=["billing", "other"])
        assert 0 <= metrics["accuracy"] <= 1
        assert 0 <= metrics["macro_f1"] <= 1
        assert metrics["n_samples"] == 4

    def test_escalation_metrics(self):
        from src.evaluation.escalation_metrics import compute_escalation_metrics
        y_true = ["escalate", "escalate", "auto_handle", "auto_handle"]
        y_pred = ["escalate", "auto_handle", "auto_handle", "escalate"]
        metrics = compute_escalation_metrics(y_true, y_pred)
        assert metrics["precision"] == 0.5
        assert metrics["recall"] == 0.5
        assert metrics["false_negative_rate"] == 0.5

    def test_reply_metrics(self):
        from src.evaluation.reply_metrics import compute_reply_metrics
        gen = ["Hello, how can I help?", "Try restarting the app"]
        ref = ["Hi there! How can I assist?", "Please restart the application"]
        metrics = compute_reply_metrics(gen, ref)
        assert "rouge_l" in metrics
        assert "bleu" in metrics
        assert metrics["n_samples"] == 2


# ── Pipeline & Generator Integration Tests ─────────────────────────────
class TestPipelineIntegration:
    def test_pipeline_process_message_mocked(self):
        from src.pipeline import SupportAgentPipeline

        with patch.object(SupportAgentPipeline, "_build_retrieval_index"):
            with patch("src.pipeline.ReplyRetriever") as MockRetriever:
                with patch("src.pipeline.LLMClassifier") as MockClassifier:
                    with patch("src.pipeline.ReplyGenerator") as MockGenerator:
                        with patch("src.pipeline.EscalationDecider") as MockDecider:
                            mock_clf_inst = MockClassifier.return_value
                            mock_clf_inst.classify.return_value = {
                                "intent": "playback_issue",
                                "confidence": 0.92,
                                "reasoning": "User songs keep stopping.",
                            }

                            mock_ret_inst = MockRetriever.return_value
                            mock_ret_inst.retrieve.return_value = [
                                {"customer_message": "test", "brand_reply": "reply", "similarity_score": 0.85}
                            ]

                            mock_gen_inst = MockGenerator.return_value
                            mock_gen_inst.generate.return_value = {
                                "reply_text": "Restart your app! /AI",
                                "grounding_sources": ["Example 1"],
                                "confidence": 0.9,
                            }

                            mock_dec_inst = MockDecider.return_value
                            mock_dec_inst.decide.return_value = {
                                "decision": "auto_handle",
                                "reason": "High confidence safe issue.",
                                "confidence": 0.88,
                            }

                            pipeline = SupportAgentPipeline(api_key="mock_key")
                            result = pipeline.process_message("My music keeps pausing")

                            assert result["intent"] == "playback_issue"
                            assert result["reply_text"] == "Restart your app! /AI"
                            assert result["escalation_decision"] == "auto_handle"
                            assert result["intent_confidence"] == 0.92

    def test_reply_generator_prompt_building(self):
        from src.reply.generator import ReplyGenerator
        prompt = ReplyGenerator._build_prompt(
            message="My music stopped",
            intent="playback_issue",
            thread_context=["[customer] Hello"],
            retrieved_examples=[{
                "customer_message": "won't play",
                "brand_reply": "Try reinstalling /AI",
                "similarity_score": 0.88,
            }],
        )
        assert "Classified intent: playback_issue" in prompt
        assert "Similar past conversations" in prompt
        assert "Try reinstalling /AI" in prompt
        assert "My music stopped" in prompt


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

