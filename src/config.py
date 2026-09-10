"""
Centralized configuration for the Hiver Support Agent.
All paths, model settings, and thresholds in one place.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
GOLDEN_SET_DIR = PROJECT_ROOT / "evaluation" / "golden_set"
RESULTS_DIR = PROJECT_ROOT / "evaluation" / "results"

RAW_CSV_PATH = DATA_RAW_DIR / "twcs.csv"
PROCESSED_CONVERSATIONS_PATH = DATA_PROCESSED_DIR / "spotify_conversations.jsonl"
GOLDEN_SET_PATH = GOLDEN_SET_DIR / "golden_set.jsonl"
EMBEDDINGS_INDEX_PATH = DATA_PROCESSED_DIR / "reply_embeddings.npz"

# ── Brand ──────────────────────────────────────────────────────────────
BRAND_AUTHOR_ID = "SpotifyCares"

# ── LLM Settings (Groq) ───────────────────────────────────────────────
LLM_API_KEY = os.getenv("GROQ_API_KEY", "")
LLM_MODEL = "openai/gpt-oss-120b"       # works with JSON mode directly on Groq
LLM_JUDGE_MODEL = "openai/gpt-oss-120b"

# ── Intent Classification ─────────────────────────────────────────────
INTENT_LABELS = [
    "account_access",
    "playback_issue",
    "billing",
    "premium_features",
    "playlist_library",
    "device_sync",
    "family_plan",
    "bug_report",
    "feedback_request",
    "other",
]

# ── Escalation Thresholds ─────────────────────────────────────────────
ESCALATION_CONFIDENCE_THRESHOLD = 0.7    # below this → escalate
AUTO_HANDLE_CONFIDENCE_THRESHOLD = 0.85  # above this → auto-handle

# Intents that almost always need human intervention
ESCALATION_INTENTS = {"billing", "account_access"}

# ── Retrieval Settings ─────────────────────────────────────────────────
RETRIEVER_TOP_K = 5
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# ── Evaluation Settings ───────────────────────────────────────────────
GOLDEN_SET_SIZE = 200
LLM_JUDGE_DIMENSIONS = [
    "relevance",
    "accuracy",
    "tone",
    "helpfulness",
    "completeness",
]

# ── Subsample Size ─────────────────────────────────────────────────────
MAX_CONVERSATIONS = 5000  # work with a manageable subsample
