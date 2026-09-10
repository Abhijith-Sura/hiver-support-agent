"""
Retrieval system for finding similar historical SpotifyCares conversations.
Uses sentence-transformers embeddings + cosine similarity.
"""
import json
import logging
from pathlib import Path

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from src.config import EMBEDDING_MODEL, EMBEDDINGS_INDEX_PATH, RETRIEVER_TOP_K

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


class ReplyRetriever:
    """Retrieve similar historical conversations using sentence embeddings."""

    def __init__(self, embedding_model_name: str | None = None):
        from sentence_transformers import SentenceTransformer
        model_name = embedding_model_name or EMBEDDING_MODEL
        logger.info("Loading embedding model: %s", model_name)
        self.model = SentenceTransformer(model_name)
        self.embeddings: np.ndarray | None = None
        self.conversations: list[dict] = []

    def build_index(self, conversations: list[dict], save_path: Path | None = None) -> None:
        """Encode all customer messages and store embeddings + conversation data."""
        save_path = save_path or EMBEDDINGS_INDEX_PATH
        save_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info("Building retrieval index from %d conversations…", len(conversations))
        messages = [conv["customer_message"] for conv in conversations]
        self.embeddings = self.model.encode(messages, show_progress_bar=True,
                                             batch_size=64, normalize_embeddings=True)
        self.conversations = conversations

        # Save index
        np.savez_compressed(
            save_path,
            embeddings=self.embeddings,
        )
        # Save conversation metadata alongside
        meta_path = save_path.with_suffix(".jsonl")
        with open(meta_path, "w", encoding="utf-8") as f:
            for conv in conversations:
                f.write(json.dumps(conv, ensure_ascii=False) + "\n")

        logger.info("Saved retrieval index to %s (%d entries).", save_path, len(conversations))

    def load_index(self, load_path: Path | None = None) -> None:
        """Load pre-built index from disk."""
        load_path = load_path or EMBEDDINGS_INDEX_PATH
        if not load_path.exists():
            raise FileNotFoundError(f"No index found at {load_path}. Run build_index first.")

        logger.info("Loading retrieval index from %s…", load_path)
        data = np.load(load_path)
        self.embeddings = data["embeddings"]

        meta_path = load_path.with_suffix(".jsonl")
        self.conversations = []
        with open(meta_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.conversations.append(json.loads(line))

        logger.info("Loaded index with %d entries.", len(self.conversations))

    def retrieve(self, query: str, top_k: int | None = None) -> list[dict]:
        """Find top-k most similar historical conversations."""
        top_k = top_k or RETRIEVER_TOP_K
        if self.embeddings is None or not self.conversations:
            raise RuntimeError("Index not loaded. Call build_index or load_index first.")

        query_emb = self.model.encode([query], normalize_embeddings=True)
        similarities = cosine_similarity(query_emb, self.embeddings)[0]

        top_indices = np.argsort(similarities)[::-1][:top_k]
        results = []
        for idx in top_indices:
            conv = self.conversations[idx]
            results.append({
                "customer_message": conv["customer_message"],
                "brand_reply": conv["brand_reply"],
                "thread_context": conv.get("thread_context", []),
                "similarity_score": round(float(similarities[idx]), 4),
            })
        return results
