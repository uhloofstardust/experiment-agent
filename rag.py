"""
rag.py
------
Minimal local RAG: chunk the markdown files in knowledge/, embed chunks with
TF-IDF (no external embedding API needed), and retrieve by cosine similarity.
"""

from __future__ import annotations

import glob
import os
from typing import List, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

KNOWLEDGE_DIR = os.path.join(os.path.dirname(__file__), "knowledge")


def _chunk_text(text: str, source: str) -> List[str]:
    """Split a markdown doc into paragraph-level chunks."""
    raw_chunks = [c.strip() for c in text.split("\n\n") if c.strip()]
    return raw_chunks


class KnowledgeBase:
    def __init__(self, knowledge_dir: str = KNOWLEDGE_DIR):
        self.chunks: List[str] = []
        self.sources: List[str] = []

        for path in sorted(glob.glob(os.path.join(knowledge_dir, "*.md"))):
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
            for chunk in _chunk_text(text, os.path.basename(path)):
                self.chunks.append(chunk)
                self.sources.append(os.path.basename(path))

        if not self.chunks:
            raise RuntimeError(f"No knowledge chunks found in {knowledge_dir}")

        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.matrix = self.vectorizer.fit_transform(self.chunks)

    def search(self, query: str, top_k: int = 3) -> List[Tuple[str, str, float]]:
        """Return top_k (source, chunk_text, score) tuples for the query."""
        query_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(query_vec, self.matrix)[0]
        ranked = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)[:top_k]
        return [(self.sources[i], self.chunks[i], float(sims[i])) for i in ranked]


def search_knowledge(query: str, top_k: int = 3) -> List[dict]:
    kb = _get_kb()
    results = kb.search(query, top_k=top_k)
    return [{"source": s, "text": t, "score": round(sc, 3)} for s, t, sc in results]


_kb_singleton = None


def _get_kb() -> KnowledgeBase:
    global _kb_singleton
    if _kb_singleton is None:
        _kb_singleton = KnowledgeBase()
    return _kb_singleton
