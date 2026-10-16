"""Semantic retrieval over a local JSON collection of reference patent records.

This prototype does not fetch patents from the internet. Add real, sourced records to
 data/reference_patents.json before treating results as prior-art research.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Module-level cache: the SentenceTransformer is loaded once per process,
# not once per ManagerAgent construction.
_MODEL_CACHE: dict[str, SentenceTransformer] = {}


def _get_embedder(name: str, device: str) -> SentenceTransformer:
    key = f"{name}|{device}"
    if key not in _MODEL_CACHE:
        _MODEL_CACHE[key] = SentenceTransformer(name, device=device)
    return _MODEL_CACHE[key]


class PriorArtRetrievalAgent:
    def __init__(
        self,
        records_path: str = "data/reference_patents.json",
        embedding_model: str = "all-MiniLM-L6-v2",
        top_k: int = 10,
    ):
        self.records_path = Path(records_path)
        self.embedding_model_name = embedding_model
        self.top_k = top_k
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = _get_embedder(embedding_model, device)
        self.device = device
        self._records: list[dict] | None = None
        self._record_embeddings: np.ndarray | None = None
        self._records_mtime: float | None = None

    def warmup(self) -> None:
        """Pre-load records and embeddings. Safe to call from a background thread
        while the first LLM stage is running."""
        records = self._load_records()
        self._embed_records(records)

    def _load_records(self) -> list[dict]:
        mtime = self.records_path.stat().st_mtime if self.records_path.exists() else None
        if self._records is not None and mtime == self._records_mtime:
            return self._records
        if not self.records_path.exists():
            raise FileNotFoundError(
                f"Reference collection not found: {self.records_path}. Create it using the README schema."
            )
        with self.records_path.open("r", encoding="utf-8") as file:
            records = json.load(file)
        if not isinstance(records, list):
            raise ValueError("Reference collection must be a JSON list of records.")
        normalized = []
        for index, item in enumerate(records):
            if not isinstance(item, dict):
                continue
            title = str(item.get("title", "")).strip()
            abstract = str(item.get("abstract", "")).strip()
            claims = str(item.get("claims", "")).strip()
            if not (title or abstract or claims):
                continue
            normalized.append({
                "patent_id": str(item.get("patent_id", f"UNKNOWN-{index + 1}")),
                "title": title,
                "abstract": abstract,
                "claims": claims,
                "url": str(item.get("url", "")),
                "source": str(item.get("source", "local reference collection")),
            })
        self._records = normalized
        self._records_mtime = mtime
        # Invalidate embeddings so they are recomputed for the new data.
        self._record_embeddings = None
        return normalized

    @staticmethod
    def _document_text(record: dict) -> str:
        parts = []
        if record.get("title"):
            parts.append(f"Title: {record['title']}")
        if record.get("abstract"):
            parts.append(f"Abstract: {record['abstract']}")
        if record.get("claims"):
            parts.append(f"Claims: {record['claims']}")
        return "\n".join(parts)

    def _embed_records(self, records: list[dict]) -> np.ndarray:
        if self._record_embeddings is not None:
            return self._record_embeddings
        texts = [self._document_text(record) for record in records]
        if not texts:
            self._record_embeddings = np.empty((0, 0), dtype=np.float32)
            return self._record_embeddings

        # Persist embeddings; recompute only if the data or model name changes.
        self.records_path.parent.mkdir(parents=True, exist_ok=True)
        cache_key = hashlib.sha256(
            (self.embedding_model_name + "\n" + json.dumps(records, sort_keys=True)).encode("utf-8")
        ).hexdigest()[:20]
        cache_path = self.records_path.parent / f"embeddings_{cache_key}.npy"
        embeddings = None
        if cache_path.exists():
            try:
                embeddings = np.load(cache_path)
                if embeddings.shape[0] != len(texts):
                    embeddings = None  # stale/corrupt cache, recompute
            except Exception:
                embeddings = None
        if embeddings is None:
            embeddings = self.model.encode(
                texts, batch_size=32, show_progress_bar=False,
                convert_to_numpy=True, normalize_embeddings=True,
            )
            np.save(cache_path, embeddings)
        self._record_embeddings = embeddings
        return embeddings

    def search(self, invention_analysis: dict, generated_queries: dict) -> dict:
        records = self._load_records()
        if not records:
            return {"results": [], "searched_records": 0, "message": "The reference collection is empty."}

        raw_queries = generated_queries.get("queries", [])
        if not isinstance(raw_queries, list):
            raw_queries = []
        queries = [str(q).strip() for q in raw_queries if str(q).strip()]

        summary = str(invention_analysis.get("short_search_summary", "")).strip()
        raw_features = invention_analysis.get("technical_features", [])
        if not isinstance(raw_features, list):
            raw_features = []
        features = [str(f).strip() for f in raw_features if str(f).strip()]

        targeted_texts = queries + features
        if not targeted_texts and invention_analysis.get("proposed_solution"):
            targeted_texts = [str(invention_analysis["proposed_solution"]).strip()]
        if not summary and not targeted_texts:
            return {
                "results": [],
                "searched_records": len(records),
                "message": "No usable search queries were generated.",
            }

        search_texts = ([summary] if summary else []) + targeted_texts
        corpus_embeddings = self._embed_records(records)
        query_embeddings = self.model.encode(
            search_texts, batch_size=32, show_progress_bar=False,
            convert_to_numpy=True, normalize_embeddings=True,
        )
        scores = cosine_similarity(query_embeddings, corpus_embeddings)
        # If a broad summary exists, blend it with the strongest targeted match.
        # Otherwise, use the strongest query/feature match so the first query is not dropped.
        if summary and targeted_texts:
            combined_scores = 0.4 * scores[0] + 0.6 * np.max(scores[1:], axis=0)
        elif summary:
            combined_scores = scores[0]
        else:
            combined_scores = np.max(scores, axis=0)
        order = np.argsort(combined_scores)[::-1][: self.top_k]

        results = []
        for index in order:
            record = records[int(index)]
            results.append({
                **record,
                "similarity_score": round(float(combined_scores[index]) * 100, 2),
                "matched_query_scores": [round(float(score) * 100, 2) for score in scores[:, int(index)]],
            })
        return {
            "results": results,
            "searched_records": len(records),
            "embedding_model": self.embedding_model_name,
            "device": self.device,
            "message": "Scores are retrieval-ranking signals, not legal novelty probabilities.",
        }