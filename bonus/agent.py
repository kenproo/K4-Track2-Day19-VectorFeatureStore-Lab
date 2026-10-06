"""Hybrid Memory Agent combining Vector Store (Episodic) + Feast Feature Store (Profile).

Deliverable for Day 19 Bonus Challenge.
"""
from __future__ import annotations

import sys
import uuid
from pathlib import Path
from typing import Any

from feast import FeatureStore
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.embeddings import Embedder


class HybridMemoryAgent:
    """Agent combining episodic memory in Qdrant with stable user profile in Feast."""

    def __init__(
        self,
        feast_repo_path: Path | str | None = None,
        collection_name: str = "episodic_memory",
    ) -> None:
        self.repo_path = Path(feast_repo_path or (ROOT / "app" / "feast_repo"))
        self.fs = FeatureStore(repo_path=str(self.repo_path))
        self.embedder = Embedder()
        self.collection_name = collection_name
        self.client = QdrantClient(":memory:")
        self._point_id = 0

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=self.embedder.dim, distance=Distance.COSINE),
        )

    def remember(self, text: str, user_id: str = "u_001") -> None:
        """Add a new piece of episodic memory for this user."""
        vec = next(self.embedder.embed([text])).tolist()
        self._point_id += 1
        point = PointStruct(
            id=self._point_id,
            vector=vec,
            payload={
                "user_id": user_id,
                "text": text,
            },
        )
        self.client.upsert(
            collection_name=self.collection_name,
            points=[point],
        )

    def recall(self, query: str, user_id: str = "u_001", top_k: int = 3) -> str:
        """Retrieve top-K memories + user profile features and assemble context."""
        # 1. Fetch user profile + velocity features from Feast
        features_to_fetch = [
            "user_profile_features:reading_speed_wpm",
            "user_profile_features:preferred_language",
            "user_profile_features:topic_affinity",
            "query_velocity_features:queries_last_hour",
            "query_velocity_features:distinct_topics_24h",
        ]
        feature_dict: dict[str, Any] = {}
        try:
            res = self.fs.get_online_features(
                features=features_to_fetch,
                entity_rows=[{"user_id": user_id}],
            ).to_dict()
            feature_dict = {k: v[0] for k, v in res.items()}
        except Exception:
            # Fallback if Feast store has missing entity
            feature_dict = {
                "reading_speed_wpm": 200,
                "preferred_language": "vi",
                "topic_affinity": "cloud",
                "queries_last_hour": 5,
                "distinct_topics_24h": 3,
            }

        # 2. Vector search in Qdrant filtered by user_id
        q_vec = next(self.embedder.embed([query])).tolist()
        user_filter = Filter(
            must=[
                FieldCondition(
                    key="user_id",
                    match=MatchValue(value=user_id),
                )
            ]
        )
        hits = self.client.query_points(
            collection_name=self.collection_name,
            query=q_vec,
            query_filter=user_filter,
            limit=top_k,
        ).points

        retrieved_texts = [h.payload["text"] for h in hits]

        # 3. Assemble unified context string
        affinity = feature_dict.get("topic_affinity", "chung")
        speed = feature_dict.get("reading_speed_wpm", 200)
        lang = feature_dict.get("preferred_language", "vi")
        velocity = feature_dict.get("queries_last_hour", 0)

        lines = [
            f"[User Profile] User: {user_id} | Ngôn ngữ: {lang} | Tốc độ đọc: {speed} wpm | Chủ đề quan tâm: {affinity}",
            f"[Recent Activity] Tần suất 1h qua: {velocity} truy vấn",
            f"[Episodic Memories (Top-{len(retrieved_texts)})]:",
        ]
        if retrieved_texts:
            for idx, text in enumerate(retrieved_texts, 1):
                lines.append(f"  {idx}. {text}")
        else:
            lines.append("  (Chưa có ký ức liên quan)")

        return "\n".join(lines)
