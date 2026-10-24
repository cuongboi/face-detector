"""Qdrant wrapper. Several vectors per person (one per enrollment pose), grouped at search time."""

import uuid
from dataclasses import dataclass

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    FilterSelector,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    QueryRequest,
    VectorParams,
)


@dataclass
class Match:
    person_id: int
    name: str
    employee_id: str
    score: float
    pose: str | None = None


class VectorStore:
    def __init__(self, url: str, collection: str, dim: int = 512):
        self.client = QdrantClient(location=url) if url == ":memory:" else QdrantClient(url=url)
        self.collection = collection
        self.dim = dim
        self.ensure_collection()

    def ensure_collection(self) -> None:
        if self.client.collection_exists(self.collection):
            return
        self.client.create_collection(
            self.collection, vectors_config=VectorParams(size=self.dim, distance=Distance.COSINE)
        )
        self.client.create_payload_index(
            self.collection, "person_id", field_schema=PayloadSchemaType.INTEGER
        )

    def add_faces(
        self, person_id: int, name: str, employee_id: str, faces: dict[str, np.ndarray]
    ) -> None:
        """faces: pose name -> L2-normalized embedding."""
        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=emb.astype(float).tolist(),
                payload={
                    "person_id": person_id, "name": name,
                    "employee_id": employee_id, "pose": pose,
                },
            )
            for pose, emb in faces.items()
        ]
        self.client.upsert(self.collection, points=points, wait=True)

    def search(self, embedding: np.ndarray, top_k: int = 10) -> Match | None:
        """Best person: max similarity over that person's pose vectors."""
        return self.search_batch(embedding[None], top_k)[0]

    def search_batch(self, embeddings: np.ndarray, top_k: int = 10) -> list[Match | None]:
        """`search` for every face of a frame in one round trip."""
        if len(embeddings) == 0:
            return []
        requests = [
            QueryRequest(query=e.astype(float).tolist(), limit=top_k, with_payload=True)
            for e in embeddings
        ]
        responses = self.client.query_batch_points(self.collection, requests)
        return [self._best(r.points) for r in responses]

    @staticmethod
    def _best(hits) -> Match | None:
        best: Match | None = None
        for hit in hits:
            p = hit.payload or {}
            if best is None or hit.score > best.score:
                best = Match(p["person_id"], p["name"], p["employee_id"], hit.score, p.get("pose"))
        return best

    def delete_person(self, person_id: int) -> None:
        self.client.delete(
            self.collection,
            points_selector=FilterSelector(
                filter=Filter(
                    must=[FieldCondition(key="person_id", match=MatchValue(value=person_id))]
                )
            ),
            wait=True,
        )

    def count(self) -> int:
        return self.client.count(self.collection, exact=True).count
