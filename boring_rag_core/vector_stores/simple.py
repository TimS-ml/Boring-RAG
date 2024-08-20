import json
from enum import Enum
import numpy as np
from typing import List, Dict, Any, Optional
from boring_rag_core.schema import Document
from dataclasses import dataclass, field, asdict

# TODO
from llama_index.core.indices.query.embedding_utils import (
    get_top_k_embeddings,
    get_top_k_embeddings_learner,
    get_top_k_mmr_embeddings,
)

from boring_rag_core.vector_stores.types import VectorStoreQuery, VectorStoreQueryResult


@dataclass
class SimpleVectorStoreData:
    """
    SimpleVectorStoreData can be seen as an abstraction layer 
    that decouples the underlying storage structure from the Document/BaseNode objects of LlamaIndex
    """
    embedding_dict: Dict[str, List[float]] = field(default_factory=dict)
    metadata_dict: Dict[str, Dict[str, Any]] = field(default_factory=dict)


class SimpleVectorStore:
    def __init__(self, data: Optional[SimpleVectorStoreData] = None):
        self.data = data or SimpleVectorStoreData()

    def add(self, nodes: List[Document]) -> List[str]:
        """Add documents to the vector store."""
        node_ids = []
        for node in nodes:
            node_id = node.id_
            embedding = node.embedding
            metadata = node.metadata
            
            if node_id and embedding:
                self.data.embedding_dict[node_id] = embedding
                self.data.metadata_dict[node_id] = metadata
                node_ids.append(node_id)
        
        return node_ids

    def delete(self, node_ids: List[str]) -> None:
        """Delete documents from the vector store."""
        for node_id in node_ids:
            self.data.embedding_dict.pop(node_id, None)
            self.data.metadata_dict.pop(node_id, None)

    def query(self, query: VectorStoreQuery) -> VectorStoreQueryResult:
        """Query the vector store for similar documents."""
        if query.query_embedding is None:
            raise ValueError("Query embedding is required.")

        scores = []
        for node_id, node in self.data.items():
            if node.embedding is not None:
                score = self._cosine_similarity(query.query_embedding, node.embedding)
                scores.append((node_id, score, node))

        scores.sort(key=lambda x: x[1], reverse=True)
        top_k = min(query.similarity_top_k, len(scores))
        top_results = scores[:top_k]

        nodes = [result[2] for result in top_results]
        similarities = [result[1] for result in top_results]
        ids = [result[0] for result in top_results]
        # metadata = self.data.metadata_dict.get(node_id, {})

        return VectorStoreQueryResult(
            nodes=nodes,
            similarities=similarities,
            ids=ids
        )

    # TODO: remove this with self-implemented get_top_k_embeddings
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    # def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
    #     """Calculate cosine similarity between two vectors."""
    #     dot_product = sum(a * b for a, b in zip(vec1, vec2))
    #     magnitude1 = sum(a * a for a in vec1) ** 0.5
    #     magnitude2 = sum(b * b for b in vec2) ** 0.5
    #     return dot_product / (magnitude1 * magnitude2)

    def persist(self, persist_path: str) -> None:
        """Persist the vector store to a file."""
        with open(persist_path, 'w') as f:
            json.dump(asdict(self.data), f)

    @classmethod
    def from_persist_path(cls, persist_path: str) -> 'SimpleVectorStore':
        """Load a vector store from a file."""
        with open(persist_path, 'r') as f:
            data = json.load(f)
        return cls(SimpleVectorStoreData(**data))

    def to_dict(self) -> Dict[str, Any]:
        """Convert the vector store to a dictionary."""
        return asdict(self.data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SimpleVectorStore':
        """Create a vector store from a dictionary."""
        return cls(SimpleVectorStoreData(**data))
