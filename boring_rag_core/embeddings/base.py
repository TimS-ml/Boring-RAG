from pathlib import Path
from abc import ABC, abstractmethod
from typing import (
    List, 
    Optional, 
    Any, 
    Dict
)
from abc import ABC, abstractmethod
from enum import Enum
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from boring_rag_core.schema import Document, TransformComponent
from boring_utils.utils import cprint, tprint, get_device

Embedding = List[float]


class SimilarityMode(str, Enum):
    """Modes for similarity/distance."""
    DEFAULT = "cosine"
    DOT_PRODUCT = "dot_product"
    EUCLIDEAN = "euclidean"


def mean_agg(embeddings: List[Embedding]) -> Embedding:
    """Mean aggregation for embeddings."""
    return np.array(embeddings).mean(axis=0).tolist()


def similarity(
    embedding1: Embedding,
    embedding2: Embedding,
    mode: SimilarityMode = SimilarityMode.DEFAULT,
) -> float:
    """Get embedding similarity."""
    if mode == SimilarityMode.EUCLIDEAN:
        return -float(np.linalg.norm(np.array(embedding1) - np.array(embedding2)))
    elif mode == SimilarityMode.DOT_PRODUCT:
        return np.dot(embedding1, embedding2)
    else:
        product = np.dot(embedding1, embedding2)
        norm = np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
        return product / norm


class BaseEmbedding(TransformComponent, ABC):
    model_name: str
    embed_batch_size: int

    @abstractmethod
    def _embed(
        self,
        sentences: List[str],
        prompt_name: Optional[str] = None,
    ) -> List[List[float]]:
        """Embed sentences."""

    @abstractmethod
    def get_query_embedding(self, query: str) -> Embedding:
        """Embed the input query."""

    @abstractmethod
    def get_text_embedding(self, text: str) -> Embedding:
        """Embed the input text."""

    def get_text_embeddings(self, texts: List[str]) -> List[Embedding]:
        """Embed the input sequence of text."""
        return [self.get_text_embedding(text) for text in texts]

    def get_text_embedding_batch(self, texts: List[str], **kwargs: Any) -> List[Embedding]:
        """Get a list of text embeddings, with batching."""
        results = []
        for i in range(0, len(texts), self.embed_batch_size):
            batch = texts[i:i + self.embed_batch_size]
            results.extend(self.get_text_embeddings(batch))
        return results

    def get_agg_embedding_from_queries(
        self,
        queries: List[str],
        agg_fn: Optional[callable] = None,
    ) -> Embedding:
        """Get aggregated embedding from multiple queries."""
        query_embeddings = [self.get_query_embedding(query) for query in queries]
        agg_fn = agg_fn or mean_agg
        return agg_fn(query_embeddings)

    def similarity(
        self,
        embedding1: Embedding,
        embedding2: Embedding,
        mode: SimilarityMode = SimilarityMode.DEFAULT,
    ) -> float:
        """Get embedding similarity."""
        return similarity(embedding1=embedding1, embedding2=embedding2, mode=mode)

    def __call__(self, nodes: List[Document], **kwargs: Any) -> List[Document]:
        """Transform nodes by adding embeddings."""
        embeddings = self.get_text_embedding_batch(
            [node.text for node in nodes],
            **kwargs,
        )
        for node, embedding in zip(nodes, embeddings):
            node.embedding = embedding
        return nodes

    @classmethod
    def class_name(cls) -> str:
        """Get class name."""
        return cls.__name__

    @abstractmethod
    def embed_documents(self, documents: List[Document]) -> List[Document]:
        """Embed a list of documents."""

