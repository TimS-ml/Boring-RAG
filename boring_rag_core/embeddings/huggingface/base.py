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
from boring_utils.utils import cprint, tprint


DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en"
DEFAULT_EMBED_INSTRUCTION = "Represent the document for retrieval: "
DEFAULT_QUERY_INSTRUCTION = "Represent the question for retrieving supporting documents: "

DEFAULT_HUGGINGFACE_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_EMBED_BATCH_SIZE = 32


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
    def _get_query_embedding(self, query: str) -> Embedding:
        """Embed the input query."""

    @abstractmethod
    def _get_text_embedding(self, text: str) -> Embedding:
        """Embed the input text."""

    def _get_text_embeddings(self, texts: List[str]) -> List[Embedding]:
        """Embed the input sequence of text."""
        return [self._get_text_embedding(text) for text in texts]

    def get_text_embedding_batch(self, texts: List[str], **kwargs: Any) -> List[Embedding]:
        """Get a list of text embeddings, with batching."""
        results = []
        for i in range(0, len(texts), self.embed_batch_size):
            batch = texts[i:i + self.embed_batch_size]
            results.extend(self._get_text_embeddings(batch))
        return results

    def get_agg_embedding_from_queries(
        self,
        queries: List[str],
        agg_fn: Optional[callable] = None,
    ) -> Embedding:
        """Get aggregated embedding from multiple queries."""
        query_embeddings = [self._get_query_embedding(query) for query in queries]
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


class HuggingFaceEmbedding(BaseEmbedding):
    def __init__(
        self,
        model_name: str = DEFAULT_HUGGINGFACE_EMBEDDING_MODEL,
        max_length: int = 512,
        normalize: bool = True,
        embed_batch_size: int = DEFAULT_EMBED_BATCH_SIZE,
        device: Optional[str] = None,
        **kwargs: Any,
    ):
        super().__init__()
        self.model_name = model_name
        self.max_length = max_length
        self.normalize = normalize
        self.embed_batch_size = embed_batch_size
        self._device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        self._model = SentenceTransformer(model_name, device=self._device)
        self._model.max_seq_length = max_length

    def _get_query_embedding(self, query: str) -> List[float]:
        embeddings = self._model.encode([query], 
                                        normalize_embeddings=self.normalize,
                                        batch_size=1)
        return embeddings[0].tolist()

    def _get_text_embedding(self, text: str) -> List[float]:
        embeddings = self._model.encode([text], 
                                        normalize_embeddings=self.normalize,
                                        batch_size=1)
        return embeddings[0].tolist()

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        embeddings = self._model.encode(texts, 
                                        normalize_embeddings=self.normalize,
                                        batch_size=self.embed_batch_size)
        return embeddings.tolist()

    def embed_documents(self, documents: List[Document]) -> List[Document]:
        texts = [doc.text for doc in documents]
        embeddings = self._get_text_embeddings(texts)
        
        for doc, embedding in zip(documents, embeddings):
            doc.embedding = embedding
        
        return documents

    @classmethod
    def class_name(cls) -> str:
        return "HuggingFaceEmbedding"
