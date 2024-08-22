import json
from enum import Enum
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Callable
from boring_rag_core.schema import Document
from dataclasses import dataclass, field, asdict

from boring_rag_core.indices.query.embedding_utils import (
    get_top_k_embeddings,
    get_top_k_embeddings_learner,
    get_top_k_mmr_embeddings,
)
from boring_rag_core.vector_stores.types import (
    VectorStoreQuery, 
    VectorStoreQueryResult, 
    VectorStoreQueryMode,
    FilterOperator, FilterCondition,
    MetadataFilter, MetadataFilters
)

LEARNER_MODES = {
    VectorStoreQueryMode.SVM,
    VectorStoreQueryMode.LINEAR_REGRESSION,
    VectorStoreQueryMode.LOGISTIC_REGRESSION,
}
MMR_MODE = VectorStoreQueryMode.MMR


def _build_metadata_filter_fn(
    filters: Optional[MetadataFilters] = None,
) -> Callable[[Dict[str, Any]], bool]:
    """Build a filter function based on MetadataFilters."""
    if not filters:
        return lambda _: True

    def filter_fn(metadata: Dict[str, Any]) -> bool:
        for filter in filters.filters:
            if not _apply_single_filter(metadata, filter):
                if filters.condition == FilterCondition.AND:
                    return False
            else:
                if filters.condition == FilterCondition.OR:
                    return True
        return filters.condition == FilterCondition.AND

    return filter_fn

def _apply_single_filter(metadata: Dict[str, Any], filter: MetadataFilter) -> bool:
    """Apply a single MetadataFilter."""
    value = metadata.get(filter.key)
    if value is None:
        return False

    if filter.operator == FilterOperator.EQ:
        return value == filter.value
    elif filter.operator == FilterOperator.NE:
        return value != filter.value
    elif filter.operator == FilterOperator.GT:
        return value > filter.value
    elif filter.operator == FilterOperator.GTE:
        return value >= filter.value
    elif filter.operator == FilterOperator.LT:
        return value < filter.value
    elif filter.operator == FilterOperator.LTE:
        return value <= filter.value
    elif filter.operator == FilterOperator.IN:
        return value in filter.value
    elif filter.operator == FilterOperator.NIN:
        return value not in filter.value
    elif filter.operator == FilterOperator.CONTAINS:
        return filter.value in value
    # elif filter.operator == FilterOperator.NOT_CONTAINS:
    #     return filter.value not in value
    else:
        raise ValueError(f"Unsupported filter operator: {filter.operator}")


@dataclass
class SimpleVectorStoreData:
    """
    SimpleVectorStoreData can be seen as an abstraction layer 
    that decouples the underlying storage structure from the Document/BaseNode objects of LlamaIndex
    """
    embedding_dict: Dict[str, List[float]] = field(default_factory=dict)
    metadata_dict: Dict[str, Dict[str, Any]] = field(default_factory=dict)


class SimpleVectorStore:
    """
    ref:
        https://docs.llamaindex.ai/en/stable/examples/low_level/vector_store/
    """
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

    def query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        """Query the vector store."""
        if query.query_embedding is None:
            raise ValueError("Query embedding is required.")

        # Apply filters
        filtered_ids, filtered_embeddings = self._apply_filters(query)

        # Perform similarity search based on query mode
        if query.mode == VectorStoreQueryMode.DEFAULT:
            top_similarities, top_ids = get_top_k_embeddings(
                query.query_embedding,
                filtered_embeddings,
                similarity_top_k=query.similarity_top_k,
                embedding_ids=filtered_ids,
            )
        elif query.mode in LEARNER_MODES:
            top_similarities, top_ids = get_top_k_embeddings_learner(
                query.query_embedding,
                filtered_embeddings,
                similarity_top_k=query.similarity_top_k,
                embedding_ids=filtered_ids,
                query_mode=query.mode,
            )
        elif query.mode == MMR_MODE:
            mmr_threshold = kwargs.get("mmr_threshold", None)
            top_similarities, top_ids = get_top_k_mmr_embeddings(
                query.query_embedding,
                filtered_embeddings,
                similarity_top_k=query.similarity_top_k,
                embedding_ids=filtered_ids,
                mmr_threshold=mmr_threshold,
            )
        else:
            raise ValueError(f"Invalid query mode: {query.mode}")

        return VectorStoreQueryResult(similarities=top_similarities, ids=top_ids)

    def _apply_filters(self, query: VectorStoreQuery) -> Tuple[List[str], List[List[float]]]:
        """
        Apply filter[s] to the vector store metadata.
        The source code of MetadataFilter includes key, value, and operator. 
        For example, the metadata we want to filter is internal_score=3 in the metadata_dict
        """
        filtered_ids = []
        filtered_embeddings = []

        filter_fn = _build_metadata_filter_fn(query.filters)

        for node_id, embedding in self.data.embedding_dict.items():
            if self._check_filters(node_id, query, filter_fn):
                filtered_ids.append(node_id)
                filtered_embeddings.append(embedding)

        return filtered_ids, filtered_embeddings

    def _check_filters(
            self, 
            node_id: str, 
            query: VectorStoreQuery, 
            filter_fn: Callable[[Dict[str, Any]], bool]
        ) -> bool:
        """Check if a node passes all metadata filters."""
        if query.filters:
            metadata = self.data.metadata_dict.get(node_id, {})
            if not filter_fn(metadata):
                return False
        
        if query.node_ids and node_id not in query.node_ids:
            return False

        return True
