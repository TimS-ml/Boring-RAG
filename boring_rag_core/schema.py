from enum import Enum, auto
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Optional,
)

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import uuid

class ObjectType(str, Enum):
    """
    Usage:
        llama-index-core/llama_index/core/indices/multi_modal/retriever.py

    in class MultiModal Retriever:
    ...
    if (not self._vector_store.stores_text) or (
        source_node is not None and source_node.node_type != ObjectType.TEXT
    ):
    """
    TEXT = auto()
    IMAGE = auto()
    INDEX = auto()
    DOCUMENT = auto()


@dataclass
class Document:
    """
    Check the Document type:
        llama-index-core/llama_index/core/schema.py

    _id, embedding, metadata, relationships, etc.
    ImageDocument
    """

    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    id_: str = field(default_factory=lambda: str(uuid.uuid4()))
    embedding: Optional[List[float]] = None
    relationships: Dict[str, Any] = field(default_factory=dict)
    start_char_idx: Optional[int] = None
    end_char_idx: Optional[int] = None

    @classmethod
    def get_type(cls) -> str:
        """Get Object type."""
        return ObjectType.TEXT


class TransformComponent(ABC):
    @abstractmethod
    def __call__(self, nodes: List[Document], **kwargs: Any) -> List[Document]:
        """Transform nodes."""

