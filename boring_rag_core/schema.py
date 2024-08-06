from enum import Enum, auto
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Optional,
)
from typing_extensions import Self

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import uuid
import json

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

    @classmethod
    def class_name(cls) -> str:
        """
        Get the class name, used as a unique ID in serialization.

        This provides a key that makes serialization robust against actual class
        name changes.
        """
        return "base_component"

    def to_dict(self, **kwargs: Any) -> Dict[str, Any]:
        data = self.dict(**kwargs)
        data["class_name"] = self.class_name()
        return data

    def to_json(self, **kwargs: Any) -> str:
        data = self.to_dict(**kwargs)
        return json.dumps(data)

    # TODO: return type here not supported by current mypy version
    @classmethod
    def from_dict(cls, data: Dict[str, Any], **kwargs: Any) -> Self:  # type: ignore
        if isinstance(kwargs, dict):
            data.update(kwargs)

        data.pop("class_name", None)
        return cls(**data)

    @classmethod
    def from_json(cls, data_str: str, **kwargs: Any) -> Self:  # type: ignore
        data = json.loads(data_str)
        return cls.from_dict(data, **kwargs)

    def json(self, **kwargs: Any) -> str:
        return self.to_json(**kwargs)

    def dict(self, **kwargs: Any) -> Dict[str, Any]:
        data = super().dict(**kwargs)
        data["class_name"] = self.class_name()
        return data
