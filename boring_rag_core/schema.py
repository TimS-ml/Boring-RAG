from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Optional,
)

from dataclasses import dataclass, field
import uuid

@dataclass
class SimpleDocument:
    """
    Check the Document type:
        llama-index-core/llama_index/core/schema.py

        _id, embedding, metadata
    """

    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    id_: str = field(default_factory=lambda: str(uuid.uuid4()))
    embedding: Optional[List[float]] = None
    relationships: Dict[str, Any] = field(default_factory=dict)
    start_char_idx: Optional[int] = None
    end_char_idx: Optional[int] = None

