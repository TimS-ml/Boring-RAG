from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Optional,
)

from dataclasses import dataclass, field


@dataclass
class SimpleDocument:
    """
    Check the Document type:
        llama-index-core/llama_index/core/schema.py

        _id, embedding, metadata
    """
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
