"""Node parser interface."""

from abc import ABC, abstractmethod
from typing import (
    Any, 
    Callable, 
    Dict, 
    Union,
    Iterable,
    List, 
    Sequence
)
from tqdm.auto import tqdm
from boring_rag_core.schema import Document, TransformComponent


def get_tqdm_iterable(items: Iterable, show_progress: bool, desc: str) -> Iterable:
    return tqdm(items, desc=desc) if show_progress else items


def build_nodes_from_splits(
        text_splits: List[str],
        document: Document,
    ):
    """
    Ref:
        llama-index-core/llama_index/core/node_parser/node_utils.py
    """
    return [Document(
                text=split, metadata=document.metadata.copy(), 
                embedding=document.embedding,
                relationships=document.relationships,
        ) for split in text_splits]


class TextSplitter(TransformComponent, ABC):
    # @abstractmethod
    # def split_text(self, text: str) -> Union[List[str], List[Document]]:
    #     ...

    @abstractmethod
    def split_text(self, input_data: Union[str, Document]) -> List[Document]:
        ...

    def split_texts(self, texts: List[str]) -> Union[List[str], List[Document]]:
        """call split_text on each text in texts and flatten the result."""
        nested_texts = [self.split_text(text) for text in texts]
        return [item for sublist in nested_texts for item in sublist]

    def __call__(self, nodes: Sequence[Document], show_progress: bool = False, **kwargs: Any) -> List[Document]:
        """from the _parse_nodes function"""
        all_nodes: List[Document] = []
        nodes_with_progress = get_tqdm_iterable(nodes, show_progress, "Parsing nodes")
        for node in nodes_with_progress:
            splits = self.split_text(node.text)
            all_nodes.extend(build_nodes_from_splits(splits, node))
        return all_nodes

