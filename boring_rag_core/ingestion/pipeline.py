import os
from pydantic import BaseModel, Field
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Optional,
    List, 
    Sequence,
    Callable,
    Union,
)

from boring_rag_core.node_parser.text.sentence import SimpleSentenceSplitter
from boring_rag_core.embeddings.huggingface.base import HuggingFaceEmbedding
from boring_rag_core.schema import Document, TransformComponent

# TODO: revisit this after implementing boring_rag_core.vector_store.base
# from boring_rag_core.vector_store.base import BaseVectorStore
BaseVectorStore = None


def run_transformations(
    nodes: List[Document],
    transformations: Sequence[TransformComponent],
    in_place: bool = True,
    **kwargs: Any,
) -> List[Document]:
    """
    Run a series of transformations on a set of nodes.
    NOTE: We skip the IngestionCache implementaiton for now 

    Args:
        nodes: The nodes to transform.
        transformations: The transformations to apply to the nodes.

    Returns:
        The transformed nodes.
    """
    if not in_place:
        nodes = list(nodes)

    for transform in transformations:
        nodes = transform(nodes, **kwargs)

    return nodes


class IngestionPipeline(BaseModel):
    """
    ref:
        llama-index-core/llama_index/core/ingestion/pipeline.py

    NOTE: there is no IngestionCache implementation for now, 
    so no persist / load function, replaced with to_dict and from_dict

    TODO: no data preprocessing, duplication remove for vector_store now, until I am happy :)
    """

    name: str = Field(
        default="default",
        description="Unique name of the ingestion pipeline",
    )
    transformations: List[TransformComponent] = Field(default_factory=list)
    vector_store: Optional[BaseVectorStore] = None

    @classmethod
    def get_default_transformations(cls) -> List[TransformComponent]:
        return [
            SimpleSentenceSplitter.from_defaults(),
            HuggingFaceEmbedding()
        ]

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **data):
        super().__init__(**data)
        if not self.transformations:
            self.transformations = self.get_default_transformations()

    def run(self, documents: Sequence[Document]) -> List[Document]:
        processed_docs = []
        for doc in documents:
            current_docs = [doc]
            # TODO: support more transformations
            for transform in self.transformations:
                if isinstance(transform, SimpleSentenceSplitter):
                    current_docs = transform.split_text(current_docs[0])
                elif isinstance(transform, HuggingFaceEmbedding):
                    current_docs = transform.embed_documents(current_docs)
            processed_docs.extend(current_docs)
        return processed_docs

    def batch_embed_documents(self, documents: List[Document], batch_size: int = 32) -> List[Document]:
        embedding_model = next((t for t in self.transformations if isinstance(t, HuggingFaceEmbedding)), None)
        if embedding_model is None:
            raise ValueError("No embedding model found in transformations")
        
        embedded_docs = []
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i+batch_size]
            embedded_batch = embedding_model.embed_documents(batch)
            embedded_docs.extend(embedded_batch)
        
        return embedded_docs

    def _prepare_inputs(
        self, nodes: Optional[List[Document]]
    ) -> List[Document]:
        """Prepare the input nodes for the pipeline."""
        input_nodes: List[Document] = []

        if nodes is not None:
            input_nodes += nodes

        if self.vector_store is not None:
            input_nodes += self.vector_store.embeddings

        return input_nodes

    def _handle_duplicates(
        self,
        nodes: List[Document],
    ) -> List[Document]:
        """Check if there is any duplicate nodes and remove them. But not doc level."""

        current_hashes = []
        nodes_to_run = []
        for node in nodes:
            if node.id_ not in current_hashes:
                nodes_to_run.append(node)
                current_hashes.append(node.id_)

        return nodes_to_run

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "transformations": [t.to_dict() for t in self.transformations]
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'IngestionPipeline':
        name = data.get("name", "default")
        transformations = [TransformComponent.from_dict(t) for t in data["transformations"]]
        return cls(name=name, transformations=transformations)


if __name__ == '__main__':
    from pathlib import Path
    from boring_rag_core.readers.base import PDFReader
    from boring_rag_core.embeddings.huggingface.base import HuggingFaceEmbedding
    from boring_utils.utils import cprint, tprint
    
    pdf_path = Path(os.getenv('DATA_DIR')) / 'nutrition' / 'human-nutrition-text_ch1.pdf'
    reader = PDFReader()
    documents = reader.load_data(file=pdf_path)
    
    pipeline = IngestionPipeline()
    processed_documents = pipeline.run(documents)
    
    # Get query embedding
    embedding = HuggingFaceEmbedding()
    query = "Tell me something about nutrition."
    query_embedding = embedding.get_text_embedding(query)
    
    # Calculate similarity
    for doc in processed_documents[:10]:
        similarity = embedding.similarity(query_embedding, doc.embedding)
        print(f"Similarity with document {doc.id_}: {similarity}")
    
    # Batch embed documents
    batch_size = 32
    embedded_documents = pipeline.batch_embed_documents(documents, batch_size)
