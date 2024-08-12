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


class IngestionPipeline(BaseModel):
    transformations: List[TransformComponent] = Field(default_factory=list)
    
    class Config:
        arbitrary_types_allowed = True

    @classmethod
    def get_default_transformations(cls) -> List[TransformComponent]:
        return [
            SimpleSentenceSplitter.from_defaults(),
            HuggingFaceEmbedding()
        ]

    def __init__(self, **data: Any):
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

    def get_query_embedding(self, query: str) -> List[float]:
        embedding_model = next((t for t in self.transformations if isinstance(t, HuggingFaceEmbedding)), None)
        if embedding_model is None:
            raise ValueError("No embedding model found in transformations")
        return embedding_model.get_text_embedding(query)

    def calculate_similarity(self, query_embedding: List[float], document_embedding: List[float]) -> float:
        embedding_model = next((t for t in self.transformations if isinstance(t, HuggingFaceEmbedding)), None)
        if embedding_model is None:
            raise ValueError("No embedding model found in transformations")
        return embedding_model.similarity(query_embedding, document_embedding)

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

    def to_dict(self) -> dict:
        return {
            "transformations": [t.to_dict() for t in self.transformations]
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'IngestionPipeline':
        transformations = [TransformComponent.from_dict(t) for t in data["transformations"]]
        return cls(transformations=transformations)


if __name__ == '__main__':
    from pathlib import Path
    from boring_rag_core.readers.base import PDFReader
    from boring_utils.utils import cprint, tprint
    
    pdf_path = Path(os.getenv('DATA_DIR')) / 'nutrition' / 'human-nutrition-text_ch1.pdf'
    reader = PDFReader()
    documents = reader.load_data(file=pdf_path)
    
    pipeline = IngestionPipeline()
    processed_documents = pipeline.run(documents)  # TODO: fix this line
    
    # Get query embedding
    query = "Tell me something about nutrition."
    query_embedding = pipeline.get_query_embedding(query)
    
    # Calculate similarity
    for doc in processed_documents:
        similarity = pipeline.calculate_similarity(query_embedding, doc.embedding)
        print(f"Similarity with document {doc.id_}: {similarity}")
    
    # Batch embed documents
    batch_size = 32
    embedded_documents = pipeline.batch_embed_documents(documents, batch_size)
