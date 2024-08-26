from abc import ABC, abstractmethod
from typing import List, Optional, Sequence, Any
from pydantic.v1 import BaseModel, Field

from boring_rag_core.schema import Document, TransformComponent
from boring_rag_core.ingestion.pipeline import IngestionPipeline

# StorageContext: includes BaseIndexStore, BaseVectorStore, ...
from boring_rag_core.storage.storage_context import StorageContext


class BaseIndex(ABC, BaseModel):
    storage_context: StorageContext = Field(default_factory=StorageContext)
    ingestion_pipeline: IngestionPipeline = Field(default_factory=IngestionPipeline)
    
    class Config:
        arbitrary_types_allowed = True

    @classmethod
    def from_documents(
        cls,
        documents: Sequence[Document],
        storage_context: Optional[StorageContext] = None,
        transformations: Optional[List[TransformComponent]] = None,
        **kwargs: Any
    ):
        storage_context = storage_context or StorageContext()
        instance = cls(storage_context=storage_context)
        
        if transformations:
            instance.ingestion_pipeline = IngestionPipeline(transformations=transformations)
        
        processed_documents = instance.ingestion_pipeline.run(documents)
        instance._build_index_from_documents(processed_documents, **kwargs)
        return instance

    @abstractmethod
    def _build_index_from_documents(self, documents: List[Document], **kwargs: Any):
        """Build the index from documents."""

    @abstractmethod
    def insert(self, document: Document, **insert_kwargs: Any) -> None:
        """Insert a document into the index."""

    @abstractmethod
    def delete(self, doc_id: str, **delete_kwargs: Any) -> None:
        """Delete a document from the index."""

    @abstractmethod
    def update(self, document: Document, **update_kwargs: Any) -> None:
        """Update a document in the index."""

    @abstractmethod
    def query(self, query_str: str, **query_kwargs: Any) -> Any:
        """Query the index."""

    # def get_document(self, doc_id: str) -> Optional[Document]:
    #     """Retrieve a document from the index by its ID."""
    #     return self.storage_context.docstore.get(doc_id)
    # 
    #  def persist(self, persist_dir: str):
    #      """Persist the index to disk."""
    #      self.storage_context.persist(persist_dir)
    #
    #  @classmethod
    #  def load(cls, load_dir: str):
    #      """Load the index from disk."""
    #      storage_context = StorageContext.from_persist_dir(load_dir)
    #      return cls(storage_context=storage_context)
