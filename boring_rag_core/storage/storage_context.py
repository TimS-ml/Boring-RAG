from typing import Optional, Dict, Any
from boring_rag_core.vector_stores.simple import SimpleVectorStore


class StorageContext:
    def __init__(self, vector_store: Optional[SimpleVectorStore] = None):
        self.vector_store = vector_store or SimpleVectorStore()

    @classmethod
    def from_defaults(cls, vector_store: Optional[SimpleVectorStore] = None):
        return cls(vector_store=vector_store)

    def persist(self, persist_dir: str):
        self.vector_store.persist(persist_dir)

    @classmethod
    def from_persist_dir(cls, persist_dir: str):
        vector_store = SimpleVectorStore.from_persist_path(persist_dir)
        return cls(vector_store=vector_store)

    def to_dict(self) -> Dict:
        return {
            "vector_store": self.vector_store.to_dict()
        }

    @classmethod
    def from_dict(cls, data: Dict):
        vector_store = SimpleVectorStore.from_dict(data["vector_store"])
        return cls(vector_store=vector_store)
