from typing import List, Dict, Any, Optional
from boring_rag_core.schema import Document


@dataclass
class VectorStoreQuery:
    query_embedding: Optional[List[float]] = None
    similarity_top_k: int = 1
    # doc_ids: Optional[List[str]] = None
    node_ids: Optional[List[str]] = None
    query_str: Optional[str] = None
    

@dataclass
class VectorStoreQueryResult:
    nodes: Optional[List[Document]] = None
    similarities: Optional[List[float]] = None
    ids: Optional[List[str]] = None

