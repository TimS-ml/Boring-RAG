import pytest
from boring_rag_core.vector_stores.simple import SimpleVectorStore
from boring_rag_core.schema import Document
from boring_rag_core.vector_stores.types import VectorStoreQuery
from boring_utils.utils import cprint


@pytest.fixture
def sample_documents():
    return [
        Document(text="First document", id_="1", embedding=[1.0, 0.0, 0.0]),
        Document(text="Second document", id_="2", embedding=[0.0, 1.0, 0.0]),
        Document(text="Third document", id_="3", embedding=[0.0, 0.0, 1.0]),
    ]

def test_vector_store_add(sample_documents):
    store = SimpleVectorStore()
    ids = store.add(sample_documents)
    
    assert len(ids) == 3
    assert set(ids) == {"1", "2", "3"}

def test_vector_store_query(sample_documents):
    store = SimpleVectorStore()
    store.add(sample_documents)
    
    query = VectorStoreQuery(query_embedding=[1.0, 0.5, 0.0], similarity_top_k=2)
    result = store.query(query)
    
    cprint(result)
    assert len(result.nodes) == 2
    assert result.nodes[0].id_ == "1"  # Most similar should be the first document
