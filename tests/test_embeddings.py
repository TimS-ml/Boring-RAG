import pytest
from boring_rag_core.embeddings.huggingface.base import HuggingFaceEmbedding
from boring_rag_core.schema import Document


@pytest.fixture
def embedding_model():
    return HuggingFaceEmbedding()

def test_text_embedding(embedding_model):
    text = "This is a test sentence."
    embedding = embedding_model.get_text_embedding(text)
    
    assert isinstance(embedding, list)
    assert len(embedding) > 0
    assert all(isinstance(x, float) for x in embedding)

def test_query_embedding(embedding_model):
    query = "What is the capital of France?"
    embedding = embedding_model.get_query_embedding(query)
    
    assert isinstance(embedding, list)
    assert len(embedding) > 0

def test_document_embedding(embedding_model):
    docs = [Document(text="First document."), Document(text="Second document.")]
    embedded_docs = embedding_model.embed_documents(docs)
    
    assert len(embedded_docs) == 2
    assert all(doc.embedding is not None for doc in embedded_docs)
