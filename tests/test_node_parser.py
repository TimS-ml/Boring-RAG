import pytest
from boring_rag_core.node_parser.text.sentence import SimpleSentenceSplitter
from boring_rag_core.schema import Document
from boring_utils.utils import cprint


@pytest.fixture
def sample_text():
    return "This is a test. It has three sentences. How many chunks will it create?"

def test_sentence_splitter(sample_text):
    splitter = SimpleSentenceSplitter.from_defaults(chunk_size=2, chunk_overlap=1)
    chunks = splitter.split_text(sample_text)
    
    assert len(chunks) > 1
    # cprint(chunks)
    assert chunks[0].text == "This is a test. It has three sentences."
    assert chunks[1].text == "It has three sentences. How many chunks will it create?"

def test_sentence_splitter_with_document():
    doc = Document(text="Sentence one. Sentence two. Sentence three.", metadata={"source": "test"})
    splitter = SimpleSentenceSplitter.from_defaults(chunk_size=2, chunk_overlap=0)
    chunks = splitter.split_text(doc)
    
    assert len(chunks) == 2
    assert chunks[0].metadata == doc.metadata
    assert chunks[0].start_char_idx == 0
    assert chunks[1].start_char_idx > 0
