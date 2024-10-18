import os
import pytest
from pathlib import Path
from boring_rag_core.readers.base import PDFReader
from boring_rag_core.schema import Document


@pytest.fixture
def sample_pdf_path():
    pdf_path = Path(os.getenv('DATA_DIR')) / 'nutrition' / 'human-nutrition-text_ch1.pdf'
    return pdf_path

def test_pdf_reader(sample_pdf_path):
    reader = PDFReader()
    documents = reader.load_data(file=sample_pdf_path)
    
    assert len(documents) > 0
    assert isinstance(documents[0], Document)
    assert documents[0].text
    assert documents[0].metadata
    assert "file_name" in documents[0].metadata
    assert "page_number" in documents[0].metadata

def test_pdf_reader_full_document(sample_pdf_path):
    reader = PDFReader(return_full_document=True)
    documents = reader.load_data(file=sample_pdf_path)
    
    assert len(documents) == 1
    assert isinstance(documents[0], Document)
    assert documents[0].text
    assert "page_count" in documents[0].metadata
