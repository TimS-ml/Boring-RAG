from tqdm.auto import tqdm
from pathlib import Path

from abc import ABC, abstractmethod
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Optional,
)

from boring_rag_core.schema import Document


class BaseReader(ABC):
    """Utilities for loading data from a directory."""

    def lazy_load_data(self, *args: Any, **load_kwargs: Any):
        """Load data from the input directory lazily."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not provide lazy_load_data method currently"
        )

    def load_data(self, *args: Any, **load_kwargs: Any):
        """Load data from the input directory."""
        return list(self.lazy_load_data(*args, **load_kwargs))


class PDFReader(BaseReader):
    """
    Ref:
        llama-index-integrations/readers/llama-index-readers-file/llama_index/readers/file/docs/base.py -> PDFReader (pypdf)
        llama-index-integrations/readers/llama-index-readers-file/llama_index/readers/file/pymu_pdf/base.py -> PDFMuPDFReader (fitz)
        llama-index-integrations/readers/llama-index-readers-pdf-table/llama_index/readers/pdf_table/base.py -> PDFTableReader (camelot)
    """

    def __init__(self, return_full_document: bool = False):
        self.return_full_document = return_full_document

    def load_data(self, file: Path, extra_info: Dict[str, Any] = None) -> List[Document]:
        if not isinstance(file, Path):
            file = Path(file)

        try:
            import fitz
        except ImportError:
            raise ImportError("PyMuPDF (fitz) is required to read PDF files: `pip install PyMuPDF`")

        docs = []
        with fitz.open(file) as pdf:
            num_pages = len(pdf)

            if self.return_full_document:
                full_text = ""
                for page in pdf:
                    full_text += page.get_text()
                metadata = {
                    "file_name": file.name,
                    "page_count": num_pages,
                    "page_char_count": len(full_text),
                    "page_word_count": len(full_text.split()),
                    "page_sentence_count_raw": len(full_text.split(". ")),
                    "page_token_count": len(full_text) // 4,
                }
                if extra_info:
                    metadata.update(extra_info)
                docs.append(Document(
                    text=full_text, 
                    metadata=metadata,
                    start_char_idx=0,
                    end_char_idx=len(full_text)
                ))
            else:
                char_index = 0
                for page_num, page in enumerate(pdf):
                    page_text = page.get_text()
                    start_idx = char_index
                    end_idx = char_index + len(page_text)
                    metadata = {
                        "file_name": file.name,
                        "page_label": str(page_num + 1),
                        "page_number": page_num + 1,
                        "page_char_count": len(page_text),
                        "page_word_count": len(page_text.split()),
                        "page_sentence_count_raw": len(page_text.split(". ")),
                        "page_token_count": len(page_text) // 4,
                    }
                    if extra_info:
                        metadata.update(extra_info)
                    docs.append(Document(
                        text=page_text, 
                        metadata=metadata,
                        start_char_idx=start_idx,
                        end_char_idx=end_idx
                    ))
                    char_index = end_idx

        return docs

    def load_data_pypdf(self, file: Path, extra_info: Dict[str, Any] = None) -> List[Document]:
        """
        We might need to do postprocessing like this:

        def process_text(text: str) -> str:
            text = re.sub(r'\s+', ' ', text)
            text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
            text = re.sub(r'(\w+)\s(?=[a-z])', r'\1', text)
            return text.strip()
        """

        if not isinstance(file, Path):
            file = Path(file)

        try:
            import pypdf
        except ImportError:
            raise ImportError("pypdf is required to read PDF files: `pip install pypdf`")

        pdf = pypdf.PdfReader(file)
        num_pages = len(pdf.pages)
        docs = []

        if self.return_full_document:
            text = "\n".join(pdf.pages[page].extract_text() for page in range(num_pages))
            metadata = {
                "file_name": file.name,
                "page_count": num_pages,
                "page_char_count": len(text),
                "page_word_count": len(text.split()),
                "page_sentence_count_raw": len(text.split(". ")),
                "page_token_count": len(text) // 4,
            }
            if extra_info:
                metadata.update(extra_info)
            docs.append(Document(
                text=text, 
                metadata=metadata,
                start_char_idx=0,
                end_char_idx=len(text)
            ))
        else:
            char_index = 0
            for page in range(num_pages):
                page_text = pdf.pages[page].extract_text()
                start_idx = char_index
                end_idx = char_index + len(page_text)
                metadata = {
                    "file_name": file.name,
                    "page_label": pdf.page_labels[page],
                    "page_number": page + 1,
                    "page_char_count": len(page_text),
                    "page_word_count": len(page_text.split()),
                    "page_sentence_count_raw": len(page_text.split(". ")),
                    "page_token_count": len(page_text) // 4,
                }
                if extra_info:
                    metadata.update(extra_info)
                docs.append(Document(
                    text=page_text, 
                    metadata=metadata,
                    start_char_idx=start_idx,
                    end_char_idx=end_idx
                ))
                char_index = end_idx

        return docs


if __name__ == '__main__':
    import os
    from boring_utils.utils import cprint

    pdf_path = Path(os.getenv('DATA_DIR')) / 'nutrition' / 'human-nutrition-text_ch1.pdf'
    reader = PDFReader()
    documents = reader.load_data(file=pdf_path)
    print()
    
    if documents:
        cprint(len(documents), c='red')

        for id in range(1, 3):
            if id < len(documents):
                cprint(id)
                cprint(documents[id].text[:100])
                cprint(documents[id].metadata)    
                cprint(documents[id].id_)
                cprint(documents[id].start_char_idx, documents[id].end_char_idx)
            print()
