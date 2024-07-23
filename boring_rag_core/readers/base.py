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

from boring_rag_core.schema import SimpleDocument


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
        llama-index-integrations/readers/llama-index-readers-file/llama_index/readers/file/docs/base.py
    """

    def __init__(self, return_full_document: bool = False):
        self.return_full_document = return_full_document

    def load_data(self, file: Path, extra_info: Dict[str, Any] = None) -> List[SimpleDocument]:
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
            docs.append(SimpleDocument(
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
                docs.append(SimpleDocument(
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

    pdf_path = Path(os.getenv('DATA_DIR', '.')) / 'nutrition' / 'human-nutrition-text.pdf'
    reader = PDFReader()
    documents = reader.load_data(file=pdf_path)
    print()
    
    if documents:
        cprint(len(documents), c='red')

        id = 1
        for _ in range(2):
            if id < len(documents):
                cprint(id)
                cprint(documents[id].text[:100])
                cprint(documents[id].metadata)    
                cprint(documents[id].id_)
                cprint(documents[id].start_char_idx, documents[id].end_char_idx)
                id += 1
            print()
