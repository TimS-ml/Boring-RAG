import re
from tqdm.auto import tqdm
from pathlib import Path
from pydantic import Field
from abc import ABC, abstractmethod
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Optional,
    List, 
    Callable
)

from boring_rag_core.schema import Document
from boring_rag_core.node_parser.interface import TextSplitter
from dataclasses import dataclass, field

def default_sentence_splitter() -> Callable[[str], List[str]]:
    import nltk
    nltk.download('punkt', quiet=True)
    return nltk.sent_tokenize

def default_sentence_splitter_spacy() -> Callable[[str], List[str]]:
    from spacy.lang.en import English
    nlp: English = field(default_factory=lambda: English())
    return nlp.add_pipe("sentencizer")


SENTENCE_CHUNK_OVERLAP = 2
DEFAULT_CHUNK_SIZE = 10


@dataclass
class SimpleSentenceSplitter(TextSplitter):
    # chunk_size: int = Field(
    #     default=DEFAULT_CHUNK_SIZE,
    #     description="The number of sentences in each chunk.",
    # )
    # chunk_overlap: int = Field(
    #     default=SENTENCE_CHUNK_OVERLAP,
    #     description="The number of sentences to overlap between chunks.",
    # )
    # separator: str = Field(
    #     default=" ", description="Separator for joining sentences"
    # )
    # _sentence_splitter: Callable[[str], List[str]] = field(default_factory=default_sentence_splitter)

    def __init__(
        self,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = SENTENCE_CHUNK_OVERLAP,
        separator: str = " ",
        sentence_splitter: Optional[Callable[[str], List[str]]] = field(default_factory=default_sentence_splitter)
    ):
        """Initialize with parameters."""
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separator = separator
        self._sentence_splitter = sentence_splitter

        if self.chunk_overlap > self.chunk_size:
            raise ValueError(
                f"Got a larger chunk overlap ({self.chunk_overlap}) than chunk size "
                f"({self.chunk_size}), should be smaller."
            )
    
    """
    def split_text(self, text: str) -> List[dict]:
        doc = self.nlp(text)

        # to sentensces
        sentences = [str(sent) for sent in doc.sents]

        # to chunks
        chunks = self._split_into_chunks(sentences)

        processed_chunks = []
        for chunk in chunks:
            processed_chunk = self._process_chunk(chunk)
            processed_chunks.append(processed_chunk)

        return processed_chunks

    def _process_chunk(self, chunk: List[str]) -> dict:
        joined_chunk = " ".join(chunk).replace("  ", " ").strip()
        joined_chunk = re.sub(r'\.([A-Z])', r'. \1', joined_chunk)

        chunk_dict = {
            "sentence_chunk": joined_chunk,
            "chunk_char_count": len(joined_chunk),
            "chunk_word_count": len(joined_chunk.split()),
            "chunk_token_count": len(joined_chunk) // 4
        }
        return chunk_dict
    """

    def split_text(self, text: str) -> List[Document]:
        sentences = self._sentence_splitter(text)
        chunks = self._split_into_chunks(sentences)

        # return [' '.join(chunk) for chunk in chunks]
        return [self._create_document_from_chunk(chunk) for chunk in chunks]

    def _create_document_from_chunk(self, chunk: List[str]) -> Document:
        joined_chunk = self.separator.join(chunk).strip()
        joined_chunk = re.sub(r'\.([A-Z])', r'. \1', joined_chunk)
        
        return Document(
            text=joined_chunk,
            metadata={
                "chunk_char_count": len(joined_chunk),
                "chunk_word_count": len(joined_chunk.split()),
                "chunk_token_count": len(joined_chunk) // 4
            }
        )

    def _split_into_chunks(self, sentences: List[str]) -> List[List[str]]:
        """including overlap"""
        chunks = []
        for i in range(0, len(sentences), self.chunk_size - self.chunk_overlap):
            chunk = sentences[i:i + self.chunk_size]
            chunks.append(chunk)
        return chunks

    @classmethod
    def from_defaults(cls, 
                      chunk_size: int = DEFAULT_CHUNK_SIZE, 
                      chunk_overlap: int = SENTENCE_CHUNK_OVERLAP,
                      sentence_splitter: Optional[Callable[[str], List[str]]] = None
    ) -> 'SimpleSentenceSplitter':
        return cls(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            sentence_splitter=default_sentence_splitter() if sentence_splitter is None else sentence_splitter
        )


if __name__ == '__main__':
    import os
    from boring_utils.utils import cprint
    from boring_rag_core.readers.base import PDFReader

    pdf_path = Path(os.getenv('DATA_DIR')) / 'nutrition' / 'human-nutrition-text_ch1.pdf'
    reader = PDFReader()
    documents = reader.load_data(file=pdf_path)
    cprint(len(documents), c='red')
    cprint(documents[0].metadata)    

    splitter = SimpleSentenceSplitter.from_defaults()
    
    for doc in documents:
        chunks = splitter.split_text(doc.text)
        cprint(len(chunks), c='green')
        # doc.chunks = chunks
        # doc.metadata["num_chunks"] = len(chunks)

