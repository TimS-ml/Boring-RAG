import re
from tqdm.auto import tqdm
from pathlib import Path
from pydantic.v1 import Field
from abc import ABC, abstractmethod
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Optional,
    List, 
    Callable,
    Union
)

from boring_rag_core.schema import Document
from boring_rag_core.node_parser.interface import TextSplitter
from dataclasses import dataclass, field
from boring_utils.helpers import DEBUG

if DEBUG > 0:
    from boring_utils.utils import cprint

def default_sentence_splitter() -> Callable[[str], List[str]]:
    import nltk
    nltk.download('punkt', quiet=True)
    return nltk.sent_tokenize

def default_sentence_splitter_spacy() -> Callable[[str], List[str]]:
    from spacy.lang.en import English
    nlp = English()
    nlp.add_pipe("sentencizer")

    def spacy_sentence_splitter(text: str) -> List[str]:
        doc = nlp(text)
        return [str(sent) for sent in doc.sents]

    return spacy_sentence_splitter


SENTENCE_CHUNK_OVERLAP = 2
DEFAULT_CHUNK_SIZE = 10


@dataclass
class SimpleSentenceSplitter(TextSplitter):
    chunk_size: int = Field(
        default=DEFAULT_CHUNK_SIZE,
        description="The number of sentences in each chunk.",
    )
    chunk_overlap: int = Field(
        default=SENTENCE_CHUNK_OVERLAP,
        description="The number of sentences to overlap between chunks.",
    )
    separator: str = Field(
        default=" ", description="Separator for joining sentences"
    )
    sentence_splitter: Callable[[str], List[str]] = field(default_factory=default_sentence_splitter)
    # include_metadata: bool = True  # llama index's include_metadata is mainly for the chunk size correction
    
    def __post_init__(self):
        if self.chunk_overlap > self.chunk_size:
            raise ValueError(
                f"Got a larger chunk overlap ({self.chunk_overlap}) than chunk size "
                f"({self.chunk_size}), should be smaller."
            )
        self._sentence_splitter = self.sentence_splitter
    
    """
    def split_text(self, text: str) -> List[dict]:
        doc = self.nlp(text)

        # to sentensces
        sentences = [str(sent) for sent in doc.sents]

        # to chunks
        chunks = self._split_into_chunks(sentences)

        processed_chunks = []
        for chunk in chunks:
            processed_chunk = self._create_document_from_chunk(chunk)
            processed_chunks.append(processed_chunk)

        return processed_chunks

    def _create_document_from_chunk(self, chunk: List[str]) -> dict:
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

    def split_text(self, input_data: Union[str, Document]) -> List[Document]:
        """
        Input can accept either a string or a Document as input.
        Ignore the embedding and relationships for now.
        """

        if isinstance(input_data, Document):
            text = input_data.text
            metadata = input_data.metadata.copy()
            base_start_idx = input_data.start_char_idx or 0
        else:
            text = input_data
            metadata = {}
            base_start_idx = 0
    
        chunks = self._split_text(text)
        documents = []
        
        current_idx = 0
        for chunk in chunks:
            chunk_text = self.separator.join(chunk).replace("  ", " ").strip()
            # if DEBUG: cprint(chunk_text, chunk, c='blue')
            start_idx = base_start_idx + current_idx
            end_idx = start_idx + len(chunk_text)
            
            doc = Document(
                chunk_text, 
                metadata,
                start_char_idx=start_idx,
                end_char_idx=end_idx
            )
            documents.append(doc)
            
            # Update current_idx to the start of the non-overlapping part of the next chunk
            non_overlapping_text = self.separator.join(chunk[:-self.chunk_overlap])
            current_idx += len(non_overlapping_text)
            if chunk != chunks[-1]:  # If not the last chunk, add separator length
                current_idx += len(self.separator)
        
        return documents

    def _split_text(self, text: str) -> List[List[str]]:
        text = self._preprocess_text(text)
        sentences = self._sentence_splitter(text)
        if DEBUG: cprint(text, sentences, c='blue')

        chunks = self._split_into_chunks(sentences)
        if DEBUG: cprint(chunks, c='blue')

        return chunks

    def _preprocess_text(self, text: str) -> str:
        """I could not found the preprocess or postprocess in the llama index, but I do need this"""
        # Remove page numbers and headers/footers
        text = re.sub(r'\d+\s*\|\s*[\w\s]+$', '', text, flags=re.MULTILINE)
        # Replace newlines with spaces
        text = re.sub(r'\s+', ' ', text)
        # Ensure proper spacing after periods
        text = re.sub(r'\.(?=[A-Z])', '. ', text)
        return text.strip()

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
                      separator: str = " ",
                      sentence_splitter: Optional[Callable[[str], List[str]]] = None,
    ) -> 'SimpleSentenceSplitter':
        return cls(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separator=separator,
            sentence_splitter=default_sentence_splitter() if sentence_splitter is None else sentence_splitter,
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
    
    # for doc in documents:
    #     chunks = splitter.split_text(doc.text)
    #     cprint(len(chunks), c='green')
    #     # doc.chunks = chunks
    #     # doc.metadata["num_chunks"] = len(chunks)

    tmp_doc = documents[5]
    tmp_chunks = splitter.split_text(tmp_doc.text)
    cprint(tmp_doc.text, tmp_doc.metadata)
    cprint(len(tmp_chunks), c='red')
    cprint(tmp_chunks[0].text, tmp_chunks[0].metadata)
