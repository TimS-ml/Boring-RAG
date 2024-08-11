from pathlib import Path
from abc import ABC, abstractmethod
from typing import (
    List, 
    Optional, 
    Any, 
    Dict
)
from abc import ABC, abstractmethod
from enum import Enum
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from boring_rag_core.embeddings.base import BaseEmbedding
from boring_rag_core.schema import Document, TransformComponent
from boring_utils.utils import cprint, tprint, get_device

Embedding = List[float]

# DEFAULT_HUGGINGFACE_EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"  # ~438M
# DEFAULT_HUGGINGFACE_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"  # ~134M
DEFAULT_HUGGINGFACE_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # ~70M
DEFAULT_EMBED_INSTRUCTION = "Represent the document for retrieval: "
DEFAULT_QUERY_INSTRUCTION = "Represent the question for retrieving supporting documents: "
DEFAULT_EMBED_BATCH_SIZE = 32


class HuggingFaceEmbedding(BaseEmbedding):
    """
    ref:
        https://www.sbert.net/docs/package_reference/sentence_transformer/SentenceTransformer.html
    """
    def __init__(
        self,
        model_name: str = DEFAULT_HUGGINGFACE_EMBEDDING_MODEL,
        max_length: int = 512,
        normalize: bool = True,
        embed_batch_size: int = DEFAULT_EMBED_BATCH_SIZE,
        query_instruction: Optional[str] = DEFAULT_QUERY_INSTRUCTION,
        text_instruction: Optional[str] = DEFAULT_EMBED_INSTRUCTION,
        cache_folder: Optional[str] = None,
        trust_remote_code: bool = False,
        device: Optional[str] = None,
        **model_kwargs,
    ):
        super().__init__()
        self.model_name = model_name
        self.max_length = max_length
        self.normalize = normalize
        self.embed_batch_size = embed_batch_size
        self._device = device if device else get_device(return_str=True)
        
        # NOTE: in llama_index, get_query_instruct_for_model_name mainly for INSTRUCTOR / BGE detection
        #       but we will skip that for now 
        self._model = SentenceTransformer(
                    model_name, 
                    trust_remote_code=trust_remote_code,
                    device=self._device,
                    cache_folder=cache_folder,
                    prompts={
                        "query": query_instruction,
                        "text": text_instruction,
                    },
                    **model_kwargs,
                )
        self._model.max_seq_length = max_length

    def _embed(
        self,
        sentences: List[str],
        prompt_name: Optional[str] = None,
    ) -> List[List[float]]:
        """Embed sentences."""
        return self._model.encode(
            sentences,
            batch_size=self.embed_batch_size,
            prompt_name=prompt_name,
            normalize_embeddings=self.normalize,
        ).tolist()

    def get_query_embedding(self, query: str, prompt_name="query") -> Embedding:
        embeddings = self._embed([query], prompt_name=prompt_name)[0]
        return embeddings

    def get_text_embedding(self, text: str) -> Embedding:
        embeddings = self._embed([text])[0]
        return embeddings

    def get_text_embeddings(self, texts: List[str]) -> List[Embedding]:
        embeddings = self._embed(texts)
        return embeddings

    def embed_documents(self, documents: List[Document]) -> List[Document]:
        texts = [doc.text for doc in documents]
        embeddings = self.get_text_embeddings(texts)
        
        for doc, embedding in zip(documents, embeddings):
            doc.embedding = embedding
        
        return documents

    @classmethod
    def class_name(cls) -> str:
        return "HuggingFaceEmbedding"


if __name__ == '__main__':
    import os
    from boring_utils.utils import cprint, tprint
    from boring_rag_core.readers.base import PDFReader

    pdf_path = Path(os.getenv('DATA_DIR')) / 'nutrition' / 'human-nutrition-text_ch1.pdf'
    reader = PDFReader()
    documents = reader.load_data(file=pdf_path)
    cprint(len(documents), c='red')
    cprint(documents[0].metadata)    

    embedding = HuggingFaceEmbedding()

    tprint('Single Query Embedding')
    query_embed = embedding.get_text_embedding("Tell me something about nutrition.")
    cprint(query_embed[:5])

    tprint('Calc Embedding')
    documents = embedding.embed_documents(documents[:3])  # just to speed the things up...
    cprint(documents[0].embedding[:5])
    cprint(documents[0].metadata)

    tprint('Calc Similarity')
    query_embed_sim_0 = embedding.similarity(
           query_embed,
           documents[0].embedding,
           )
    query_embed_sim_1 = embedding.similarity(
           query_embed,
           documents[1].embedding,
           )
    cprint(query_embed_sim_0, query_embed_sim_1)

    # # test save
    # embed_path = Path(os.getenv('DATA_DIR')) / 'nutrition' / 'test_embed.txt'
    # save_embedding(documents[0].embedding, embed_path)

