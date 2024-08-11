from pathlib import Path
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

def save_embedding(embedding: List[float], file_path: str) -> None:
    """Save embedding to file."""
    with open(file_path, "w") as f:
        f.write(",".join([str(x) for x in embedding]))


def load_embedding(file_path: str) -> List[float]:
    """Load embedding from file. Will only return first embedding in file."""
    with open(file_path) as f:
        for line in f:
            embedding = [float(x) for x in line.strip().split(",")]
            break
        return embedding


if __name__ == '__main__':
    import os
    from boring_utils.utils import cprint
    from boring_rag_core.readers.base import PDFReader
    from boring_rag_core.embeddings.huggingface import HuggingFaceEmbedding
    from boring_rag_core.embeddings.utils import save_embedding

    pdf_path = Path(os.getenv('DATA_DIR')) / 'nutrition' / 'human-nutrition-text_ch1.pdf'
    reader = PDFReader()
    documents = reader.load_data(file=pdf_path)
    cprint(len(documents), c='red')
    cprint(documents[0].metadata)    

    embedding = HuggingFaceEmbedding()
    documents = embedding.embed_documents(documents[:3])
    cprint(documents[0].embedding)

    # test save
    embed_path = Path(os.getenv('DATA_DIR')) / 'nutrition' / 'test_embed.txt'
    save_embedding(documents[0].embedding, embed_path)

    # test load
    embedding = load_embedding(embed_path)
    cprint(embedding)
