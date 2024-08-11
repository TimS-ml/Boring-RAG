# %%
import os
import re

from pathlib import Path
from spacy.tokens import Doc
from tqdm.auto import tqdm
from random import random

from boring_utils.utils import cprint, tprint

# %%[markdown]
# PDF Import

# %%
from boring_rag_core.readers.base import PDFReader
tprint('PDF Import')

# pdf_path = Path(os.getenv('DATA_DIR')) / 'nutrition' / 'human-nutrition-text.pdf'
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

# %%[markdown]
# Sentence Splitter 
# NodeParser -> TextSplitter -> MetadataAwareTextSplitter -> SentenceSplitter

# %%
from boring_rag_core.node_parser.text.sentence import SimpleSentenceSplitter, default_sentence_splitter_spacy
tprint('Sentence Splitter')

splitter = SimpleSentenceSplitter.from_defaults()
# splitter = SimpleSentenceSplitter.from_defaults(
#     sentence_splitter=default_sentence_splitter_spacy()
# )


# %%[markdown]
# Split to chunk (let's test doc 0)
# [Chunking Strategies for LLM Applications | Pinecone](https://www.pinecone.io/learn/chunking-strategies/)
# [MTEB Leaderboard - a Hugging Face Space by mteb](https://huggingface.co/spaces/mteb/leaderboard)
# [Text Splitters | 🦜️🔗 LangChain](https://python.langchain.com/v0.1/docs/modules/data_connection/document_transformers/)

# %%
tmp_doc = documents[5]
# tmp_chunks = splitter.split_text(tmp_doc.text)  # do not copy metadata and start_char_idx
tmp_chunks = splitter.split_text(tmp_doc)

tprint('doc node preview', sep='-')
cprint(tmp_doc.text, tmp_doc.metadata)
print()

tprint('separator node preview', sep='-')
cprint('NOTE: pay attention to the chunk overlap', c='red')

cprint(len(tmp_chunks), c='red')

for id in range(2):
    cprint(tmp_chunks[id].text, tmp_chunks[id].metadata)
    cprint(tmp_chunks[id].start_char_idx, tmp_chunks[id].end_char_idx)

cprint(tmp_chunks[0].start_char_idx == tmp_doc.start_char_idx)

# %%[markdown]
# Embeddings
# Checkout `llama-index-integrations/embeddings/llama-index-embeddings-huggingface/`

# %%
from boring_rag_core.embeddings.huggingface.base import HuggingFaceEmbedding

embedding = HuggingFaceEmbedding()
documents = splitter.split_text(documents[0])  # only split chunk 0 for now

tprint('Single Query Embedding')
query_embed = embedding.get_text_embedding("Tell me something about nutrition.")
cprint(query_embed[:5])

tprint('Calc Embedding', sep='-')
documents = embedding.embed_documents(documents[:3])  # just to speed the things up...
cprint(documents[0].embedding[:5])
cprint(documents[0].metadata)

tprint('Calc Similarity', sep='-')
query_embed_sim_0 = embedding.similarity(
       query_embed,
       documents[0].embedding,
       )
query_embed_sim_1 = embedding.similarity(
       query_embed,
       documents[1].embedding,
       )
cprint(query_embed_sim_0, query_embed_sim_1)



# %%
# import IPython; IPython.embed()

# %%[markdown]
# Indexing
# Basically we pack SentenceSplitter and HuggingFaceEmbedding together into IngestionPipeline._get_default_transformations
# VectorStoreIndex.from_documents -> as_query_engine / as_chat_engine / as_retriever
