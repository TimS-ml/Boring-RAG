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

# import IPython; IPython.embed()

