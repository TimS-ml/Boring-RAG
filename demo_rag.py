# %%
import os
import re

from pathlib import Path
from spacy.tokens import Doc
from tqdm.auto import tqdm
from random import random

from boring_utils.utils import cprint
from boring_rag_core.readers.base import PDFReader

# %%[markdown]
# PDF Import

# %%
pdf_path = Path(os.getenv('DATA_DIR')) / 'nutrition' / 'human-nutrition-text.pdf'
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
from boring_rag_core.node_parser.text.sentence import SimpleSentenceSplitter

splitter = SimpleSentenceSplitter.from_defaults()


# %%[markdown]
# Split to chunk (let's test doc 0)

# %%
tmp_doc = documents[5]
tmp_chunks = splitter.split_text(tmp_doc.text)
cprint(tmp_doc.text, tmp_doc.metadata)
cprint(len(tmp_chunks), c='red')
cprint(tmp_chunks[0].text, tmp_chunks[0].metadata)

# import IPython; IPython.embed()

