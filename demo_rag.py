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
from spacy.lang.en import English
nlp = English()
nlp.add_pipe("sentencizer")

# persudo code
# https://docs.llamaindex.ai/en/stable/module_guides/loading/node_parsers/
for doc in tqdm(documents):
    # item is a dataclass with keys: text, metadata, id_, etc.
    doc.sentences = list(nlp(doc.text).sents)

    # Make sure all sentences are strings
    doc.sentences = [str(sentence) for sentence in doc.sentences]

    # Count the sentences
    doc.metadata["page_sentence_count_spacy"] = len(doc.sentences)


# random.sample(documents, k=1)


# %%[markdown]
# Split to chunk 

# %%
# Define split size to turn groups of sentences into chunks
num_sentence_chunk_size = 10

# Create a function that recursively splits a list into desired sizes
def split_list(input_list: list,
               slice_size: int) -> list[list[str]]:
    """
    Splits the input_list into sublists of size slice_size (or as close as possible).

    For example, a list of 17 sentences would be split into two lists of [[10], [7]]
    """
    return [input_list[i:i + slice_size] for i in range(0, len(input_list), slice_size)]


# Loop through pages and texts and split sentences into chunks
for item in tqdm(documents):
    item["sentence_chunks"] = split_list(input_list=item["sentences"],
                                         slice_size=num_sentence_chunk_size)
    item["num_chunks"] = len(item["sentence_chunks"])


# Split each chunk into its own item
pages_and_chunks = []
for item in tqdm(documents):
    for sentence_chunk in item["sentence_chunks"]:
        chunk_dict = {}
        chunk_dict["page_number"] = item["page_number"]

        # Join the sentences together into a paragraph-like structure, aka a chunk (so they are a single string)
        joined_sentence_chunk = "".join(sentence_chunk).replace("  ", " ").strip()
        joined_sentence_chunk = re.sub(r'\.([A-Z])', r'. \1', joined_sentence_chunk) # ".A" -> ". A" for any full-stop/capital letter combo
        chunk_dict["sentence_chunk"] = joined_sentence_chunk

        # Get stats about the chunk
        chunk_dict["chunk_char_count"] = len(joined_sentence_chunk)
        chunk_dict["chunk_word_count"] = len([word for word in joined_sentence_chunk.split(" ")])
        chunk_dict["chunk_token_count"] = len(joined_sentence_chunk) / 4 # 1 token = ~4 characters

        pages_and_chunks.append(chunk_dict)

# How many chunks do we have?
len(pages_and_chunks)
