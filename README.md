# 😑😑😑 Boring RAG 😑😑😑 

(WIP) A RAG with LLM that generates boring text. Implemented everything from scratch.

Ref:
- [Stanford CS25: V3 I Retrieval Augmented Language Models - YouTube](https://www.youtube.com/watch?v=mE7IDf2SmJg)
- https://github.com/mrdbourke/simple-local-rag/tree/main


# TODO
- [x] Preprocessing + PDF Reader
- [x] Chunking, SentenceSplit
- [x] Embedding chunks
- [x] Save the embedding
- [x] Similarity Search
- [ ] (TBD) Embedding pooling

- [x] BaseIndex Draft
- [x] Ingestion Pipeline: Pack into splitter and embedding into IngestionPipeline
- [x] Storage Context

- [ ] BaseEmbedding.similarity vs SimpleVectorStore.query vs RetrieverQueryEngine._query
- [ ] Retrieval
- [ ] Generation

- [ ] Re-org demo_rag.py


# BUGs
- [x] PDF doc needs post processing 
- [x] Chunking's metadata is not saved
- [x] Pydantic requires 1.10.14
