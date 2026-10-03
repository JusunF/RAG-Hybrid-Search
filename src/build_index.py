"""
build_index.py
----------------
INGESTION pipeline — run this once, or whenever documents/ changes.
Chunks + embeds + indexes everything, then saves it to disk so query time
never has to redo this expensive work.
"""

import os
import pickle
from chunking import build_corpus
from sparse_retriever import SparseRetriever
from dense_retriever import DenseRetriever


def build_and_save(documents_folder: str, index_path: str, chunk_size: int = 40, overlap: int = 10):
    print(f"Loading and chunking documents from {documents_folder}...")
    chunks = build_corpus(documents_folder, chunk_size, overlap)
    print(f"  {len(chunks)} chunks created")

    print("Building sparse (BM25) index...")
    sparse = SparseRetriever(chunks)

    print("Computing dense embeddings (this is the slow part)...")
    dense = DenseRetriever(chunks)

    print(f"Saving index to {index_path}...")
    with open(index_path, "wb") as f:
        pickle.dump({
            "chunks": chunks,
            "sparse": sparse,
            "dense_embeddings": dense.embeddings,
            "embedder_model_name": dense.embedder.model_name,
        }, f)
    print("Done.")


if __name__ == "__main__":
    here = os.path.dirname(__file__)
    build_and_save(
        documents_folder=os.path.join(here, "..", "documents"),
        index_path=os.path.join(here, "..", "index.pkl"),
    )