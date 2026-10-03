"""
query.py
----------
QUERY-TIME pipeline — run this any time you want to ask questions.
Loads the pre-built index from build_index.py instead of re-chunking and
re-embedding everything from scratch. This is the fast path.
"""

import os
import sys
import pickle

sys.path.insert(0, os.path.dirname(__file__))  # so pickle can resolve "chunking", "sparse_retriever", etc.

from dense_retriever import DenseRetriever, SentenceTransformerEmbedder
from hybrid_retriever import HybridRetriever


def load_index(index_path: str):
    print("Loading index...")
    with open(index_path, "rb") as f:
        data = pickle.load(f)

    chunks = data["chunks"]
    sparse = data["sparse"]

    # Rebuild a lightweight embedder (loads the cached model — fast after first download)
    embedder = SentenceTransformerEmbedder(data["embedder_model_name"])
    dense = DenseRetriever(chunks, embedder=embedder, embeddings=data["dense_embeddings"])

    print(f"Loaded {len(chunks)} chunks.")
    return HybridRetriever(sparse, dense)


def main():
    index_path = os.path.join(os.path.dirname(__file__), "..", "index.pkl")
    hybrid = load_index(index_path)

    print("\nAsk a question (or type 'quit' to exit):")
    while True:
        query = input("\n> ").strip()
        if query.lower() in ("quit", "exit"):
            break
        if not query:
            continue

        results = hybrid.search(query, top_k=3)
        print()
        for chunk, score in results:
            print(f"  {score:.3f}  [{chunk.chunk_id}]  {chunk.text[:100]}...")


if __name__ == "__main__":
    main()