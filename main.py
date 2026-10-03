import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from chunking import build_corpus
from sparse_retriever import SparseRetriever
from dense_retriever import DenseRetriever
from hybrid_retriever import HybridRetriever

chunks = build_corpus("documents", chunk_size=400, overlap=10)
sparse = SparseRetriever(chunks)
dense = DenseRetriever(chunks)
hybrid = HybridRetriever(sparse, dense)

query = "Python"
print("Sparse:", [(c.chunk_id, round(s,2)) for c,s in sparse.search(query,3)])
print("Dense:", [(c.chunk_id, round(s,2)) for c,s in dense.search(query,3)])
print("Hybrid:", [(c.chunk_id, round(s,3)) for c,s in hybrid.search(query,3)])