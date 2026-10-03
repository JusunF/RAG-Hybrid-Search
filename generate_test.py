import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from chunking import build_corpus
from sparse_retriever import SparseRetriever
from dense_retriever import DenseRetriever
from hybrid_retriever import HybridRetriever
from rag_pipeline import extractive_answer
from rag_pipeline import generate_with_anthropic

chunks = build_corpus("documents", chunk_size=40, overlap=10)
hybrid = HybridRetriever(SparseRetriever(chunks), DenseRetriever(chunks))

query = "What programming language is good for data sciences?"
results = hybrid.search(query, top_k=3)

print(extractive_answer(query, results))

# answer = generate_with_anthropic(query, results)
# print("Question:", query)
# print("Answer:", answer)