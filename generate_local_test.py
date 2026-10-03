import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from query import load_index
from rag_pipeline import generate_with_ollama

hybrid = load_index(os.path.join(os.path.dirname(__file__), "index.pkl"))

query = "What does python mean?"
results = hybrid.search(query, top_k=3)

answer = generate_with_ollama(query, results)
print("Q:", query)
print("A:", answer)