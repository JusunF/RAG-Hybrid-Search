# RAG with Hybrid Search

A Retrieval-Augmented Generation (RAG) system built from scratch, combining
keyword search (BM25) and semantic search (neural embeddings) through
hybrid fusion — with a complete ingestion/query pipeline and free local
LLM generation. Includes a demo knowledge base of VALORANT esports info
(VCT format, pro teams, player transfers).

## Why hybrid search?

Keyword search (BM25) is precise but blind to meaning — it can't match
"PRX" to "Paper Rex" unless the text says both. Semantic search
(embeddings) understands meaning and paraphrasing, but can miss exact
terms, codes, or names. This project runs both and fuses their rankings
with **Reciprocal Rank Fusion (RRF)**, so a query like *"what happened to
aspas"* (no word overlap with the source text) and a query like *"team
prx"* (an exact abbreviation) both retrieve correctly.

## Architecture