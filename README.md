# RAG with Hybrid Search

A Retrieval-Augmented Generation (RAG) system built from scratch in Python,
combining BM25 keyword search and neural embedding search through
Reciprocal Rank Fusion (RRF) — with a persisted index, free local LLM
generation via Ollama, and a measured evaluation harness. The demo
knowledge base covers VALORANT esports: VCT tournament format, VALORANT
Champions 2025 results, and profiles of top teams (NRG, Fnatic, Paper Rex)
and player transfers.

## The pipeline

This project has two separate pipelines that run at different times.

### 1. Ingestion — `src/build_index.py` (run once, or after editing `documents/`)

```
documents/*.txt
      │
      ▼
  chunking.py        split each document into overlapping ~40-word chunks
      │               (10-word overlap, so no sentence is cleanly severed
      │                at a chunk boundary)
      ▼
  ┌───────────────┴───────────────┐
  ▼                                ▼
sparse_retriever.py          dense_retriever.py
BM25Okapi index over         sentence-transformers (all-MiniLM-L6-v2)
stopword-filtered,           encodes each chunk into a 384-dimension
lowercased word tokens       vector, normalized for cosine similarity
  │                                │
  └───────────────┬───────────────┘
                   ▼
              index.pkl
     (chunks + BM25 index + embedding matrix, saved to disk)
```

### 2. Query — `src/query.py` (run anytime after the index exists)

```
your question
      │
      ▼
  load index.pkl           no re-chunking, no re-embedding the corpus —
      │                     only the new question gets processed
      ▼
  ┌───────────────┴───────────────┐
  ▼                                ▼
BM25 search                  embed question → cosine similarity
(exact word match)           search (meaning match)
  │                                │
  └───────────────┬───────────────┘
                   ▼
         hybrid_retriever.py
    Reciprocal Rank Fusion — merges both
    ranked lists by RANK POSITION, not raw
    score (BM25 scores and cosine similarities
    aren't on comparable scales)
                   ▼
            top-k chunks returned
                   │
         (optional) rag_pipeline.py
         build_prompt() assembles chunks + question
         into a prompt → sent to a local LLM via Ollama
         (generate_with_ollama) → natural-language answer
```

**Why two separate pipelines, not one script:** embedding a corpus is the
expensive step. Re-running it on every single question would be wasteful
and slow at any real scale. Ingestion does that work once and saves the
result; query time only ever has to embed one short string — the
incoming question.

## The documents (`documents/`)

Six plain-text files, each covering one topic, chunked into 28 total
passages:

| File | Covers |
|---|---|
| VCT format doc | How the VALORANT Champions Tour is structured — Kickoff, Stage 1/2, Masters, Champions; the 2026 format changes (triple-elimination Kickoff, Path to Champions for Challengers teams) |
| Champions 2025 results | NRG's 3-2 grand final win over Fnatic in Paris, tournament dates, prize pool, MVP |
| NRG team profile | Roster (Ethan, s0m, FNS, mada, Verno), Ethan's two Champions titles, 2026 roster changes |
| Fnatic team profile | Roster (Boaster, Alfajer, crashies, kaajak, Veqaj), their three 2025 runner-up finishes |
| Paper Rex team profile | Roster (f0rsakeN, Jinggg, and others), Masters Toronto win, the "PRX" abbreviation |
| Notable transfers | aspas's move to MIBR, Demon1 to Leviatan, Derke's move to Team Vitality |

Run `dir documents` (Windows) to see the actual filenames in your copy —
this table describes content, not exact file names, since `doc1.txt`
etc. aren't self-descriptive.

**A deliberate lesson baked into this corpus:** esports content is full
of abbreviations (PRX, FNC, IGL, EMEA) that a real user will query with.
Documents need to spell those out explicitly at least once — BM25 and
embeddings can only match what's actually written in the text, not what a
human reader would infer.

## What's inside `index.pkl`

A single pickled Python dictionary — the project's hand-built equivalent
of a vector database:

```python
{
    "chunks": [...],              # every Chunk object: id, source doc, text
    "sparse": <SparseRetriever>,  # the full BM25 index, ready to search
    "dense_embeddings": <array>,  # shape (28, 384) — one 384-number vector
                                   # per chunk
    "embedder_model_name": "all-MiniLM-L6-v2",
}
```

Two things worth knowing about why it's built this way:

- **The embedding model itself is never pickled** — only its *name*.
  Reloading a cached sentence-transformers model from disk is fast;
  repickling a large neural network object every time would be wasteful.
  `query.py` reconstructs a fresh embedder from the saved name and
  attaches the saved `dense_embeddings` array to it, skipping re-encoding
  entirely.
- **It is not automatically kept in sync with `documents/`.** Editing a
  `.txt` file has no effect until `build_index.py` is rerun. This mirrors
  a real operational concern in production RAG systems — re-indexing has
  to be triggered deliberately or on a schedule.

`index.pkl` is a build artifact, not source data — it's excluded via
`.gitignore` and regenerated locally with `build_index.py`.

## Setup

```bash
pip install -r requirements.txt
```

First run of `build_index.py` or `query.py` downloads the embedding model
(~90MB) from Hugging Face — one-time, then cached locally.

For local, free LLM generation (optional):
```bash
# install Ollama from https://ollama.com, then:
ollama pull llama3.2
```

## Usage

```bash
# 1. Build the index (run once, or after editing documents/)
python src/build_index.py

# 2. Ask questions interactively (retrieval only, no LLM needed)
python src/query.py

# 3. Measure retrieval quality (sparse vs dense vs hybrid recall@k)
python evaluate.py
```

Generating a full natural-language answer (requires Ollama running):
```python
import sys; sys.path.insert(0, "src")
from query import load_index
from rag_pipeline import generate_with_ollama

hybrid = load_index("index.pkl")
results = hybrid.search("who won valorant champions 2025", top_k=3)
print(generate_with_ollama("who won valorant champions 2025", results))
```

## Key design decisions

- **RRF over weighted score fusion** — fuses on rank position, not raw
  score, since BM25 scores and cosine similarities live on incompatible
  scales and can't be blended directly without normalization assumptions.
- **Stopword filtering in BM25** — without it, filler words
  ("is", "for", "good") let BM25 award partial credit to completely
  irrelevant chunks just for sharing common words with the query.
- **Overlap in chunking** — each chunk shares its last 10 words with the
  next chunk's first 10, so a sentence spanning a chunk boundary is never
  fully lost to either half.

## Limitations

- `index.pkl` + brute-force cosine similarity scans every chunk on every
  query — fine at dozens of chunks, too slow at real scale. A production
  system would use an approximate nearest-neighbor vector database
  (FAISS, Pinecone, Weaviate).
- The knowledge base reflects whatever was true when the documents were
  written — esports rosters and results change constantly and this is
  not live data.
- Small local models (Llama 3.2) can still misjudge genuinely ambiguous
  questions even when the retrieved context is correct.

## Possible next steps

- Swap `index.pkl` for FAISS or a hosted vector database to test at scale
- Add a cross-encoder reranker after fusion for a precision boost on the
  top candidates
- Expand `evaluate.py`'s hand-built test set and track recall@k as the
  corpus grows