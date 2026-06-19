# Golgi — Healthcare Search Engine
## Instructions for Claude Code

This is a real production-quality project. Every file must be fully implemented — no stubs, no TODOs in final code. Follow the architecture exactly as described.

---

## Project Overview

Golgi is an LLM-powered intelligent search & retrieval system over 50+ healthcare sources (PubMed, NIH, WHO, FDA). It uses:
- Hybrid BM25 + FAISS dense retrieval
- Sentence-transformers embeddings (all-MiniLM-L6-v2)
- Cross-encoder re-ranking (ms-marco-MiniLM-L-6-v2)
- Query expansion
- Intent-aware LangChain orchestration routing queries to the right source APIs
- Claude (claude-sonnet-4-6) for cited LLM responses
- RAGAS evaluation harness
- Streamlit UI

---

## Environment Variables (in .env)

```
ANTHROPIC_API_KEY=your_key_here
PUBMED_API_KEY=your_ncbi_key_here
PUBMED_EMAIL=your_email_here
```

---

## Module Specifications

### 1. `ingestion/fetchers.py`
Implement async fetchers for each source:

- `fetch_pubmed(query, max_results=20)` — use NCBI E-utilities API (esearch + efetch). Returns list of dicts: `{id, title, abstract, url, source, date}`
- `fetch_nih_reporter(query, max_results=10)` — use NIH RePORTER API v2 `https://api.reporter.nih.gov/v2/projects/search`. Returns project title, abstract, PI name, url.
- `fetch_fda_drugs(query, max_results=10)` — use openFDA API `https://api.fda.gov/drug/label.json`. Returns drug name, indications, warnings, url.
- `fetch_who_iris(query, max_results=10)` — use WHO IRIS OAI-PMH or search API. Returns title, description, url.
- `fetch_all(query)` — runs all 4 fetchers concurrently with `asyncio.gather`, returns combined list tagged by source.

### 2. `ingestion/chunker.py`
- `chunk_documents(docs, chunk_size=512, overlap=50)` — takes list of doc dicts, splits `abstract` or `description` field using LangChain `RecursiveCharacterTextSplitter`. Returns list of `{chunk_text, metadata: {id, title, url, source, date}}`.
- `embed_and_store(chunks, collection_name="golgi")` — embeds chunks using `sentence-transformers/all-MiniLM-L6-v2`, stores in ChromaDB persistent collection at `./data/processed/chroma`. Returns collection.

### 3. `retrieval/dense.py`
- `DenseRetriever` class
  - `__init__(collection_name)` — loads existing Chroma collection
  - `search(query, k=20)` — embeds query, returns top-k chunks with scores as `{chunk_text, metadata, score}`

### 4. `retrieval/sparse.py`
- `BM25Retriever` class
  - `__init__(chunks)` — builds BM25 index from list of chunk texts using `rank_bm25` library
  - `search(query, k=20)` — tokenizes query, returns top-k with BM25 scores

### 5. `retrieval/hybrid.py`
- `HybridRetriever` class
  - `__init__(dense_retriever, bm25_retriever, alpha=0.5)` — alpha controls dense vs sparse weight
  - `search(query, k=20)` — runs both retrievers, normalises scores to [0,1], combines as `alpha * dense_score + (1-alpha) * sparse_score`, returns top-k merged results
  - `expand_query(query)` — uses Claude to generate 2 alternative phrasings of the query, runs hybrid search on all 3, deduplicates and returns merged top-k

### 6. `reranking/cross_encoder.py`
- `CrossEncoderReranker` class
  - `__init__()` — loads `cross-encoder/ms-marco-MiniLM-L-6-v2` from sentence-transformers
  - `rerank(query, candidates, top_k=5)` — scores all query-candidate pairs, returns top_k sorted by cross-encoder score

### 7. `orchestration/router.py`
- `IntentRouter` class using LangChain
  - `detect_intent(query)` — uses Claude to classify query into one of: `drug_info`, `clinical_evidence`, `guidelines`, `research_funding`, `general`
  - Each intent maps to preferred sources:
    - `drug_info` → FDA first, then PubMed
    - `clinical_evidence` → PubMed first, then NIH
    - `guidelines` → WHO first, then NIH
    - `research_funding` → NIH RePORTER
    - `general` → all sources equally
  - `route_and_fetch(query)` — detects intent, fetches from sources in priority order, returns combined docs with intent label

### 8. `orchestration/pipeline.py`
- `GolgiPipeline` class — the main end-to-end pipeline
  - `__init__()` — initialises all components
  - `run(query)` → dict with keys:
    - `intent`: detected intent string
    - `answer`: LLM-generated answer string (from Claude, grounded only in retrieved context)
    - `sources`: list of `{title, url, source, relevance_score}` — top 5 cited sources
    - `chunks_used`: the actual text chunks passed to Claude
    - `retrieval_trace`: list of steps taken (for UI display)
  - Claude system prompt must instruct: answer ONLY from provided context, cite sources by [1], [2] etc., say "I cannot find this in the retrieved sources" if not found

### 9. `evaluation/ragas_eval.py`
- `run_evaluation(test_queries_path)` — loads test queries from JSON file
- Each test query: `{question, ground_truth_answer, expected_sources}`
- Runs pipeline on each, collects `{question, answer, contexts, ground_truth}`
- Uses RAGAS to compute: `faithfulness`, `answer_relevancy`, `context_recall`, `context_precision`
- Saves results to `evaluation/results.json` and prints summary table

### 10. `evaluation/test_queries.json`
Create 10 realistic test queries covering all intents:
```json
[
  {"question": "What are the side effects of metformin?", "ground_truth": "...", "expected_sources": ["FDA"]},
  ...
]
```

### 11. `ui/app.py`
Streamlit app with:
- Header: "Golgi — Healthcare Intelligence Search"
- Search bar (large, centered)
- Source filter checkboxes: PubMed, NIH, WHO, FDA (all checked by default)
- On search:
  - Show spinner: "Searching 50+ healthcare sources..."
  - Display intent badge (e.g., 🔬 Clinical Evidence)
  - Show answer in a clean card with numbered citations inline
  - Expandable "Sources" section: cards per source with title, source badge, URL link, relevance score bar
  - Expandable "Retrieval trace" section: shows steps taken (intent detected → sources fetched → chunks retrieved → reranked → answer generated)
  - Show metrics: retrieval time (ms), chunks searched, sources found
- Sidebar: About section, API status indicators

### 12. `tests/test_retrieval.py`
Unit tests for:
- BM25 returns results for known queries
- Dense retriever returns results
- Hybrid merger produces scores in [0,1]
- Cross-encoder reranks correctly (top result should change)

### 13. `requirements.txt`
```
anthropic
langchain
langchain-anthropic
langchain-community
chromadb
sentence-transformers
rank-bm25
streamlit
ragas
datasets
httpx
python-dotenv
pytest
```

---

## Code Quality Standards
- All functions have docstrings
- Type hints on all function signatures
- Async where appropriate (fetchers)
- Error handling: if a source API fails, log warning and continue (never crash the pipeline)
- No hardcoded API keys — always use `os.getenv()`
- Logging via Python `logging` module, not print statements

## Implementation Order
1. `ingestion/fetchers.py` — get real data flowing first
2. `ingestion/chunker.py` — process and store it
3. `retrieval/dense.py` + `retrieval/sparse.py` — both retrievers
4. `retrieval/hybrid.py` — merge them
5. `reranking/cross_encoder.py` — add reranking
6. `orchestration/router.py` + `orchestration/pipeline.py` — wire everything together
7. `ui/app.py` — the demo
8. `evaluation/ragas_eval.py` — evaluation last
