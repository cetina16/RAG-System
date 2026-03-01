# Production-Ready RAG System

A fully deployable **Retrieval-Augmented Generation (RAG)** system built with FastAPI, LangGraph, FAISS, and Mistral. Designed to ingest messy real-world PDFs, retrieve relevant passages using hybrid search, and generate grounded responses with source citations.

Includes a **built-in web UI** served at `/` — no separate frontend server needed.

---

## Aim

Standard LLMs hallucinate when asked about private or domain-specific knowledge. RAG solves this by grounding every response in retrieved documents — the model can only answer from what it actually finds. This system demonstrates a production-grade RAG pipeline with:

- PDF ingestion with text cleaning and duplicate detection
- Hybrid retrieval (semantic + keyword) for higher recall
- Cross-encoder re-ranking for higher precision
- A stateful LangGraph pipeline with query rewriting
- Full observability via structured logging and per-node tracing
- A clean REST API with SSE streaming support
- A built-in chat UI served directly by FastAPI
- Rate limiting and duplicate ingest protection

---

## Architecture

```
  Browser
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│                      FastAPI Server                          │
│                                                             │
│   GET  /                     → Web UI (index.html)          │
│   GET  /api/v1/health        → index stats + cache info     │
│   POST /api/v1/ingest        → PDF upload (rate: 10/min)    │
│   POST /api/v1/query         → RAG query + SSE streaming    │
│   GET  /api/v1/documents     → list indexed files           │
│   DELETE /api/v1/index       → reset FAISS index            │
└────────────┬──────────────────────────┬─────────────────────┘
             │                          │
  ┌──────────▼──────────┐   ┌───────────▼───────────────────┐
  │  INGESTION PIPELINE  │   │   QUERY PIPELINE (LangGraph)  │
  │                      │   │                               │
  │  MD5 duplicate check │   │  ┌─────────────────────────┐  │
  │  PyMuPDF extraction  │   │  │   query_rewriter        │  │
  │  ftfy text cleaning  │   │  │   LLM → 3 query variants│  │
  │  512-tok chunking    │   │  └────────────┬────────────┘  │
  │  BGE-base embeddings │   │               ↓               │
  │  FAISS IndexFlatIP   │◄──│  ┌─────────────────────────┐  │
  │  (disk persisted)    │   │  │   retriever_node        │  │
  └──────────────────────┘   │  │   BM25 + FAISS + RRF    │  │
                             │  │   top-20 docs           │  │
                             │  └────────────┬────────────┘  │
                             │               ↓               │
                             │  ┌─────────────────────────┐  │
                             │  │   reranker_node         │  │
                             │  │   cross-encoder → top-5 │  │
                             │  └────────────┬────────────┘  │
                             │               ↓               │
                             │  ┌─────────────────────────┐  │
                             │  │   generator_node        │  │
                             │  │   Mistral LLM + cites   │  │
                             │  └─────────────────────────┘  │
                             └───────────────────────────────┘
                                            │
                            ┌───────────────▼──────────────┐
                            │   TTLCache (SHA256 key)       │
                            │   cache hit → skip pipeline  │
                            └──────────────────────────────┘
```

### LangGraph Pipeline Flow

```
START
  │
  ▼
query_rewriter    LLM expands query into 3 variants for broader recall
  │
  ▼
retriever_node    BM25 (keyword) + FAISS (semantic) on all variants
  │               Reciprocal Rank Fusion merges results → top-20 docs
  │
  ├─ [0 docs found] ──────────────────────────────────┐
  │                                                    │
  ▼                                                    │
reranker_node     cross-encoder scores each (query,    │
  │               passage) pair → reorders → top-5     │
  │                                                    │
  ▼                                                    ▼
generator_node    builds prompt with context + query → LLM → response + [Doc N] citations
  │
  ▼
END
```

---

## Web UI

A chat interface is served at `http://localhost:8000` — no React, no separate build step, no extra server. It is a single static HTML file served by FastAPI.

**Features:**
- Drag-and-drop PDF upload with progress bar
- Duplicate file detection (re-uploading the same PDF shows an error)
- Live index stats (document count, cache size)
- List of all ingested files in the sidebar
- Chat interface with streaming token output
- Citations displayed under each response (source file, page, rerank score)
- Per-node latency trace shown in non-streaming mode
- One-click index reset with confirmation dialog

---

## Tech Stack

| Layer | Technology | Why |
|---|---|---|
| API framework | FastAPI | Async, auto-docs (Swagger), SSE streaming, static file serving |
| Frontend | Vanilla HTML/CSS/JS | Zero build step, served directly by FastAPI, no framework overhead |
| Pipeline orchestration | LangGraph | Stateful graph with conditional edges, easy to extend |
| Document processing | PyMuPDF (`fitz`) | Best-in-class messy PDF extraction |
| Text cleaning | ftfy | Fixes Unicode mojibake, encoding errors |
| Chunking | LangChain `RecursiveCharacterTextSplitter` | Respects sentence/paragraph boundaries |
| Embeddings | `BAAI/bge-base-en-v1.5` (sentence-transformers) | Strong free model, 768-dim, MPS/CUDA aware |
| Vector store | FAISS `IndexFlatIP` | Fast cosine search, local, no infra needed |
| Keyword search | BM25 (`rank_bm25`) | Captures exact matches that semantic search misses |
| Fusion | Reciprocal Rank Fusion (RRF) | Parameter-free, robust merging of ranked lists |
| Re-ranking | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Precise (query, passage) scoring after first-stage retrieval |
| LLM | Mistral (API) / Stub | Swappable via env var, works without API key |
| Logging | structlog | Structured JSON logs, per-request traces |
| Caching | cachetools `TTLCache` | In-memory LRU + TTL, no Redis dependency |
| Rate limiting | slowapi | Per-IP rate limiting on ingest endpoint |
| Containerisation | Docker + docker-compose | Single-command deployment |

---

## Project Structure

```
RAG-System/
├── app/
│   ├── main.py                        # FastAPI app + lifespan + static file serving
│   ├── static/
│   │   └── index.html                 # Full web UI (chat + upload + index management)
│   ├── api/routes/
│   │   ├── health.py                  # GET  /api/v1/health
│   │   ├── ingest.py                  # POST /api/v1/ingest (rate limited, dedup)
│   │   ├── query.py                   # POST /api/v1/query (+ SSE streaming)
│   │   └── documents.py               # GET  /api/v1/documents, DELETE /api/v1/index
│   ├── core/
│   │   ├── config.py                  # Pydantic settings from .env
│   │   └── logging.py                 # structlog configuration
│   ├── ingestion/
│   │   ├── loader.py                  # PyMuPDF page extraction + metadata
│   │   ├── cleaner.py                 # Unicode fix, boilerplate removal
│   │   └── chunker.py                 # Recursive chunking with chunk_id
│   ├── embeddings/
│   │   └── embedder.py                # Singleton sentence-transformer wrapper
│   ├── vectorstore/
│   │   └── faiss_store.py             # FAISS index, persist/load, similarity search
│   ├── retrieval/
│   │   ├── hybrid_retriever.py        # BM25 + FAISS + RRF
│   │   └── reranker.py                # Cross-encoder re-ranking
│   ├── graph/
│   │   ├── state.py                   # RAGState TypedDict
│   │   ├── rag_graph.py               # StateGraph compile + run_rag_pipeline()
│   │   └── nodes/
│   │       ├── query_rewriter.py
│   │       ├── retriever_node.py
│   │       ├── reranker_node.py
│   │       └── generator_node.py
│   ├── llm/
│   │   └── mistral_client.py          # Protocol + Stub + real MistralAPIClient
│   ├── cache/
│   │   └── query_cache.py             # TTLCache keyed by SHA256(query+top_k)
│   ├── evaluation/
│   │   ├── metrics.py                 # precision@k, recall@k, MRR, hit_rate
│   │   └── evaluator.py               # Batch eval runner over JSONL eval sets
│   └── observability/
│       ├── tracer.py                  # Span collector (per-node timing)
│       └── logger.py                  # Structured request logger
├── data/
│   ├── pdfs/                          # Drop source PDFs here
│   └── faiss_index/                   # Auto-generated: index.faiss, docstore.pkl
├── tests/
│   ├── test_ingestion.py              # Cleaner + chunker unit tests
│   ├── test_retrieval.py              # RRF + HybridRetriever tests (mocked store)
│   └── test_evaluation.py             # All metrics with known inputs
├── scripts/
│   ├── ingest_pdfs.py                 # CLI bulk ingest with progress bar
│   └── evaluate.py                    # CLI evaluation report
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

---

## Design Trade-offs

### FAISS vs. managed vector DB (Pinecone, Qdrant)
**Chose FAISS** for zero infrastructure overhead — no Docker service, no API key, runs fully local. Trade-off: no built-in metadata filtering, no distributed scale, and the BM25 index must be rebuilt in memory (not persistent). For production at scale, migrating to Qdrant adds ~50 lines of code.

### Hybrid retrieval (BM25 + FAISS) vs. semantic-only
**Chose hybrid** because pure semantic search misses exact keyword matches (e.g. acronyms, version numbers, proper names). BM25 captures these; FAISS captures conceptual similarity. RRF fusion is parameter-free and consistently outperforms either alone.

### Two-stage retrieval (retrieve 20 → rerank to 5) vs. retrieve 5 directly
**Chose two-stage** because embedding models optimise for recall, not precision. Retrieving a larger candidate set (20) then re-ranking with a cross-encoder gives much higher precision. The cross-encoder is slower per pair but only runs on 20 candidates, not the whole index.

### LangGraph vs. plain function chain
**Chose LangGraph** because it provides: a typed state object, conditional edges (e.g. skip reranker if no docs found), per-node observability via span tracing, and a clear extension point for agentic loops (iterative retrieval, tool use).

### Vanilla frontend vs. React/Next.js
**Chose a single HTML file** served by FastAPI. Zero build pipeline, zero npm, no separate dev server. Works out of the box and stays in the same Docker image. Easy to replace with React later if needed.

### In-memory cache vs. Redis
**Chose `cachetools.TTLCache`** to eliminate the Redis dependency. Cache is lost on restart. Swapping to Redis requires ~20 lines of change in `app/cache/query_cache.py`.

### Stub LLM vs. requiring a real API key
**Chose a stub** so the full pipeline runs and is testable without a Mistral account. The `get_llm_client()` factory auto-switches to the real API when `MISTRAL_API_KEY` is set in `.env`.

---

## How to Run

### Prerequisites
- Python 3.11+
- (Optional) Mistral API key for real LLM responses

### Local development

```bash
# 1. Clone and set up virtual environment
git clone <repo-url>
cd RAG-System
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env
# Edit .env — set MISTRAL_API_KEY if you have one (optional)

# 4. Start the server
uvicorn app.main:app --reload

# Open in browser:
# http://localhost:8000        → Web UI
# http://localhost:8000/docs  → Swagger API docs
```

### Docker

```bash
cp .env.example .env              # edit as needed
docker-compose up --build
# → http://localhost:8000
```

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/v1/health` | Server status, index doc count, cache stats |
| `POST` | `/api/v1/ingest` | Upload a PDF (multipart/form-data), rate: 10/min |
| `POST` | `/api/v1/query` | Ask a question, optional SSE streaming |
| `GET` | `/api/v1/documents` | List all files in the FAISS index |
| `DELETE` | `/api/v1/index` | Wipe the FAISS index and reset all state |

### Health check
```bash
curl http://localhost:8000/api/v1/health
```
```json
{"status": "ok", "index_documents": 15, "cache": {"size": 0, "maxsize": 256, "ttl": 300}}
```

### Ingest a PDF
```bash
curl -X POST http://localhost:8000/api/v1/ingest \
  -F "file=@your_document.pdf"
```
```json
{
  "filename": "your_document.pdf",
  "pages_extracted": 5,
  "chunks_created": 15,
  "total_index_size": 15,
  "duration_ms": 462.3,
  "duplicate": false
}
```
Uploading the same file twice returns `409 Conflict`.

### Query
```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "How do I access the system via SSH?", "top_k": 10}'
```
```json
{
  "query": "How do I access the system via SSH?",
  "rewritten_query": "methods to connect to the system using SSH keys",
  "response": "To access via SSH: run `ssh username@login.ai.lrz.de` [Doc 1]. For passwordless login use ssh-copy-id [Doc 3].",
  "citations": [
    {
      "doc_index": 1,
      "source": "your_document.pdf",
      "page": 0,
      "title": "Your Document",
      "snippet": "ssh username@login.ai.lrz.de -o ServerAliveInterval=30...",
      "rerank_score": -10.62
    }
  ],
  "cache_hit": false,
  "trace": {
    "total_latency_ms": 9832.9,
    "node_timings_ms": {
      "query_rewriter": 1150.5,
      "retriever": 338.3,
      "reranker": 1963.7,
      "generator": 5217.8
    }
  }
}
```

### Streaming query (SSE)
```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "How do I submit a GPU job?", "top_k": 10, "stream": true}'
```
```
data: {"type": "meta", "citations": [...], "rewritten_query": "..."}
data: {"type": "token", "content": "To "}
data: {"type": "token", "content": "submit "}
...
data: [DONE]
```

### List indexed documents
```bash
curl http://localhost:8000/api/v1/documents
```
```json
{
  "files": [
    {"filename": "LRZ Intro.pdf", "chunks": 15, "pages": [0, 1, 2, 3, 4]}
  ],
  "total_chunks": 15
}
```

### Reset the index
```bash
curl -X DELETE http://localhost:8000/api/v1/index
```
```json
{"message": "Index reset successfully."}
```

### Bulk ingest via CLI
```bash
python scripts/ingest_pdfs.py --dir data/pdfs
```

---

## Evaluation

Create a JSONL evaluation set (chunk IDs are returned in query `citations`):
```jsonl
{"query": "How do I log in via SSH?", "relevant_doc_ids": ["your_document.pdf::p0::c0"]}
{"query": "What is SLURM?", "relevant_doc_ids": ["your_document.pdf::p1::c2"]}
```

Run evaluation:
```bash
python scripts/evaluate.py --eval-set data/eval.jsonl --k 5
```
```
==================================================
Evaluation Results (k=5, n=2 queries)
==================================================
  precision@5          0.4000
  recall@5             0.8000
  mrr                  1.0000
  hit_rate@5           1.0000
==================================================
```

### Run tests
```bash
pytest tests/ -v
# 43 passed
```

---

## Configuration

All settings are controlled via `.env` (see `.env.example`):

| Variable | Default | Description |
|---|---|---|
| `MISTRAL_API_KEY` | `""` | If empty, uses the stub LLM |
| `MISTRAL_MODEL` | `mistral-large-latest` | Model name |
| `EMBEDDING_MODEL` | `BAAI/bge-base-en-v1.5` | Sentence-transformer model |
| `FAISS_INDEX_PATH` | `data/faiss_index` | Where to persist the index |
| `RETRIEVAL_TOP_K` | `20` | First-stage retrieval count |
| `RERANK_TOP_K` | `5` | Final docs passed to LLM |
| `RRF_K` | `60` | RRF constant (higher = less score spread) |
| `CACHE_MAX_SIZE` | `256` | Max cached queries |
| `CACHE_TTL_SECONDS` | `300` | Cache expiry in seconds |

---

## Connecting a Real LLM

The system ships with a stub LLM that returns placeholder responses. To enable real generation:

**Option 1 — Mistral API**
```bash
# In .env
MISTRAL_API_KEY=your_key_here
MISTRAL_MODEL=mistral-large-latest
```

**Option 2 — Local Ollama (Mixtral)**
```bash
ollama pull mixtral
```
Then set in `.env`:
```
MISTRAL_API_BASE=http://localhost:11434/v1
MISTRAL_API_KEY=ollama
MISTRAL_MODEL=mixtral
```

**Option 3 — Groq (fast, free tier)**
```
MISTRAL_API_BASE=https://api.groq.com/openai/v1
MISTRAL_API_KEY=your_groq_key
MISTRAL_MODEL=mixtral-8x7b-32768
```
