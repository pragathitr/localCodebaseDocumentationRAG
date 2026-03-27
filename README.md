# Local RAG System with LLM Evaluation Framework
 
A fully local retrieval-augmented generation (RAG) pipeline over FastAPI documentation, built without any external APIs. Designed to answer natural-language questions about a codebase using semantic search over a structured document hierarchy, with a custom evaluation framework to rigorously measure retrieval and generation quality.
 
---
 
## Why This Exists
 
Most RAG tutorials stop at "chunk your docs, embed them, retrieve, generate." That approach ignores two real problems:
 
1. **Chunking destroys context.** Splitting documents naively breaks the logical hierarchy between headings, explanations, and code blocks. A retrieved chunk about `CORSMiddleware` is useless if it's missing the surrounding `add_middleware` call.
2. **There's no way to know if it's working.** Without a structured evaluation framework, you can't tell whether your retrieval is actually finding the right content or whether the model is hallucinating.
 
This project addresses both. It implements a **parent-child chunking strategy** that preserves document structure, and a **two-layer evaluation framework** combining keyword-based retrieval metrics with an LLM-as-judge scoring system.
 
---
 
## System Architecture
 
```
FastAPI Docs (HTML)
        │
        ▼
  [scraper.py]
  Scrapes and cleans raw documentation pages
        │
        ▼
  [chunker.py]
  Markdown-aware parent-child chunking
  Splits by heading structure, preserves code blocks
  Tags chunks with code coverage metadata
        │
        ▼
  [indexer.py]
  Embeds child chunks via Ollama (nomic-embed-text)
  Stores vectors in FAISS index
  Maintains parent-child mapping in memory
        │
        ▼
  [query.py]
  Semantic search over child chunks
  Retrieves parent context for matched children
  Generates answers via Ollama (Llama 3.2:3b)
        │
        ▼
  [evaluate.py]
  Runs retrieval metrics + LLM judge across 10 structured questions
  Outputs aggregate scores to data/evaluation_results.json
```
 
---
 
## Key Design Decisions
 
### 1. Parent-Child Chunking
 
The chunker splits documents by heading structure into **parent sections** (full logical units) and **child chunks** (smaller, embeddable units). This gives the retrieval system the precision of small chunks while preserving the coherence of full sections for answer generation.
 
- **338 parent sections** and **971 child chunks** indexed over 27 FastAPI tutorial pages
- **360 child chunks tagged with code coverage metadata** (`has_code: true`) enabling retrieval-aware filtering — so queries about implementation details can prioritize chunks that actually contain code
 
When a child chunk is matched, the system retrieves its parent section for answer generation. This means the model always sees the full surrounding context, not just the matched fragment.
 
### 2. Fully Local Stack
 
Every component runs locally — no OpenAI API, no external embedding service, no cloud dependencies:
 
| Component | Tool |
|---|---|
| Embeddings | `all-MiniLM-L6-v2` via sentence-transformers |
| Generation | `Llama 3.2:3b` via Ollama |
| Vector store | FAISS (L2 index, in-memory) |
| LLM judge | `Llama 3.2:3b` via Ollama |
 
This makes the system fully reproducible and cost-free to run.
 
### 3. Two-Layer Evaluation
 
Rather than eyeballing outputs, the eval framework measures quality across two independent layers:
 
**Layer 1 — Retrieval metrics (deterministic):**
- **Hit@5**: Does any expected keyword appear in the top-5 retrieved parents?
- **Keyword Recall**: What fraction of expected keywords appear across all retrieved parent texts?
- **Avg Distance**: Mean FAISS cosine distance of matched children (retrieval confidence proxy)
- **Code Coverage**: Did any matched chunk contain code?
 
**Layer 2 — LLM-as-judge (Llama 3.2):**
The judge scores each generated answer on three criteria (1–5 scale):
- **Relevance**: Does the answer address the question?
- **Completeness**: Does it cover the topic adequately?
- **Hallucination-free**: Does it avoid inventing information? (5 = no hallucination)
 
The judge prompt enforces structured JSON output and strips markdown fences before parsing, making scoring deterministic enough to aggregate across runs.
 
---
 
## Evaluation Results
 
Evaluated across 10 structured FastAPI questions covering CORS, authentication, file uploads, database connections, background tasks, path parameters, request validation, middleware, exception handling, and testing.
 
| Metric | Score |
|---|---|
| Hit@5 Rate | **100%** (10/10) |
| Mean Keyword Recall | **0.90** (90%) |
| Mean Relevance (LLM judge) | **4.67 / 5** |
| Mean Completeness (LLM judge) | **4.22 / 5** |
| Mean Hallucination-free (LLM judge) | **4.44 / 5** |
| LLM Judge Successes | **9 / 10** |
 
The single judge failure was a JSON parse error, not a retrieval or generation failure.
 
---
 
## Example Evaluation Questions
 
```json
[
  {
    "id": 1,
    "question": "How do I enable CORS in FastAPI?",
    "expected_keywords": ["CORSMiddleware", "allow_origins", "add_middleware"]
  },
  {
    "id": 7,
    "question": "How do I validate request body in FastAPI?",
    "expected_keywords": ["Pydantic", "BaseModel"]
  },
  {
    "id": 10,
    "question": "How do I test FastAPI endpoints?",
    "expected_keywords": ["TestClient", "pytest"]
  }
]
```
 
---
 
## Failure Analysis & Observations
 
Running the eval surfaced concrete failure patterns worth understanding — not just aggregate scores.
 
### Retrieval failures
 
**Q2 (Authentication)** had the lowest keyword recall at 33% — only `OAuth2` was found in retrieved parent texts; `security` and `Depends` were missed. The retrieval correctly identified the security tutorial page but landed on a parent section (`FastAPI utilities`) that was too high-level to contain the specific implementation keywords. This points to a chunking granularity problem: the parent section boundary was drawn at too coarse a level, swallowing the specific content into a neighboring section.
 
**Q4 (Database connections)** had 67% recall — `SQLAlchemy` and `database` were found but `session` was not. The retrieved parents covered the SQL tutorial page but landed on sections about running the app rather than session management. A metadata filter on `has_code` at query time would likely fix this, since session management code is almost always in a code block.
 
### Generation failures
 
**Q8 (Custom middleware)** and **Q10 (Testing endpoints)** both received hallucination-free scores of 3/5 from the judge. Looking at the actual answers:
 
- Q8's answer invents an incorrect `app.add_middleware()` usage pattern — passing an instance with a `name` keyword argument that doesn't exist in FastAPI's API
- Q10's answer is factually correct but the judge penalized it for not mentioning alternative testing approaches, which it interpreted as a potential omission rather than a scope decision
 
This reveals a known weakness of LLM-as-judge: the judge conflates *incompleteness* with *hallucination*, inflating hallucination scores for answers that are correct but narrow in scope.
 
### What this tells us about the system
 
The pipeline's retrieval layer is strong — 100% Hit@5 means the right content is always in the retrieved set. The failure mode is precision within that set: the wrong parent section gets promoted for answer generation even when the right child was matched. A re-ranking step over retrieved parents (e.g. scoring parent text against the query before passing to the generator) would likely close the keyword recall gap without changing the retrieval architecture.
 
---
 
## Stack
 
- **Python** — end-to-end pipeline
- **Ollama** — local LLM inference (Llama 3.2:3b, nomic-embed-text)
- **FAISS** — vector similarity search
- **FastAPI docs** — corpus (27 tutorial pages, scraped and chunked)
