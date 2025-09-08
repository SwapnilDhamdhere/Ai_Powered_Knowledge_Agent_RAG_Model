# AI Knowledge Agent

**Semantic search + RAG (Retrieval-Augmented Generation)** using **FastAPI**, **Qdrant** (vector DB) and **Ollama GPT-OSS** (local LLM). This repository provides a simple, modular pipeline to upload documents, create embeddings, store them in Qdrant, and ask contextual questions that the LLM answers using retrieved document chunks.

---

## Table of contents

* [Overview](#overview)
* [Features](#features)
* [Architecture (high level)](#architecture-high-level)
* [Prerequisites](#prerequisites)
* [Quickstart — Run locally (step-by-step)](#quickstart--run-locally-step-by-step)
* [Environment variables (`.env`) example](#environment-variables-env-example)
* [API Endpoints](#api-endpoints)

  * [Upload Document](#upload-document)
  * [Ask (query)](#ask-query)
  * [Health & metadata](#health--metadata)
* [Examples (curl)](#examples-curl)
* [Development notes & file map](#development-notes--file-map)
* [Production recommendations](#production-recommendations)
* [Troubleshooting](#troubleshooting)
* [Next steps / Roadmap](#next-steps--roadmap)
* [License & Contributing](#license--contributing)

---

## Overview

This project implements a Retrieval-Augmented Generation (RAG) pipeline:

1. Upload a document (PDF or TXT).
2. Extract and clean text, split into chunks.
3. Generate embeddings with Ollama (or configured embedder).
4. Insert vector points into Qdrant with metadata (source, chunk index, content).
5. For queries: embed the query → semantic search in Qdrant → send concatenated top chunks as context to ollama GPT-OSS to generate a contextual answer.

The codebase is intentionally modular (routes, services, db, utils) to make it easy to extend and harden.

---

## Features

* Simple upload endpoint for PDFs/TXT.
* Chunking and embedding creation per chunk.
* Vector storage and semantic search using Qdrant.
* Context-aware answer generation using local Ollama LLM (GPT-OSS).
* Health & readiness checks for Qdrant and Ollama.
* Structured logging and clear error handling.

---

## Architecture (high level)

```
[Client] --(upload)--> [FastAPI Upload Route] -> parse -> chunk -> embed -> Qdrant
[Client] --(query)--> [FastAPI Ask Route] -> embed(query) -> Qdrant search -> context -> Ollama -> answer
```

Main components:

* `app.routes` — HTTP endpoints.
* `app.services` — business logic: embeddings, qdrant ops, search, ollama calls.
* `app.db` — qdrant client initialization and collection setup.
* `app.utils` — file handling, text cleaning, PDF parsing, splitting.
* `app.core` — config, logger, exceptions.

---

## Prerequisites

* Python 3.10+ (3.11 recommended)
* Docker or Podman (optional but highly recommended for running Qdrant and Ollama models)
* Enough RAM & disk for the Ollama model you choose (20B model requires lots of resources)

Recommended components to run locally:

* Qdrant running on `localhost:6333`.
* Ollama server running on `localhost:11434` with the desired model loaded (e.g. `gpt-oss:20b`) and an embeddings model available (e.g. `nomic-embed-text`).

Quick Qdrant run (docker):

```bash
# runs Qdrant with persistent storage
docker run -d --name qdrant -p 6333:6333 -v $(pwd)/qdrant_storage:/qdrant/storage qdrant/qdrant
```

Ollama installation and model setup is outside the scope of this README — follow the Ollama docs to install & pull models. Ensure Ollama is reachable at the `OLLAMA_HOST` configured in `.env`.

---

## Quickstart — Run locally (step-by-step)

1. **Clone repository**

```bash
git clone <this-repo-url>
cd <repo-folder>
```

2. **Create and activate a Python virtual environment**

```bash
python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Create a `.env` file** (see example below). Place it at project root.

5. **Start Qdrant and Ollama** (if not already running)

6. **Run the app (development)**

```bash
# start with reload for local dev
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

7. **Open API docs**

* Swagger UI: `http://localhost:8000/docs`
* Redoc: `http://localhost:8000/redoc`

---

## Environment variables (`.env`) example

Create a `.env` at the repo root with values similar to this:

```ini
APP_NAME=AI Knowledge Agent
APP_VERSION=1.0.0

OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=gpt-oss:20b
# (optional) embedding model name if different
OLLAMA_EMBEDDINGS_MODEL=nomic-embed-text

QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION=knowledge_base
QDRANT_VECTOR_SIZE=768
QDRANT_DISTANCE=COSINE

CHUNK_SIZE=512
DEBUG=false
PORT=8000
```

**Note:** `QDRANT_VECTOR_SIZE` must match the dimensionality of the embeddings produced by your embeddings model (e.g. 768 for `nomic-embed-text`).

---

## API Endpoints

### Upload Document

* **URL:** `POST /api/docs/upload`
* **Description:** Accepts a file upload (`multipart/form-data`). Currently supports **.pdf** and **.txt** files.
* **Request:** `file` field (multipart)
* **Response:** `DocumentUploadResponse` JSON

```json
{
  "message": "File 'example.pdf' uploaded and processed successfully.",
  "chunks": 12,
  "source": "example.pdf"
}
```

### Ask (query)

* **URL:** `POST /api/ask/`
* **Description:** Sends a query string; the backend will:

  1. Generate embedding for query.
  2. Search Qdrant (top\_k chunks).
  3. Combine chunks as context and call Ollama for an answer.
* **Request Body (JSON):**

```json
{
  "query": "What is the refund policy for product X?"
}
```

* **Response (`AskResponse`):**

```json
{
  "answer": "<LLM-generated plain text answer>",
  "sources": ["document1.pdf", "document2.txt"]
}
```

### Health & Metadata

* `GET /` — app metadata, docs links and UTC time.
* `GET /healthz` — lightweight alive check.
* `GET /readyz` — readiness check (verifies Qdrant collection and Ollama health).

---

## Examples (curl)

**Upload PDF**

```bash
curl -X POST "http://localhost:8000/api/docs/upload" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/your/file.pdf"
```

**Ask a question**

```bash
curl -X POST "http://localhost:8000/api/ask/" \
  -H "Content-Type: application/json" \
  -d '{"query": "Explain the refund policy"}'
```

**Health check**

```bash
curl http://localhost:8000/readyz
```

---

## Development notes & file map

Key modules to look into:

* `app/main.py` — FastAPI app instance, lifespan hooks, health endpoints.
* `app/routes/upload_routes.py` — file upload handler and orchestration for parsing, embedding & qdrant insert.
* `app/routes/ask_routes.py` — query endpoint.
* `app/services/embeddings_service.py` — calls Ollama embedding API.
* `app/services/ollama_service.py` — chat (answer generation) + health check.
* `app/services/qdrant_service.py` — upsert and search helpers for Qdrant.
* `app/utils/*` — file handling, PDF parsing, text splitting and cleaning utilities.
* `app/core` — configuration, logger, and custom exceptions.

---

## Production recommendations

1. **Security**

   * Restrict CORS origins (`app.add_middleware(CORSMiddleware, allow_origins=[...])`).
   * Add authentication (JWT) for protected endpoints.
   * Use HTTPS / TLS on production.

2. **Resilience**

   * Implement retries & exponential backoff for Ollama and Qdrant calls (e.g., using `tenacity`).
   * Add a fallback LLM/model if Ollama is unavailable.

3. **Performance**

   * Cache embeddings by document fingerprint (hash) to avoid duplicate work.
   * Use bulk upserts for Qdrant and tune `top_k` for search vs performance.

4. **Monitoring & Observability**

   * Add Prometheus metrics and expose them (e.g., `prometheus-fastapi-instrumentator`).
   * Collect logs centrally (ELK/CloudWatch) and set alerting for failures.

5. **Testing & CI/CD**

   * Add unit tests for utils and services, integration tests for endpoints.
   * Containerize (Docker) and add GitHub Actions or similar for CI.

6. **Storage & Cleanup**

   * Use persistent volume for Qdrant storage in production.
   * Implement retention/soft-delete if documents change or should be removed.

---

## Troubleshooting

* **No embeddings or wrong vector size**: verify `QDRANT_VECTOR_SIZE` matches the embedding dimension. Check the embedding response format.
* **`Ollama` connection errors**: ensure Ollama is running and reachable at `OLLAMA_HOST`. Use `/api/tags` or health endpoint of Ollama to test.
* **Qdrant not reachable**: ensure the Qdrant container is running and that `QDRANT_HOST`/`QDRANT_PORT` are correct.
* **Uploads fail for large files**: consider increasing request size limits on the server or splitting large PDFs before upload.
* **Scanning PDF (image-based)**: this repo does not include OCR by default. Add OCR (e.g. Tesseract) to support scanned docs.

---

## Next steps / Roadmap (suggestions)

* Add authentication & role-based access control.
* Implement embedding caching & duplicate detection.
* Add DOCX support and OCR for scanned PDFs.
* Add automated tests (pytest) and CI pipeline.
* Add usage analytics (query counts, latency) and cost monitoring for model usage.

---

## License & Contributing

* This project is provided as-is. Add your preferred license file (e.g., MIT) if you plan to open-source it.
* Contributions are welcome — please open issues and PRs.

---

If you want, I can:

* Produce a **diagram (16:9)** of this flow (PNG/SVG).
* Add a **Dockerfile** and `docker-compose.yml` to make local setup easier.
* Add an example **`.github/workflows/ci.yml`** for CI.

Tell me which of these you'd like next.
