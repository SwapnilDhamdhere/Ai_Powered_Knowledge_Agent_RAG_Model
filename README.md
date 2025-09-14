# AI Knowledge Agent 🚀

[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)](https://example.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![Qdrant](https://img.shields.io/badge/Qdrant-ready-orange)](https://qdrant.tech/)
[![Ollama](https://img.shields.io/badge/Ollama-GPT--OSS-lightgrey)](https://ollama.com/)

---

## 📌 Project Overview

**AI Knowledge Agent** is a production-friendly template that combines **FastAPI**, **Qdrant**, and **Ollama (GPT‑OSS)** to provide fast semantic search and Retrieval‑Augmented Generation (RAG) over your document corpus. It’s built to be modular, configurable via environment variables, and easy to extend.

---

## ✨ Features

* 📄 **Document Upload** — Upload and index documents (PDF, TXT).
* 🔍 **Semantic & Hybrid Search** — Retrieve relevant chunks using embedding vectors or hybrid search.
* 🧠 **RAG Pipeline** — Use retrieved context + Ollama LLM for grounded answers.
* ⚡ **FastAPI** — Lightweight, high-performance REST APIs with health checks.
* 🛢 **Qdrant** — Vector storage for efficient similarity search.
* 🤖 **Ollama GPT‑OSS** — Open-source LLM for on-premise generation.
* 🧩 **Modular Architecture** — Services & routes separated for maintainability.

---

## 🛠️ Quick Start

1. **Clone the repo**

```bash
git clone https://github.com/your-repo/ai-knowledge-agent.git
cd ai-knowledge-agent
```

2. **Create & activate virtual environment**

```bash
python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows
.venv\Scripts\activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Create `.env`** (example below)

```env
APP_NAME="AI Knowledge Agent"
APP_VERSION="0.1.0"
APP_DESCRIPTION="Semantic search + RAG with GPT-OSS (Ollama) + Qdrant via FastAPI."

# Ollama
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=gpt-oss:20b
OLLAMA_EMBEDDINGS_MODEL=nomic-embed-text

# Qdrant
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION=knowledge_base
QDRANT_VECTOR_SIZE=768
QDRANT_DISTANCE=COSINE

# Chunking & Search
CHUNK_SIZE=512
TOP_K=8
MIN_CHUNKS=3
MIN_RELEVANCE=0.6

# Debug
DEBUG=true
```

5. **Run Qdrant (Docker)**

```bash
docker run -d --name qdrant -p 6333:6333 \
    -v ./qdrant_storage:/qdrant/storage \
    qdrant/qdrant
```

6. **Start FastAPI**

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

---

## 📡 API Endpoints (Examples)

### Upload documents

**POST** `/api/docs/upload`

Upload `.pdf` or `.txt` files to be chunked, embedded, and indexed in Qdrant.

**curl** example:

```bash
curl -X POST "http://0.0.0.0:8000/api/docs/upload" \
  -F "file=@sample.pdf"
```

**Response (example)**:

```json
{
  "message": "File uploaded and indexed successfully",
  "file_name": "sample.pdf",
  "chunks_indexed": 12
}
```

---

### Ask a question (RAG)

**POST** `/api/ask`

Query the system; it will decide whether to respond with LLM-only or LLM+Docs (RAG).

**curl** example:

```bash
curl -X POST "http://0.0.0.0:8000/api/ask" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is Retrieval-Augmented Generation?"}'
```

**Response (example)**:

```json
{
  "answer": "Retrieval-Augmented Generation (RAG) is a technique that combines semantic search with LLMs to provide more accurate answers.",
  "sources": [
    {
      "title": "RAG_Overview.pdf",
      "page": 2,
      "snippet": "RAG enhances LLMs by grounding responses in relevant documents."
    }
  ],
  "generated_by": "AI+Docs",
  "confidence": 0.93
}
```

---

## 📁 Project Structure

```
ai-knowledge-agent/
├── app/
│   ├── main.py             # FastAPI entrypoint
│   ├── routes/             # API endpoints
│   ├── services/           # Ollama & RAG logic
│   ├── db/qdrant_init.py   # Qdrant client setup
│   ├── core/
│   │   ├── config.py       # App configuration
│   │   ├── logger.py       # JSON logging
│   │   └── exceptions.py   # Error handling
│   └── models/             # Request/response schemas
├── requirements.txt
├── README.md
└── .env
```

---

## 🐞 Troubleshooting

* **Qdrant Connection Refused**

  * Ensure Qdrant container is running: `docker ps`
  * Verify port `6333` is reachable

* **Ollama API Not Responding**

  * Confirm Ollama is installed and running locally
  * Check available models: `ollama list`

* **CORS Errors**

  * Configure `allow_origins` in `app/main.py` (restrict to trusted origins in production)

---

## 🚀 Roadmap

* 🔐 Add authentication & session management
* 📡 Streaming responses from Ollama
* 🐳 Provide Docker Compose for full stack deployment
* 🔄 Improve retry logic & error handling
* ✅ Unit tests for services & routes

---

## ✅ License

MIT License © 2025

---

*Generated with ❤️ — AI Knowledge Agent*
