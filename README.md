# AI-Powered Knowledge Agent 🚀

An **AI-powered knowledge agent** built with **FastAPI**, **Qdrant**, and **Ollama GPT-OSS**, enabling **semantic search** and **Retrieval-Augmented Generation (RAG)** on your documents.

---

## **Features** ✨

* 📄 **Document Upload** → Store & index documents in Qdrant.
* 🔍 **Semantic Search** → Fetch relevant chunks using embeddings.
* 🧠 **RAG Pipeline** → Combine semantic context + GPT-OSS reasoning.
* ⚡ **FastAPI** → High-performance REST APIs.
* 🛢 **Qdrant** → Vector database for efficient similarity search.
* 🤖 **Ollama GPT-OSS** → Open-source LLM for generating answers.
* 🧩 **Modular Design** → Clean architecture, easy to extend.

---

## **Setup Instructions** 🛠️

### **1. Clone the Repository**

```bash
git clone https://github.com/your-repo/ai-knowledge-agent.git
cd ai-knowledge-agent
```

### **2. Create & Activate Virtual Environment**

```bash
python -m venv .venv
source .venv/bin/activate   # Linux / Mac
.venv\\Scripts\\activate   # Windows
```

### **3. Install Dependencies**

```bash
pip install -r requirements.txt
```

### **4. Set Up Environment Variables**

Create a `.env` file:

```env
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=ai_knowledge
OLLAMA_API=http://localhost:11434
EMBEDDINGS_MODEL=nomic-embed-text
GPT_MODEL=gpt-oss:20b
```

### **5. Run Qdrant via Podman/Docker**

```bash
podman run -d --name qdrant -p 6333:6333 \
    -v D:/qdrant_storage:/qdrant/storage \
    docker.io/qdrant/qdrant
```

### **6. Start FastAPI Server**

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

---

## **API Usage** 📡

### **POST** `/api/docs/upload`

Upload documents (PDF/TXT) for indexing.

**Request:**

```bash
curl -X POST "http://0.0.0.0:8000/api/docs/upload" \
  -F "file=@sample.pdf"
```

**Response:**

```json
{
    "message": "File uploaded and indexed successfully",
    "file_name": "sample.pdf",
    "chunks_indexed": 12
}
```

---

### **POST** `/api/ask`

Ask a question and get an AI-generated answer with optional document context.

**Request:**

```bash
curl -X POST "http://0.0.0.0:8000/api/ask" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is Retrieval-Augmented Generation?"
  }'
```

**Response:**

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
    "generated_by": "AI-only",
    "confidence": 0.93
}
```

#### **Response Fields Explained**

| Field             | Type   | Description                                                                                                                                                              |
| ----------------- | ------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **answer**        | string | AI-generated answer based on semantic search and GPT-OSS reasoning.                                                                                                      |
| **sources**       | array  | List of document references supporting the answer. Can be empty (`[]`) if no documents are found.                                                                        |
| **generated\_by** | string | Indicates how the response was generated:<br>• `AI-only` → Answer purely from GPT-OSS without context.<br>• `AI+Docs` → Answer uses semantic context from uploaded docs. |
| **confidence**    | float  | Confidence score of the generated answer, ranging from `0.0` (low) to `1.0` (high).                                                                                      |

---

### **Health & Metadata Endpoints**

| Endpoint   | Description                         |
| ---------- | ----------------------------------- |
| `/`        | Basic app info                      |
| `/healthz` | Checks if app is alive              |
| `/readyz`  | Checks Qdrant & Ollama connectivity |

---

## **Project Structure** 📂

```
ai-knowledge-agent/
├── app/
│   ├── main.py             # FastAPI entrypoint
│   ├── routes/             # Upload & Ask APIs
│   ├── services/           # Ollama integration
│   ├── db/qdrant_init.py   # Qdrant client setup
│   ├── core/config.py      # Configurations
│   └── __init__.py         # Logger & metadata
├── requirements.txt
├── README.md
└── .env
```

---

## **Troubleshooting** 🐞

### **1. Qdrant connection refused**

* Ensure Qdrant is running: `podman ps`
* Check port: `6333`

### **2. Ollama health check failed**

* Make sure Ollama is running locally.
* Check model availability: `ollama list`

### **3. CORS Issues**

* Update CORS settings in `app/main.py` → restrict `allow_origins` in production.

---

## **Next Steps** 🚀

* [ ] Add authentication & session management.
* [ ] Integrate streaming responses.
* [ ] Add Docker + docker-compose for full local deployment.
* [ ] Improve retry logic for Ollama and Qdrant.

---

## **License** 📜

MIT License © 2025
