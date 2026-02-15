# RAG API (FastAPI + ChromaDB + Ollama)

A simple Retrieval-Augmented Generation (RAG) API that retrieves context from a Chroma vector database (ChromaDB) and uses a local LLM (TinyLlama via Ollama) to answer questions.

## Prerequisites
- Python 3.10+
- Ollama installed and running
- TinyLlama pulled:
```bash
ollama pull tinyllama
```

## Setup
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```


## Create embeddings (build the Chroma DB)
```bash
python embed.py
```
## Run the API
```bash
uvicorn app:app --reload
```

## Test the API

### Swagger UI
Open:
- http://127.0.0.1:8000/docs

### curl
```bash
curl -X POST "http://127.0.0.1:8000/query" -G --data-urlencode "q=What is Kubernetes?"
