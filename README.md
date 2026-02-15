# RAG-API-with-FastAPI
# Network RAG API (FastAPI + ChromaDB + Ollama)

A simple Retrieval-Augmented Generation (RAG) API that retrieves context from a Chroma vector database and uses a local LLM (TinyLlama via Ollama) to answer questions.

## Setup
```bash
python -m venv venv
source venv/bin/activate  
pip install -r requirements.txt
