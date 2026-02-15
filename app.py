from fastapi import FastAPI  # Web framework for building HTTP APIs
import chromadb              # Vector database for storing/searching document embeddings
import ollama                # Python client to call a local Ollama LLM
import os                    # For environment variables
import logging               # For logging

logging.basicConfig(level=logging.INFO,
                    format= "%(asctime)s %(levelname)s %(message)s",
                    )

MODEL_NAME = os.getenv("MODEL_NAME", "tinyllama")  # Default to "tinyllama" if not set
logging.info(f"Using model: {MODEL_NAME}")

# Create a FastAPI application instance.
# This "app" object is what Uvicorn runs to serve API endpoints.
app = FastAPI()

# Create (or open) a persistent ChromaDB database stored on disk in the folder "./db".
# "PersistentClient" means embeddings + documents survive after the program stops.
chroma = chromadb.PersistentClient(path="./db")

# Get an existing collection named "docs", or create it if it doesn't exist.
collection = chroma.get_or_create_collection("docs")

# Define an HTTP POST endpoint at /query.
# Clients will call POST /query to ask a question.
@app.post("/query")
def query(q: str):

    # query_texts=[q] -> searching using one query string.
    # n_results=1     -> return only the top 1 most similar document.
    #
    # Chroma will:
    # - embed the query `q` using the collection's embedding function
    # - run similarity search against stored embeddings
    # - return the best match(es)
    results = collection.query(query_texts=[q], n_results=1)

    # Extract the retrieved context text:
    # results["documents"][0][0] means:
    # - [0] -> first query
    # - [0] -> top document for that query
    #
    context = results["documents"][0][0] if results["documents"] else ""

    # Call the local LLM via Ollama to generate an answer.
    # model="tinyllama" -> Ollama have this model installed/pulled already.
    answer = ollama.generate(
        model=MODEL_NAME,
        prompt=f"Context:\n{context}\n\nQuestion: {q}\n\nAnswer clearly and concisely:"
    )

    logging.info(f"Query: {q}")

    # Return JSON response to the API caller.
    # Ollama's response is a dict-like object
    return {"answer": answer["response"]}

@app.post("/add")
def add_knowledge(text: str):
    """Add new content to the knowledge base dynamically."""
    try:
        # Generate a unique ID for this document
        import uuid
        doc_id = str(uuid.uuid4())
        
        # Add the text to Chroma collection
        collection.add(documents=[text], ids=[doc_id])
        logging.info(f"Added document with ID {doc_id} to Chroma collection.")
        
        return {
            "status": "success",
            "message": "Content added to knowledge base",
            "id": doc_id
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

@app.get("/health")
def health_check():
    """Simple health check endpoint."""
    return {"status": "ok"}
