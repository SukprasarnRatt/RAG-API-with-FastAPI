import chromadb  # Import the ChromaDB Python library (a local vector database)

# Create a *persistent* Chroma client.
# "persistent" means Chroma will save data to disk so it remains after your script ends.
# It will store its files inside the folder "./db" (relative to where you run the script).
client = chromadb.PersistentClient(path="./db")

# Get an existing collection named "docs" OR create it if it doesn't exist yet.
# A "collection" is like a table/index in a database for storing documents + (optionally) embeddings.
# Index table name: "docs"
collection = client.get_or_create_collection("docs")

# Open the text file "k8s.txt" and read the entire contents into one string.
# - "r" means read mode.
# - f.read() reads the whole file at once.
with open("k8s.txt", "r") as f:
    text = f.read()

# Add the text into the Chroma collection.
# documents=[text] -> you are storing ONE document (the whole file as a single document string)
# ids=["k8s"]      -> you give that document a unique ID called "k8s"
#
# IMPORTANT DETAIL:
# - Whether an "embedding" is actually created depends on how your Chroma collection is configured.
#   In many setups, you must provide an embedding function when creating the collection,
#   or provide embeddings=... here.
# - If you do NOT provide an embedding function or embeddings,
#   Chroma will still store the raw document text, but similarity search may not work yet.
collection.add(documents=[text], ids=["k8s"])

# Print a message to your terminal so you know the script reached this point.
# The message says "Embedding stored in Chroma",
# but again: an embedding is only stored if an embedding function/embeddings were provided.
print("Embedding stored in Chroma")
