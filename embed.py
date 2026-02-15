import chromadb  # Import the ChromaDB Python library (a local vector database)

# Create a *persistent* Chroma client.
# "persistent" means Chroma will save data to disk so it remains after your script ends.
# It will store its files inside the folder "./db" (relative to where you run the script).
client = chromadb.PersistentClient(path="./db")

# Get an existing collection named "docs" OR create it if it doesn't exist yet.
# Index table name: "docs"
collection = client.get_or_create_collection("docs")

# Open the text file "k8s.txt" and read the entire contents into one string.
with open("k8s.txt", "r") as f:
    text = f.read()

# Add the text into the Chroma collection.
# documents=[text] -> storing ONE document (the whole file as a single document string)
# ids=["k8s"]      -> give that document a unique ID called "k8s"

collection.add(documents=[text], ids=["k8s"])

print("Embedding stored in Chroma")
