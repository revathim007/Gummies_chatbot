# 4_chromadb_demo.py
# This script demonstrates how to use ChromaDB, a popular vector database.

import chromadb
from chromadb.utils import embedding_functions

# 1. Initialize the Chroma Client
# 'Persistent' means it saves data to a folder. 'HttpClient' would connect to a server.
client = chromadb.Client() 

# 2. Create a Collection (like a Table in SQL)
# We'll name it 'my_knowledge_base'
collection = client.create_collection(name="my_knowledge_base")

# 3. Add Documents
# In a real app, you'd add thousands of documents.
# ChromaDB will automatically convert these texts into vectors (embeddings)
# using its default model.
print("--- Adding Documents to ChromaDB ---")
collection.add(
    documents=[
        "The capital of France is Paris.",
        "The Great Wall of China is visible from space (mostly).",
        "Python is a versatile programming language.",
        "The sun is a star at the center of our solar system.",
        "A vector database stores data as numerical representations."
    ],
    ids=["id1", "id2", "id3", "id4", "id5"]
)
print("Documents added successfully!")

# 4. Query the Database
# We search using 'natural language', not keywords!
query_text = "What is a vector database?"
print(f"\n--- Querying for: '{query_text}' ---")

results = collection.query(
    query_texts=[query_text],
    n_results=2 # Give me the top 2 most similar results
)

print("\nTop Results:")
for i in range(len(results['documents'][0])):
    doc = results['documents'][0][i]
    distance = results['distances'][0][i]
    print(f"- {doc} (Distance: {distance:.4f})")

"""
Note: 
- 'Distance' is the opposite of Similarity. Lower distance = Higher similarity.
- ChromaDB handles the 'Vectorization' (Text -> Vector) and 
  'Similarity Search' (Math comparison) automatically.
"""
