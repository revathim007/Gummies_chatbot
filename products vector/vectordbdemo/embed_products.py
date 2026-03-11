import json
import chromadb
from chromadb.utils import embedding_functions
import os
from dotenv import load_dotenv

load_dotenv()

def embed_products():
    # 1. Load products.json
    print("Loading products.json...")
    with open('products.json', 'r') as f:
        products = json.load(f)
    
    # 2. Initialize ChromaDB
    client = chromadb.PersistentClient(path="./chroma_db")
    
    # 3. Setup Embedding Function
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key and "sk-" in openai_key:
        print("Using OpenAI embeddings...")
        emb_fn = embedding_functions.OpenAIEmbeddingFunction(
            api_key=openai_key,
            model_name="text-embedding-3-small"
        )
        coll_name = "gummy_rag_collection_v3"
    else:
        print("Error: OpenAI API Key missing or invalid. Please check .env")
        return

    # 4. Create or Get Collection
    # To ensure a fresh start for V3
    try:
        client.delete_collection(name=coll_name)
    except:
        pass
        
    collection = client.get_or_create_collection(
        name=coll_name,
        embedding_function=emb_fn
    )

    # 5. Prepare Data
    documents = []
    metadatas = []
    ids = []

    for product in products:
        # Convert product into a rich text description
        description = f"""
        Product Name: {product['product name']}
        Category: {product['category']}
        Flavor: {product['flavor']}
        Price: {product['price']}
        Benefits: {product['benefits']}
        Ingredients: {product['ingredients']}
        Stock status: {product['stock availability']}
        """.strip()
        
        documents.append(description)
        metadatas.append(product) # Store original fields as metadata
        ids.append(product['id'])

    # 6. Add to Collection
    print(f"Adding {len(documents)} products to ChromaDB...")
    try:
        collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        print("Embeddings stored successfully using OpenAI!")
    except Exception as e:
        if "insufficient_quota" in str(e).lower() or "rate_limit" in str(e).lower():
            print("OpenAI Quota reached! Falling back to local embeddings...")
            # Switch to local collection
            local_coll_name = "gummy_rag_local_v3"
            try: client.delete_collection(name=local_coll_name)
            except: pass
            
            local_collection = client.create_collection(
                name=local_coll_name,
                embedding_function=embedding_functions.DefaultEmbeddingFunction()
            )
            local_collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            print("Embeddings stored successfully using LOCAL fallback!")
        else:
            print(f"Error: {e}")

if __name__ == "__main__":
    embed_products()
