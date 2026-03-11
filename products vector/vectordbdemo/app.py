import os
from flask import Flask, render_template, request, jsonify
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from chatbot import GummyBot

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

app = Flask(__name__, template_folder=os.path.join(BASE_DIR, "templates"))

# Initialize GummyBot
bot = GummyBot()

# Original Search Collection for traditional search
client = chromadb.PersistentClient(path=os.path.join(BASE_DIR, "chroma_db"))
openai_key = os.getenv("OPENAI_API_KEY")
if openai_key and "sk-" in openai_key:
    search_emb_fn = embedding_functions.OpenAIEmbeddingFunction(
        api_key=openai_key,
        model_name="text-embedding-3-small"
    )
else:
    search_emb_fn = embedding_functions.DefaultEmbeddingFunction()

try:
    search_collection = client.get_or_create_collection(
        name="product_collection", 
        embedding_function=search_emb_fn
    )
except ValueError:
    # If conflict, fallback to default (which is likely what's persisted)
    search_emb_fn = embedding_functions.DefaultEmbeddingFunction()
    search_collection = client.get_or_create_collection(
        name="product_collection",
        embedding_function=search_emb_fn
    )

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    query = request.form.get('query')
    if not query:
        return render_template('index.html', results=[])
    
    results = search_collection.query(
        query_texts=[query],
        n_results=6
    )
    
    formatted_results = []
    for i in range(len(results['documents'][0])):
        formatted_results.append({
            'document': results['documents'][0][i],
            'metadata': results['metadatas'][0][i],
            'distance': results['distances'][0][i],
            'id': results['ids'][0][i]
        })
        
    return render_template('index.html', query=query, results=formatted_results)

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_message = data.get('message')
    history = data.get('history', [])  # Get conversation history from frontend
    
    if not user_message:
        return jsonify({"reply": "I didn't catch that. Could you please repeat?"})
    
    reply = bot.ask(user_message, history=history)
    return jsonify({"reply": reply})

if __name__ == '__main__':
    app.run(debug=True, port=5001)
