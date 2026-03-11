import os
import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

class GummyBot:
    def __init__(self):
        # 1. Initialize Clients
        self.client = chromadb.PersistentClient(path="./chroma_db")
        
        # Load products data directly for aggregation
        try:
            import json
            with open("products.json", "r") as f:
                self.products_data = json.load(f)
        except:
            self.products_data = []

        openai_key = os.getenv("OPENAI_API_KEY")
        self.openai_client = None
        self.use_openai = False

        if openai_key and "sk-" in openai_key:
            try:
                self.emb_fn = embedding_functions.OpenAIEmbeddingFunction(
                    api_key=openai_key,
                    model_name="text-embedding-3-small"
                )
                
                # Test the embedding function to detect quota limits early
                self.emb_fn(["test"])
                
                self.openai_client = OpenAI(api_key=openai_key)
                self.coll_name = "gummy_rag_collection_v3"
                self.use_openai = True
                
                self.collection = self.client.get_or_create_collection(
                    name=self.coll_name,
                    embedding_function=self.emb_fn
                )
            except Exception as e:
                print(f"Chatbot Init: OpenAI access failed ({e}). Falling back to local mode.")
                self.use_openai = False

        if not self.use_openai:
            self.emb_fn = embedding_functions.DefaultEmbeddingFunction()
            self.coll_name = "gummy_rag_local_v3"
            self.collection = self.client.get_or_create_collection(
                name=self.coll_name,
                embedding_function=self.emb_fn
            )

    def _format_product_card(self, product):
        """Creates a rich HTML snippet for a product inside the chat."""
        name = product.get('product name', 'Unknown Product')
        price = product.get('price', '€--')
        img = product.get('image_url', 'https://via.placeholder.com/150')
        return f"""
        <div class="chat-product-card">
            <img src="{img}" alt="{name}">
            <div class="card-details">
                <strong>{name}</strong><br>
                <span class="price-tag">{price}</span>
            </div>
        </div>
        """

    def ask(self, user_question, history=None):
        if history is None:
            history = []
            
        q_lower = user_question.lower()

        # 1. Handle Aggregation Queries Directly
        if "how many" in q_lower or "total product" in q_lower or "all product" in q_lower:
            count = len(self.products_data)
            names = [p['product name'] for p in self.products_data]
            response = f"We currently have **{count}** gummy products available! 🍬<br><br>Here they are:<br>"
            for p in self.products_data:
                response += f"• {p['product name']} ({p['price']})<br>"
            return response

        try:
            # 2. Retrieve relevant products via RAG
            results = self.collection.query(
                query_texts=[user_question],
                n_results=1 # Just get the best match for rich card
            )
            
            # Get data for rich response
            best_match_meta = results['metadatas'][0][0] if results['metadatas'] and results['metadatas'][0] else None
            
            context = ""
            for i in range(len(results['documents'][0])):
                context += f"--- Product Info ---\n{results['documents'][0][i]}\n\n"

            # 3. Generate answer
            if self.openai_client and self.use_openai:
                try:
                    messages = [{"role": "system", "content": f"""
                    You are 'Gummy Sunny Bot'. Answer product questions clearly.
                    If you mention a specific product from the context, the system will append a card for it. 
                    Be helpful and friendly.
                    
                    CONTEXT:
                    {context}
                    """}]
                    
                    for msg in history[-6:]:
                        messages.append(msg)
                    messages.append({"role": "user", "content": user_question})

                    response = self.openai_client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=messages,
                        temperature=0
                    )
                    text_ans = response.choices[0].message.content
                    
                    # Append rich card if we have a match
                    if best_match_meta:
                        text_ans += "<br>" + self._format_product_card(best_match_meta)
                    return text_ans

                except Exception as e:
                    if "insufficient_quota" in str(e).lower():
                        if best_match_meta:
                            return f"I found the **{best_match_meta.get('product name')}** for you! It's priced at **{best_match_meta.get('price')}**.<br>" + self._format_product_card(best_match_meta)
                        return f"I found some info for you! (AI is in Economy Mode)."
                    raise e
            else:
                if best_match_meta:
                    return f"I'm Sunny! I found the **{best_match_meta.get('product name')}** for you at **{best_match_meta.get('price')}**.<br>" + self._format_product_card(best_match_meta)
                return "I'm Sunny! I can see our catalog, let me know what you need!"

        except Exception as e:
            print(f"Error in GummyBot: {e}")
            return "Oops! I ran into an error while searching for the best gummies for you."

if __name__ == "__main__":
    # Quick test if run directly
    try:
        bot = GummyBot()
        print(bot.ask("What gummies help with immunity?"))
    except Exception as e:
        print(e)
