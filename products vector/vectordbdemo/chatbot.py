import json
import os
import re
from typing import List, Optional, TypedDict

import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

# LangChain and LangGraph imports
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))


class AgentState(TypedDict):
    query: str
    history: List[dict]
    context: str
    answer: str
    best_match: Optional[dict]


class GummyBot:
    def __init__(self):
        self.persist_directory = os.path.join(BASE_DIR, "chroma_db")
        try:
            with open(os.path.join(BASE_DIR, "products.json"), "r", encoding="utf-8") as f:
                self.products_data = json.load(f)
        except Exception:
            self.products_data = []

        openai_key = os.getenv("OPENAI_API_KEY")
        self.use_openai = False
        self.coll_name = "gummy_rag_local_v3"

        if openai_key and "sk-" in openai_key:
            self.use_openai = True
            self.coll_name = "gummy_rag_collection_v3"
            self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
            self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        else:
            self.embeddings = None
            self.llm = None

        if self.use_openai:
            self.vectorstore = Chroma(
                persist_directory=self.persist_directory,
                collection_name=self.coll_name,
                embedding_function=self.embeddings,
            )
            self._setup_graph()

        self.client = chromadb.PersistentClient(path=self.persist_directory)
        self.emb_fn = embedding_functions.DefaultEmbeddingFunction()
        self.collection = self.client.get_or_create_collection(
            name="gummy_rag_local_v3",
            embedding_function=self.emb_fn,
        )
        self.search_collection = self._get_product_collection()
        self.category_keywords = {
            "sleep": ["sleep", "restful", "melatonin"],
            "skin": ["skin", "glow", "brightening"],
            "hair": ["hair", "biotin"],
            "immunity": ["immune", "immunity"],
            "stress": ["stress", "calm"],
            "kids": ["kids", "children"],
            "protein": ["protein"],
            "omega 3": ["omega"],
            "beauty": ["beauty"],
        }
        self.greeting_words = {"hi", "hello", "hey", "yo", "hiya", "hola"}
        self.product_terms = {
            "product",
            "products",
            "item",
            "items",
            "gummy",
            "gummies",
            "category",
            "categories",
            "price",
            "prices",
            "cost",
            "costs",
            "available",
            "availability",
            "feature",
            "features",
            "compare",
            "comparison",
            "recommend",
            "recommendation",
            "suggest",
            "suggestion",
            "buy",
            "show",
        }
        self.capability_phrases = {
            "what can you do",
            "how can you help",
            "help me",
            "help",
        }
        self.health_terms = {
            "health",
            "medical",
            "medicine",
            "doctor",
            "disease",
            "symptom",
            "symptoms",
            "pain",
            "fever",
            "cold",
            "cough",
            "infection",
            "treatment",
            "diagnosis",
            "diagnose",
            "allergy",
            "allergies",
            "side effect",
            "side effects",
            "dosage",
            "prescription",
            "blood pressure",
            "diabetes",
            "heart",
            "stomach",
            "headache",
        }

    def _get_product_collection(self):
        """Load the main searchable product collection used by the web UI."""
        try:
            return self.client.get_collection(
                name="product_collection",
                embedding_function=self.emb_fn,
            )
        except Exception:
            return self.client.get_collection(name="product_collection")

    def _setup_graph(self):
        """Sets up the LangGraph workflow."""
        builder = StateGraph(AgentState)

        def retrieve_node(state: AgentState):
            query = state["query"]
            docs = self.vectorstore.similarity_search(query, k=3)

            context = ""
            best_match = None
            if docs:
                best_match = docs[0].metadata
                for doc in docs:
                    context += f"--- Product Info ---\n{doc.page_content}\n\n"
            return {"context": context, "best_match": best_match}

        def generate_node(state: AgentState):
            query = state["query"]
            history = state["history"]
            context = state["context"]

            system_prompt = f"""
            You are 'Gummy Sunny Bot'. Be warm, polite, and conversational.
            Answer follow-up questions naturally using the conversation history.
            When the user asks about products, answer clearly and helpfully.
            When the user asks a general question, still reply politely even if it is not a product request.
            If you mention a specific product from the context, the system may append a card for it.

            CONTEXT:
            {context}
            """

            messages = [{"role": "system", "content": system_prompt}]
            for msg in history[-6:]:
                messages.append(msg)
            messages.append({"role": "user", "content": query})

            response = self.llm.invoke(messages)
            return {"answer": response.content}

        builder.add_node("retrieve", retrieve_node)
        builder.add_node("generate", generate_node)
        builder.add_edge(START, "retrieve")
        builder.add_edge("retrieve", "generate")
        builder.add_edge("generate", END)

        self.memory = MemorySaver()
        self.app = builder.compile(checkpointer=self.memory)

    def _format_product_card(self, product):
        """Create an HTML product card for chat replies."""
        name = product.get("title") or product.get("product name") or "Unknown Product"
        price = product.get("price", "--")
        img = product.get("image_url", "https://via.placeholder.com/150")
        return f"""
        <div class="chat-product-card">
            <img src="{img}" alt="{name}">
            <div class="card-details">
                <strong>{name}</strong><br>
                <span class="price-tag">{price}</span>
            </div>
        </div>
        """

    def _search_products(self, query, n_results=3):
        """Return top matching products from the real product catalog only."""
        results = self.search_collection.query(query_texts=[query], n_results=n_results)
        return results["metadatas"][0] if results.get("metadatas") and results["metadatas"][0] else []

    def _all_products(self):
        results = self.search_collection.get(include=["metadatas"])
        return results.get("metadatas", [])

    def _infer_category(self, product):
        explicit = (product.get("category") or "").strip()
        if explicit:
            return explicit

        title = (product.get("title") or product.get("product name") or "").lower()
        for category, keywords in self.category_keywords.items():
            if any(keyword in title for keyword in keywords):
                return category.title()
        return "Other"

    def _category_summary(self):
        counts = {}
        for product in self._all_products():
            category = self._infer_category(product)
            counts[category] = counts.get(category, 0) + 1
        return dict(sorted(counts.items()))

    def _detect_need(self, query):
        q_lower = query.lower()
        for category, keywords in self.category_keywords.items():
            if any(keyword in q_lower for keyword in keywords):
                return category
        return None

    def _tokenize(self, query):
        return re.findall(r"[a-z0-9']+", query.lower())

    def _is_greeting(self, query):
        tokens = self._tokenize(query)
        return bool(tokens) and all(token in self.greeting_words for token in tokens)

    def _is_capability_query(self, query):
        q_lower = query.lower().strip()
        return any(phrase in q_lower for phrase in self.capability_phrases)

    def _is_product_query(self, query):
        tokens = set(self._tokenize(query))
        if not tokens:
            return False

        if self._is_greeting(query) or self._is_capability_query(query):
            return False

        if "total" in tokens and ("product" in tokens or "products" in tokens):
            return True

        if tokens.intersection(self.product_terms):
            return True

        need = self._detect_need(query)
        if need:
            verbs = {"need", "want", "looking", "show", "recommend", "suggest", "find", "compare"}
            request_phrases = (
                "for ",
                "under ",
                "best ",
                "show me",
                "looking for",
                "recommend",
                "suggest",
                "compare",
                "price of",
            )
            q_lower = query.lower()
            return bool(tokens.intersection(verbs)) or any(phrase in q_lower for phrase in request_phrases)

        return False

    def _detect_intent(self, query):
        q_lower = query.lower().strip()
        if self._is_greeting(query):
            return "greeting"
        if self._is_capability_query(query):
            return "capability"
        if "how many" in q_lower or "total product" in q_lower or "all product" in q_lower:
            return "summary"
        if self._is_product_query(query):
            return "product"
        return "non_product"

    def _is_health_query(self, query):
        q_lower = query.lower()
        return any(term in q_lower for term in self.health_terms)

    def _build_polite_intro(self, need, user_question):
        if need == "sleep":
            return (
                "Of course. I would be happy to help with that. "
                "If you are looking for support with better sleep, these are a few products you may like:"
            )
        if need == "skin":
            return (
                "Of course. I would be happy to help with that. "
                "If you are looking for skin support, these are a few products you may like:"
            )
        if need == "hair":
            return (
                "Of course. I would be happy to help with that. "
                "If you are looking for hair care support, these are a few products you may like:"
            )
        if need == "immunity":
            return (
                "Of course. I would be happy to help with that. "
                "If you are looking for immune support, these are a few products you may like:"
            )
        if need:
            return (
                f"Of course. I would be happy to help with that. "
                f"If you are looking for {need} support, these are a few products you may like:"
            )
        return (
            f"Of course. I would be happy to help with that. "
            f"Here are a few products related to **{user_question}**:"
        )

    def _small_talk_reply(self, query):
        q_lower = query.lower().strip()
        if self._is_greeting(query):
            return "Hey! How can I help you today?"
        if "your name" in q_lower or "who are you" in q_lower:
            return "I'm Gummy Sunny Bot. I can help with product suggestions and product-related questions."
        if "thank" in q_lower:
            return "You're welcome. Let me know if you'd like help finding or comparing products."
        if self._is_capability_query(query):
            return (
                "I can help you find products, compare items, check prices, and answer product-related questions."
            )
        if "how are you" in q_lower:
            return "I am doing well, thank you for asking. How may I help you today?"
        if "joke" in q_lower:
            return "I mainly help with product-related questions, but I can still chat a little."
        return None

    def _health_fallback_reply(self):
        return (
            "I'm sorry, but I can't give reliable medical advice here. "
            "For any health-related concern, you should talk to a doctor or consult a healthcare professional."
        )

    def ask(self, user_question, history=None):
        if history is None:
            history = []

        q_lower = user_question.lower().strip()
        intent = self._detect_intent(user_question)
        small_talk = self._small_talk_reply(user_question)
        if small_talk:
            return small_talk

        if intent == "summary":
            count = self.search_collection.count()
            category_counts = self._category_summary()
            response = f"Certainly! We currently have **{count}** products in total.<br><br>"
            response += "Here is the category-wise count:<br>"
            for category, qty in category_counts.items():
                response += f"- **{category}**: {qty}<br>"
            return response

        if intent == "product":
            try:
                matches = self._search_products(user_question, n_results=3)
                if matches:
                    need = self._detect_need(user_question)
                    response = self._build_polite_intro(need, user_question) + "<br><br>"
                    response += "".join(self._format_product_card(product) for product in matches)
                    return response
            except Exception as e:
                print(f"Product Search Error: {e}")

        if self.use_openai and intent != "non_product":
            try:
                config = {"configurable": {"thread_id": "default_user"}}
                inputs = {
                    "query": user_question,
                    "history": history,
                    "context": "",
                    "answer": "",
                    "best_match": None,
                }

                final_state = self.app.invoke(inputs, config)
                text_ans = final_state["answer"]
                best_match = final_state["best_match"]

                if best_match and intent == "product":
                    text_ans += "<br>" + self._format_product_card(best_match)
                return text_ans
            except Exception as e:
                print(f"LangGraph Error: {e}")

        try:
            if intent == "product":
                results = self.collection.query(query_texts=[user_question], n_results=1)
                best_match_meta = results["metadatas"][0][0] if results["metadatas"] and results["metadatas"][0] else None
                if best_match_meta:
                    return f"I'm Sunny! I found this product for **{user_question}**:<br>{self._format_product_card(best_match_meta)}"

            if self._is_health_query(user_question):
                return self._health_fallback_reply()

            return "I mainly help with product-related questions, but I'm happy to guide you if you're looking for a product, price, category, or comparison."
        except Exception as e:
            print(f"Local Fallback Error: {e}")
            if self._is_health_query(user_question):
                return self._health_fallback_reply()
            return "Oops! I ran into an error while searching for the best gummies for you."


if __name__ == "__main__":
    bot = GummyBot()
    print(bot.ask("What gummies help with immunity?"))
