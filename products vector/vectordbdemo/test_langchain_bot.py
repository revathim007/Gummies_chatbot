import os
from typing import Annotated, List, TypedDict, Optional
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

# Define the State
class AgentState(TypedDict):
    query: str
    history: List[dict]
    context: str
    answer: str
    best_match: Optional[dict]

# Initialize LLM and Vector Store
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Load existing collection
# Note: The original code uses PersistentClient(path="./chroma_db")
# and collection "gummy_rag_collection_v3"
persist_directory = "./chroma_db"
collection_name = "gummy_rag_collection_v3"

# We'll use the same embedding function if possible
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

vectorstore = Chroma(
    persist_directory=persist_directory,
    collection_name=collection_name,
    embedding_function=embeddings
)

# Nodes
def retrieve(state: AgentState):
    print("--- RETRIEVING ---")
    query = state["query"]
    
    # Simple retrieval
    docs = vectorstore.similarity_search(query, k=1)
    
    context = ""
    best_match = None
    if docs:
        best_match = docs[0].metadata
        for doc in docs:
            context += f"--- Product Info ---\n{doc.page_content}\n\n"
            
    return {"context": context, "best_match": best_match}

def generate(state: AgentState):
    print("--- GENERATING ---")
    query = state["query"]
    history = state["history"]
    context = state["context"]
    
    system_prompt = f"""
    You are 'Gummy Sunny Bot'. Answer product questions clearly.
    If you mention a specific product from the context, the system will append a card for it. 
    Be helpful and friendly.
    
    CONTEXT:
    {context}
    """
    
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history[-6:])
    messages.append({"role": "user", "content": query})
    
    response = llm.invoke(messages)
    
    return {"answer": response.content}

# Build Graph
builder = StateGraph(AgentState)
builder.add_node("retrieve", retrieve)
builder.add_node("generate", generate)

builder.add_edge(START, "retrieve")
builder.add_edge("retrieve", "generate")
builder.add_edge("generate", END)

# Compile
memory = MemorySaver()
app = builder.compile(checkpointer=memory)

# Test
if __name__ == "__main__":
    config = {"configurable": {"thread_id": "1"}}
    inputs = {
        "query": "What gummies help with immunity?",
        "history": [],
        "context": "",
        "answer": "",
        "best_match": None
    }
    
    for output in app.stream(inputs, config):
        for key, value in output.items():
            print(f"Output from node '{key}':")
            # print(value)
    
    # Print final result
    final_state = app.get_state(config)
    print("\nFinal Answer:")
    print(final_state.values["answer"])
    print("\nBest Match Metadata:")
    print(final_state.values["best_match"])
