# 3_ai_examples.py
# This script shows how text data is turned into vectors in AI models.

"""
In AI, we use Embedding Models to turn words, sentences, or even images 
into vectors. This process is called 'Vectorization' or 'Embedding'.

Instead of manually picking features like 'sweetness' or 'sourness',
an AI model automatically finds hundreds or thousands of 'features'
that represent the meaning of the text.
"""

# Let's imagine a simple 3D embedding for some words
# [Is it about food?, Is it positive?, Is it tech-related?]
word_embeddings = {
    "Pizza": [0.95, 0.8, 0.1],
    "Burger": [0.98, 0.7, 0.15],
    "Laptop": [0.1, 0.9, 0.95],
    "Keyboard": [0.05, 0.85, 0.98],
    "Apple": [0.9, 0.9, 0.2]  # Apple fruit
}

def compare(word1, word2):
    import math
    
    v1 = word_embeddings[word1]
    v2 = word_embeddings[word2]
    
    # Cosine Similarity
    dot = sum(a * b for a, b in zip(v1, v2))
    mag1 = math.sqrt(sum(a**2 for a in v1))
    mag2 = math.sqrt(sum(b**2 for b in v2))
    
    similarity = dot / (mag1 * mag2)
    print(f"Similarity between '{word1}' and '{word2}': {similarity:.4f}")

print("--- Real-world AI Vector Concept ---")
compare("Pizza", "Burger")   # Both food, both positive
compare("Laptop", "Keyboard") # Both tech
compare("Pizza", "Laptop")   # One food, one tech

"""
Real AI Models like OpenAI's 'text-embedding-3-small' create vectors
with 1,536 dimensions! This allows them to capture extremely subtle
differences in meaning.
"""
