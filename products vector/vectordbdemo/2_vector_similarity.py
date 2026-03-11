# 2_vector_similarity.py
# This script explains how to measure similarity between vectors.

import math

def dot_product(v1, v2):
    """Calculates the dot product of two vectors."""
    return sum(x * y for x, y in zip(v1, v2))

def magnitude(v):
    """Calculates the magnitude (length) of a vector."""
    return math.sqrt(sum(x**2 for x in v))

def cosine_similarity(v1, v2):
    """
    Cosine Similarity measures the angle between two vectors.
    Range: -1 to 1 (1 = identical direction, 0 = perpendicular, -1 = opposite)
    Formula: (A · B) / (||A|| * ||B||)
    """
    numerator = dot_product(v1, v2)
    denominator = magnitude(v1) * magnitude(v2)
    
    if not denominator:
        return 0.0
        
    return numerator / denominator

# Fruit Vectors: [Sweetness, Sourness]
apple = [8, 2]
lemon = [1, 9]
orange = [6, 4]
lime = [1, 10]

print("--- Calculating Vector Similarity ---")

# Compare Apple and Orange (Expected: High similarity)
sim_apple_orange = cosine_similarity(apple, orange)
print(f"Similarity between Apple and Orange: {sim_apple_orange:.4f}")

# Compare Apple and Lemon (Expected: Low similarity)
sim_apple_lemon = cosine_similarity(apple, lemon)
print(f"Similarity between Apple and Lemon:  {sim_apple_lemon:.4f}")

# Compare Lemon and Lime (Expected: Very high similarity)
sim_lemon_lime = cosine_similarity(lemon, lime)
print(f"Similarity between Lemon and Lime:   {sim_lemon_lime:.4f}")

"""
Conclusion:
The cosine similarity tells us how similar the 'profile' of the items are.
Even if a lemon is bigger than a lime (different magnitudes), their 
direction in the 'sweetness-sourness' space is almost the same!
"""
