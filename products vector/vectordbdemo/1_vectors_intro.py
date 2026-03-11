# 1_vectors_intro.py
# This script explains what vectors are in the context of AI and Data Science.

"""
What is a Vector?
In simple terms, a vector is just a list of numbers. 
In AI, these numbers represent 'features' or 'characteristics' of data.

Example: Representing Fruits as Vectors
Let's say we want to represent fruits based on two features:
1. Sweetness (0 to 10)
2. Sourness (0 to 10)
"""

# Defining vectors for different fruits [Sweetness, Sourness]
apple = [8, 2]
lemon = [1, 9]
orange = [6, 4]
lime = [1, 10]

print("--- Vector Representations ---")
print(f"Apple vector:  {apple}")
print(f"Lemon vector:  {lemon}")
print(f"Orange vector: {orange}")
print(f"Lime vector:   {lime}")

"""
Why do we use vectors?
By converting data into numbers, we can use math to:
- Compare items (How similar is an orange to a lemon?)
- Group items (Cluster sweet fruits together)
- Find patterns (Predict if a new fruit is sweet or sour)
"""

def plot_vectors():
    # In a real scenario, you'd use matplotlib to visualize this.
    # For now, let's just describe the 'coordinates'.
    print("\n--- Visualizing Vectors (Mental Map) ---")
    print("Imagine a 2D graph where X-axis is Sweetness and Y-axis is Sourness.")
    print(f"Apple is at (8, 2) - Far right (sweet), Low down (not sour)")
    print(f"Lemon is at (1, 9) - Far left (not sweet), High up (very sour)")
    print(f"Orange is at (6, 4) - Middle right, Middle low")

if __name__ == "__main__":
    plot_vectors()
