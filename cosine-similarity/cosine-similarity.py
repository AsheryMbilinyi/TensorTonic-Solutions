import numpy as np

def cosine_similarity(a, b):
    """
    Compute cosine similarity between two 1D NumPy arrays.
    Returns: float in [-1, 1]
    """
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    norms = np.linalg.norm(a) * np.linalg.norm(b)
    return (a@b)/norms if norms else 0