from collections import Counter
import numpy as np

def bag_of_words_vector(tokens, vocab):
    """
    Returns: np.ndarray of shape (len(vocab),), dtype=int
    """
    tokens_frequency = Counter(tokens)
    result = []
    for word in vocab:
        if word in tokens_frequency:
            result.append(tokens_frequency[word])
        else:
            result.append(0)

    return np.array(result,dtype=int)
        
    