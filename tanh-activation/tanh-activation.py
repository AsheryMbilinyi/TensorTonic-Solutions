import numpy as np

def tanh(x):
    """
    Implement Tanh activation function.
    """
    x = np.array(x,dtype=float)
    numerator = np.exp(x)-np.exp(-x)
    denominator = np.exp(x) + np.exp(-x)
    return numerator/denominator