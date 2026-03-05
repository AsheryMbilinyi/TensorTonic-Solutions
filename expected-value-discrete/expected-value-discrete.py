import numpy as np

def expected_value_discrete(x, p):
    """
    Returns: float expected value
    """

    x = np.array(x,dtype=float)
    p = np.array(p,dtype=float)

    if abs(np.sum(p)-1) > 1e-6:
        raise ValueError("Probabilities must sum to 1")

    if x.shape != p.shape:
        return
    
    return np.sum(x*p)
