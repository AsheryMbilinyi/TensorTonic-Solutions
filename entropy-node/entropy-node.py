import numpy as np

def entropy_node(y):
    """
    Compute entropy for a single node using stable logarithms.
    """
    # Write code here
    if y is None:
        return 0

    y = np.array(y, dtype = float)

    unique_vals, counts = np.unique(y,return_counts=True)

    probabilities = counts / counts.sum()

    return -np.sum (probabilities * np.log2(probabilities) )