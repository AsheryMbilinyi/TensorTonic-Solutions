import numpy as np

def entropy_node(y):
    """
    Compute entropy for a single node using stable logarithms.
    """
    # Write code here
    if y is None:
        return 0

    y = np.array(y, dtype = float)
    classes, counts = np.unique(y,return_counts=True)
    p = counts / counts.sum()
    p = p[p>0]
    return -np.sum (p * np.log2(p) )