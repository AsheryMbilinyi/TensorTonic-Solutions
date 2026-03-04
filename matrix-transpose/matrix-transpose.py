import numpy as np

def matrix_transpose(A):
    """
    Return the transpose of matrix A (swap rows and columns).
    """
    B = []

    for j in range(len(A[0])):
        row = []
        for i in range(len(A)):
            row.append(A[i][j])
        B.append(row)
            
    return np.array(B)