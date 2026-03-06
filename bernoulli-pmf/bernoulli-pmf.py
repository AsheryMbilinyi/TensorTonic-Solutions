import numpy as np

def bernoulli_pmf_and_moments(x, p):
    """
    Compute Bernoulli PMF and distribution moments.
    """
    x = np.array(x,dtype=float)
    pmf = (p**x)*((1-p)**(1-x))
    return (pmf,p,p*(1-p))