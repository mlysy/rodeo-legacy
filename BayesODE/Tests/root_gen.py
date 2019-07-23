import numpy as np
from math import exp

def root_gen(r0, p):
    """
    Creates p geometrically decaying CAR model roots.
    """
    roots = np.zeros(p)
    r = r0
    
    for k in range(p):
        roots[k] = -r
        r = exp(r0*(k+1))

    return roots