import numpy as np
from math import exp
        
def _mou_car(roots, sigma=1., test=False):
    """
    Construct the `Gamma`, `Sigma`, `Sigma_tilde` and Q matrices out of the CAR specification.
    Return `Gamma` and `Sigma` for tests.
    """
    delta = np.array(-roots)
    D = np.diag(delta)
    p = len(roots)
    Q = np.zeros((p, p))
    row = np.ones(p)
        
    for i in range(p):
        Q[i] = row
        row = row*roots
    Q_inv = np.linalg.pinv(Q)

    if test:
        Sigma = np.zeros((p, p))
        Sigma[p-1, p-1] = sigma*sigma
        Gamma = np.zeros((p,p)) # Q*D*Q^-1
        Gamma[range(p-1),range(1,p)] = -1.
        Gamma[p-1] = np.linalg.multi_dot([Q[p-1], D, Q_inv])
        
        return Gamma, Sigma


    Sigma = np.zeros(p)
    Sigma[p-1] = sigma * sigma
    Sigma_tilde = np.matmul(Q_inv * Sigma, Q_inv.T)
    
    return Sigma_tilde, Q
