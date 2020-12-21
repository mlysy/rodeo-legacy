import numpy as np

def ibm_state(dt, q, sigma):
    """
    Calculate the state transition matrix and variance matrix using the q-times integrated 
    Brownian motion for the Kalman solver.
        
    Args:
        dt (float): The step size between simulation points.
        q: q-times integrated Brownian Motion.
        sigma (float): Parameter in the q-times integrated Brownian Motion.

    Returns:
        (tuple):
        - **A** (ndarray(n_dim_roots, n_dim_roots)): The state transition matrix defined in 
          Kalman solver.
        - **Q** (ndarray(n_dim_roots, n_dim_roots)): The state variance matrix defined in
          Kalman solver.

    """
    A = np.zeros((q, q), order='F')
    Q = np.zeros((q, q), order='F')
    for i in range(q):
        for j in range(q):
            if i<=j:
                A[i, j] = dt**(j-i)/np.math.factorial(j-i)
    
    for i in range(q):
        for j in range(q):
            num = dt**(2*q+1-i-j)
            denom = (2*q+1-i-j)*np.math.factorial(q-i)*np.math.factorial(q-j)
            Q[i, j] = sigma**2*num/denom
    return A, Q

def ibm_init(dt, n_deriv_prior, sigma):
    """
    Calculates the initial parameters necessary for the Kalman solver with the q-times
    integrated Brownian Motion.

    Args:
        dt (float): The step size between simulation points.
        n_deriv_prior (list(int)): Dimension of the prior.
        sigma (list(float)): Parameter in variance matrix.
        
    Returns:
        (dict):
        - **wgt_state** (ndarray(p, p)) Transition matrix defining the solution prior; :math:`T`.
        - **mu_state** (ndarray(p)): Transition_offsets defining the solution prior; denoted by :math:`\lambda`.
        - **var_state** (ndarray(p, p)) Variance matrix defining the solution prior; :math:`R`.

    """
    n_var = len(n_deriv_prior)
    mu_state = np.zeros(sum(n_deriv_prior))
    wgt_state = [None]*n_var
    var_state = [None]*n_var
    for i in range(n_var):
        wgt_state[i], var_state[i] = ibm_state(dt, n_deriv_prior[i], sigma[i])
    
    if n_var == 1:
        wgt_state = wgt_state[0]
        var_state = var_state[0]
    
    init = {"wgt_state":wgt_state,  "mu_state":mu_state,
            "var_state":var_state}
    return init
