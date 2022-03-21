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
    A = np.zeros((q+1, q+1), order='F')
    Q = np.zeros((q+1, q+1), order='F')
    for i in range(q+1):
        for j in range(q+1):
            if i<=j:
                A[i, j] = dt**(j-i)/np.math.factorial(j-i)
    
    for i in range(q+1):
        for j in range(q+1):
            num = dt**(2*q+1-i-j)
            denom = (2*q+1-i-j)*np.math.factorial(q-i)*np.math.factorial(q-j)
            Q[i, j] = sigma**2*num/denom
    return A, Q

def ibm_init(dt, n_order, sigma):
    """
    Calculates the initial parameters necessary for the Kalman solver with the q-times
    integrated Brownian Motion.

    Args:
        dt (float): The step size between simulation points.
        n_order (list(int)): Dimension of the prior.
        sigma (list(float)): Parameter in variance matrix.
        
    Returns:
        (dict):
        - **wgt_state** (ndarray(n_block, p, p)) Transition matrix defining the solution prior; :math:`Q_n`.
        - **mu_state** (ndarray(n_block, p)): Transition_offsets defining the solution prior; denoted by :math:`c_n`.
        - **var_state** (ndarray(n_block, p, p)) Variance matrix defining the solution prior; :math:`R_n`.

    """
    n_block = len(n_order)
    p = max(n_order)
    mu_state = np.zeros((n_block, p))
    wgt_state = [None]*n_block
    var_state = [None]*n_block
    for i in range(n_block):
        wgt_state[i], var_state[i] = ibm_state(dt, n_order[i]-1, sigma[i])
    
    #if n_var == 1:
    #    wgt_state = wgt_state[0]
    #    var_state = var_state[0]
    
    init = {"wgt_state":wgt_state,  "mu_state":mu_state,
            "var_state":var_state}
    return init
