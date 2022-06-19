import jax
import jax.numpy as jnp
import jax.scipy as jsp


def _factorial(x):
    """
    JAX factorial function.

    It's actually the gamma function shifted such that `_factorial(x) = x!` for integer-valued `x`.

    Args:
        x (int): Integer.
    
    Returns:
        (int): Factorial of x.

    """
    return jnp.exp(jsp.special.gammaln(x+1.0))


def ibm_state(dt, q, sigma):
    """
    Calculate the state transition matrix and variance matrix of q-times integrated Brownian motion.

    The q-times integrated Brownian motion process :math:`X_t` is such that its q-th order derivative :math:`X^{q}_t = d^q/dt^q X_t` is :math:`\sigma B_t`, i.e., Brownian motion scaled by :math:`\sigma`.

    Args:
        dt (float): The step size between simulation points.
        q (int): The number of times to integrate the underlying Brownian motion.
        sigma (float): Parameter in the q-times integrated Brownian Motion.

    Returns:
        (tuple):
        - **A** (ndarray(q+1, q+1)): The state transition matrix defined in
          Kalman solver.
        - **Q** (ndarray(q+1, q+1)): The state variance matrix defined in
          Kalman solver.

    """
    I, J = jnp.meshgrid(jnp.arange(q+1), jnp.arange(q+1),
                        indexing="ij", sparse=True)
    mesh = J-I
    #A = jnp.maximum(dt**mesh/_factorial(mesh),
    #                jnp.zeros((q+1, q+1)))

    A = jnp.nan_to_num(dt**mesh/_factorial(mesh), 0)
    mesh = (2.0*q+1.0) - I - J
    num = dt**mesh
    den = mesh * _factorial(q - I) * _factorial(q-J)
    Q = sigma**2 * num/den

    # A = np.zeros((q, q), order='F')
    # Q = np.zeros((q, q), order='F')
    # for i in range(q):
    #     for j in range(q):
    #         if i <= j:
    #             A[i, j] = dt**(j-i)/np.math.factorial(j-i)

    # for i in range(q):
    #     for j in range(q):
    #         num = dt**(2*q+1-i-j)
    #         denom = (2*q+1-i-j)*np.math.factorial(q-i)*np.math.factorial(q-j)
    #         Q[i, j] = sigma**2*num/denom
    return A, Q


def ibm_init(dt, n_order, sigma):
    """
    Calculates the initial parameters necessary for the Kalman solver with the p-1 times
    integrated Brownian Motion.

    Args:
        dt (float): The step size between simulation points.
        n_order (ndarray(n_block)): Dimension of the prior.
        sigma (ndarray(n_block)): Parameter in variance matrix.

    Returns:
        (dict):
        - **wgt_state** (ndarray(n_block, p, p)) Transition matrix defining the solution prior; :math:`Q`.
        - **mu_state** (ndarray(n_block, p)): Transition offsets defining the solution prior; denoted by :math:`c`.
        - **var_state** (ndarray(n_block, p, p)) Variance matrix defining the solution prior; :math:`R`.

    """
    n_block = len(n_order)
    p = max(n_order)
    mu_state = jnp.zeros((n_block, p))
    #wgt_state = [None]*n_block
    #var_state = [None]*n_block    

    wgt_state, var_state = jax.vmap(lambda b:
        ibm_state(dt, p-1, sigma[b]))(jnp.arange(n_block))
    
    # mu_state = jax.vmap(lambda b:
        # x0[b] - jnp.matmul(wgt_state[b], x0[b]))(jnp.arange(n_block))
    #for i in range(n_block):
    #    wgt_state[i], var_state[i] = ibm_state(dt, n_order[i]-1, sigma[i])
    
    #wgt_state = jnp.array(wgt_state)
    #var_state = jnp.array(var_state)
    #if n_var == 1:
    #    wgt_state = wgt_state[0]
    #    var_state = var_state[0]
    
    init = {"wgt_state": wgt_state,  "mu_state": mu_state,
            "var_state": var_state}
    return init
