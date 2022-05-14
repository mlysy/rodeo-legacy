import jax
import jax.numpy as jnp

from rodeo.jax.ibm_init import ibm_init
from rodeo.jax.kalmantv import _solveV

def bridge_prior(A, Q, R, Y, n_res, n_block):
    r"""
    Transforming the parameter names and computing transitional offset needed for the interrogation algorithm.

    Args:
        A (ndarray(n_res, n_block, p, n_by)): Transition offsets defining the solution prior; denoted by :math:`A_n`.
        Q (ndarray(n_res, n_block, p, p)): Transition matrix defining the solution prior; :math:`Q_n`.
        R (ndarray(n_res, n_block, p, p)) Variance matrix defining the solution prior; :math:`R_n`.
        Y (ndarray(n_y, n_res, n_block, n_by)): Observations.
        n_res (int): The resolution number between observations.
        n_block (int): Number of blocks of X_n.

    Returns:
        (dict):
        - **wgt_state** (ndarray(n_eval, n_block, p, p)) Transition matrix defining the solution prior; :math:`Q_n`.
        - **mu_state** (ndarray(n_eval, n_block, p)): Transition_offsets defining the solution prior; denoted by :math:`c_n`.
        - **var_state** (ndarray(n_eval, n_block, p, p)) Variance matrix defining the solution prior; :math:`R_n`.

    """
    n_y = Y.shape[0]
    mu_state = jax.vmap(lambda t:
                jax.vmap(lambda r: 
                jax.vmap(lambda b: jnp.matmul(A[r, b], Y[t, r, b]))
                        (jnp.arange(n_block)))(jnp.arange(n_res)))(jnp.arange(n_y))
    mu_state = jnp.reshape(mu_state, newshape=(n_y*n_res, n_block, -1))
    wgt_state = jnp.tile(Q.T, n_y).T
    var_state = jnp.tile(R.T, n_y).T
    
    init = {"wgt_state": wgt_state, "mu_state": mu_state, "var_state": var_state}
    return init


def bridge_pars(W, bridge_init, Omega, n_res, n_block):
    r"""
    Calculates the parameters A_n, Q_n, R_n used in the bridge proposal for the Kalman filter with the formula
    `X_n | X_{n-1}, Y ~ N(Q_n X_{n-1} + A_n y_{n/m}, R_n)`.

    Args:
        W (ndarray(n_block, p, p)): Transitional matrix in the observations; `Y~N(WX, \Omega).
        Omega (ndarray(n_block, n_y, n_y)): Variance matrix in the observations; `Y~N(WX, \Omega).
        bridge_init (dict): Dictionary containing the initial parameters computed using the ibm prior.
        n_res (int): The resolution number between observations.
        n_block (int): Number of blocks of X_n.
        
    Returns:
        (dict):
        - **A** (ndarray(n_res, n_block, p, n_by)) Transition offsets defining the solution prior; denoted by :math:`A_n`.
        - **Q** (ndarray(n_res, n_block, p, p)): Transition matrix defining the solution prior; :math:`Q_n`.
        - **R** (ndarray(n_res, n_block, p, p)) Variance matrix defining the solution prior; :math:`R_n`.

    """
    Qt = bridge_init['wgt_state']
    Rt = bridge_init['var_state']
    #Q0 = Qt[0] # <- need to iterate (Q_2)
    #R0 = Rt[0] # <- need to iterate (R_2)
    Q1 = Qt[1] # <- constant (Q_1)
    R1 = Rt[1] # <- constant (R_1)
    y_dim = Omega.shape[1]
    eye = jnp.eye(y_dim)
    
    def vmap_res(r):
        def vmap_block(b):
            WQ2 = jnp.matmul(W[b], Qt[r, b])
            AS_X = jnp.matmul(WQ2, R1[b])
            mu_X = jnp.matmul(WQ2, Q1[b])
            Sigma_y = (
                jnp.matmul(AS_X, WQ2.T) + 
                jnp.linalg.multi_dot([W[b], Rt[r, b], W[b].T]) +
                Omega[b])
            Sigma_yinv = _solveV(Sigma_y, eye)
            An = jnp.matmul(AS_X.T, Sigma_yinv)
            Qn = Q1[b] - jnp.matmul(An, mu_X)
            Rn = R1[b] - jnp.matmul(An, AS_X)

            return An, Qn, Rn
        return jax.vmap(lambda b: vmap_block(b))(jnp.arange(n_block))
    A, Q, R = jax.vmap(lambda r: vmap_res(r))(jnp.arange(n_res)[::-1])
    
    return A, Q, R


def ibm_bridge_init(n_res, dt, n_order, sigma, Y, W, Omega):
    r"""
    Calculates the initial parameters necessary for the Kalman solver with the p-1 times integrated Brownian Motion via the bridge proposal.

    Args:
        n_res (int): The resolution number between observations.
        dt (float): The step size between simulation points.
        n_order (ndarray(n_block)): Dimension of the prior.
        sigma (ndarray(n_block)): Parameter in variance matrix.
        Y (ndarray(n_y, n_block, n_by)): Observations.
        W (ndarray(n_block, p, p)): Transitional matrix in the observations; `Y~N(WX, \Omega).
        Omega (ndarray(n_block, n_y, n_y)): Variance matrix in the observations; `Y~N(WX, \Omega).

    Returns:
        (dict):
        - **wgt_state** (ndarray(n_eval, n_block, p, p)) Transition matrix defining the solution prior; :math:`Q_n`.
        - **mu_state** (ndarray(n_eval, n_block, p)): Transition_offsets defining the solution prior; denoted by :math:`c_n`.
        - **var_state** (ndarray(n_eval, n_block, p, p)) Variance matrix defining the solution prior; :math:`R_n`.

    """
    n_block = len(n_order)
    init = jax.vmap(lambda r: ibm_init(r*dt, n_order, sigma))(jnp.arange(max(2, n_res)))
    Y = jnp.repeat(Y[:, jnp.newaxis, :, :], n_res, axis=1)
    A, Q, R = bridge_pars(W, init, Omega, n_res, n_block)
    init = bridge_prior(A, Q, R, Y, n_res, n_block)
    
    return init
