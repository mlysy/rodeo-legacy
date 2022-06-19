import numpy as np
import jax
import jax.numpy as jnp
from functools import partial
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from rodeo.jax.ode_block_solve import *

# lorenz ode for jax
def lorenz(X, t, theta):
    rho, sigma, beta = theta
    x, y, z = X[:, 0]
    dx = -sigma*x + sigma*y
    dy = rho*x - y -x*z
    dz = -beta*z + x*y
    return jnp.array([[dx], [dy], [dz]])

# lorenz ode for odeint
def lorenz0(X, t, theta):
    rho, sigma, beta = theta
    x, y, z = X
    dx = -sigma*x + sigma*y
    dy = rho*x - y -x*z
    dz = -beta*z + x*y
    return jnp.array([dx, dy, dz])

def lorenz_compute(key, x0_block, theta, tmin, tmax, n_eval, W_block, prior, draws, method):
    r"""
    Computes the posteriors for the graph.

    Args:
        key (PRNGKey): PRNG key.
        x0_block (ndarray(n_block, n_deriv_prior)): Initial value that in block form.
        theta (ndarray(n_theta)) : Specific :math:`\theta` for the Lorenz system.
        tmin (float) : First time point of the time interval to be evaluated; :math:`a`.
        tmax (float) : Last time point of the time interval to be evaluated; :math:`b`.
        n_eval (int) : Number of discretization points of the time interval that is evaluated, such that discretization timestep is :math:`dt = (b-a)/N`.
        W_block (ndarray(n_block, 1, n_deriv_prior)) : Corresponds to the :math:`W` matrix in the ODE equation.
        prior (dict) : Parameters required to use the rodeo solver; :math:`R`, :math:`Q`, :math:`c`.
        draws (int) : Number of samples we need to draw from the kalman solver.
        method (string) : Interrogation method.

    Returns:
        (ndarray(draws, n_eval, n_block, p)): Simulated solution from the rodeo solver.

    """
    key, *subkeys = jax.random.split(key, num=draws+1)
    subkeys = jnp.array(subkeys)
    if method=="rodeo":
        sim_jit = jax.jit(solve_sim, static_argnums=(1, 6))
        mv_jit = jax.jit(solve_mv, static_argnums=(1, 6))
        Xn = jax.vmap(lambda i:
                      sim_jit(subkeys[i], lorenz, x0_block, theta, tmin, tmax,
                              n_eval, W_block, **prior)
                     )(jnp.arange(draws))
        key, subkey = jax.random.split(key)
        Xmean = mv_jit(subkey, lorenz, x0_block, theta, tmin, tmax, 
                       n_eval, W_block, **prior)[0]
        Xmean = jnp.array([Xmean])
        Xn = jnp.concatenate([Xn, Xmean])
    else:
        sim_jit = jax.jit(solve_sim, static_argnums=(1, 6, 11))
        Xn = jax.vmap(lambda i:
                      sim_jit(subkeys[i], lorenz, x0_block, theta, tmin, tmax,
                              n_eval, W_block, **prior, interrogate=interrogate_chkrebtii)
                     )(jnp.arange(draws))
    np.save(('saves/lorenz{}.npy').format(method), Xn)
    return Xn
        
def lorenz_graph(x0_block, theta, tmin, tmax, n_eval, W_block, prior, draws, method="rodeo", load_calcs=True):
    r"""
    Produces the graph for the Lorenz63 example in the paper.

    Args:
        x0_block (ndarray(n_block, n_deriv_prior)): Initial value that in block form.
        theta (ndarray(n_theta)) : Specific :math:`\theta` for the Lorenz system.
        tmin (float) : First time point of the time interval to be evaluated; :math:`a`.
        tmax (float) : Last time point of the time interval to be evaluated; :math:`b`.
        n_eval (int) : Number of discretization points of the time interval that is evaluated, such that discretization timestep is :math:`dt = (b-a)/N`.
        W_block (ndarray(n_block, 1, n_deriv_prior)) : Corresponds to the :math:`W` matrix in the ODE equation.
        prior (dict) : Parameters required to use the rodeo solver; :math:`R`, :math:`Q`, :math:`c`.
        draws (int) : Number of samples we need to draw from the kalman solver.
        method (string) : Interrogation method.
    
    Returns:
        (figure): Figure for the lorenz example in the rodeo paper.

    """
    key = jax.random.PRNGKey(0)
    tseq = np.linspace(tmin, tmax, n_eval+1)
    exact = odeint(lorenz0, x0_block[:, 0], tseq, args=(theta, ))
    ylabel = ['x', 'y', 'z']
    n_var = len(ylabel)

    if load_calcs:
        Xn = np.load(('saves/lorenz{}.npy').format(method))
    else:
        Xn = lorenz_compute(key, x0_block, theta, tmin, tmax, n_eval,  W_block, prior, draws, method)
    
    fig, axs = plt.subplots(n_var, 1, figsize=(20, 10))
    for prow in range(n_var):
        for i in range(draws):
            if i == (draws - 1):
                axs[prow].plot(tseq, Xn[i, :, prow, 0],
                        color="gray", alpha=1, label="rodeo draws")
                axs[prow].set_ylabel(ylabel[prow])
            else:
                axs[prow].plot(tseq, Xn[i, :, prow, 0],
                        color="gray", alpha=1)
                
        axs[prow].plot(tseq, exact[:, prow], label='odeint', color="orange")
        if method=="rodeo":
            axs[prow].plot(tseq, Xn[-1, :, prow, 0],
                           label="rodeo mean", color="green")
    axs[0].legend(loc='upper left', bbox_to_anchor=[1, 1])
    fig.tight_layout()
    fig.set_rasterized(True)
    plt.show()
    return fig
