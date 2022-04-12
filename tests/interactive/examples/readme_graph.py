"""
Python file for the sole purpose of producing the graphs in the README file.
"""
import numpy as np
from math import sin, cos
import jax
import jax.numpy as jnp
from jax import random
import matplotlib.pyplot as plt

from rodeo.jax.ibm_init import ibm_init
from rodeo.jax.ode_solve import *
from rodeo.jax.utils import zero_pad
from euler_approx import euler_approx

# Example ODE Exact Solution for x_t^{(0)}
def ode_exact_x(t):
    return (-3*cos(t) + 2*sin(t) - sin(2*t))/3

# Example ODE Exact Solution for x_t^{(1)}
def ode_exact_x1(t):
    return (-2*cos(2*t) + 3*sin(t) + 2*cos(t))/3

# Example ode written for Euler Approximation
def ode_euler(x,t):
    return np.array([x[1], sin(2*t) -x[0]])

sim_jit = jax.jit(solve_sim, static_argnums=(1, 6))
mv_jit = jax.jit(solve_mv, static_argnums=(1, 6))

# Helper function to draw samples from Kalman solver
def readme_kalman_draw(fun, n_order, n_eval, tmin, tmax, sigma, w_mat, x0, draws):
    key = random.PRNGKey(0)
    key, *subkeys = random.split(key, num=draws+1)
    dt = (tmax-tmin)/n_eval
    p = n_order[0]
    W_pad, x0_pad = zero_pad(w_mat, x0, p)
    prior = ibm_init(dt, n_order, sigma)
    X  = np.zeros((draws, n_eval+1, 1, p))
    for i in range(draws):
        X[i] = sim_jit(subkeys[i], fun, x0_pad, None, tmin, tmax,
                       n_eval, W_pad, **prior)
    # X = jax.vmap(lambda i:
    #              sim_jit(subkeys[i], fun, x0_pad, None, tmin, tmax,
    #                      n_eval, W_pad, **prior)
    #             )(jnp.arange(draws))
    return X

def readme_kalman_mv(fun, n_order, n_eval, tmin, tmax, sigma, w_mat, x0):
    key = random.PRNGKey(0)
    dt = (tmax-tmin)/n_eval
    p = n_order[0]
    W_pad, x0_pad = zero_pad(w_mat, x0, p)
    prior = ibm_init(dt, n_order, sigma)
    #mv_jit = jax.jit(solve_mv, static_argnums=(1, 6))
    X = mv_jit(key, fun, x0_pad, None, tmin, tmax,
               n_eval, W_pad, **prior)[0]
    return X

def readme_solve(fun, n_deriv_prior, n_obs, tmin, tmax, n_eval, w_mat, sigma, init, draws):
    """
    Calculates kalman_ode, euler_ode, and exact_ode on the given grid for the README ode.

    Args:
        fun (function): Higher order ODE function :math:`w x_t = F(x_t, t)` 
            taking arguments :math:`x` and :math:`t`.
        n_deriv_prior (ndarray(n_var)): Dimensions of the CAR(p) process.
        n_obs (int): Number of observations.
        tmin (float): First time point of the time interval to be evaluated; :math:`a`.
        tmax (float): Last time point of the time interval to be evaluated; :math:`b`.
        n_eval (int): Number of discretization points of the time interval that is evaluated, 
            such that discretization timestep is :math:`dt = (b-a)/N`.
        w_mat (ndarray(n_var, q+1)): Corresponds to the :math:`W` matrix in the ODE equation.
        sigma (float): Scale parameter.
        init (ndarray(q+1)) or (ndarray(p)): The initial values of :math:`x_L` or :math:`X_L = (x_L, y_L)`.
        draws (int): Number of samples we need to draw from the kalman solver.

    Returns:
        (tuple):
        - **tseq** (ndarray(n_eval+1)): Time discretization points for :math:`t = 0,1/N,\ldots,1`.
        - **Xt** (ndarray(draws, n_eval+1, n_deriv)): Draws of the solution process :math:`X_t` at times
          :math:`t = 0,1/N,\ldots,1`.
        - **x_euler** (ndarray(n_eval+1, 2)): Euler approximation of the solution process at
          times :math:`t = 0,1/N,\ldots,1`.
        - **x_exact** (ndarray(n_eval+1, 2)): Exact solution at times :math:`t = 0,1/N,\ldots,1`.

    """
    tseq = np.linspace(tmin, tmax, n_eval+1)
    n_order = jnp.array([n_deriv_prior]*n_obs)
    Xt = readme_kalman_draw(fun, n_order, n_eval, tmin, tmax, sigma, w_mat, init, draws)
    Xm = readme_kalman_mv(fun, n_order, n_eval, tmin, tmax, sigma, w_mat, init)
    x_euler = euler_approx(ode_euler, tseq, init[0])
    x_exact = np.zeros((n_eval+1, 2))
    for i,t in enumerate(tseq):
        x_exact[i, 0] = ode_exact_x(t)
        x_exact[i, 1] = ode_exact_x1(t)

    return tseq, Xt, x_euler, x_exact, Xm

# Function that produces the graph as shown in README
def readme_graph(fun, n_deriv_prior, n_obs, tmin, tmax, w_mat, init, draws):
    """
    Produces the graph in README file.

    Args:
        fun (function) : Higher order ODE function :math:`W x_t = F(x_t, t)` 
            taking arguments :math:`x` and :math:`t`.
        n_deriv_prior (ndarray(n_var)): Dimensions of the CAR(p) process.
        n_obs (int): Number of observations.
        tmin (float) : First time point of the time interval to be evaluated; :math:`a`.
        tmax (float) : Last time point of the time interval to be evaluated; :math:`b`.
        w_mat (ndarray(n_var, q+1)) : Corresponds to the :math:`W` matrix in the ODE equation.
        init (ndarray(q+1)) or (ndarray(p)): The initial values of :math:`x_L` or :math:`X_L = (x_L, y_L)`.
        draws (int): Number of samples we need to draw from the kalman solver.
    
    """
    # Initialize variables for the graph
    dim_deriv = 2
    N = [50, 100, 200]
    Sigma = jnp.array([[.05], [.05], [.001]])
    dim_example = len(N)
    tseq = [None] * dim_example
    Xn = [None] * dim_example
    Xmean = [None] * dim_example
    x_euler = [None] * dim_example
    x_exact = [None] * dim_example

    for i in range(dim_example):
        tseq[i], Xn[i], x_euler[i], x_exact[i], Xmean[i] = readme_solve(fun=fun,
                                                                        n_deriv_prior=n_deriv_prior,
                                                                        n_obs=n_obs,
                                                                        tmin=tmin, 
                                                                        tmax=tmax, 
                                                                        n_eval=N[i],
                                                                        w_mat=w_mat,
                                                                        sigma=Sigma[i], 
                                                                        init=init,
                                                                        draws=draws)

    fig, axs = plt.subplots(dim_deriv, dim_example, figsize=(20, 10))
    for prow in range(dim_deriv):
        for pcol in range(dim_example):
            # plot Kalman draws
            for i in range(draws):
                if i == (draws - 1):
                    axs[prow, pcol].plot(tseq[pcol], Xn[pcol][i,:, 0, prow], 
                                        color="lightgray", alpha=.3, label="rodeo draws")
                else:
                    axs[prow, pcol].plot(tseq[pcol], Xn[pcol][i,:, 0, prow], 
                                        color="lightgray", alpha=.3)
            # plot Euler and Exact
            axs[prow, pcol].plot(tseq[pcol], x_euler[pcol][:,prow], 
                                label="Euler")
            axs[prow, pcol].plot(tseq[pcol], x_exact[pcol][:,prow], 
                                label="Exact")
            # plot Kalman mean
            axs[prow, pcol].plot(tseq[pcol], Xmean[pcol][:, 0, prow],
                                 label="rodeo mean")
            # set legend and title
            axs[prow, pcol].set_title("$x^{(%s)}_t$;   $N=%s$" % (prow, N[pcol]))
            #axs[prow, pcol].set_ylabel("$x^{(%s)}_t$" % (prow))
    axs[0, -1].legend(loc='upper left', bbox_to_anchor=[1, 1])
    
    fig.tight_layout()
    plt.show()
    return fig
