import numpy as np
from math import sin
import jax
import jax.numpy as jnp
from jax import random, config
import matplotlib.pyplot as plt

from rodeo.jax.ibm_init import ibm_init
from rodeo.jax.ode_solve import *
from rodeo.jax.utils import zero_pad
from readme_graph import readme_graph
config.update("jax_enable_x64", True)

# ODE function
def ode_fun(x, t, theta=None):
    return jnp.array([[jnp.sin(2*t) - x[0, 0]]])

def chkrebtii_example():
    r"Produces the graph in Figure 1 of the paper."
    # Produce a Pseudo-RNG key
    key = random.PRNGKey(0)

    # LHS vector of ODE
    # 2.  Define the IVP

    W = jnp.array([[[0.0, 0.0, 1.0]]])  # LHS vector of ODE
    x0 = jnp.array([[-1., 0., 1.]])  # initial value for the IVP

    # Time interval on which a solution is sought.
    tmin = 0.
    tmax = 10.

    # 3.  Define the prior process

    # problem setup and intialization
    n_deriv = 2  # number of derivatives in IVP
    n_obs = 1  
    n_deriv_prior = 4

    # zero padding
    W_pad, x0_pad = zero_pad(W, x0, n_deriv_prior)
    # IBM process scale factor
    sigma = jnp.array([.5]*n_obs)

    # 4.  Instantiate the ODE solver object.

    n_points = 80  # number of steps in which to discretize the time interval.
    dt = (tmax-tmin)/n_points  # step size

    # generate the Kalman parameters corresponding to the prior
    n_order = jnp.array([n_deriv_prior]*n_obs)
    prior = ibm_init(dt, n_order, sigma)

    # 5.  Evaluate the ODE solution

    # jit-compile the solver
    sim_jit = jax.jit(solve_sim, static_argnums=(1, 6))
    mv_jit = jax.jit(solve_mv, static_argnums=(1, 6))

    # deterministic output: posterior mean
    key, subkey = jax.random.split(key)
    mut, Sigmat = mv_jit(
        key = subkey,
        fun = ode_fun, 
        x0 = x0_pad,
        theta = None,
        tmin = tmin,
        tmax = tmax,
        n_eval = n_points,
        wgt_meas = W_pad,
        **prior)

    # probabilistic output: draw from posterior
    key, subkey = jax.random.split(key)
    xt = sim_jit(
        key = subkey,
        fun = ode_fun, 
        x0 = x0_pad,
        theta = None,
        tmin = tmin,
        tmax = tmax,
        n_eval = n_points,
        wgt_meas = W_pad,
        **prior)
    
    # Produces the graph in Figure 1
    draws = 100
    readme_graph(ode_fun, n_deriv_prior, n_obs, tmin, tmax, W, x0, draws)
    return

if __name__ == '__main__':
    chkrebtii_example()
    