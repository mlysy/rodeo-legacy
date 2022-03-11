import numpy as np
from math import sin
import jax
import jax.numpy as jnp
from jax import random, jit
import matplotlib.pyplot as plt

from rodeo.ibm import ibm_init
from rodeo.jax.KalmanODE import *
from rodeo.utils import indep_init, zero_pad
from readme_graph import readme_graph

# ODE function
@jit
def ode_fun(x, t, theta=None):
    return jnp.array([jnp.sin(2*t) - x[0]])

def chkrebtii_example():
    r"Produces the graph in Figure 1 of the paper."
    # Produce a Pseudo-RNG key
    key = random.PRNGKey(0)

    # LHS vector of ODE
    # 2.  Define the IVP

    W = np.array([[0.0, 0.0, 1.0]])  # LHS vector of ODE
    x0 = np.array([-1., 0., 1.])  # initial value for the IVP

    # Time interval on which a solution is sought.
    tmin = 0
    tmax = 10

    # 3.  Define the prior process
    #
    # (Perhaps best to describe this in text, not code comments)
    #
    # We're going to use a solution prior that has one more derivative than as specified in the IVP.  
    # To do this, we'll pad the original IVP with zeros, for which we have the convenience function 
    # zero_pad().

    n_deriv = [2]  # number of derivatives in IVP
    n_deriv_prior = [4]  # number of derivatives in IBM prior

    # zero padding
    W_pad = jnp.array(zero_pad(W, n_deriv, n_deriv_prior))
    x0_pad = jnp.array(zero_pad(x0, n_deriv, n_deriv_prior))

    # IBM process scale factor
    sigma = [.5]

    # 4.  Instantiate the ODE solver object.

    n_points = 80  # number of steps in which to discretize the time interval.
    dt = (tmax-tmin)/n_points  # step size

    # generate the Kalman parameters corresponding to the prior
    prior = ibm_init(dt, n_deriv_prior, sigma)
    prior = dict((k, jnp.array(v)) for k, v in prior.items())

    # 5.  Evaluate the ODE solution

    # deterministic output: posterior mean
    mut, Sigmat = solve_mv(fun = ode_fun, 
                           x0 = x0_pad,
                           tmin = tmin,
                           tmax = tmax,
                           n_eval = n_points,
                           wgt_meas = W_pad,
                           **prior)

    # probabilistic output: draw from posterior
    xt = solve_sim(fun = ode_fun, 
                   x0 = x0_pad,
                   tmin = tmin,
                   tmax = tmax,
                   n_eval = n_points,
                   wgt_meas = W_pad,
                   **prior,
                   key=key)
    
    # Produces the graph in Figure 1
    draws = 100
    readme_graph(ode_fun, n_deriv, n_deriv_prior, tmin, tmax, W, x0, draws)
    return

if __name__ == '__main__':
    chkrebtii_example()
    