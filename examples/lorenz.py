import numpy as np
import matplotlib.pyplot as plt

from rodeo.car import car_init
from rodeo.jax.KalmanODE import *
import jax
import jax.numpy as jnp
from jax import random, jit
from rodeo.utils import indep_init, zero_pad
from lorenz_graph import lorenz_graph

# RHS of ODE
@jit
def lorenz(X, t, theta):
    rho, sigma, beta = theta
    p = len(X)//3
    x, y, z = X[p*0], X[p*1], X[p*2]
    dx = -sigma*x + sigma*y
    dy = rho*x - y -x*z
    dz = -beta*z + x*y
    return jnp.array([dx, dy, dz])

def lorenz_example():
    r"Produces the graph in Figure 2 of the paper."
    # Produce a Pseudo-RNG key
    key = random.PRNGKey(0)

    # theta for this example
    theta = jnp.array([28, 10, 8/3])

    # Initial value, x0, for the IVP
    x0 = [-12, -5, 38]
    v0 = [70, 125, -124/3]
    X0 = np.column_stack([x0, v0])
    
    # prior process definition
    n_deriv = [1, 1, 1] # number of derivatives in IVP
    n_deriv_prior = [3, 3, 3] # number of derivatives in IBM prior

    # LHS Matrix of ODE
    W_mat = np.zeros((len(n_deriv), sum(n_deriv)+len(n_deriv)))
    for i in range(len(n_deriv)): W_mat[i, sum(n_deriv[:i])+i+1] = 1

    # pad the inputs
    W_pad = jnp.array(zero_pad(W_mat, n_deriv, n_deriv_prior))

    # Time interval on which a solution is sought.
    tmin = 0
    tmax = 20

    # The rest of the parameters can be tuned according to ODE
    # For this problem, we will use
    tau = np.array([1.3, 1.3, 1.3])
    sigma = np.array([.5, .5, .5])

    n_points = 5000 # number of steps in which to discretize the time interval.
    dt = (tmax-tmin)/n_points # step size

    # generate the Kalman parameters corresponding to the prior
    ode_init, x0_pad = car_init(dt, n_deriv_prior, tau, sigma, X0)
    x0_pad = jnp.array(x0_pad)
    kinit = indep_init(ode_init, n_deriv_prior)
    kinit = dict((k, jnp.array(v)) for k, v in kinit.items())

    # Run the solver to get an approximation
    xt = solve_sim(fun = lorenz, 
                   x0 = x0_pad,
                   tmin = tmin,
                   tmax = tmax,
                   n_eval = n_points,
                   wgt_meas = W_pad,
                   **kinit,
                   key=key,
                   theta=theta)

    # Produce the graph in Figure 2
    draws = 1000
    lorenz_graph(lorenz, n_deriv, n_deriv_prior, tmin, tmax, n_points, W_mat, tau, sigma, X0, theta, draws)
    return

if __name__ == '__main__':
    lorenz_example()
