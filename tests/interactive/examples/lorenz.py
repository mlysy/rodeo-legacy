import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from jax.config import config
from rodeo.jax.car_init import *
from rodeo.jax.ode_block_solve import *
from rodeo.jax.utils import zero_pad
from lorenz_graph import lorenz_graph
config.update("jax_enable_x64", True)

def lorenz(X, t, theta):
    rho, sigma, beta = theta
    x, y, z = X[:, 0]
    dx = -sigma*x + sigma*y
    dy = rho*x - y -x*z
    dz = -beta*z + x*y
    return jnp.array([[dx], [dy], [dz]])

def lorenz_example():
    r"Produces the graph in Figure 2 of the paper."
    # Produce a Pseudo-RNG key
    key = jax.random.PRNGKey(0)

    # theta for this example
    theta = jnp.array([28, 10, 8/3])

    # Initial value, x0, for the IVP
    x0 = [-12, -5, 38]
    v0 = [70, 125, -124/3]
    X0 = jnp.column_stack([x0, v0])
    
    # prior process definition
    n_deriv = 1 # number of derivatives in IVP
    n_obs = 3
    n_deriv_prior = 3
    n_order = jnp.array([3, 3, 3]) # number of derivatives in IBM prior

    # LHS Matrix of ODE
    W_mat = np.zeros((n_obs, 1, n_deriv+1))
    W_mat[:, :, 1] = 1
    # W_block = jnp.array(W_mat)

    # Padded version of W
    W_block, _ = zero_pad(W_mat, X0, n_deriv_prior)
    
    # Time interval on which a solution is sought.
    tmin = 0.
    tmax = 20.

    # The rest of the parameters can be tuned according to ODE
    # For this problem, we will use
    tau = np.array([1.3, 1.3, 1.3])
    sigma = np.array([.5, .5, .5])
    n_points = 5000 # number of steps in which to discretize the time interval.
    dt = (tmax-tmin)/n_points # step size

    # generate the Kalman parameters corresponding to the prior
    key, subkey = jax.random.split(key)
    x0_block = car_initial_draw(subkey, n_order, tau, sigma, X0)
    prior = car_init(dt, n_order, tau, sigma)

    # jit-compile the solver
    sim_jit = jax.jit(solve_sim, static_argnums=(1, 6))

    # Run the solver to get an approximation
    key, subkey = jax.random.split(key)
    xt = sim_jit(key = subkey,
                 fun = lorenz, 
                 x0 = x0_block,
                 theta = theta,
                 tmin = tmin,
                 tmax = tmax,
                 n_eval = n_points,
                 wgt_meas = W_block,
                 **prior)

    # Produce the graph in Figure 2
    # set font size
    plt.rcParams.update({'font.size': 20})
    draws = 1000
    
    # rodeo
    method = "rodeo"
    figure = lorenz_graph(x0_block, theta, tmin, tmax, n_points, W_block, prior, draws, method)
    figure.tight_layout()
    
    ## chkrebtii
    method = "chkrebtii"
    figure = lorenz_graph(x0_block, theta, tmin, tmax, n_points, W_block, prior, draws, method)
    figure.tight_layout()
    
    return 
    
if __name__ == '__main__':
    lorenz_example()

