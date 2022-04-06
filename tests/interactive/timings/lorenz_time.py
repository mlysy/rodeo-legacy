from timeit import default_timer as timer
import numpy as np
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import jax.random as random
from scipy.integrate import odeint
from numba import njit

from rodeo.jax.ibm_init import ibm_init
from rodeo.jax.ode_block_solve import *

def ode_fun_jax(X, t, theta):
    rho, sigma, beta = theta
    x, y, z = X[:, 0]
    dx = -sigma*x + sigma*y
    dy = rho*x - y -x*z
    dz = -beta*z + x*y
    return jnp.array([[dx], [dy], [dz]])

@njit
def ode_fun(X, t, theta, out=None):
    rho, sigma, beta = theta
    x, y, z = X
    out = -sigma*x + sigma*y, rho*x - y - x*z, -beta*z + x*y
    return out

def ode_fun2(X, t, theta, out=None):
    rho, sigma, beta = theta
    x, y, z = X
    out = -sigma*x + sigma*y, rho*x - y - x*z, -beta*z + x*y
    return out

# problem setup and intialization
n_deriv = 1  # Total state
n_obs = 3  # Total measures
n_deriv_prior = 3

# it is assumed that the solution is sought on the interval [tmin, tmax].
n_eval = 3000
tmin = 0.
tmax = 20.
theta = np.array([28, 10, 8/3])
thetaj = jnp.array(theta)

# The rest of the parameters can be tuned according to ODE
# For this problem, we will use
sigma = jnp.array([.5, .5, .5])

# Initial value, x0, for the IVP
W_mat = np.zeros((n_obs, 1, n_deriv_prior))
W_mat[:, :, 1] = 1
W_block = jnp.array(W_mat)

ode0 = np.array([-12, -5, 38])
X0 = jnp.array([[-12, 70], [-5, 125], [38, -124/3]])
pad_dim = n_deriv_prior - n_deriv - 1
x0_block = jnp.pad(X0, [(0, 0), (0, pad_dim)])

# Get parameters needed to run the solver
dt = (tmax-tmin)/n_eval
n_order = jnp.array([n_deriv_prior]*n_obs)
ode_init = ibm_init(dt, n_order, sigma)

# Jit solver
key = jax.random.PRNGKey(0)
sim_jit = jax.jit(solve_sim, static_argnums=(1, 6))
sim_jit(key=key, fun=ode_fun_jax,
        x0=x0_block, theta=thetaj,
        tmin=tmin, tmax=tmax, n_eval=n_eval,
        wgt_meas=W_block, **ode_init)

# Timings
n_loops = 100

# Jax
start = timer()
for i in range(n_loops):
    _ = sim_jit(key=key, fun=ode_fun_jax,
                x0=x0_block, theta=thetaj,
                tmin=tmin, tmax=tmax, n_eval=n_eval,
                wgt_meas=W_block, **ode_init)
end = timer()
time_jax = (end - start)/n_loops

# odeint
tseq = np.linspace(tmin, tmax, n_eval+1)
_ = odeint(ode_fun, ode0, tseq, args=(theta, ))
start = timer()
for i in range(n_loops):
    _ = odeint(ode_fun, ode0, tseq, args=(theta,))
end = timer()
time_ode = (end - start)/n_loops

# odeint2 
start = timer()
for i in range(n_loops):
    _ = odeint(ode_fun2, ode0, tseq, args=(theta,))
end = timer()
time_ode2 = (end - start)/n_loops

print(time_jax)
print(time_ode)
print(time_ode2)
