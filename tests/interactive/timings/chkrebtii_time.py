from timeit import default_timer as timer
import numpy as np
import jax
import jax.numpy as jnp
from scipy.integrate import odeint
from numba import njit

from rodeo.jax.ibm_init import ibm_init
from rodeo.jax.ode_solve import *
#import rodeo.jax.ode_solve as jode


def ode_fun_jax(x, t, theta=None):
    return jnp.array([[jnp.sin(2*t) - x[0, 0]]])

@njit
def ode_fun(x_t, t, theta=None):
    return np.array([x_t[1], np.sin(2*t) - x_t[0]])

# problem setup and intialization
n_deriv = 2  # Total state
n_obs = 1  # Total measures
n_deriv_prior = 4

# it is assumed that the solution is sought on the interval [tmin, tmax].
n_eval = 75
tmin = 0.
tmax = 10.

# The rest of the parameters can be tuned according to ODE
# For this problem, we will use
sigma = .5

# Initial value, x0, for the IVP
W_block = jnp.array([[[0.0, 0.0, 1.0, 0.0]]])

# Initial x0 for odeint
ode0 = jnp.array([-1., 0.])

# Initial x0 for jax block
x0_block = jnp.array([[-1., 0., 1., 0.]])

# Get parameters needed to run the solver
dt = (tmax-tmin)/n_eval
n_order = jnp.array([n_deriv_prior]*n_obs)
sigma = jnp.array([sigma]*n_obs)
ode_init = ibm_init(dt, n_order, sigma)

# Jit solver
key = jax.random.PRNGKey(0)
sim_jit = jax.jit(solve_sim, static_argnums=(1, 6))
sim_jit(key=key, fun=ode_fun_jax,
        x0=x0_block, theta=None,
        tmin=tmin, tmax=tmax, n_eval=n_eval,
        wgt_meas=W_block, **ode_init)

# Timings
n_loops = 10000

# odeint
tseq = np.linspace(tmin, tmax, n_eval+1)
_ = odeint(ode_fun, ode0, tseq)
start = timer()
for i in range(n_loops):
    _ = odeint(ode_fun, ode0, tseq)
end = timer()
time_ode = (end - start)/n_loops

# Jax block
start = timer()
for i in range(n_loops):
    _ = sim_jit(key=key, fun=ode_fun_jax,
                x0=x0_block, theta=None,
                tmin=tmin, tmax=tmax, n_eval=n_eval,
                wgt_meas=W_block, **ode_init)
end = timer()
time_jax = (end - start)/n_loops

print("Number of times faster jax is compared to odeint {}".format(time_ode/time_jax))

