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
import rodeo.jax.ode_solve as jode


def ode_fun_jax(x, t, theta=None):
    return jnp.array([[jnp.sin(2*t) - x[0, 0]]])

def ode_fun_jax2(x, t, theta=None):
    return jnp.array([jnp.sin(2*t) - x[0]])

@njit
def ode_fun(x_t, t, theta=None):
    return np.array([x_t[1], np.sin(2*t) - x_t[0]])

# problem setup and intialization
n_deriv = 2  # Total state
n_obs = 1  # Total measures
n_deriv_prior = 4

# it is assumed that the solution is sought on the interval [tmin, tmax].
n_eval = 80
tmin = 0.
tmax = 10.

# The rest of the parameters can be tuned according to ODE
# For this problem, we will use
sigma = .5

# Initial value, x0, for the IVP
W = jnp.array([[0.0, 0.0, 1.0, 0.0]])
W_block = jnp.array([[[0.0, 0.0, 1.0, 0.0]]])
x0 = jnp.array([-1., 0., 1., 0.])
x0_block = jnp.array([[-1., 0., 1., 0.]])

# Get parameters needed to run the solver
dt = (tmax-tmin)/n_eval
n_order = jnp.array([n_deriv_prior]*n_obs)
sigma = jnp.array([sigma]*n_obs)
ode_init = ibm_init(dt, n_order, sigma)
ode_init2 = dict((k, v[0]) for k, v in ode_init.items())

# Jit solver
key = jax.random.PRNGKey(0)
sim_jit = jax.jit(solve_sim, static_argnums=(1, 6))
sim_jit(key=key, fun=ode_fun_jax,
        x0=x0_block, theta=None,
        tmin=tmin, tmax=tmax, n_eval=n_eval,
        wgt_meas=W_block, **ode_init)
sim_jit2 = jax.jit(jode.solve_sim, static_argnums=(1, 6))
sim_jit2(key=key, fun=ode_fun_jax2,
         x0=x0, theta=None,
         tmin=tmin, tmax=tmax, n_eval=n_eval,
         wgt_meas=W, **ode_init2)

# Timings
n_loops = 1000

# Jax block
start = timer()
for i in range(n_loops):
    _ = sim_jit(key=key, fun=ode_fun_jax,
                x0=x0_block, theta=None,
                tmin=tmin, tmax=tmax, n_eval=n_eval,
                wgt_meas=W_block, **ode_init)
end = timer()
time_jax = (end - start)/n_loops

# Jax
start = timer()
for i in range(n_loops):
    _ = sim_jit2(key=key, fun=ode_fun_jax2,
                 x0=x0, theta=None,
                 tmin=tmin, tmax=tmax, n_eval=n_eval,
                 wgt_meas=W, **ode_init2)
end = timer()
time_jax2 = (end - start)/n_loops

# odeint
tseq = np.linspace(tmin, tmax, n_eval+1)
_ = odeint(ode_fun, x0[0:2], tseq)
start = timer()
for i in range(n_loops):
    _ = odeint(ode_fun, x0[0:2], tseq)
end = timer()
time_ode = (end - start)/n_loops

print(time_jax)
print(time_jax2)
print(time_ode)
