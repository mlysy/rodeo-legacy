from timeit import default_timer as timer
import numpy as np
import jax
import jax.numpy as jnp
from scipy.integrate import odeint
from numba import njit

from rodeo.jax.ibm_init import ibm_init
from rodeo.jax.ode_solve import *

def ode_fun_jax(X_t, t, theta):
    "FitzHugh-Nagumo ODE."
    a, b, c = theta
    V, R = X_t[:,0]
    return jnp.array([[c*(V - V*V*V/3 + R)],
                    [-1/c*(V - a + b*R)]])

@njit
def ode_fun(X_t, t, theta):
    a, b, c = theta
    V, R = X_t
    return np.array([c*(V - V*V*V/3 + R), -1/c*(V - a + b*R)])

# problem setup and intialization
n_deriv = 1  # Total state
n_obs = 2  # Total measures
n_deriv_prior = 3

# it is assumed that the solution is sought on the interval [tmin, tmax].
n_eval = 400
tmin = 0.
tmax = 40.
theta = np.array([0.2, 0.2, 3])
thetaj = jnp.array(theta)

# The rest of the parameters can be tuned according to ODE
# For this problem, we will use
sigma = .1
sigma = jnp.array([sigma]*n_obs)

# Initial value, x0, for the IVP
W_mat = np.zeros((n_obs, 1, n_deriv_prior))
W_mat[:, :, 1] = 1
W_block = jnp.array(W_mat)

# Initial x0 for odeint
ode0 = np.array([-1., 1.])

# Initial x0 for jax block
x0_block = jnp.array([[-1., 1., 0.], [1., 1/3, 0.]])

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
n_loops = 1000

# Jax block
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
_ = odeint(ode_fun, ode0, tseq, args=(theta,))
start = timer()
for i in range(n_loops):
    _ = odeint(ode_fun, ode0, tseq, args=(theta,))
end = timer()
time_ode = (end - start)/n_loops

print("Number of times faster jax is compared to odeint {}".format(time_ode/time_jax))
