from timeit import default_timer as timer
import numpy as np
import jax
import jax.numpy as jnp
from scipy.integrate import odeint
from numba import njit

from diffrax import diffeqsolve, Dopri5, ODETerm, SaveAt, PIDController
from rodeo.jax.ibm_init import ibm_init
from rodeo.jax.ode_solve import *

from rodeo.ibm import ibm_init as ibm_init_nb
from rodeo.utils import indep_init
import ode_solve as rodeonb

import sys
sys.path.append("..")
from examples.euler_solve import euler

def ode_fun_jax(X_t, t, theta):
    "FitzHugh-Nagumo ODE."
    a, b, c = theta
    V, R = X_t[:,0]
    return jnp.array([[c*(V - V*V*V/3 + R)],
                    [-1/c*(V - a + b*R)]])

# ode function used by jax non block
def ode_fun_jax2(X, t, theta):
    "FitzHugh-Nagumo ODE function for jax."
    a, b, c = theta
    p = len(X)//2
    V, R = X[0], X[p]
    return jnp.array([c*(V - V*V*V/3 + R),
                      -1/c*(V - a + b*R)])

@njit
def ode_fun(X_t, t, theta):
    a, b, c = theta
    V, R = X_t
    return np.array([c*(V - V*V*V/3 + R), -1/c*(V - a + b*R)])

def ode_fun_rax(t, X_t, theta):
    a, b, c = theta
    V, R = X_t
    return jnp.array([c*(V - V*V*V/3 + R), -1/c*(V - a + b*R)])

def ode_fun_euler(X_t, t, theta):
    a, b, c = theta
    V, R = X_t
    return jnp.array([c*(V - V*V*V/3 + R), -1/c*(V - a + b*R)])

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

# Initial W for jax block
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

# Initial W for jax non block
W = np.zeros((n_obs, jnp.sum(n_order)))
W[0, 1] = 1
W[1, n_deriv_prior+1] = 1
W = jnp.array(W)

# Initial x0 for non block
x0_state = x0_block.flatten()

# Ger parameters for non block
ode_init2 = ibm_init_nb(dt, n_order, sigma)
kinit = indep_init(ode_init2, n_order)
ode_initnb = dict((k, jnp.array(v)) for k, v in kinit.items())

# Jit solver
key = jax.random.PRNGKey(0)
sim_jit = jax.jit(solve_sim, static_argnums=(1, 6))
sim_jit(key=key, fun=ode_fun_jax,
        x0=x0_block, theta=thetaj,
        tmin=tmin, tmax=tmax, n_eval=n_eval,
        wgt_meas=W_block, **ode_init)

# Jit non block solver
sim_jit2 = jax.jit(rodeonb.solve_sim, static_argnums=(1, 6))
sim_jit2(key=key, fun=ode_fun_jax2,
         x0=x0_state, theta=thetaj,
         tmin=tmin, tmax=tmax, n_eval=n_eval,
         wgt_meas=W, **ode_initnb) 

# Timings
n_loops = 100

# Jax block
start = timer()
for i in range(n_loops):
    _ = sim_jit(key=key, fun=ode_fun_jax,
                x0=x0_block, theta=thetaj,
                tmin=tmin, tmax=tmax, n_eval=n_eval,
                wgt_meas=W_block, **ode_init)
end = timer()
time_jax = (end - start)/n_loops

# Jax non block
start = timer()
for i in range(n_loops):
    _ = sim_jit2(key=key, fun=ode_fun_jax2,
                 x0=x0_state, theta=thetaj,
                 tmin=tmin, tmax=tmax, n_eval=n_eval,
                 wgt_meas=W, **ode_initnb)
end = timer()
time_jaxnb = (end - start)/n_loops


# odeint
tseq = np.linspace(tmin, tmax, n_eval+1)
_ = odeint(ode_fun, ode0, tseq, args=(theta,))
start = timer()
for i in range(n_loops):
    _ = odeint(ode_fun, ode0, tseq, args=(theta,))
end = timer()
time_ode = (end - start)/n_loops

# diffrax
tseq = np.linspace(tmin, tmax, n_eval+1)
term = ODETerm(ode_fun_rax)
solver = Dopri5()
saveat = SaveAt(ts=tseq)
stepsize_controller = PIDController(rtol=1e-5, atol=1e-5)
sol = diffeqsolve(term, solver, args = thetaj, t0=tmin, t1=tmax, dt0=dt, y0=jnp.array(ode0), saveat=saveat,
                  stepsize_controller=stepsize_controller)
start = timer()
for i in range(n_loops):
    _ = diffeqsolve(term, solver, args = thetaj, t0=tmin, t1=tmax, dt0=dt, y0=jnp.array(ode0), saveat=saveat,
                    stepsize_controller=stepsize_controller)
end = timer()
time_rax = (end - start)/n_loops

# euler
n_eval = 4000
euler_sim = euler(ode_fun_euler, ode0, theta, tmin, tmax, n_eval)
start = timer()
for i in range(n_loops):
    _ = euler(ode_fun_euler, ode0, theta, tmin, tmax, n_eval)
end = timer()
time_euler = (end - start)/n_loops

print("Number of times faster jax is compared to odeint {}".format(time_ode/time_jax))
print("Number of times faster jax is compared to diffrax {}".format(time_rax/time_jax))
print("Number of times faster jax is compared to non-blocking {}".format(time_jaxnb/time_jax))
print("Number of times faster jax is compared to euler {}".format(time_euler/time_jax))
