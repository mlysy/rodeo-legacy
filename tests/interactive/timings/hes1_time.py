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
    P, M, H = jnp.exp(X_t[:, 0])
    a, b, c, d, e, f, g = theta
    x1 = -a*H + b*M/P - c
    x2 = -d + e/(1+P*P)/M
    x3 = -a*P + f/(1+P*P)/H - g
    return jnp.array([[x1], [x2], [x3]])

def ode_fun_jax2(X_t, t, theta):
    "FitzHugh-Nagumo ODE."
    P, M, H = jnp.exp(X_t[::3])
    a, b, c, d, e, f, g = theta
    x1 = -a*H + b*M/P - c
    x2 = -d + e/(1+P*P)/M
    x3 = -a*P + f/(1+P*P)/H - g
    return jnp.array([x1, x2, x3])

@njit
def ode_fun(X_t, t, theta):
    P, M, H = np.exp(X_t)
    a, b, c, d, e, f, g = theta
    x1 = -a*H + b*M/P - c
    x2 = -d + e/(1+P*P)/M
    x3 = -a*P + f/(1+P*P)/H - g
    return np.array([x1, x2, x3])

def ode_fun_rax(t, X_t, theta):
    P, M, H = jnp.exp(X_t)
    a, b, c, d, e, f, g = theta
    x1 = -a*H + b*M/P - c
    x2 = -d + e/(1+P*P)/M
    x3 = -a*P + f/(1+P*P)/H - g
    return jnp.array([x1, x2, x3])

def ode_fun_euler(X_t, t, theta):
    P, M, H = jnp.exp(X_t)
    a, b, c, d, e, f, g = theta
    x1 = -a*H + b*M/P - c
    x2 = -d + e/(1+P*P)/M
    x3 = -a*P + f/(1+P*P)/H - g
    return jnp.array([x1, x2, x3])

# problem setup and intialization
n_deriv = 1  # Total state
n_obs = 3  # Total measures
n_deriv_prior = 3

# it is assumed that the solution is sought on the interval [tmin, tmax].
n_eval = 120
tmin = 0.
tmax = 240.
theta = np.array([0.022, 0.3, 0.031, 0.028, 0.5, 20, 0.3])
thetaj = jnp.array(theta)

# The rest of the parameters can be tuned according to ODE
# For this problem, we will use
sigma = 0.00001
sigma = jnp.array([sigma]*n_obs)

# Initial value, x0, for the IVP
W_mat = np.zeros((n_obs, 1, n_deriv_prior))
W_mat[:, :, 1] = 1
W_block = jnp.array(W_mat)

# Initial x0 for odeint
ode0 = np.log(np.array([1.439, 2.037, 17.904]))

# Initial x0 for jax block
x0 = jnp.log(jnp.array([[1.439], [2.037], [17.904]]))
v0 = ode_fun_jax(x0, 0, theta)
X0 = jnp.concatenate([x0, v0],axis=1)
pad_dim = n_deriv_prior - n_deriv - 1
x0_block = jnp.pad(X0, [(0, 0), (0, pad_dim)])

# Get parameters needed to run the solver
dt = (tmax-tmin)/n_eval
n_order = jnp.array([n_deriv_prior]*n_obs)
ode_init = ibm_init(dt, n_order, sigma)

# Initial W for jax non block
W = np.zeros((n_obs, jnp.sum(n_order)))
for i in range(n_obs):
    W[i, n_deriv+i*n_deriv_prior] = 1
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

# Jax non block
start = timer()
for i in range(n_loops):
    _ = sim_jit2(key=key, fun=ode_fun_jax2,
                 x0=x0_state, theta=thetaj,
                 tmin=tmin, tmax=tmax, n_eval=n_eval,
                 wgt_meas=W, **ode_initnb)
end = timer()
time_jaxnb = (end - start)/n_loops

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

# odeint
tseq = np.linspace(tmin, tmax, n_eval+1)
_ = odeint(ode_fun, ode0, tseq, args=(theta,))
start = timer()
for i in range(n_loops):
    _ = odeint(ode_fun, ode0, tseq, args=(theta,))
end = timer()
time_ode = (end - start)/n_loops

# euler
n_eval = 2000
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
