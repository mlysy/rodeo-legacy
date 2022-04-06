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

# ode function used by jax
def ode_fun_jax(X_t, t, theta):
    "SEIRAH ODE function"
    p = len(X_t)//6
    S, E, I, R, A, H = X_t[:, 0]
    N = S + E + I + R + A + H
    b, r, alpha, D_e, D_I, D_q= theta
    D_h = 30
    x1 = -b*S*(I + alpha*A)/N
    x2 = b*S*(I + alpha*A)/N - E/D_e
    x3 = r*E/D_e - I/D_q - I/D_I
    x4 = (I + A)/D_I + H/D_h
    x5 = (1-r)*E/D_e - A/D_I
    x6 = I/D_q - H/D_h
    return jnp.array([[x1], [x2], [x3], [x4], [x5], [x6]])

@njit
def ode_fun(X_t, t, theta, out=None):
    "SEIRAH ODE function"
    S, E, I, R, A, H = X_t
    N = S + E + I + R + A + H
    b, r, alpha, D_e, D_I, D_q = theta
    D_h = 30
    dS = -b*S*(I + alpha*A)/N
    dE = b*S*(I + alpha*A)/N - E/D_e
    dI = r*E/D_e - I/D_q - I/D_I
    dR = (I + A)/D_I + H/D_h
    dA = (1-r)*E/D_e - A/D_I
    dH = I/D_q - H/D_h
    out = np.array([dS, dE, dI, dR, dA, dH])
    return out

def ode_fun2(X_t, t, theta, out=None):
    "SEIRAH ODE function"
    S, E, I, R, A, H = X_t
    N = S + E + I + R + A + H
    b, r, alpha, D_e, D_I, D_q = theta
    D_h = 30
    dS = -b*S*(I + alpha*A)/N
    dE = b*S*(I + alpha*A)/N - E/D_e
    dI = r*E/D_e - I/D_q - I/D_I
    dR = (I + A)/D_I + H/D_h
    dA = (1-r)*E/D_e - A/D_I
    dH = I/D_q - H/D_h
    out = np.array([dS, dE, dI, dR, dA, dH])
    return out

# problem setup and intialization
n_deriv = 1  # Total state
n_obs = 6  # Total measures
n_deriv_prior = 3

# it is assumed that the solution is sought on the interval [tmin, tmax].
n_eval = 50
tmin = 0.
tmax = 10.
theta = np.array([2.23, 0.034, 0.55, 5.1, 2.3, 0.36])
thetaj = jnp.array(theta)

# The rest of the parameters can be tuned according to ODE
# For this problem, we will use
sigma = jnp.array([.5]*n_obs)

# Initial value, x0, for the IVP
W_mat = np.zeros((n_obs, 1, n_deriv_prior))
W_mat[:, :, 1] = 1
W_block = jnp.array(W_mat)

ode0 = np.array([63804435, 15492, 21752, 0, 618013, 93583])
x0 = jnp.array([[63804435], [15492], [21752], [0], [618013], [93583]])
v0 = ode_fun_jax(x0, 0, theta)
X0 = jnp.concatenate([x0, v0],axis=1)
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
n_loops = 1000

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
