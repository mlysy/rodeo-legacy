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

# ode function used by jax
def ode_fun_jax(X_t, t, theta):
    "SEIRAH ODE function"
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

# ode function used by non block
def ode_fun_jax2(X_t, t, theta):
    "SEIRAH ODE function"
    S, E, I, R, A, H = X_t[::3]
    N = S + E + I + R + A + H
    b, r, alpha, D_e, D_I, D_q= theta
    D_h = 30
    x1 = -b*S*(I + alpha*A)/N
    x2 = b*S*(I + alpha*A)/N - E/D_e
    x3 = r*E/D_e - I/D_q - I/D_I
    x4 = (I + A)/D_I + H/D_h
    x5 = (1-r)*E/D_e - A/D_I
    x6 = I/D_q - H/D_h
    return jnp.array([x1, x2, x3, x4, x5, x6])

@njit
def ode_fun(X_t, t, theta):
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

def ode_fun_rax(t, X_t, theta):
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
    out = jnp.array([dS, dE, dI, dR, dA, dH])
    return out

def ode_fun_euler(X_t, t, theta):
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
    out = jnp.array([dS, dE, dI, dR, dA, dH])
    return out

def _logpost(y_meas, Xt, gamma):
    return jnp.sum(jsp.stats.norm.logpdf(x=y_meas, loc=Xt, scale=gamma))

def logpost_rodeo(theta, y_meas, gamma):
    Xt = solve_sim(key=key, fun=ode_fun_jax,
                    x0=x0_block, theta=theta,
                    tmin=tmin, tmax=tmax, n_eval=n_eval,
                    wgt_meas=W_block, **ode_init)[:,:,0]
    return _logpost(y_meas, Xt, gamma)

def logpost_diffrax(theta, y_meas, gamma):
    Xt = diffeqsolve(term, solver, args = theta, t0=tmin, t1=tmax, dt0=dt, y0=jnp.array(ode0), saveat=saveat,
                      stepsize_controller=stepsize_controller).ys
    return _logpost(y_meas, Xt, gamma)

def logpost_nbrodeo(theta, y_meas, gamma):
    Xt = rodeonb.solve_sim(key=key, fun=ode_fun_jax2,
                  x0=x0_state, theta=theta,
                  tmin=tmin, tmax=tmax, n_eval=n_eval,
                  wgt_meas=W, **ode_initnb)[:,::n_deriv_prior]
    return _logpost(y_meas, Xt, gamma)

# problem setup and intialization
n_deriv = 1  # Total state
n_obs = 6  # Total measures
n_deriv_prior = 3

# it is assumed that the solution is sought on the interval [tmin, tmax].
n_eval = 150
tmin = 0.
tmax = 60.
theta = np.array([2.23, 0.034, 0.55, 5.1, 2.3, 0.36])
thetaj = jnp.array(theta)

# The rest of the parameters can be tuned according to ODE
# For this problem, we will use
sigma = jnp.array([.5]*n_obs)

# W matrix for the IVP
W_mat = np.zeros((n_obs, 1, n_deriv_prior))
W_mat[:, :, 1] = 1
W_block = jnp.array(W_mat)

# Initial x0 for odeint
ode0 = np.array([63804435., 15492., 21752., 0., 618013., 93583.])

# Initial x0 for jax block
x0 = jnp.array([[63804435], [15492], [21752], [0], [618013], [93583]])
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

# # odeint
tseq = np.linspace(tmin, tmax, n_eval+1)
y_meas = odeint(ode_fun, ode0, tseq, args=(theta, ))
# start = timer()
# for i in range(n_loops):
#     _ = odeint(ode_fun, ode0, tseq, args=(theta,))
# end = timer()
# time_ode = (end - start)/n_loops

# jit grad for diffrax and rodeo
gamma = 0.001
grad_jit1 = jax.jit(jax.grad(logpost_rodeo))
grad_jit2 = jax.jit(jax.grad(logpost_diffrax))
grad_jit3 = jax.jit(jax.grad(logpost_nbrodeo))



# diffrax grad
start = timer()
for i in range(n_loops):
    _ = grad_jit2(thetaj, y_meas, gamma)
end = timer()
time_raxgrad = (end - start)/n_loops

# non-block grad
start = timer()
for i in range(n_loops):
    _ = grad_jit3(thetaj, y_meas, gamma)
end = timer()
time_nbgrad = (end - start)/n_loops

# rodeo grad
start = timer()
for i in range(n_loops):
    _ = grad_jit1(thetaj, y_meas, gamma)
end = timer()
time_jaxgrad = (end - start)/n_loops

# # euler
# n_eval = 150
# euler_sim = euler(ode_fun_euler, ode0, theta, tmin, tmax, n_eval)
# start = timer()
# for i in range(n_loops):
#     _ = euler(ode_fun_euler, ode0, theta, tmin, tmax, n_eval)
# end = timer()
# time_euler = (end - start)/n_loops

# print("Number of times faster jax is compared to odeint {}".format(time_ode/time_jax))
print("Number of times faster jax is compared to diffrax {}".format(time_rax/time_jax))
print("Number of times faster jax is compared to non-blocking {}".format(time_jaxnb/time_jax))
# print("Number of times faster jax is compared to euler {}".format(time_euler/time_jax))
print("Number of times faster jax is compared to diffrax for grad {}".format(time_raxgrad/time_jaxgrad))
print("Number of times faster jax is compared to non-blocking for grad {}".format(time_nbgrad/time_jaxgrad))
