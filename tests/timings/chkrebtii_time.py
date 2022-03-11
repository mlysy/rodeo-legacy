import warnings
import numba
import numpy as np
import jax
import jax.numpy as jnp
from jax import random, jit
import getopt
import sys
from numba.core.errors import NumbaPerformanceWarning
from timer import *
from timeit import default_timer as timer
from numba import njit
from scipy.integrate import odeint
from math import sin
import sys
sys.path.append("..")
from KalmanODE_py import KalmanODE_py
from rodeo.utils.utils import rand_mat, indep_init, zero_pad
from rodeo.ibm import ibm_init
from rodeo.numba.KalmanODE import KalmanODE as KalmanODE_num
from rodeo.cython.KalmanODE import KalmanODE as KalmanODE_cy
from rodeo.eigen.KalmanODE import KalmanODE as KalmanODE_c
from rodeo.eigen.KalmanODE2 import KalmanODE as KalmanODE_c2
from rodeo.jax.KalmanODE import *
from rodeo.tests.ode_functions import chkrebtii_fun as ode_fun_nd
from rodeo.tests.ode_functions_ctuple import chkrebtii_fun as ode_fun_ct

# pick ode function
use_ctuple = False
opts, args = getopt.getopt(sys.argv[1:], "c")
for o, a in opts:
    use_ctuple = o == "-c"

warnings.simplefilter('ignore', category=NumbaPerformanceWarning)


# ode function used by jax
@jit
def ode_fun_jax(x, t, theta=None):
    return jnp.array([jnp.sin(2*t) - x[0]])

@njit
def ode_fun2(x, t, theta=None, x_out=None):
    if x_out is None:
        x_out = np.zeros(1)
    x_out[0] = sin(2*t) - x[0]
    return x_out

# ode function used by odeint (output is different)

@njit
def f(x_t, t, theta=None):
    return [x_t[1], sin(2*t) - x_t[0]]


# problem setup and intialization
# LHS vector of ODE
w_mat = np.array([[0.0, 0.0, 1.0]])

n_deriv = [2]
n_deriv_prior = [4]

# it is assumed that the solution is sought on the interval [tmin, tmax].
n_eval = 80
tmin = 0
tmax = 10

# The rest of the parameters can be tuned according to ODE
# For this problem, we will use
sigma = [.5]

# Initial value, x0, for the IVP
x0 = np.array([-1., 0., 1.])

# Get parameters needed to run the solver
dt = (tmax-tmin)/n_eval
# All necessary parameters are in kinit, namely, T, c, R, W
W = zero_pad(w_mat, n_deriv, n_deriv_prior)
x0_state = zero_pad(x0, n_deriv, n_deriv_prior)
kinit = ibm_init(dt, n_deriv_prior, sigma)
z_state = rand_mat(2*n_eval, sum(n_deriv_prior))
theta = np.empty(0)

# pick ode function with ndarray or ctuple inputs
ode_fun = ode_fun_ct if use_ctuple else ode_fun_nd
if use_ctuple:
    theta = tuple(theta)


# Timings
n_loops = 1000
# C++
kode_c = KalmanODE_c(W, tmin, tmax, n_eval, ode_fun, **kinit)
kode_c.z_state = z_state
time_c = timing(kode_c, x0_state, W, theta, n_loops)

# C++2
kode_c2 = KalmanODE_c2(W, tmin, tmax, n_eval, ode_fun, **kinit)
kode_c2.z_state = z_state
time_c2 = timing(kode_c2, x0_state, W, theta, n_loops)

# cython
kode_cy = KalmanODE_cy(W, tmin, tmax, n_eval,
                       ode_fun, **kinit)  # Initialize the class
kode_cy.z_state = z_state
time_cy = timing(kode_cy, x0_state, W, theta, n_loops)

# numba
kode_num = KalmanODE_num(W, tmin, tmax, n_eval, ode_fun2, **kinit, z_state=z_state)
# Need to run once to compile KalmanTV
_ = kode_num.solve_sim(x0_state, W, theta)
time_num = timing(kode_num, x0_state, W, theta, n_loops)

# jax
key = random.PRNGKey(0)
kinit2 = dict((k, jnp.array(v)) for k, v in kinit.items())
kode_jax = solve_sim(ode_fun_jax, jnp.array(x0_state), tmin, tmax, n_eval, jnp.array(W), **kinit2, key=key) 
start = timer()
for i in range(n_loops):
    kode_jax = solve_sim(ode_fun_jax, jnp.array(x0_state), tmin, tmax, n_eval, jnp.array(W), **kinit2, key=key) 
end = timer()
time_jax = (end - start)/n_loops

# python
kode_py = KalmanODE_py(W, tmin, tmax, n_eval, ode_fun, **kinit)
kode_py.z_state = z_state
time_py = timing(kode_py, x0_state, W, theta, n_loops)

# odeint
tseq = np.linspace(tmin, tmax, n_eval+1)
_ = f(x0[0:2], tseq[0], theta=None)
time_det = det_timing(f, x0[0:2], tseq, n_loops)

print("Cython is {}x faster than Python".format(time_py/time_cy))
print("Numba is {}x faster than Python".format(time_py/time_num))
print("C++ is {}x faster than Python".format(time_py/time_c))
print("C++2 is {}x faster than Python".format(time_py/time_c2))
print("Jax is {}x faster than Python".format(time_py/time_jax))
print("ode is {}x faster than Python".format(time_py/time_det))
