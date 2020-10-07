import numpy as np
from scipy.integrate import odeint
import numba
from numba import njit
from math import sin
from timer import *
import warnings
from numba.core.errors import NumbaPerformanceWarning
warnings.simplefilter('ignore', category=NumbaPerformanceWarning)

from probDE.ibm import ibm_init
from probDE.utils.utils import rand_mat, indep_init, zero_pad
from probDE.cython.KalmanODE import KalmanODE as KalmanODE_cy
from probDE.tests.KalmanODE import KalmanODE as KalmanODE_c
from kalmanode_numba import KalmanODE as KalmanODE_num
from KalmanODE_py import KalmanODE_py


# ode function used by cython, C++, python
def ode_fun(x, t, theta=None, x_out=None):
    if x_out is None:
        x_out = np.zeros(1)
    x_out[0] = sin(2*t) - x[0]
    return

# ode function used by numba
@njit
def ode_fun2(x, t, theta=None, x_out=None):
    if x_out is None:
        x_out = np.zeros(1)
    x_out[0] = sin(2*t) - x[0]
    return x_out

# ode function used by odeint (output is different)
@njit
def f(x_t, t):
    return [x_t[1], sin(2*t) - x_t[0]]

# problem setup and intialization
# LHS vector of ODE
w_mat = np.array([[0.0, 0.0, 1.0]])

n_obs = 1
n_deriv = [3]
n_deriv_prior = [4]
p = sum(n_deriv_prior)

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
ode_init = ibm_init(dt, n_deriv_prior, sigma)
kinit = indep_init(ode_init, n_deriv_prior)
z_states = rand_mat(2*(n_eval+1), p)
theta = np.empty(0)

# Timings
n_loops = 1000
# C++
kode_c = KalmanODE_c(p, n_obs, tmin, tmax, n_eval, ode_fun, **kinit) 
kode_c.z_states = z_states
time_c = timing(kode_c, x0_state, W, theta, n_loops)

# cython
kode_cy = KalmanODE_cy(p, n_obs, tmin, tmax, n_eval, ode_fun, **kinit) # Initialize the class
kode_cy.z_states = z_states
time_cy = timing(kode_cy, x0_state, W, theta, n_loops)

# numba
kode_num = KalmanODE_num(p, n_obs, tmin, tmax, n_eval, ode_fun2, kinit['mu_state'], 
                         kinit['wgt_state'], kinit['var_state'], z_states)
_ = kode_num.solve(x0_state, W, theta, True) # Need to run once to compile KalmanTV
time_num = timing(kode_num, x0_state, W, theta, n_loops, True)

# python
kode_py = KalmanODE_py(p, n_obs, tmin, tmax, n_eval, ode_fun, **kinit)
kode_py.z_states = z_states
time_py = timing(kode_py, x0_state, W, theta, n_loops)

# odeint
tseq = np.linspace(tmin, tmax, n_eval+1)
time_det = det_timing(f, [-1, 0], tseq, n_loops)

print("Cython is {}x faster than Python".format(time_py/time_cy))
print("Numba is {}x faster than Python".format(time_py/time_num))
print("C++ is {}x faster than Python".format(time_py/time_c))
print("ode is {}x faster than Python".format(time_py/time_det))