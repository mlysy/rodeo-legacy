import warnings
import numba
import numpy as np
import getopt
import sys
from numba.core.errors import NumbaPerformanceWarning
from timer import *
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
from rodeo.tests.ode_functions import fitz_fun as ode_fun_nd
from rodeo.tests.ode_functions_ctuple import fitz_fun as ode_fun_ct

# pick ode function
use_ctuple = False
opts, args = getopt.getopt(sys.argv[1:], "c")
for o, a in opts:
    use_ctuple = o == "-c"

warnings.simplefilter('ignore', category=NumbaPerformanceWarning)


# ode function used by cython, C++, python
# def ode_fun(X, t, theta, X_out):
#     "FitzHugh-Nagumo ODE function."
#     a, b, c = theta
#     n_deriv1 = len(X)//2
#     V, R = X[0], X[n_deriv1]
#     X_out[0] = c*(V - V**3/3 + R)
#     X_out[1] = -1/c*(V - a + b*R)
#     return

# ode function used by numba, odeint

@njit
def ode_fun2(X, t, theta, out=None):
    "FitzHugh-Nagumo ODE function."
    if out is None:
        out = np.zeros(2)
    a, b, c = theta
    n_deriv1 = len(X)//2
    V, R = X[0], X[n_deriv1]
    out[:] = c*(V - V*V*V/3 + R), -1/c*(V - a + b*R)
    return out


# problem setup and intialization
n_deriv = [1, 1]  # Total state
n_deriv_prior = [3, 3]
p = sum(n_deriv_prior)

# it is assumed that the solution is sought on the interval [tmin, tmax].
tmin = 0
tmax = 40
# h = 0.1 # step size
#n_eval = int((tmax-tmin)/h)
n_eval = 400

# The rest of the parameters can be tuned according to ODE
# For this problem, we will use
n_var = 2
sigma = [.1]*n_var

# Initial value, x0, for the IVP
x0 = np.array([-1., 1.])
X0 = np.array([-1, 1, 1, 1/3])
w_mat = np.array([[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
W = zero_pad(w_mat, n_deriv, n_deriv_prior)
x0_state = zero_pad(X0, n_deriv, n_deriv_prior)
theta = np.array([0.2, 0.2, 3])

# Get parameters needed to run the solver
dt = (tmax-tmin)/n_eval
ode_init = ibm_init(dt, n_deriv_prior, sigma)
kinit = indep_init(ode_init, n_deriv_prior)
z_state = rand_mat(2*n_eval, p)

# pick ode function with ndarray or ctuple inputs
ode_fun = ode_fun_ct if use_ctuple else ode_fun_nd
if use_ctuple:
    theta = tuple(theta)

# Timings
n_loops = 100
# C++
kode_c = KalmanODE_c(W, tmin, tmax, n_eval, ode_fun, **kinit)
kode_c.z_state = z_state
time_c = timing(kode_c, x0_state, W, theta, n_loops)

# C++2
kode_c2 = KalmanODE_c2(W, tmin, tmax, n_eval, ode_fun, **kinit)
kode_c2.z_state = z_state
time_c2 = timing(kode_c2, x0_state, W, theta, n_loops)

# cython
kode_cy = KalmanODE_cy(W, tmin, tmax, n_eval, ode_fun, **kinit)  
kode_cy.z_state = z_state
time_cy = timing(kode_cy, x0_state, W, theta, n_loops)

# numba
kode_num = KalmanODE_num(W, tmin, tmax, n_eval, ode_fun2, **kinit, z_state=z_state)
# Need to run once to compile KalmanTV
_ = kode_num.solve_sim(x0_state, W, theta)
time_num = timing(kode_num, x0_state, W, theta, n_loops)

# python
kode_py = KalmanODE_py(W, tmin, tmax, n_eval, ode_fun, **kinit)
kode_py.z_state = z_state
time_py = timing(kode_py, x0_state, W, theta, n_loops)

# odeint
tseq = np.linspace(tmin, tmax, n_eval+1)
time_det = det_timing(ode_fun, x0, tseq, n_loops, theta)

print("Cython is {}x faster than Python".format(time_py/time_cy))
print("Numba is {}x faster than Python".format(time_py/time_num))
print("C++ is {}x faster than Python".format(time_py/time_c))
print("C++2 is {}x faster than Python".format(time_py/time_c2))
print("ode is {}x faster than Python".format(time_py/time_det))
