import warnings
import numba
import numpy as np
import getopt
import sys
from numba.core.errors import NumbaPerformanceWarning
from timer import *
from numba import njit
from scipy.integrate import odeint
import sys
sys.path.append("..")
from KalmanODE_py import KalmanODE_py
from rodeo.utils.utils import rand_mat, indep_init, zero_pad
from rodeo.ibm import ibm_init
from rodeo.numba.KalmanODE import KalmanODE as KalmanODE_num
from rodeo.cython.KalmanODE import KalmanODE as KalmanODE_cy
from rodeo.eigen.KalmanODE import KalmanODE as KalmanODE_c
from rodeo.eigen.KalmanODE2 import KalmanODE as KalmanODE_c2
from rodeo.tests.ode_functions import lorenz_fun as ode_fun_nd
from rodeo.tests.ode_functions_ctuple import lorenz_fun as ode_fun_ct

# pick ode function
use_ctuple = False
opts, args = getopt.getopt(sys.argv[1:], "c")
for o, a in opts:
    use_ctuple = o == "-c"

warnings.simplefilter('ignore', category=NumbaPerformanceWarning)


# ode function used by cython, C++, python

# def ode_fun(X, t, theta, out):
#     rho, sigma, beta = theta
#     p = len(X)//3
#     x, y, z = X[p*0], X[p*1], X[p*2]
#     out[0] = -sigma*x + sigma*y
#     out[1] = rho*x - y - x*z
#     out[2] = -beta*z + x*y
#     return

# ode function used by numba, odeint


@njit
def ode_fun2(X, t, theta, out=None):
    if out is None:
        out = np.zeros(3)
    rho, sigma, beta = theta
    p = len(X)//3
    x, y, z = X[p*0], X[p*1], X[p*2]
    out[:] = -sigma*x + sigma*y, rho*x - y - x*z, -beta*z + x*y
    return out


# problem setup and intialization
n_deriv = [1, 1, 1] 
n_deriv_prior = [3, 3, 3]
p = sum(n_deriv_prior)

# LHS Matrix of ODE
W_mat = np.zeros((len(n_deriv), sum(n_deriv) + len(n_deriv)))
for i in range(len(n_deriv)): 
    W_mat[i, sum(n_deriv[:i])+i+1] = 1

# it is assumed that the solution is sought on the interval [tmin, tmax].
n_eval = 3000
tmin = 0
tmax = 20
theta = np.array([28, 10, 8/3])

# The rest of the parameters can be tuned according to ODE
# For this problem, we will use
sigma = np.array([.5, .5, .5])

# Initial value, x0, for the IVP
x0 = [-12, -5, 38]
X0 = np.array([-12, 70, -5, 125, 38, -124/3])

# Get parameters needed to run the solver
dt = (tmax-tmin)/n_eval
W = zero_pad(W_mat, n_deriv, n_deriv_prior)
x0_state = zero_pad(X0, n_deriv, n_deriv_prior)
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
kode_cy = KalmanODE_cy(W, tmin, tmax, n_eval,
                       ode_fun, **kinit)  # Initialize the class
kode_cy.z_state = z_state
time_cy = timing(kode_cy, x0_state, W, theta, n_loops)

# numba
kode_num = KalmanODE_num(W, tmin, tmax, n_eval, ode_fun2, **kinit, z_state=z_state)
# Need to run once to compile KalmanTV
_ = kode_num.solve_sim(x0_state, W, np.asarray(theta))
time_num = timing(kode_num, x0_state, W, np.asarray(theta), n_loops)

# python
kode_py = KalmanODE_py(W, tmin, tmax, n_eval, ode_fun, **kinit)
kode_py.z_state = z_state
time_py = timing(kode_py, x0_state, W, theta, n_loops)

# odeint
tseq = np.linspace(tmin, tmax, n_eval+1)
time_det = det_timing(ode_fun, x0, tseq, n_loops, theta)

p# odeint2
# Need to run once to compile jitted ode function
_ = odeint(ode_fun2, x0, tseq, args=(theta, ))
time_det2 = det_timing(ode_fun2, x0, tseq, n_loops, theta)

print("Cython is {}x faster than Python".format(time_py/time_cy))
print("Numba is {}x faster than Python".format(time_py/time_num))
print("C++ is {}x faster than Python".format(time_py/time_c))
print("C++2 is {}x faster than Python".format(time_py/time_c2))
print("ode is {}x faster than Python".format(time_py/time_det))
print("ode2 is {}x faster than Python".format(time_py/time_det2))
