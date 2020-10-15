from KalmanODE_py import KalmanODE_py
from kalmanode_numba import KalmanODE as KalmanODE_num
from probDE.tests.KalmanODE import KalmanODE as KalmanODE_c
from probDE.tests.ode_functions import fitz_fun as fitz
from probDE.cython.KalmanODE import KalmanODE as KalmanODE_cy
from probDE.utils.utils import rand_mat, indep_init, zero_pad
from probDE.ibm import ibm_init
import numpy as np
from scipy.integrate import odeint
import numba
from numba import njit
from timer import *
import warnings
from numba.core.errors import NumbaPerformanceWarning
warnings.simplefilter('ignore', category=NumbaPerformanceWarning)


# ode function used by cython, C++, python
# def fitz(X, t, theta, X_out):
#     "FitzHugh-Nagumo ODE function."
#     a, b, c = theta
#     n_deriv1 = len(X)//2
#     V, R = X[0], X[n_deriv1]
#     X_out[0] = c*(V - V**3/3 + R)
#     X_out[1] = -1/c*(V - a + b*R)
#     return

# ode function used by numba, odeint

@njit
def fitz2(X, t, theta, out=None):
    "FitzHugh-Nagumo ODE function."
    if out is None:
        out = np.zeros(2)
    a, b, c = theta
    n_deriv1 = len(X)//2
    V, R = X[0], X[n_deriv1]
    out[:] = c*(V - V*V*V/3 + R), -1/c*(V - a + b*R)
    return out


# problem setup and intialization
n_deriv = [2, 2]  # Total state
n_deriv_prior = [3, 3]
p = sum(n_deriv_prior)
n_obs = 2  # Total measures

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
z_states = rand_mat(2*(n_eval+1), p)

# Timings
n_loops = 100
# C++
kode_c = KalmanODE_c(p, n_obs, tmin, tmax, n_eval, fitz, **kinit)
kode_c.z_states = z_states
time_c = timing(kode_c, x0_state, W, theta, n_loops)

# cython
kode_cy = KalmanODE_cy(p, n_obs, tmin, tmax, n_eval,
                       fitz, **kinit)  # Initialize the class
kode_cy.z_states = z_states
time_cy = timing(kode_cy, x0_state, W, theta, n_loops)

# numba
kode_num = KalmanODE_num(p, n_obs, tmin, tmax, n_eval, fitz2, kinit['mu_state'],
                         kinit['wgt_state'], kinit['var_state'], z_states)
# Need to run once to compile KalmanTV
_ = kode_num.solve(x0_state, W, np.asarray(theta))
time_num = timing(kode_num, x0_state, W, np.asarray(theta), n_loops)

# python
kode_py = KalmanODE_py(p, n_obs, tmin, tmax, n_eval, fitz, **kinit)
kode_py.z_states = z_states
time_py = timing(kode_py, x0_state, W, theta, n_loops//10)

# odeint
tseq = np.linspace(tmin, tmax, n_eval+1)
time_det = det_timing(fitz2, x0, tseq, n_loops*10, theta)

print("Cython is {}x faster than Python".format(time_py/time_cy))
print("Numba is {}x faster than Python".format(time_py/time_num))
print("C++ is {}x faster than Python".format(time_py/time_c))
print("ode is {}x faster than Python".format(time_py/time_det))
