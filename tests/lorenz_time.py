from KalmanODE_py import KalmanODE_py
from kalmanode_numba import KalmanODE as KalmanODE_num
from probDE.tests.ode_functions import lorenz_fun as lorenz
from probDE.tests.KalmanODE import KalmanODE as KalmanODE_c
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

# def lorenz(X, t, theta, out):
#     rho, sigma, beta = theta
#     p = len(X)//3
#     x, y, z = X[p*0], X[p*1], X[p*2]
#     out[0] = -sigma*x + sigma*y
#     out[1] = rho*x - y - x*z
#     out[2] = -beta*z + x*y
#     return

# ode function used by numba, odeint


@njit
def lorenz2(X, t, theta, out=None):
    if out is None:
        out = np.zeros(3)
    rho, sigma, beta = theta
    p = len(X)//3
    x, y, z = X[p*0], X[p*1], X[p*2]
    out[:] = -sigma*x + sigma*y, rho*x - y - x*z, -beta*z + x*y
    return out


# problem setup and intialization
n_obs = 3
n_deriv = [2, 2, 2]
n_deriv_prior = [3, 3, 3]
p = sum(n_deriv_prior)

# LHS Matrix of ODE
W_mat = np.zeros((len(n_deriv), sum(n_deriv)))
for i in range(len(n_deriv)):
    W_mat[i, sum(n_deriv[:i])+1] = 1

# it is assumed that the solution is sought on the interval [tmin, tmax].
n_eval = 3000
tmin = 0
tmax = 20
theta = np.array([28, 10, 8/3])

# The rest of the parameters can be tuned according to ODE
# For this problem, we will use
tau = np.array([0.1, 1, 10])
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
z_states = rand_mat(2*(n_eval+1), p)

# Timings
n_loops = 100
# C++
kode_c = KalmanODE_c(p, n_obs, tmin, tmax, n_eval, lorenz, **kinit)
kode_c.z_states = z_states
time_c = timing(kode_c, x0_state, W, theta, n_loops)

# cython
kode_cy = KalmanODE_cy(p, n_obs, tmin, tmax, n_eval,
                       lorenz, **kinit)  # Initialize the class
kode_cy.z_states = z_states
time_cy = timing(kode_cy, x0_state, W, theta, n_loops)

# numba
kode_num = KalmanODE_num(p, n_obs, tmin, tmax, n_eval, lorenz2, kinit['mu_state'],
                         kinit['wgt_state'], kinit['var_state'], z_states)
# Need to run once to compile KalmanTV
_ = kode_num.solve(x0_state, W, np.asarray(theta), True)
time_num = timing(kode_num, x0_state, W, np.asarray(theta), n_loops, True)

# python
kode_py = KalmanODE_py(p, n_obs, tmin, tmax, n_eval, lorenz, **kinit)
kode_py.z_states = z_states
time_py = timing(kode_py, x0_state, W, theta, n_loops//10)

# odeint
tseq = np.linspace(tmin, tmax, n_eval+1)
time_det = det_timing(lorenz2, x0, tseq, n_loops*10, theta)

print("Cython is {}x faster than Python".format(time_py/time_cy))
print("Numba is {}x faster than Python".format(time_py/time_num))
print("C++ is {}x faster than Python".format(time_py/time_c))
print("ode is {}x faster than Python".format(time_py/time_det))
