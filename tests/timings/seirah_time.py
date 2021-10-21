import warnings
import numba
import numpy as np
import getopt
import sys
from numba.core.errors import NumbaPerformanceWarning
from timer import *
from numba import njit, jit
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
from rodeo.tests.ode_functions import seirah_fun as ode_fun_nd
from rodeo.tests.ode_functions_ctuple import seirah_fun as ode_fun_ct

# pick ode function
use_ctuple = False
opts, args = getopt.getopt(sys.argv[1:], "c")
for o, a in opts:
    use_ctuple = o == "-c"


# # ode function used by cython, C++, python

# def ode_fun(X, t, theta, X_out):
#     p = len(X)//5
#     M, S, E, I, R = X[::p]
#     N = M+S+E+I+R
#     Lambda, delta, beta, mu, epsilon, gamma = theta
#     X_out[0] = Lambda - delta*M - mu*M
#     X_out[1] = delta*M - beta*S*I/N - mu*S
#     X_out[2] = beta*S*I/N - (epsilon + mu)*E
#     X_out[3] = epsilon*E - (gamma + mu)*I
#     X_out[4] = gamma*I - mu*R
#     return

# ode function used by numba, odeint

@njit
def ode_fun2(X_t, t, theta, out=None):
    "SEIRAH ODE function"
    if out is None:
        out = np.zeros(6)
    p = len(X_t)//6
    S, E, I, R, A, H = X_t[::p]
    N = S + E + I + R + A + H
    b, r, alpha, D_e, D_I, D_q = theta
    D_h = 30
    dS = -b*S*(I + alpha*A)/N
    dE = b*S*(I + alpha*A)/N - E/D_e
    dI = r*E/D_e - I/D_q - I/D_I
    dR = (I + A)/D_I + H/D_h
    dA = (1-r)*E/D_e - A/D_I
    dH = I/D_q - H/D_h
    out[:] = np.array([dS, dE, dI, dR, dA, dH])
    return out


# problem setup and intialization
n_deriv = [1]*6 # number of derivatives in IVP
n_deriv_prior = [3]*6 # number of derivatives in IBM prior
p = sum(n_deriv_prior)

# it is assumed that the solution is sought on the interval [tmin, tmax].
n_eval = 50
tmin = 0
tmax = 10
theta = np.array([2.23, 0.034, 0.55, 5.1, 2.3, 0.36])

# The rest of the parameters can be tuned according to ODE
# For this problem, we will use
n_var = 6
sigma = [.1]*n_var


# Initial value, x0, for the IVP
W_mat = np.zeros((len(n_deriv), sum(n_deriv)+len(n_deriv)))
for i in range(len(n_deriv)): 
    W_mat[i, sum(n_deriv[:i])+i+1] = 1
W = zero_pad(W_mat, n_deriv, n_deriv_prior)

x0 = np.array([63804435, 15492, 21752, 0, 618013, 93583])
v0 = ode_fun2(x0, 0, theta)
X0 = np.ravel([x0, v0], 'F')
x0_state = zero_pad(X0, n_deriv, n_deriv_prior)

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

# # C++
kode_c2 = KalmanODE_c2(W, tmin, tmax, n_eval, ode_fun, **kinit)
kode_c2.z_state = z_state
time_c2 = timing(kode_c2, x0_state, W, theta, n_loops)

# # cython
kode_cy = KalmanODE_cy(W, tmin, tmax, n_eval, ode_fun, **kinit)  # Initialize the class
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

print("Cython is {}x faster than Python".format(time_py/time_cy))
print("Numba is {}x faster than Python".format(time_py/time_num))
print("C++ is {}x faster than Python".format(time_py/time_c))
print("C++2 is {}x faster than Python".format(time_py/time_c2))
print("ode is {}x faster than Python".format(time_py/time_det))
