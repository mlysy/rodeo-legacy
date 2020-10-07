import numpy as np
from scipy.integrate import odeint
import numba
from numba import njit
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
def mseir(X, t, theta, X_out):
    p = len(X)//5
    M, S, E, I, R = X[::p]
    N = M+S+E+I+R
    Lambda, delta, beta, mu, epsilon, gamma = theta
    X_out[0] = Lambda - delta*M - mu*M
    X_out[1] = delta*M - beta*S*I/N - mu*S
    X_out[2] = beta*S*I/N - (epsilon + mu)*E
    X_out[3] = epsilon*E - (gamma + mu)*I
    X_out[4] = gamma*I - mu*R
    return

# ode function used by numba, odeint
@njit
def mseir2(X_t, t, theta, out=None):
    "MSEIR ODE function"
    if out is None:
        out = np.zeros(5)
    p = len(X_t)//5
    M, S, E, I, R = X_t[::p]
    N = M+S+E+I+R
    Lambda, delta, beta, mu, epsilon, gamma = theta
    dM = Lambda - delta*M - mu*M
    dS = delta*M - beta*S*I/N - mu*S
    dE = beta*S*I/N - (epsilon + mu)*E
    dI = epsilon*E - (gamma + mu)*I
    dR = gamma*I - mu*R
    out[:] = np.array([dM, dS, dE, dI, dR])
    return out

# problem setup and intialization
n_deriv = [2]*5 # Total state
n_deriv_prior = [3]*5
p = sum(n_deriv_prior)
n_obs = 5 # Total measures

# it is assumed that the solution is sought on the interval [tmin, tmax].
n_eval = 50
tmin = 0
tmax = 20
theta = np.array([1.1, 0.7, 0.4, 0.005, 0.02, 0.03])

# The rest of the parameters can be tuned according to ODE
# For this problem, we will use
n_var = 5
sigma = [.1]*n_var

# Initial value, x0, for the IVP
W_mat = np.zeros((len(n_deriv), sum(n_deriv)))
for i in range(len(n_deriv)): W_mat[i, sum(n_deriv[:i])+1] = 1
W = zero_pad(W_mat, n_deriv, n_deriv_prior)

x0 = np.array([1000, 100, 50, 3, 3])
v0 = mseir2(x0, 0, theta)
X0 = np.ravel([x0, v0], 'F')
x0_state = zero_pad(X0, n_deriv, n_deriv_prior)

# Get parameters needed to run the solver
dt = (tmax-tmin)/n_eval
ode_init = ibm_init(dt, n_deriv_prior, sigma)
kinit = indep_init(ode_init, n_deriv_prior)
z_states = rand_mat(2*(n_eval+1), p)

# Timings
n_loops = 1000
# C++
kode_c = KalmanODE_c(p, n_obs, tmin, tmax, n_eval, mseir, **kinit) 
kode_c.z_states = z_states
time_c = timing(kode_c, x0_state, W, theta, n_loops)

# cython
kode_cy = KalmanODE_cy(p, n_obs, tmin, tmax, n_eval, mseir, **kinit) # Initialize the class
kode_cy.z_states = z_states
time_cy = timing(kode_cy, x0_state, W, theta, n_loops)

# numba
kode_num = KalmanODE_num(p, n_obs, tmin, tmax, n_eval, mseir2, kinit['mu_state'], 
                         kinit['wgt_state'], kinit['var_state'], z_states)
_ = kode_num.solve(x0_state, W, np.asarray(theta), True) # Need to run once to compile KalmanTV
time_num = timing(kode_num, x0_state, W, np.asarray(theta), n_loops, True)

# python
kode_py = KalmanODE_py(p, n_obs, tmin, tmax, n_eval, mseir, **kinit)
kode_py.z_states = z_states
time_py = timing(kode_py, x0_state, W, theta, n_loops//10)

# odeint
tseq = np.linspace(tmin, tmax, n_eval+1)
time_det = det_timing(mseir2, x0, tseq, n_loops*10, theta)

print("Cython is {}x faster than Python".format(time_py/time_cy))
print("Numba is {}x faster than Python".format(time_py/time_num))
print("C++ is {}x faster than Python".format(time_py/time_c))
print("ode is {}x faster than Python".format(time_py/time_det))