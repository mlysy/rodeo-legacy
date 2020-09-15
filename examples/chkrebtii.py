import numpy as np
from math import sin
import matplotlib.pyplot as plt

from probDE.car import car_init
from probDE.cython.KalmanODE import KalmanODE
from probDE.utils import indep_init
from readme_graph import readme_graph

def ode_fun(x_out, x, t, theta=None):
    x_out[0] = sin(2*t) - x[0]
    return

def chkrebtii_example():
    r"Produces the graph in Figure 1 of the paper."
    # LHS vector of ODE
    w_mat = np.array([[0.0, 0.0, 1.0]])

    # These parameters define the order of the ODE and the CAR(p) process
    n_obs = 1
    n_deriv = 4
    n_deriv_var = [4]

    # it is assumed that the solution is sought on the interval [tmin, tmax].
    n_eval = 200
    tmin = 0
    tmax = 10

    # The rest of the parameters can be tuned according to ODE
    # For this problem, we will use
    tau = [50]
    sigma = [.001]

    # Initial value, x0, for the IVP
    x0 = np.array([[-1., 0., 1.]])

    # Get parameters needed to run the solver
    dt = (tmax-tmin)/n_eval
    # All necessary parameters are in kinit, namely, T, c, R, W
    kinit, W, x0_state = indep_init(car_init(n_deriv_var, tau, sigma, dt, x0), w_mat, n_deriv)

    # Initialize the Kalman class
    kalmanode = KalmanODE(n_deriv, n_obs, tmin, tmax, n_eval, ode_fun, **kinit)
    # Run the solver to get an approximation
    kalman_sim = kalmanode.solve(x0_state, W, mv=False, sim=True)

    # Produces the graph in Figure 1
    draws = 100
    readme_graph(ode_fun, n_deriv, n_obs, tmin, tmax, w_mat, x0, draws)
    return

if __name__ == '__main__':
    chkrebtii_example()
    