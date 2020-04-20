"""
Python file for the sole purpose of producing the graphs in the Lane-Emden example.
"""
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt

from probDE.Kalman.ode_init import car_init, indep_ode_init
from probDE.cython.KalmanTest.KalmanODE import KalmanODE

def lane_exact(t):
    return 1/sqrt(1+t**2/3)

def lane_exact1(t):
    return -t/3*(1+t**2/3)**(-3/2)

def lane_graph(fun, n_state, n_meas, tmin, tmax, n_eval, w_vec, tau, sigma, init, draws):
    """
    Produces the graph for the Lane-Emden example in tutorial.

    Args:
        fun (function) : Higher order ODE function :math:`w x_t = F(x_t, t)` 
            taking arguments :math:`x` and :math:`t`.
        n_state (int) : Size of the CAR(p) process; :math:`p`.
        n_meas (int) : Size of the observed state.
        tmin (float) : First time point of the time interval to be evaluated; :math:`a`.
        tmax (float) : Last time point of the time interval to be evaluated; :math:`b`.
        n_eval (int) : Number of discretization points of the time interval that is evaluated, 
            such that discretization timestep is :math:`dt = (b-a)/N`.
        w_vec (ndarray(q+1)) : Corresponds to the :math:`w` vector in the ODE equation.
        tau (float) : Decorrelation time.
        sigma (float) : Scale parameter.
        init (ndarray(p)) : The initial values of :math:`X_L = (x_L, y_L)`.
        draws (int) : Number of samples we need to draw from the kalman solver.

    """
    # Get exact solution
    tseq = np.linspace(tmin, tmax, n_eval+1)
    title = ["$x^{(0)}_t$", "$x^{(1)}_t$"]
    n_var = len(title)
    exact_lane = np.zeros((n_eval+1, n_var))
    for i,t in enumerate(tseq):
        exact_lane[i, 0] = lane_exact(t)
        exact_lane[i, 1] = lane_exact1(t)

    Xn_lane = np.zeros((draws, n_eval+1, n_state))
    dt = (tmax-tmin)/n_eval
    for i in range(draws):
        # Get parameters needed to run the solver
        kinit = indep_ode_init([car_init(n_state, tau, sigma, dt, w_vec, init)], n_state)
        x0_state = kinit[-1]

        # Run the solver
        kalmanode = KalmanODE.initialize(kinit, n_state, n_meas, tmin, tmax, n_eval, fun)
        Xn_lane[i] = kalmanode.solve(x0_state, mv=False, sim=True)

    _, axs = plt.subplots(ncols=n_var, figsize=(15, 5))
    for pcol in range(n_var):
        for i in range(draws):
            if i == draws-1:
                axs[pcol].plot(tseq, Xn_lane[i, :, pcol], color="gray", alpha=.3, label='Kalman')
            else:
                axs[pcol].plot(tseq, Xn_lane[i, :, pcol], color='gray', alpha=.3)
        axs[pcol].set_title(title[pcol])
        axs[pcol].plot(tseq, exact_lane[:, pcol], label = 'exact')
        axs[pcol].legend(loc='upper left')
    plt.show()
