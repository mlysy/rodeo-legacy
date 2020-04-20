"""
Python file for the sole purpose of producing the graphs in the README file.
"""
import numpy as np
from math import sin, cos
import matplotlib.pyplot as plt

from probDE.Kalman.ode_init import car_init, indep_ode_init
from probDE.cython.KalmanTest.KalmanODE import KalmanODE
from probDE.Examples.euler_approx import euler_approx

# Example ODE Exact Solution for x_t^{(0)}
def ode_exact_x(t):
    return (-3*cos(t) + 2*sin(t) - sin(2*t))/3

# Example ODE Exact Solution for x_t^{(1)}
def ode_exact_x1(t):
    return (-2*cos(2*t) + 3*sin(t) + 2*cos(t))/3

# Example ode written for Euler Approximation
def ode_euler(x,t):
    return np.array([x[1], sin(2*t) -x[0]])

# Helper function to draw samples from Kalman solver
def readme_kalman_draw(fun, n_state, n_meas, n_eval, tmin, tmax, tau, sigma, w, init, draws):
    #roots = root_gen(r0, p)
    #X = kalman_solver(fun, tmin, tmax, n_eval, mu, sigma, roots, w, init, draws)
    # Get parameters needed to run the solver
    dt = (tmax-tmin)/n_eval
    X = np.zeros((draws, n_eval+1, n_state))
    for i in range(draws):
        kinit = indep_ode_init([car_init(n_state, tau, sigma, dt, w, init)], n_state)
        x0_state = kinit[-1]
        kalmanode = KalmanODE.initialize(kinit, n_state, n_meas, tmin, tmax, n_eval, fun)
        X[i]= kalmanode.solve(x0_state, mv=False, sim=True)
    return X

def readme_solve(fun, n_state, n_meas, tmin, tmax, n_eval, w, tau, sigma, init, draws):
    """
    Calculates kalman_ode, euler_ode, and exact_ode on the given grid for the README ode.

    Args:
        fun (function) : Higher order ODE function :math:`w x_t = F(x_t, t)` 
            taking arguments :math:`x` and :math:`t`.
        n_state (int) : Size of the CAR(p) process; :math:`p`.
        n_meas (int) : Size of the observed state.
        tmin (float) : First time point of the time interval to be evaluated; :math:`a`.
        tmax (float) : Last time point of the time interval to be evaluated; :math:`b`.
        n_eval (int) : Number of discretization points of the time interval that is evaluated, 
            such that discretization timestep is :math:`dt = (b-a)/N`.
        w (ndarray(q+1)) : Corresponds to the :math:`w` vector in the ODE equation.
        tau (float): Decorrelation time.
        sigma (float): Scale parameter.
        init (ndarray(q+1)) or (ndarray(p)): The initial values of :math:`x_L` or :math:`X_L = (x_L, y_L)`.
        draws (int): Number of samples we need to draw from the kalman solver.
    Returns:
        (tuple):
        - **tseq** (ndarray(n_eval+1)): Time discretization points for :math:`t = 0,1/N,\ldots,1`.
        - **Xt** (ndarray(draws, n_eval+1, n_state)): Draws of the solution process :math:`X_t` at times
          :math:`t = 0,1/N,\ldots,1`.
        - **x_euler** (ndarray(n_eval+1, 2)): Euler approximation of the solution process at
          times :math:`t = 0,1/N,\ldots,1`.
        - **x_exact** (ndarray(n_eval+1, 2)): Exact solution at times :math:`t = 0,1/N,\ldots,1`.

    """
    tseq = np.linspace(tmin, tmax, n_eval+1)
    Xt = readme_kalman_draw(fun, n_state, n_meas, n_eval, tmin, tmax, tau, sigma, w, init, draws)
    x_euler = euler_approx(ode_euler, tseq, init)
    x_exact = np.zeros((n_eval+1, 2))
    for i,t in enumerate(tseq):
        x_exact[i, 0] = ode_exact_x(t)
        x_exact[i, 1] = ode_exact_x1(t)

    return tseq, Xt, x_euler, x_exact

# Function that produces the graph as shown in README
def readme_graph(fun, n_state, n_meas, tmin, tmax, w, init, draws):
    """
    Produces the graph in README file.

    Args:
        fun (function) : Higher order ODE function :math:`W x_t = F(x_t, t)` 
            taking arguments :math:`x` and :math:`t`.
        n_state (int) : Size of the CAR(p) process; :math:`p`.
        n_meas (int) : Size of the observed state.
        tmin (float) : First time point of the time interval to be evaluated; :math:`a`.
        tmax (float) : Last time point of the time interval to be evaluated; :math:`b`.
        w (ndarray(q+1)) : Corresponds to the :math:`w` vector in the ODE equation.
        init (ndarray(q+1)) or (ndarray(p)): The initial values of :math:`x_L` or :math:`X_L = (x_L, y_L)`.
        draws (int): Number of samples we need to draw from the kalman solver.
    
    """
    # Initialize variables for the graph
    dim_deriv = len(w) - 1
    N = [50, 100, 200]
    Tau = [1/.004, 1/0.02, 1/0.02]
    Sigma = [.5, .05, .05]
    dim_example = len(N)
    tseq = [None] * dim_example
    Xn = [None] * dim_example
    x_euler = [None] * dim_example
    x_exact = [None] * dim_example

    for i in range(dim_example):
        tseq[i], Xn[i], x_euler[i], x_exact[i] = readme_solve(fun=fun,
                                                              n_state=n_state,
                                                              n_meas=n_meas, 
                                                              tmin=tmin, 
                                                              tmax=tmax, 
                                                              n_eval=N[i],
                                                              w=w,
                                                              tau=Tau[i], 
                                                              sigma=Sigma[i], 
                                                              init=init,
                                                              draws=draws)

    _, axs = plt.subplots(dim_deriv, dim_example, figsize=(20, 7))
    for prow in range(dim_deriv):
        for pcol in range(dim_example):
            # plot Kalman draws
            for i in range(draws):
                if i == (draws - 1):
                    axs[prow, pcol].plot(tseq[pcol], Xn[pcol][i,:,prow], 
                                        color="lightgray", alpha=.3, label="Kalman")
                else:
                    axs[prow, pcol].plot(tseq[pcol], Xn[pcol][i,:,prow], 
                                        color="lightgray", alpha=.3)
            # plot Euler and Exact
            axs[prow, pcol].plot(tseq[pcol], x_euler[pcol][:,prow], 
                                label="Euler")
            axs[prow, pcol].plot(tseq[pcol], x_exact[pcol][:,prow], 
                                label="Exact")
            # set legend and title
            axs[prow, pcol].set_title("$x^{(%s)}_t$;   $N=%s$" % (prow, N[pcol]))
            #axs[prow, pcol].set_ylabel("$x^{(%s)}_t$" % (prow))
            if (prow == 0) & (pcol == 0):
                axs[prow, pcol].legend(loc='upper left')
   
    plt.show()
