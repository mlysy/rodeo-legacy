"""
.. module:: readme_graph

Python file for the sole purpose of producing the graphs in the README file.

"""
import numpy as np
from math import sin, cos
import matplotlib.pyplot as plt

from BayesODE.utils.utils import root_gen
from BayesODE.Kalman.kalman_solver import kalman_solver
from BayesODE.Examples.euler_approx import euler_approx

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
def readme_kalman_draw(fun, n_eval, tmin, tmax, r0, p, sigma, mu, w, init, draws):
    roots = root_gen(r0, p)
    X = kalman_solver(fun, tmin, tmax, n_eval, mu, sigma, roots, w, init, draws)
    return X

def readme_solve(fun, p, tmin, tmax, n_eval, w, tau, sigma, init, draws):
    """
    Calculates kalman_ode, euler_ode, and exact_ode on the given grid for the README ode.

    Parameters
    ----------
    fun : function 
        Higher order ODE function :math:`W x_t = F(x_t, t)` taking arguments :math:`x` and :math:`t`.
    p : int
        Size of the CAR(p) process
    tmin : int
        First time point of the time interval to be evaluated; :math: `a`.
    tmax : int
        Last time point of the time interval to be evaluated; :math:`b`.
    n_eval : int
        Number of discretization points (:math:`N`) of the time interval that is evaluated, 
        such that discretization timestep is :math:`dt = b/N`.
    w : ndarray(q+1)
        Corresponds to the :math:`w` vector in the ODE equation.
    tau : ndarray(1)
        Decorrelation time.
    sigma : ndarray(1)
        Scale parameter.
    init : ndarray(q+1) or ndarray(p)
        The initial values of :math: `x_L` or :math:`X_L = (x_L, y_L)`.
    draws : int
        Number of samples we need to draw from the kalman solver.
        
    """
    mu = np.zeros(p)
    tseq = np.linspace(tmin, tmax, n_eval)
    Xt = readme_kalman_draw(fun, n_eval, tmin, tmax, tau, p, sigma, mu, w, init, draws)
    x_euler = euler_approx(ode_euler, tseq, init)
    x_exact = np.zeros((n_eval, 2))
    for i,t in enumerate(tseq):
        x_exact[i, 0] = ode_exact_x(t)
        x_exact[i, 1] = ode_exact_x1(t)

    return tseq, Xt, x_euler, x_exact

# Function that produces the graph as shown in README
def readme_graph(fun, p, tmin, tmax, n_eval, w, init, draws):
    """
    Produces the graph in README file.

    Parameters
    ----------
    fun : function 
        Higher order ODE function :math:`W x_t = F(x_t, t)` taking arguments :math:`x` and :math:`t`.
    p : int
        Size of the CAR(p) process
    tmin : int
        First time point of the time interval to be evaluated; :math: `a`.
    tmax : int
        Last time point of the time interval to be evaluated; :math:`b`.
    n_eval : int
        Number of discretization points (:math:`N`) of the time interval that is evaluated, 
        such that discretization timestep is :math:`dt = b/N`.
    w : ndarray(q+1)
        Corresponds to the :math:`w` vector in the ODE equation.
    init : ndarray(q+1) or ndarray(p)
        The initial values of :math: `x_L` or :math:`X_L = (x_L, y_L)`.
    draws : int
        Number of samples we need to draw from the kalman solver.
    
    """
    # Initialize variables for the graph
    dim_deriv = len(w) - 1
    N = [50, 100, 200]
    Tau = [1/.004, 1/0.01, 1/0.01]
    Sigma = [.5, .05, .001]
    dim_example = len(N)
    tseq = [None] * dim_example
    Xn = [None] * dim_example
    x_euler = [None] * dim_example
    x_exact = [None] * dim_example

    for i in range(dim_example):
        tseq[i], Xn[i], x_euler[i], x_exact[i] = readme_solve(fun=fun,
                                                              w=w, 
                                                              tmin=tmin, 
                                                              tmax=tmax, 
                                                              n_eval=N[i],
                                                              p=p, 
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
