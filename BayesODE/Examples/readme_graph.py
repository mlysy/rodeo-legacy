"""
.. module:: readme_graph

Python file for the sole purpose of producing the graphs in the README file.

"""
import numpy as np
from math import sin, cos
import matplotlib.pyplot as plt

from BayesODE.utils.utils import root_gen
from BayesODE.Kalman.kalman_initial_draw import kalman_initial_draw
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
def readme_kalman_draw(fun, n_eval, tmin, tmax, r0, p, sigma, lamb, w, x_init, draws):
    roots = root_gen(r0, p)
    X_init = kalman_initial_draw(roots, sigma, x_init, p)
    X = kalman_solver(fun, tmin, tmax, n_eval, lamb, sigma, roots, w, X_init, draws)
    return X

# Function that produces the graph as shown in README
def readme_graph(fun, p, tmin, tmax, n_eval, w, x_init, draws):
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
    X_init : ndarray(p)
        The initial values of :math:`X_L = (x_L, y_L)`.
    draws : int
        Number of samples we need to draw from the kalman solver.
    
    """
    # Variables defined for the example in README
    q = len(w) - 1
    p = q+2
    lamb = np.zeros(p)

    # N=50
    tseq_50 = np.linspace(tmin, tmax, 50)
    Xn_50 = readme_kalman_draw(fun, 50, tmin, tmax, 1/0.04, p, 0.5, lamb, w, x_init, draws)
    euler_50 = euler_approx(ode_euler, tseq_50, x_init)
    exact_50 = np.zeros((50, 2))
    for i,t in enumerate(tseq_50):
        exact_50[i, 0] = ode_exact_x(t)
        exact_50[i, 1] = ode_exact_x1(t)

    # N=100
    tseq_100 = np.linspace(tmin, tmax, 100)
    Xn_100 = readme_kalman_draw(fun, 100, tmin, tmax, 1/0.01, p, 0.5, lamb, w, x_init, draws)
    euler_100 = euler_approx(ode_euler, tseq_100, x_init)
    exact_100 = np.zeros((100, 2))
    for i,t in enumerate(tseq_100):
        exact_100[i, 0] = ode_exact_x(t)
        exact_100[i, 1] = ode_exact_x1(t)
    
    # N=200
    tseq_200 = np.linspace(tmin, tmax, 200)
    Xn_200 = readme_kalman_draw(fun, 200, tmin, tmax, 1/0.1, p, 0.001, lamb, w, x_init, draws)
    euler_200 = euler_approx(ode_euler, tseq_200, x_init)
    exact_200 = np.zeros((200, 2)) 
    for i,t in enumerate(tseq_200):
        exact_200[i, 0] = ode_exact_x(t)
        exact_200[i, 1] = ode_exact_x1(t)

    _, axs = plt.subplots(2, 3, figsize=(20, 6))
    #Plot Kalman draws
    for i in range(draws):
        if i == draws-1:
            axs[0, 0].plot(tseq_50, Xn_50[i,:,0], color="lightgray", alpha=0.3, label='Kalman')
            axs[0, 0].plot(tseq_50, euler_50[:,0], label='Euler')
            axs[0, 0].plot(tseq_50, exact_50[:,0], label='Exact')
            axs[0, 0].set_title("N=50; x^{(0)}")
            axs[0, 0].legend(loc='upper left')
            
            axs[0, 1].plot(tseq_100, Xn_100[i,:,0], color="lightgray", alpha=0.3, label='Kalman')
            axs[0, 1].plot(tseq_100, euler_100[:,0], label='Euler')
            axs[0, 1].plot(tseq_100, exact_100[:,0], label='Exact')
            axs[0, 1].set_title("N=100; x^{(0)}")
            axs[0, 1].legend(loc='upper left')
            
            axs[0, 2].plot(tseq_200, Xn_200[i,:,0], color="lightgray", alpha=0.3, label='Kalman')
            axs[0, 2].plot(tseq_200, euler_200[:,0], label='Euler')
            axs[0, 2].plot(tseq_200, exact_200[:,0], label='Exact')
            axs[0, 2].set_title("N=200; x^{(0)}")
            axs[0, 2].legend(loc='upper left')
            
            axs[1, 0].plot(tseq_50, Xn_50[i,:,1], color="lightgray", alpha=0.3, label='Kalman')
            axs[1, 0].plot(tseq_50, euler_50[:,1], label='Euler')
            axs[1, 0].plot(tseq_50, exact_50[:,1], label='Exact')
            axs[1, 0].set_title("N=50; x^{(1)}")
            axs[1, 0].legend(loc='upper left')
            
            axs[1, 1].plot(tseq_100, Xn_100[i,:,1], color="lightgray", alpha=0.3, label='Kalman')
            axs[1, 1].plot(tseq_100, euler_100[:,1], label='Euler')
            axs[1, 1].plot(tseq_100, exact_100[:,1], label='Exact')
            axs[1, 1].set_title("N=100; x^{(1)}")
            axs[1, 1].legend(loc='upper left')

            axs[1, 2].plot(tseq_200, Xn_200[i,:,1], color="lightgray", alpha=0.3, label='Kalman')
            axs[1, 2].plot(tseq_200, euler_200[:,1], label='Euler')
            axs[1, 2].plot(tseq_200, exact_200[:,1], label='Exact')
            axs[1, 2].set_title("N=200; x^{(1)}")
            axs[1, 2].legend(loc='upper left')

        else:
            axs[0,0].plot(tseq_50, Xn_50[i,:,0], color="lightgray", alpha=0.3)
            axs[0,1].plot(tseq_100, Xn_100[i,:,0], color="lightgray", alpha=0.3)
            axs[0,2].plot(tseq_200, Xn_200[i,:,0], color="lightgray", alpha=0.3)
            axs[1,0].plot(tseq_50, Xn_50[i,:,1], color="lightgray", alpha=0.3)
            axs[1,1].plot(tseq_100, Xn_100[i,:,1], color="lightgray", alpha=0.3)
            axs[1,2].plot(tseq_200, Xn_200[i,:,1], color="lightgray", alpha=0.3)
    
    plt.show()
