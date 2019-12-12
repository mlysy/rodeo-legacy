"""
.. module:: lorenz_graph

Python file for the sole purpose of producing the graphs in the Lorenz63 example.

"""
import numpy as np
import matplotlib.pyplot as plt

from probDE.utils.utils import root_gen
from probDE.Kalman.kalman_multi_solver import kalman_multi_solver

def lorenz_graph(fun, p, tmin, tmax, n_eval, W, tau, sigma, init, draws):
    """
    Produces the graph for the Lorenz63 example in tutorial.

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
    W : ndarray(q+1)
        Corresponds to the :math:`W` vector in the ODE equation.
    tau : ndarray(1)
        Decorrelation time.
    sigma : ndarray(1)
        Scale parameter.
    init : ndarray((q+1)*n_var) or ndarray(p*n_var)
        The initial values of :math: `x_L` or :math:`X_L = (x_L, y_L)`.
    draws : int
        Number of samples we need to draw from the kalman solver.
    """
    tseq = np.linspace(tmin, tmax, n_eval)
    ylabel = ['x', 'y', 'z']
    n_var = len(ylabel)
    mu = np.zeros(n_var*p)
    roots = root_gen(tau, p)
    rootlst = [roots*10]*n_var
    sigmalst = [sigma]*n_var
    Xn = kalman_multi_solver(fun, tmin, tmax, n_eval, mu, sigmalst, rootlst, W, init, draws)
    
    _, axs = plt.subplots(n_var, 1, figsize=(20, 7))
    for nrow in range(n_var):
        for ncol in range(draws):
            if ncol == (draws - 1):
                axs[nrow].plot(tseq, Xn[ncol, :, p*nrow],
                        color="gray", alpha=1, label="Kalman")
                axs[nrow].set_ylabel(ylabel[nrow])
            else:
                axs[nrow].plot(tseq, Xn[ncol, :, p*nrow],
                        color="gray", alpha=1)

        axs[nrow].legend(loc='upper left')
    plt.show()
