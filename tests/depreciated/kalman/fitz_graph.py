"""
.. module:: fitz_graph

Python file for the sole purpose of producing the graphs in the FitzHugh-Nagumo example.

"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

from probDE.utils.utils import root_gen
from probDE.Kalman.kalman_multi_solver import kalman_multi_solver

def fitz_graph(fun, fun_ode, p, tmin, tmax, W, tau, sigma, init, draws):
    """
    Produces the graph for the FitzHugh-Nagumo example in tutorial.

    Parameters
    ----------
    fun : function 
        Higher order ODE function :math:`W x_t = F(x_t, t)` taking arguments :math:`x` and :math:`t`.
    fun_ode: function
        The same function as F but compatible with odeint.
    p : int
        Size of the CAR(p) process
    tmin : int
        First time point of the time interval to be evaluated; :math: `a`.
    tmax : int
        Last time point of the time interval to be evaluated; :math:`b`.
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
    N = [400, 800, 2000, 4000, 8000]
    ylabel = ['V', 'R']
    n_var = len(ylabel)
    dim_example = len(N)
    Xn = [None]*dim_example
    tseq = [None]*dim_example
    X_exact = [None]*dim_example
    mu = np.zeros(n_var*p)
    roots = root_gen(tau, p)
    rootlst = [roots]*n_var
    sigmalst = [sigma]*n_var
    for i in range(dim_example):
        Xn[i] = kalman_multi_solver(fun, tmin, tmax, N[i], mu, sigmalst, rootlst, W, init, draws)
        tseq[i] = np.linspace(tmin, tmax, N[i])
        X_exact[i] = odeint(fun_ode, init[:,0], tseq[i])

    _, axs = plt.subplots(n_var, dim_example, figsize=(20, 7)) 
    for prow in range(n_var):
        for pcol in range(dim_example):
            for i in range(draws):
                if i == (draws - 1):
                    axs[prow, pcol].plot(tseq[pcol], Xn[pcol][i,:,p*prow], 
                                        color="lightgray", alpha=.3, label="Kalman")
                else:
                    axs[prow, pcol].plot(tseq[pcol], Xn[pcol][i,:,p*prow], 
                                        color="lightgray", alpha=.3)
            axs[prow, pcol].plot(tseq[pcol], X_exact[pcol][:, prow], label='Exact')
            axs[prow, pcol].set_title("$%s$;   $N=%s$" % (ylabel[prow], N[pcol]))
            if pcol == 0 and prow == 0:
                axs[prow, pcol].legend(loc='upper left')
            
    plt.show()
