"""
.. module:: kalman_solver

Provides a probabilistic solver for univariate ordinary differential equations (ODEs) of the form

.. math:

    w'x_t = f(x_t, t), \qquad x_L = a.

"""
import numpy as np

from BayesODE.utils.utils import zero_pad
from BayesODE.Kalman.higher_mvncond import higher_mvncond
from BayesODE.Kalman.kalman_ode_higher import kalman_ode_higher

def kalman_solver(fun, tmin, tmax, n_eval, llambda, sigma, roots, w, X_init, draws=1):
    """
    Provides a probabilistic solver for univariate ordinary differential equations (ODEs) of the form

    .. math:

    w'x_t = f(x_t, t), \qquad x_L = a.

    Parameters
    ----------
    fun : function 
        Higher order ODE function :math:`w' x_t = F(x_t, t)` taking arguments :math:`x` and :math:`t`.
    tmin : int
        First time point of the time interval to be evaluated; :math: `a`.
    tmax : int
        Last time point of the time interval to be evaluated; :math:`b`.
    n_eval : int
        Number of discretization points (:math:`N`) of the time interval that is evaluated, 
        such that discretization timestep is :math:`dt = (tmax-tmin)/N`.
    llambda : ndarray(p)
        Mean vector of the CAR(p) process.
    sigma : int
        Scale parameter of the CAR(p) process.
    roots : ndarray(p)
        Roots for the CAR(p) process.
    w : ndarray(q+1)
        Corresponds to the :math:`w` vector in the ODE equation.
    X_init : ndarray(p)
        The initial values of :math:`X_L = (x_L, y_L)`.
    
    Returns
    -------
    XSample : ndarray(n_eval, p)
        Sample solution at time t given observations from times [0...N] for :math:`t = 0,1/N,\ldots,1`.
    muX : ndarray(n_eval, p)
        Posterior mean of the solution process :math: `y_n` at times :math:`t = 0,1/N,\ldots,1`.
    varX : ndarray(n_eval, p, p)
        Posterior variance of the solution process at times :math:`t = 0,1/N,\ldots,1`.

    """
    # p in CAR(p) process
    p = len(roots)
    # grid size delta
    delta_t = np.array([(tmax - tmin)/n_eval])
    # Parameters in the State Space Model
    T, R = higher_mvncond(delta_t, roots, sigma) 
    c = llambda - T.dot(llambda.T)
    #Pad the w vector
    W_vec = zero_pad(w, p)

    if draws == 1:
        XSample, muX, varX = kalman_ode_higher(fun, X_init, tmin, tmax, n_eval-1, T, c, R, W_vec)
        return XSample, muX, varX
    else:
        XSampleDraws = np.zeros((draws, n_eval, p))
        for i in range(draws):
            XSampleDraws[i],_,_ = kalman_ode_higher(fun, X_init, tmin, tmax, n_eval-1, T, c, R, W_vec)
        return XSampleDraws