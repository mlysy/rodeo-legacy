"""
.. module:: kalman_multi_solver

Provides a probabilistic solver for multivariate ordinary differential equations (ODEs) of the form

.. math:

    W x_t = f(x_t, t), \qquad x_L = a.

"""
import numpy as np

from probDE.Kalman.multi_mvncond import multi_mvncond
from probDE.Kalman.kalman_ode_higher import kalman_ode_higher
from probDE.Kalman.kalman_initial_draw import kalman_initial_draw

def kalman_multi_solver(fun, tmin, tmax, n_eval, mu, sigmalst, rootlst, W, init, draws=1):
    """
    Provides a probabilistic solver for multivariate ordinary differential equations (ODEs) of the form

    .. math:

    W x_t = f(x_t, t), \qquad x_L = a.

    Parameters
    ----------
    fun : function 
        Higher order ODE function :math:`W x_t = F(x_t, t)` taking arguments :math:`x` and :math:`t`.
    tmin : int
        First time point of the time interval to be evaluated; :math: `a`.
    tmax : int
        Last time point of the time interval to be evaluated; :math:`b`.
    n_eval : int
        Number of discretization points (:math:`N`) of the time interval that is evaluated, 
        such that discretization timestep is :math:`dt = (tmax-tmin)/N`.
    mu : ndarray(p)
        Mean vector of the CAR(p) process.
    sigmalst : ndarray(n)
        A list of scale parameters of n CAR(p) processes.
    rootlst : ndarray(n, p)
        A list of roots for n CAR(p) processes.
    W : ndarray(n, q+1)
        Corresponds to the :math:`W` vector in the ODE equation.
    init : ndarray(n, q) or ndarray(n, p)
        Initial values.  Either `a`, or `X_L = (a, y_L)`.  In case of the former, y_L is drawn from stationary distribution condional on `x_L = a`.
    
    Returns
    -------
    XSample : ndarray(n_eval, p)
        Sample solution at time t given observations from times [0...N] for :math:`t = 0,1/N,\ldots,1`.
    meanX : ndarray(n_eval, p)
        Posterior mean of the solution process :math:`y_n` at times :math:`t = 0,1/N,\ldots,1`.
    varX : ndarray(n_eval, p, p)
        Posterior variance of the solution process at times :math:`t = 0,1/N,\ldots,1`.

    """
    # number of variates
    n = len(rootlst)
    # p in CAR(p) process
    p = len(rootlst[0])
    # grid size delta
    delta_t = np.array([(tmax - tmin)/n_eval])
    # Parameters in the State Space Model
    T_mat, R_mat = multi_mvncond(delta_t, rootlst, sigmalst) 
    c = mu - T_mat.dot(mu.T)
    
    W_mat = np.zeros((n, n*p))
    X_init = np.zeros(n*p)
    for i in range(n):
        # Pad the W matrix
        w_len = len(W[i])
        W_mat[i, p*i:p*i+w_len] = W[i]
        # initialize prior
        if len(init[i]) < p:
            X_init[p*i:p*(i+1)] = kalman_initial_draw(rootlst[i], sigmalst[i], init[i], p)
        else:
            X_init[p*i:p*(i+1)] = init[i]

    if draws == 1:
        XSample, meanX, varX = kalman_ode_higher(fun, X_init, tmin, tmax, n_eval-1, T_mat, c, R_mat, W_mat)
        return XSample, meanX, varX
    else:
        XSampleDraws = np.zeros((draws, n_eval, n*p))
        for i in range(draws):
            XSampleDraws[i],_,_ = kalman_ode_higher(fun, X_init, tmin, tmax, n_eval-1, T_mat, c, R_mat, W_mat)
        return XSampleDraws
