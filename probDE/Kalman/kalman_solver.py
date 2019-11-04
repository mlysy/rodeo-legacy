"""
.. module:: kalman_solver

Provides a probabilistic solver for univariate ordinary differential equations (ODEs) of the form

.. math:

    w'x_t = f(x_t, t), \qquad x_L = a.

"""
import numpy as np

from probDE.utils.utils import zero_pad
from probDE.Kalman.higher_mvncond import higher_mvncond
from probDE.Kalman.kalman_ode_higher import kalman_ode_higher
from probDE.Kalman.kalman_initial_draw import kalman_initial_draw

def kalman_solver(fun, tmin, tmax, n_eval, mu, sigma, roots, w, init, draws=1):
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
    mu : ndarray(p)
        Mean vector of the CAR(p) process.
    sigma : ndarray(1)
        Scale parameter of the CAR(p) process.
    roots : ndarray(p)
        Roots for the CAR(p) process.
    w : ndarray(q+1)
        Corresponds to the :math:`w` vector in the ODE equation.
    init : ndarray(q) or ndarray(p)
        Initial values.  Either `a`, or `X_L = (a, y_L)`.  In case of the former, y_L is drawn from stationary distribution condional on `x_L = a`.
    
    Returns
    -------
    XSample : ndarray(n_eval, p)
        Sample solution at time t given observations from times [0...N] for :math:`t = 0,1/N,\ldots,1`.
    meanX : ndarray(n_eval, p)
        Posterior mean of the solution process :math: `y_n` at times :math:`t = 0,1/N,\ldots,1`.
    varX : ndarray(n_eval, p, p)
        Posterior variance of the solution process at times :math:`t = 0,1/N,\ldots,1`.

    """
    # p in CAR(p) process
    p = len(roots)
    # grid size delta
    delta_t = np.array([(tmax - tmin)/n_eval])
    # Parameters in the State Space Model
    T_mat, R_mat = higher_mvncond(delta_t, roots, sigma) 
    c = mu - T_mat.dot(mu.T)
    # Pad the w vector
    W_vec = zero_pad(w, p)
    # initialize prior
    if len(init) < p:
        X_init = kalman_initial_draw(roots, sigma, init, p)
    else:
        X_init = init

    if draws == 1:
        XSample, meanX, varX = kalman_ode_higher(fun, X_init, tmin, tmax, n_eval-1, T_mat, c, R_mat, W_vec)
        return XSample, meanX, varX
    else:
        XSampleDraws = np.zeros((draws, n_eval, p))
        for i in range(draws):
            XSampleDraws[i],_,_ = kalman_ode_higher(fun, X_init, tmin, tmax, n_eval-1, T_mat, c, R_mat, W_vec)
        return XSampleDraws
