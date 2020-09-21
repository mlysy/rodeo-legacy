"""
.. module:: lorenz_graph

Python file for the sole purpose of producing the graphs in the Lorenz63 example.

"""
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

from probDE.car import car_init
from probDE.cython.KalmanODE import KalmanODE
from probDE.utils import indep_init, zero_pad

def lorenz(state, t, theta=(28, 10, 8/3)):
    r"Loren63 ODE function"
    rho, sigma, beta = theta
    x, y, z = state  # Unpack the state vector
    return -sigma*x + sigma*y, rho*x - y -x*z, -beta*z + x*y

def lorenz_graph(fun, n_deriv, n_deriv_prior, n_obs, tmin, tmax, n_eval, w_mat, tau, sigma, init, theta, draws):
    r"""
    Produces the graph for the Lorenz63 example in tutorial.

    Args:
        fun (function) : Higher order ODE function :math:`W x_t = F(x_t, t)` 
            taking arguments :math:`x` and :math:`t`.
        n_deriv (ndarray(3)) : Number of derivatives per variable in IVP.
        n_deriv_prior (ndarray(3)) : Number of derivatives per variable in prior.
        n_obs (int) : Size of the observed state.
        tmin (float) : First time point of the time interval to be evaluated; :math:`a`.
        tmax (float) : Last time point of the time interval to be evaluated; :math:`b`.
        n_eval (int) : Number of discretization points of the time interval that is evaluated, 
            such that discretization timestep is :math:`dt = (b-a)/N`.
        w_mat (ndarray(q+1)) : Corresponds to the :math:`W` matrix in the ODE equation.
        n_deriv_var (ndarray(3)) : State size of each variable.
        tau (ndarray(3)) : Decorrelation time.
        sigma (ndarray(3)) : Scale parameter.
        init (ndarray(q)) : The initial values of the ode function.
        theta (ndarray(3)) : Specific :math:`\theta` for the Lorenz system.
        draws (int) : Number of samples we need to draw from the kalman solver.

    """
    tseq = np.linspace(tmin, tmax, n_eval+1)
    exact = odeint(fun, init[:, 0], tseq, args=(theta, ))
    ylabel = ['x', 'y', 'z']
    n_var = len(ylabel)
    dt = (tmax-tmin)/n_eval
    p = sum(n_deriv_prior)
    Xn = np.zeros((draws, n_eval+1, p))
    W = zero_pad(w_mat, n_deriv, n_deriv_prior)
    ode_init, v_init = car_init(n_deriv_prior, tau, sigma, dt, init)
    kinit = indep_init(ode_init, n_deriv_prior)
    kalmanode = KalmanODE(p, n_obs, tmin, tmax, n_eval, fun, **kinit)
    for i in range(draws):
        Xn[i] = kalmanode.solve(v_init, W, theta, mv=False, sim=True)
        del kalmanode.z_states
    
    _, axs = plt.subplots(n_var, 1, figsize=(20, 7))
    for prow in range(n_var):
        for i in range(draws):
            if i == (draws - 1):
                axs[prow].plot(tseq, Xn[i, :, sum(n_deriv_prior[:prow])],
                        color="gray", alpha=1, label="Kalman")
                axs[prow].set_ylabel(ylabel[prow])
            else:
                axs[prow].plot(tseq, Xn[i, :, sum(n_deriv_prior[:prow])],
                        color="gray", alpha=1)
                
        axs[prow].plot(tseq, exact[:, prow], label='odeint')
        axs[prow].legend(loc='upper left')
    plt.show()
