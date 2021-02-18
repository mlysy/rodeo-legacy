"""
.. module:: lorenz_graph

Python file for the sole purpose of producing the graphs in the Lorenz63 example.

"""
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

from rodeo.car import car_init
from rodeo.ibm import ibm_init
from rodeo.cython.KalmanODE import KalmanODE
from rodeo.utils import indep_init, zero_pad

def lorenz_graph(fun, n_deriv, n_deriv_prior, tmin, tmax, n_eval, w_mat, tau, sigma, init, theta, draws, method="rodeo", load_calcs=False):
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
        method (string) : Interrogation method.

    """
    tseq = np.linspace(tmin, tmax, n_eval+1)
    exact = odeint(fun, init[:, 0], tseq, args=(theta, ))
    init = np.ravel([init[:, 0], init[:, 1]], order='F')
    ylabel = ['x', 'y', 'z']
    n_var = len(ylabel)
    dt = (tmax-tmin)/n_eval
    p = sum(n_deriv_prior)

    if load_calcs:
        Xn = np.load(('saves/lorenz{}.npy').format(method))
    else:
        Xn = np.zeros((draws, n_eval+1, p))
        W = zero_pad(w_mat, n_deriv, n_deriv_prior)
        v_init = zero_pad(init, n_deriv, n_deriv_prior)
        ode_init = ibm_init(dt, n_deriv_prior, sigma)
        kinit = indep_init(ode_init, n_deriv_prior)
        kalmanode = KalmanODE(W, tmin, tmax, n_eval, fun, **kinit)
        for i in range(draws):
            Xn[i] = kalmanode.solve_sim(v_init, theta=theta, method=method)
            del kalmanode.z_state
        if method=="rodeo":
            Xmean = kalmanode.solve_mv(v_init, theta=theta, method=method)[0]
            Xmean = np.array([Xmean])
            Xn = np.concatenate([Xn, Xmean])
        np.save(('saves/lorenz{}.npy').format(method), Xn)
    
    fig, axs = plt.subplots(n_var, 1, figsize=(20, 10))
    for prow in range(n_var):
        for i in range(draws):
            if i == (draws - 1):
                axs[prow].plot(tseq, Xn[i, :, sum(n_deriv_prior[:prow])],
                        color="gray", alpha=1, label="rodeo draws")
                axs[prow].set_ylabel(ylabel[prow])
            else:
                axs[prow].plot(tseq, Xn[i, :, sum(n_deriv_prior[:prow])],
                        color="gray", alpha=1)
                
        axs[prow].plot(tseq, exact[:, prow], label='odeint', color="orange")
        if method=="rodeo":
            axs[prow].plot(tseq, Xn[-1, :, sum(n_deriv_prior[:prow])],
                           label="rodeo mean", color="green")
    axs[0].legend(loc='upper left', bbox_to_anchor=[1, 1])
    fig.tight_layout()
    fig.set_rasterized(True)
    plt.show()
    return fig
