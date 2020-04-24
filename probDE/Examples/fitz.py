"""
MCMC algorithm for statistical inference and graphing in the FitzHuge-Nagumo example.
"""
import numpy as np
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import seaborn as sns

from probDE.Kalman.ode_init import car_init, indep_ode_init
from probDE.cython.KalmanTest.KalmanODE import KalmanODE

def euler(X0, n_eval, h, fun, theta):
    X_t = np.zeros((n_eval+1, len(X0)))
    X_t[0] = X0
    one_ind = int(1/h)
    for i in range(n_eval):
        X_t[i+1] = X_t[i] + fun(X_t[i], h*i, theta) * h
    return X_t[one_ind::one_ind]


def theta_plot(Theta_euler, Theta_kalman, Theta_ode, theta_true, h_eval):
    n_h, _, n_theta = Theta_euler.shape
    nrow = 2
    burn = 50
    _, axs = plt.subplots(nrow, n_theta, sharex='col', figsize=(20, 5))
    patches = [None]*(n_h+2)
    for col in range(n_theta):
        axs[0, col].set_title('$\\theta_{}$'.format(col))
        for row in range(nrow):
            axs[row, col].axvline(x=theta_true[col], linewidth=1, color='r', linestyle='dashed')
            sns.kdeplot(Theta_ode[burn:, col], ax=axs[row, col], color='black')
        for i in range(n_h):
            if col==0:
                patches[i] = mpatches.Patch(color='C{}'.format(i), label='N={}'.format(h_eval[i]))
            sns.kdeplot(Theta_euler[i, burn:, col], ax=axs[0, col])
            sns.kdeplot(Theta_kalman[i, burn:, col], ax=axs[1, col])
        
    axs[0, 0].set_ylabel('Euler')
    axs[1, 0].set_ylabel('KalmanODE')
    patches[-2] = mpatches.Patch(color='black', label='True Posterior')
    patches[-1] = mlines.Line2D([], [], color='r', linestyle='dashed', linewidth=1, label='True $\\theta$')
    axs[0,2].legend(handles=patches)
    plt.show()

class Fitz:
    # private members
    def __init__(self, n_state1, n_state2, n_state, n_meas):
        self._n_state1 = n_state1
        self._n_state2 = n_state2
        self._n_state = n_state
        self._n_meas = n_meas
    
    def _fitz(self, X_t, t, theta):
        "FitzHugh-Nagumo ODE function."
        a, b, c = theta
        V, R = X_t[0], X_t[self._n_state1]
        return np.array([c*(V - V**3/3 + R), -1/c*(V - a + b*R)])
    
    def _normal_lpdf(self, y, mean, std):
        loglik = -0.5*(np.log(2*np.pi) + np.log(std*std) + np.sum((y - mean)**2)/(std*std))
        return loglik
    
    def _lognormal_lpdf(self, y, mean, std):
        loglik = -np.log(y) + self._normal_lpdf(np.log(y), mean, std)
        return loglik
    
    def _logprior(self, Y_t, X_t, gamma, theta, theta_true, theta_sd):
        _, n_comp = Y_t.shape
        n_theta = len(theta)
        lpi = 0.
        for j in range(n_comp):
            lpi += self._normal_lpdf(Y_t[:, j], X_t[:, self._n_state1*j], gamma)
        for i in range(n_theta):
            lpi += self._lognormal_lpdf(theta[i], theta_true[i], theta_sd[i])
        return lpi
    
    def simulate(self, tmin, tmax, n_eval, x0, theta, tau, sigma, w_mat, gamma, scale=1):
        dt = (tmax - tmin)/n_eval
        kinit = indep_ode_init([car_init(self._n_state1, tau[0], sigma[0], dt, w_mat[0], x0[0], scale),
                                car_init(self._n_state2, tau[1], sigma[1], dt, w_mat[1], x0[1], scale)],
                                self._n_state)
        kode = KalmanODE.initialize(kinit, self._n_state, self._n_meas, tmin, tmax, n_eval, self._fitz)
        x0_state = kinit[-1]
        X_t = kode.solve(x0_state, theta=theta) # ODE no noise
        Y_t = X_t + gamma*np.random.randn() # include noise
        del kode
        return Y_t, X_t

    def mwg(self, n_samples, Y_t, start_ind, tmin, tmax, n_eval, w_mat, x0, theta0, 
            theta_true, theta_sd, gamma, tau, sigma, rwsd, scale, accept = False):
        
        # Get problem dimensions and initialization
        n_theta = len(theta0)
        theta_curr = theta0.copy()
        theta_prop = theta0.copy() 
        paccept = np.zeros(n_theta, dtype=int)
        Theta = np.zeros((n_samples, n_theta))
        n_obs = len(Y_t)
        n_skip = n_eval//n_obs
        # MCMC process
        dt = (tmax - tmin)/n_eval
        kinit = indep_ode_init([car_init(self._n_state1, tau[0], sigma[0], dt, w_mat[0], x0[0], scale),
                                car_init(self._n_state2, tau[1], sigma[1], dt, w_mat[1], x0[1], scale)],
                                self._n_state)
        x0_state = kinit[-1]
        kode = KalmanODE.initialize(kinit, self._n_state, self._n_meas, tmin, tmax, n_eval, self._fitz)
        X_curr = kode.solve(x0_state, theta=theta_curr)[start_ind::n_skip]
        lp_curr = self._logprior(Y_t, X_curr, gamma, theta_curr, theta_true, theta_sd)
        for i in range(n_samples):
            for j in range(n_theta):
                theta_prop[j] += rwsd[j]*np.random.randn()
                if theta_prop[j]>0:
                    v0 = self._fitz(x0_state, 0, theta_prop)
                    x0_state[[1, self._n_state1+1]] = v0
                    del kode.z_states
                    X_prop = kode.solve(x0_state, theta=theta_prop)[start_ind::n_skip]
                    lp_prop = self._logprior(Y_t, X_prop, gamma, theta_prop, theta_true, theta_sd)
                    lacc = lp_prop - lp_curr
                    if lacc > 0 or np.random.uniform() < np.exp(lacc):
                        theta_curr[j] = theta_prop[j]
                        lp_curr = lp_prop
                        paccept[j] = paccept[j] + 1
                    else:
                        theta_prop[j] = theta_curr[j]
                else:
                    theta_prop[j] = theta_curr[j]
            # storage
            Theta[i] = theta_curr
        # output
        if not accept:
            return Theta
        else:
            paccept = paccept/n_samples
            return Theta, paccept
        
    def mwg_det(self, n_samples, Y_t, h, n_eval, w_mat, x0, theta0, theta_true, 
                theta_sd, gamma, rwsd, accept = False):
        
        # Get problem dimensions and initialization
        n_theta = len(theta0)
        theta_curr = theta0.copy()
        theta_prop = theta0.copy() 
        paccept = np.zeros(n_theta, dtype=int)
        Theta = np.zeros((n_samples, n_theta))
        
        # MCMC process
        old_state1 = self._n_state1
        old_state2 = self._n_state2
        self._n_state1 = 1 # State size for deterministic is 1
        self._n_state2 = 1
        X_curr = euler(x0, n_eval, h, self._fitz, theta_curr)
        lp_curr = self._logprior(Y_t, X_curr, gamma, theta_curr, theta_true, theta_sd)
        for i in range(n_samples):
            for j in range(n_theta):
                theta_prop[j] += rwsd[j]*np.random.randn()
                if theta_prop[j]>0:
                    X_prop = euler(x0, n_eval, h, self._fitz, theta_prop)
                    lp_prop = self._logprior(Y_t, X_prop, gamma, theta_prop, theta_true, theta_sd)
                    lacc = lp_prop - lp_curr
                    if lacc > 0 or np.random.uniform() < np.exp(lacc):
                        theta_curr[j] = theta_prop[j]
                        lp_curr = lp_prop
                        paccept[j] = paccept[j] + 1
                    else:
                        theta_prop[j] = theta_curr[j]
                else:
                    theta_prop[j] = theta_curr[j]
            # storage
            Theta[i] = theta_curr
        self._n_state1 = old_state1
        self._n_state2 = old_state2
        # output
        if not accept:
            return Theta
        else:
            paccept = paccept/n_samples
            return Theta, paccept

    def mwg_ode(self, n_samples, Y_t, start_ind, tmin, tmax, n_eval, w_mat, x0, theta0, 
                theta_true, theta_sd, gamma, rwsd, accept = False):

        # Get problem dimensions and initialization
        n_theta = len(theta0)
        theta_curr = theta0.copy()
        theta_prop = theta0.copy() 
        paccept = np.zeros(n_theta, dtype=int)
        Theta = np.zeros((n_samples, n_theta))
        n_obs = len(Y_t)
        n_skip = n_eval//n_obs

        # MCMC process
        tseq = np.linspace(tmin, tmax, n_eval+1)
        X_curr = odeint(fitz0, x0, tseq, args=(theta_curr,))[start_ind::n_skip]
        self._n_state1 = 1
        self._n_state2 = 1
        lp_curr = self._logprior(Y_t, X_curr, gamma, theta_curr, theta_true, theta_sd)
        for i in range(n_samples):
            for j in range(n_theta):
                theta_prop[j] += rwsd[j]*np.random.randn()
                if theta_prop[j]>0:
                    X_prop = odeint(fitz0, x0, tseq, args=(theta_prop,))[start_ind::n_skip]
                    lp_prop = self._logprior(Y_t, X_prop, gamma, theta_prop, theta_true, theta_sd)
                    lacc = lp_prop - lp_curr
                    if lacc > 0 or np.random.uniform() < np.exp(lacc):
                        theta_curr[j] = theta_prop[j]
                        lp_curr = lp_prop
                        paccept[j] = paccept[j] + 1
                    else:
                        theta_prop[j] = theta_curr[j]
                else:
                    theta_prop[j] = theta_curr[j]
            # storage
            Theta[i] = theta_curr
        self._n_state1 = old_state1
        self._n_state2 = old_state2
        # output
        if not accept:
            return Theta
        else:
            paccept = paccept/n_samples
            return Theta, paccept
