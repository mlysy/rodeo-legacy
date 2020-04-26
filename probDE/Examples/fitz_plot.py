from scipy.integrate import odeint
import numpy as np
import scipy as sp
import scipy.stats
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import numdifftools as nd

import warnings
warnings.filterwarnings('ignore')

class fitz_plot:
    def __init__(self, n_state1, tmin, tmax, kode=None):
        self._n_state1 = n_state1
        self.tmin = tmin
        self.tmax = tmax
        self.kode = kode
    
    def loglike(self, x, mean, var):
        return np.sum(sp.stats.norm.logpdf(x=x, loc=mean, scale=var))
        
    def fitz(self, X_t, t, theta):
        "FitzHugh-Nagumo ODE function."
        a, b, c = theta
        if len(X_t) > 2:
            V, R = X_t[0], X_t[self._n_state1]
        else:
            V, R = X_t
        return np.array([c*(V - V**3/3 + R), -1/c*(V - a + b*R)])
    
    def simulate(self, x0, theta_true, gamma):
        tseq = np.linspace(self.tmin, self.tmax, self.tmax-self.tmin+1)
        X_t = odeint(self.fitz, x0, tseq, args=(theta_true, ))
        X_t = X_t[1:,:]
        e_t = np.random.default_rng().normal(loc=0.0, scale=1, size=X_t.shape)
        Y_t = X_t + gamma*e_t
        return Y_t, X_t
    
    def euler(self, x0, h, theta):
        n_eval = int((self.tmax-self.tmin)/h)
        X_t = np.zeros((n_eval+1, len(x0)))
        X_t[0] = x0
        one_ind = int(1/h)
        n_skip = int(1/h)
        for i in range(n_eval):
            X_t[i+1] = X_t[i] + self.fitz(X_t[i], h*i, theta) * h
        return X_t[one_ind::n_skip]
    
    def ode_nlpost(self, phi, Y_t, x0, h, phi_mean, phi_sd, gamma):
        theta =  np.exp(phi)
        tseq = np.linspace(self.tmin, self.tmax, self.tmax-self.tmin+1)
        X_t = odeint(self.fitz, x0, tseq, args=(theta,))[1:]
        lp = self.loglike(Y_t, X_t, gamma)
        lp += self.loglike(phi, phi_mean, phi_sd) # logprior
        return -lp
    
    def euler_nlpost(self, phi, Y_t, x0, h, phi_mean, phi_sd, gamma):
        theta = np.exp(phi)
        X_t = self.euler(x0, h, theta)
        lp = self.loglike(Y_t, X_t, gamma)
        lp += self.loglike(phi, phi_mean, phi_sd)
        return -lp
    
    def kalman_nlpost(self, phi, Y_t, x0, h, phi_mean, phi_sd, gamma):
        theta = np.exp(phi)
        one_ind = int(1/h)
        n_skip = int(1/h)
        X_t = self.kode.solve(x0, theta)
        X_t = X_t[one_ind::n_skip, [0, self._n_state1]]
        lp = self.loglike(Y_t, X_t, gamma)
        lp += self.loglike(phi, phi_mean, phi_sd)
        return -lp
    
    def phi_fit(self, obj_fun, Y_t, x0, h, theta_true, phi_sd, gamma, var_calc=True):
        phi_mean = np.log(theta_true)
        n_theta = len(theta_true)
        opt_res = sp.optimize.minimize(obj_fun, phi_mean+.1,
                                       args=(Y_t, x0, h, phi_mean, phi_sd, gamma),
                                       method='Nelder-Mead')
        phi_hat = opt_res.x
        hes = nd.Hessian(obj_fun)
        phi_fisher = hes(phi_hat, Y_t, x0, h, phi_mean, phi_sd, gamma)
        if var_calc:
            phi_cho, low = sp.linalg.cho_factor(phi_fisher)
            phi_var = sp.linalg.cho_solve((phi_cho, low), np.eye(n_theta))
            return phi_hat, phi_var
        return phi_hat, phi_fisher
    
    def Theta_sample(self, phi_hat, phi_var, n_samples):
        Phi = np.random.multivariate_normal(phi_hat, phi_var, n_samples)
        Theta = np.exp(Phi)
        return Theta
    
    def theta_plot(self, Theta_euler, Theta_kalman, theta_true, hlst):
        n_h, _, n_theta = Theta_euler.shape
        nrow = 2
        _, axs = plt.subplots(nrow, n_theta, sharex='col', figsize=(20, 5))
        patches = [None]*(n_h+1)
        for col in range(n_theta):
            axs[0, col].set_title('$\\theta_{}$'.format(col))
            for row in range(nrow):
                axs[row, col].axvline(x=theta_true[col], linewidth=1, color='r', linestyle='dashed')
            for i in range(n_h):
                if col==0:
                    patches[i] = mpatches.Patch(color='C{}'.format(i), label='h={}'.format(hlst[i]))
                sns.kdeplot(Theta_euler[i, :, col], ax=axs[0, col])
                sns.kdeplot(Theta_kalman[i, :, col], ax=axs[1, col])

        axs[0, 0].set_ylabel('Euler')
        axs[1, 0].set_ylabel('KalmanODE')
        patches[-1] = mlines.Line2D([], [], color='r', linestyle='dashed', linewidth=1, label='True $\\theta$')
        axs[0, 2].legend(handles=patches)
        plt.show()
        return
    def theta_plot_single(self, Theta, theta_true, h):
        _, n_theta = Theta.shape
        _, axs = plt.subplots(1, n_theta, sharex='col', figsize=(20, 5))
        patches = [None]*(2)
        for col in range(n_theta):
            axs[col].set_title('$\\theta_{}$'.format(col))
            axs[col].axvline(x=theta_true[col], linewidth=1, color='r', linestyle='dashed')
            sns.kdeplot(Theta[:, col], color='C0', ax=axs[col])
        patches[0] = mpatches.Patch(color='C0', label='h={}'.format(h))
        patches[-1] = mlines.Line2D([], [], color='r', linestyle='dashed', linewidth=1, label='True $\\theta$')
        axs[2].legend(handles=patches)
        plt.show()
        