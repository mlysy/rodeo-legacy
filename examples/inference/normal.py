from scipy.integrate import odeint
from .inference import inference
import numpy as np

from rodeo.jax.KalmanODE import *

class normal(inference):
    r"Inference assuming a normal prior"
    
    def simulate(self, fun, x0, theta, gamma, tseq):
        r"Get the observations assuming a normal distribution."
        X_t = odeint(fun, x0, tseq, args=(theta,))
        e_t = np.random.default_rng(0).normal(loc=0.0, scale=1, size=X_t.shape)
        Y_t = X_t + gamma*e_t
        return Y_t, X_t
    
    def kalman_solve(self, step_size, obs_size, x0, theta):
        r"Using Kalman solver to compute solutions"
        data_tseq = np.linspace(self.tmin, self.tmax, int((self.tmax-self.tmin)/obs_size)+1)
        ode_tseq = np.linspace(self.tmin, self.tmax, int((self.tmax-self.tmin)/step_size)+1)
        X_t = solve_mv(self.fun, x0, self.tmin, self.tmax, self.n_eval, self.W, **self.kinit, theta=theta)[0]
        X_t = self.thinning(ode_tseq, data_tseq, X_t)[:, self.state_ind]
        return X_t
    
    def euler_solve(self, x0, step_size, obs_size, theta):
        r"Using Euler method to compute solutions"
        data_tseq = np.linspace(self.tmin, self.tmax, int((self.tmax-self.tmin)/obs_size)+1)
        ode_tseq = np.linspace(self.tmin, self.tmax, int((self.tmax-self.tmin)/step_size)+1)
        X_t = self.euler(x0, ode_tseq, data_tseq, step_size, theta)
        return X_t
    