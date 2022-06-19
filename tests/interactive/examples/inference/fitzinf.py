from scipy.integrate import odeint
from .inference import inference
import numpy as np
import jax

from rodeo.jax.ode_solve import *

class fitzinf(inference):
    r"Inference assuming a normal prior"
    
    def ode_fun(self, X_t, t, theta):
        "Fitz ODE written for odeint"
        a, b, c = theta
        p = len(X_t)//2
        V, R = X_t[0], X_t[p]
        return jnp.array([c*(V - V*V*V/3 + R),
                        -1/c*(V - a + b*R)])

    def rax_fun(self, t, X_t, theta):
        "Fitz ODE written for diffrax"
        a, b, c = theta
        p = len(X_t)//2
        V, R = X_t[0], X_t[p]
        return jnp.array([c*(V - V*V*V/3 + R),
                        -1/c*(V - a + b*R)])

    def loglike(self, Y_t, X_t, step_size, obs_size, theta, gamma):
        data_tseq = np.linspace(self.tmin, self.tmax, int((self.tmax-self.tmin)/obs_size)+1)
        ode_tseq = np.linspace(self.tmin, self.tmax, int((self.tmax-self.tmin)/step_size)+1)
        X_t = self.thinning(ode_tseq, data_tseq, X_t)
        return self.logprior(Y_t, X_t, gamma)

    def simulate(self, x0, theta, gamma, tseq):
        r"Get the observations assuming a normal distribution."
        X_t = odeint(self.ode_fun, x0, tseq, args=(theta,))
        e_t = np.random.default_rng(0).normal(loc=0.0, scale=1, size=X_t.shape)
        Y_t = X_t + gamma*e_t
        return Y_t, X_t
    
    # def kalman_solve(self, step_size, obs_size, x0, theta):
    #     r"Using Kalman solver to compute solutions"
    #     self.key, subkey = jax.random.split(self.key)
    #     data_tseq = np.linspace(self.tmin, self.tmax, int((self.tmax-self.tmin)/obs_size)+1)
    #     ode_tseq = np.linspace(self.tmin, self.tmax, int((self.tmax-self.tmin)/step_size)+1)
    #     X_t = mv_jit(subkey, self.fun, x0, theta, self.tmin, self.tmax, self.n_eval, self.W, **self.kinit)[0]
    #     X_t = X_t[:, :, 0]
    #     X_t = self.thinning(ode_tseq, data_tseq, X_t)
    #     return X_t
    
    # def euler_solve(self, x0, step_size, obs_size, theta):
    #     r"Using Euler method to compute solutions"
    #     data_tseq = jnp.linspace(self.tmin, self.tmax, int((self.tmax-self.tmin)/obs_size)+1)
    #     ode_tseq = jnp.linspace(self.tmin, self.tmax, int((self.tmax-self.tmin)/step_size)+1)
    #     X_t = self.euler(fitz, x0, ode_tseq, data_tseq, step_size, theta)
    #     return X_t
    
    # def diffrax_solve(self, x0, step_size, obs_size, theta):
    #     term = ODETerm(fitz_rax)
    #     solver = Dopri5()
    #     tseq = jnp.linspace(self.tmin, self.tmax, int((self.tmax-self.tmin)/obs_size)+1)
    #     saveat = SaveAt(ts=tseq)
    #     stepsize_controller = PIDController(rtol=1e-5, atol=1e-5)
    #     sol = diffeqsolve(term, solver, args = theta, t0=self.tmin, t1=self.tmax, dt0=obs_size, y0=jnp.array(x0), saveat=saveat,
    #                     stepsize_controller=stepsize_controller)
    #     return sol.ys
