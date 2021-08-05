from scipy.integrate import odeint
from .inference import inference
import numpy as np

class normal(inference):
    r"Inference assuming a normal prior"
    
    def simulate(self, fun, x0, theta, gamma, tseq):
        r"Get the observations assuming a normal distribution."
        X_t = odeint(fun, x0, tseq, args=(theta,))[1:,]
        e_t = np.random.default_rng().normal(loc=0.0, scale=1, size=X_t.shape)
        Y_t = X_t + gamma*e_t
        return Y_t, X_t
    
    def kalman_solve(self, data_tseq, ode_tseq, x0, theta):
        r"Using Kalman solver to compute solutions"
        X_t = self.kode.solve_mv(x0, self.W, theta)[0]
        X_t = self.thinning(data_tseq, ode_tseq, X_t)[:, self.state_ind]
        return X_t
    
    def euler_solve(self, x0, step_size, theta):
        r"Using Euler method to compute solutions"
        X_t = self.euler(x0, step_size, theta)
        return X_t
    