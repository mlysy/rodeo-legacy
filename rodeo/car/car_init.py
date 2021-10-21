import numpy as np

from rodeo.car.car_cov import car_cov
from rodeo.car.car_mou import car_mou
from rodeo.car.car_var import car_var
from rodeo.utils.utils import mvncond

def root_gen(tau, p):
    """
    Creates p CAR model roots.

    Args:
        tau (int): First root parameter.
        p (int): Number of roots to generate.

    Returns:
        (ndarray(p)): Vector size of p roots generated.

    """
    return np.append(1/tau, np.linspace(1 + 1/(10*(p-1)), 1.1, p-1))*10

def car_initial_draw(roots, sigma, x0, p):
    """
    Computes the initial draw X0 for the kalman process :math:`X0 \sim N(c_0, 0_{pxp})` 
    where :math:`c_0 \sim p(N(\lambda, V_{\infty}) | x_0 = x0)`.

    Args:
        roots (ndarray(n_dim_roots)): Roots to the p-th order polynomial of the car(p) 
            process (roots must be negative).
        sigma (float): Parameter in mOU volatility matrix.
        x0 (ndarray(n_dim_ode)): Initial conditions of the ode.
        p (float): Size of x0_state.

    Returns:
        (ndarray(n_dim_roots)): Simulate :math:`X0 \sim N(c_0, 0_{pxp})` conditioned on x0.

    """
    q = len(x0) - 1
    if p == q+1:
        return x0
        
    x0_state = np.zeros(p)    #Initialize p sized initial X0
    V_inf = car_cov([], roots, sigma, v_infinity=True)    #Get V_inf to draw X^{{q+1} ... (p-1)}_0
    icond = np.array([True]*(q+1) + [False]*(p-q-1))   #Conditioned on all but X^{{q+1} ... (p-1)}_0
    A, b, V = mvncond(np.zeros(p), V_inf, icond)    #Get mean and variance for p(z_0 | y_0 = c) where z_0 = X_0 \setminus y_0
    z_0 = np.random.multivariate_normal(A.dot(x0) + b, V)    #Draw X^{(p-1)}_0
    x0_state[:q+1] = x0    #Replace x^{(0), (1), (2) ... (q)}}_0 with y_0
    x0_state[q+1:] = z_0
    return x0_state

def car_state(delta_t, roots, sigma):    
    """
    Calculate the state transition matrix and variance matrix using the CAR(p) process in Kalman solver.
        
    Args:
        delta_t (ndarray(1)): A vector containing the step size between simulation points.
        roots (ndarray(n_dim_roots)): Roots to the p-th order polynomial of the CAR(p) 
            process (roots must be negative).
        sigma (float): Parameter in mOU volatility matrix

    Returns:
        (tuple):
        - **wgtState** (ndarray(n_dim_roots, n_dim_roots)): The state transition matrix defined in 
          Kalman solver.
        - **varState** (ndarray(n_dim_roots, n_dim_roots)): The state variance matrix defined in
          Kalman solver.

    """
    _, Q = car_mou(roots, sigma)
    Q_inv = np.linalg.pinv(Q)
    varState = car_var(delta_t, roots, sigma)[:, :, 0]
    wgtState = np.matmul(Q * np.exp(-roots*delta_t[0]), Q_inv, order='F')
    return wgtState, varState
    
def car_init(dt, n_deriv_prior, tau, sigma, x0=None):
    """
    Calculates the initial parameters necessary for the Kalman solver.
    Args:
        dt (float): The step size between simulation points.
        n_deriv_prior (list(int)): Dimension of the prior.
        tau (list(float)): First root parameter.
        sigma (list(float)): Parameter in mOU volatility matrix.
        x0 (ndarray(n_var, q+1)): The initial value, :math:`x0`, to the ODE problem.
        
    Returns:
        (tuple):
        - **wgt_state** (ndarray(p, p)) Transition matrix defining the solution prior; :math:`T`.
        - **mu_state** (ndarray(p)): Transition_offsets defining the solution prior; denoted by :math:`\lambda`.
        - **var_state** (ndarray(p, p)) Variance matrix defining the solution prior; :math:`R`.
        - **x0_state** (ndarray(p)): Initial value of the state variable :math:`x_t` at time 
          :math:`t = 0`; :math:`x_0`.
    
    """
    delta_t = np.array([dt])
    n_var = len(n_deriv_prior)
    if x0 is not None:
        x0_state = np.zeros(sum(n_deriv_prior))

    mu_state = np.zeros(sum(n_deriv_prior))
    wgt_state = [None]*n_var
    var_state = [None]*n_var
    for i in range(n_var):
        roots = root_gen(tau[i], n_deriv_prior[i])
        if x0 is not None:
            x0_state[sum(n_deriv_prior[:i]):sum(n_deriv_prior[:i+1])] = \
                car_initial_draw(roots, sigma[i], x0[i], n_deriv_prior[i])
        wgt_state[i], var_state[i] = car_state(delta_t, roots, sigma[i])
  
    if n_var == 1:
        wgt_state = wgt_state[0]
        var_state = var_state[0]
    
    init = {"wgt_state":wgt_state,  "mu_state":mu_state,
            "var_state":var_state}
    
    if x0 is not None:
        return init, x0_state
    else:
        return init
