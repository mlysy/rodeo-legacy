import numpy as np

from probDE.utils.utils import zero_pad, root_gen
from probDE.Kalman.kalman_initial_draw import kalman_initial_draw
from probDE.Kalman.higher_mvncond import higher_mvncond

def car_init(p, tau, sigma, dt, w, x0, scale=1):
    delta_t = np.array([dt])
    wgt_meas = zero_pad(w, p)
    roots = root_gen(tau, p)*scale
    x0_state = kalman_initial_draw(roots, sigma, x0, p)
    wgt_state, var_state = higher_mvncond(delta_t, roots, sigma)
    return wgt_meas, wgt_state, var_state, x0_state

def indep_ode_init(car_init, n_state):
    n_var = len(car_init)
    wgt_meas = np.zeros((n_var, n_state), order='F')
    x0_state = np.zeros(n_state)
    wgt_state = np.zeros((n_state, n_state), order='F')
    var_state = np.zeros((n_state, n_state), order='F')
    ind = 0
    for i in range(n_var):
        wgt_meas_i, wgt_state_i, var_state_i, x0_state_i = car_init[i]
        p_i = len(x0_state_i)
        wgt_meas[i, ind:ind+p_i] = wgt_meas_i
        x0_state[ind:ind+p_i] = x0_state_i
        wgt_state[ind:ind+p_i, ind:ind+p_i] = wgt_state_i
        var_state[ind:ind+p_i, ind:ind+p_i] = var_state_i
        ind += p_i
    return wgt_meas, wgt_state, var_state, x0_state