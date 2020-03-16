cimport cython
import numpy as np
cimport numpy as np
from KalmanTVODE cimport KalmanTVODE
DTYPE = np.double
ctypedef np.double_t DTYPE_t

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cpdef kalman_ode(fun,
                 double[::1] x0_state,
                 double tmin,
                 double tmax,
                 int n_eval,
                 double[::1, :] wgt_state,
                 double[::1] mu_state, 
                 double[::1, :] var_state,
                 double[::1, :] wgt_meas, 
                 double[::1, :] z_state_sim,
                 double[::1, :] x_meass,
                 bint smooth_mv=True,
                 bint smooth_sim=False,
                 bint offline=True):
    # Dimensions of state and measure variables
    cdef int n_dim_meas = wgt_meas.shape[0]
    cdef int n_dim_state = mu_state.shape[0]
    cdef int n_steps = n_eval + 1
    # argumgents for kalman_filter and kalman_smooth
    cdef np.ndarray[DTYPE_t, ndim=2] mu_state_smooths = np.zeros((n_dim_state, n_steps),
                                                                 dtype=DTYPE, order='F')
    cdef np.ndarray[DTYPE_t, ndim=3] var_state_smooths = np.zeros((n_dim_state, n_dim_state, n_steps),
                                                                  dtype=DTYPE, order='F')
    cdef np.ndarray[DTYPE_t, ndim=2] x_state_smooths = np.zeros((n_dim_state, n_steps),
                                                                dtype=DTYPE, order='F')
    
    cdef np.ndarray[DTYPE_t, ndim=1] x_state = np.zeros(n_dim_state, dtype=DTYPE, order='F')
    cdef np.ndarray[DTYPE_t, ndim=1] x_meas = np.zeros(n_dim_meas, dtype=DTYPE, order='F') 
    cdef int t
    # forward pass
    ktvode = new KalmanTVODE(n_dim_meas, n_dim_state, n_steps, & x0_state[0])
    for t in range(n_eval):
        # kalman filter:
        if offline:
            ktvode.filter(t, & mu_state[0], & wgt_state[0, 0],
                          & var_state[0, 0], & x_meas[0], & wgt_meas[0, 0])
        else:
            ktvode.predict(t, & mu_state[0], & wgt_state[0, 0], & var_state[0, 0])
            ktvode.chkrebtii_int(& x_state[0], t, & wgt_meas[0, 0],
                                 & z_state_sim[0, 0])
            x_meas = fun(x_state, tmin + (tmax-tmin)*(t+1)/n_eval)
            ktvode.update(t, & x_meas[0], & wgt_meas[0, 0])
            
    # backward pass
    ktvode.smooth_update(& x_state_smooths[0, 0], & mu_state_smooths[0, 0],
                         & var_state_smooths[0, 0, 0], & wgt_state[0, 0],
                         & z_state_sim[0, 0], smooth_mv, smooth_sim)

    if smooth_mv and smooth_sim:
        return x_state_smooths, mu_state_smooths, var_state_smooths
    elif smooth_mv:
        return mu_state_smooths, var_state_smooths
    elif smooth_sim:
        return x_state_smooths
