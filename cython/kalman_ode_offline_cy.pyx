cimport cython
import numpy as np
cimport numpy as np
from probDE.cython.kalmantv import KalmanTV
from probDE.cython.mat_mult import mat_mult, mat_vec_mult

DTYPE = np.double
ctypedef np.double_t DTYPE_t

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cpdef kalman_ode_offline_cy(fun,
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
                          bint filter_only=False,
                          bint offline=True):
    # Dimensions of state and measure variables
    cdef int n_dim_meas = wgt_meas.shape[0]
    cdef int n_dim_state = mu_state.shape[0]
    cdef int n_steps = n_eval + 1
    # argumgents for kalman_filter and kalman_smooth
    cdef np.ndarray[DTYPE_t, ndim=1] mu_meas = np.zeros(n_dim_meas,
                                                        dtype=DTYPE, order='F')
    cdef np.ndarray[DTYPE_t, ndim=3] var_meass = np.zeros((n_dim_meas, n_dim_meas, n_steps),
                                                          dtype=DTYPE, order='F')
    cdef np.ndarray[DTYPE_t, ndim=2] mu_state_filts = np.zeros((n_dim_state, n_steps),
                                                               dtype=DTYPE, order='F')
    cdef np.ndarray[DTYPE_t, ndim=3] var_state_filts = np.zeros((n_dim_state, n_dim_state, n_steps),
                                                                dtype=DTYPE, order='F')
    cdef np.ndarray[DTYPE_t, ndim=2] mu_state_preds = np.zeros((n_dim_state, n_steps),
                                                               dtype=DTYPE, order='F')
    cdef np.ndarray[DTYPE_t, ndim=3] var_state_preds = np.zeros((n_dim_state, n_dim_state, n_steps),
                                                                dtype=DTYPE, order='F')
    cdef np.ndarray[DTYPE_t, ndim=2] mu_state_smooths = np.zeros((n_dim_state, n_steps),
                                                                 dtype=DTYPE, order='F')
    cdef np.ndarray[DTYPE_t, ndim=3] var_state_smooths = np.zeros((n_dim_state, n_dim_state, n_steps),
                                                                  dtype=DTYPE, order='F')
    cdef np.ndarray[DTYPE_t, ndim=2] x_state_smooths = np.zeros((n_dim_state, n_steps),
                                                                dtype=DTYPE, order='F')
    cdef np.ndarray[DTYPE_t, ndim=1] x_state_tt = np.zeros(n_dim_state, 
                                                           dtype=DTYPE, order='F') #Temporary state simulation for interrogation
    cdef np.ndarray[DTYPE_t, ndim=2] var_state_meas = np.zeros((n_dim_meas, n_dim_state),
                                                           dtype=DTYPE, order='F') #Temporary matrix for multi_dot
    cdef np.ndarray[DTYPE_t, ndim=1] x_meas_ = np.zeros(n_dim_meas, dtype=DTYPE, order='F') #Temporary for matrix vector multi
    cdef int t
    
    # initialize things
    mu_state_filts[:, 0] = x0_state
    mat_vec_mult(wgt_meas, x0_state, x_meas_)
    x_meass[:, 0] = x_meas_.T
    mu_state_preds[:, 0] = mu_state_filts[:, 0]
    var_state_preds[:, :, 0] = var_state_filts[:, :, 0]
    # forward pass
    ktv = KalmanTV(n_dim_meas, n_dim_state)
    for t in range(n_eval):
        # kalman filter:
        if offline:
            ktv.filter(mu_state_pred = mu_state_preds[:, t+1],
                    var_state_pred = var_state_preds[:, :, t+1],
                    mu_state_filt = mu_state_filts[:, t+1],
                    var_state_filt = var_state_filts[:, :, t+1],
                    mu_state_past = mu_state_filts[:, t],
                    var_state_past = var_state_filts[:, :, t],
                    mu_state = mu_state,
                    wgt_state = wgt_state,
                    var_state = var_state,
                    x_meas = x_meass[:, t+1],
                    mu_meas = mu_meas,
                    wgt_meas = wgt_meas,
                    var_meas = var_meass[:, :, t+1])
        else:
            # 1. predict
            ktv.predict(mu_state_pred = mu_state_preds[:, t+1],
                        var_state_pred = var_state_preds[:, :, t+1],
                        mu_state_past = mu_state_filts[:, t],
                        var_state_past = var_state_filts[:, :, t],
                        mu_state = mu_state,
                        wgt_state = wgt_state,
                        var_state = var_state)
            
            # 2. chkrebtii interrogation
            mat_mult(wgt_meas, var_state_preds[:, :, t+1], var_state_meas)
            mat_mult(var_state_meas, wgt_meas.T, var_meass[:, :, t+1])
            ktv.state_sim(x_state_tt, 
                            mu_state_preds[:, t+1], 
                            var_state_preds[:, :, t+1], 
                            z_state_sim[:, t])
            x_meass[:, t+1] = fun(x_state_tt, tmin + (tmax-tmin)*(t+1)/n_eval)
            
            # 3. update
            ktv.update(mu_state_filt = mu_state_filts[:, t+1],
                    var_state_filt = var_state_filts[:, :, t+1],
                    mu_state_pred = mu_state_preds[:, t+1],
                    var_state_pred = var_state_preds[:, :, t+1],
                    x_meas = x_meass[:, t+1],
                    mu_meas = mu_meas,
                    wgt_meas = wgt_meas,
                    var_meas = var_meass[:, :, t+1])
    if not filter_only:
        # backward pass
        mu_state_smooths[:, n_eval] = mu_state_filts[:, n_eval]
        var_state_smooths[:, :, n_eval] = var_state_filts[:, :, n_eval]
        ktv.state_sim(x_state_smooths[:, n_eval], 
                      mu_state_smooths[:, n_eval], 
                      var_state_smooths[:, :, n_eval],
                      z_state_sim[:, n_eval])

        for t in reversed(range(n_eval)):
            if smooth_mv and smooth_sim: 
                ktv.smooth(x_state_smooth = x_state_smooths[:, t],
                           mu_state_smooth = mu_state_smooths[:, t],
                           var_state_smooth = var_state_smooths[:, :, t], 
                           x_state_next = x_state_smooths[:, t+1],
                           mu_state_next = mu_state_smooths[:, t+1],
                           var_state_next = var_state_smooths[:, :, t+1],
                           mu_state_filt = mu_state_filts[:, t],
                           var_state_filt = var_state_filts[:, :, t],
                           mu_state_pred = mu_state_preds[:, t+1],
                           var_state_pred = var_state_preds[:, :, t+1],
                           wgt_state = wgt_state,
                           z_state = z_state_sim[:, (n_eval+1)+t])
            elif smooth_mv:
                ktv.smooth_mv(mu_state_smooth = mu_state_smooths[:, t],
                              var_state_smooth = var_state_smooths[:, :, t],
                              mu_state_next = mu_state_smooths[:, t+1],
                              var_state_next = var_state_smooths[:, :, t+1],
                              mu_state_filt = mu_state_filts[:, t],
                              var_state_filt = var_state_filts[:, :, t],
                              mu_state_pred = mu_state_preds[:, t+1],
                              var_state_pred = var_state_preds[:, :, t+1],
                              wgt_state = wgt_state)
            elif smooth_sim:
                ktv.smooth_sim(x_state_smooth = x_state_smooths[:, t],
                               x_state_next = x_state_smooths[:, t+1],
                               mu_state_filt = mu_state_filts[:, t],
                               var_state_filt = var_state_filts[:, :, t],
                               mu_state_pred = mu_state_preds[:, t+1],
                               var_state_pred = var_state_preds[:, :, t+1],
                               wgt_state = wgt_state,
                               z_state = z_state_sim[:, (n_eval+1)+t])
    
        if smooth_mv and smooth_sim:
            return x_state_smooths, mu_state_smooths, var_state_smooths
        elif smooth_mv:
            return mu_state_smooths, var_state_smooths
        elif smooth_sim:
            return x_state_smooths
    return mu_state_filts, var_state_filts
