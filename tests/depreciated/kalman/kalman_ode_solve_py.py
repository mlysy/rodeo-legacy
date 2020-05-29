"""
.. module:: kalman_ode_higher

Probabilistic ODE solver based on the Kalman filter and smoother.

"""
import numpy as np

from KalmanTV import KalmanTV
from probDE.utils import norm_sim


def kalman_ode_solve_py(fun, x0_state, tmin, tmax, n_eval, wgt_state, mu_state, var_state, wgt_meas, z_state_sim,
                     mu_meas, var_meass, x_meass, mu_state_filts, var_state_filts, mu_state_preds, var_state_preds,
                     mu_state_smooths, var_state_smooths, x_state_smooths, x_state_tt, smooth_mv=True, smooth_sim=False, 
                     filter_only=False):

    # Dimensions of state and measure variables
    n_dim_meas = wgt_meas.shape[0]
    n_dim_state = len(mu_state)

    # initialize things
    mu_state_filts[:, 0] = x0_state
    x_meass[:, 0] = x0_state.dot(wgt_meas.T)
    mu_state_preds[:, 0] = mu_state_filts[:, 0]
    var_state_preds[:, :, 0] = var_state_filts[:, :, 0]

    # forward pass
    KFS = KalmanTV(n_dim_meas, n_dim_state)
    for t in range(n_eval):
        mu_state_preds[:, t+1], var_state_preds[:, :, t+1] = (
            KFS.predict(mu_state_past=mu_state_filts[:, t],
                        var_state_past=var_state_filts[:, :, t],
                        mu_state=mu_state,
                        wgt_state=wgt_state,
                        var_state=var_state)
        )

        var_meass[:, :, t+1] = np.linalg.multi_dot(
            [wgt_meas, var_state_preds[:, :, t+1], wgt_meas.T])
        x_state_tt = norm_sim(z=z_state_sim[:, t],
                              mu=mu_state_preds[:, t+1],
                              V=var_state_preds[:, :, t+1])
        x_meass[:, t+1] = fun(x_state_tt, tmin + (tmax-tmin)*(t+1)/n_eval)
        mu_state_filts[:, t+1], var_state_filts[:, :, t+1] = (
            KFS.update(mu_state_pred=mu_state_preds[:, t+1],
                       var_state_pred=var_state_preds[:, :, t+1],
                       x_meas=x_meass[:, t+1],
                       mu_meas=mu_meas,
                       wgt_meas=wgt_meas,
                       var_meas=var_meass[:, :, t+1])
        )

    # backward pass
    mu_state_smooths[:, n_eval] = mu_state_filts[:, n_eval]
    var_state_smooths[:, :, n_eval] = var_state_filts[:, :, n_eval]
    x_state_smooths[:, n_eval] = norm_sim(z=z_state_sim[:, n_eval],
                                          mu=mu_state_smooths[:, n_eval],
                                          V=var_state_smooths[:, :, n_eval])
    if not filter_only:
        for t in reversed(range(n_eval)):
            if smooth_mv and smooth_sim:
                mu_state_smooths[:, t], var_state_smooths[:, :, t], x_state_smooths[:, t] = (
                    KFS.smooth(x_state_next = x_state_smooths[:, t+1],
                               mu_state_next = mu_state_smooths[:, t+1],
                               var_state_next = var_state_smooths[:, :, t+1],
                               mu_state_filt = mu_state_filts[:, t],
                               var_state_filt = var_state_filts[:, :, t],
                               mu_state_pred = mu_state_preds[:, t+1],
                               var_state_pred = var_state_preds[:, :, t+1],
                               wgt_state = wgt_state,
                               z_state = z_state_sim[:, (n_eval+1)+t])
                )
            elif smooth_mv:
                mu_state_smooths[:, t], var_state_smooths[:, :, t] = (
                    KFS.smooth_mv(mu_state_next = mu_state_smooths[:, t+1],
                                  var_state_next = var_state_smooths[:, :, t+1],
                                  mu_state_filt = mu_state_filts[:, t],
                                  var_state_filt = var_state_filts[:, :, t],
                                  mu_state_pred = mu_state_preds[:, t+1],
                                  var_state_pred = var_state_preds[:, :, t+1],
                                  wgt_state = wgt_state)
                )
            elif smooth_sim:
                x_state_smooths[:, t] = (
                    KFS.smooth_sim(x_state_next = x_state_smooths[:, t+1],
                                   mu_state_filt = mu_state_filts[:, t],
                                   var_state_filt = var_state_filts[:, :, t],
                                   mu_state_pred = mu_state_preds[:, t+1],
                                   var_state_pred = var_state_preds[:, :, t+1],
                                   wgt_state = wgt_state,
                                   z_state = z_state_sim[:, (n_eval+1)+t])
                )

        if smooth_sim and smooth_mv:
            return x_state_smooths, mu_state_smooths, var_state_smooths
        elif smooth_mv:
            return mu_state_smooths, var_state_smooths
        elif smooth_sim:
            return x_state_smooths
    return mu_state_filts, var_state_filts
    