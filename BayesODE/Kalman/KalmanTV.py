"""
Time-Varying Kalman Filter and Smoother.

Model is

.. math:: 

   x_n = c_n + T_n x_n-1 + R_n^{1/2} \epsilon_n

   y_n = d_n + W_n x_n + H_n^{1/2} \eta_n

Naming conventions:

- `meas` and `state`.
- `x`, `mu`, `var`, `wgt`.
- `_past`: `n-1|n-1` (filter)
- `_pred`: `n|n-1`
- `_filt`: `n|n`
- `_next`: `n+1|N` (smoother)
- `_smooth`: `n|N`
- `mu_n|m = E[x_n | y_0:m]`
- similarly for `Sigma_n|m` and `theta_n|m = (mu_n|m, Sigma_n|m)`.
- `x_n|m` is a draw from `p(x_n | x_n+1, y_0:m)`.

So for example we have:
- `x_n = xState[n]`
- `W_n = wgtMeas[n]`
- `E[x_n | y_0:n] = muState_filt`
- `var(x_n | y_0:N) = varState_smooth`
"""

import numpy as np
import scipy as sp

class KalmanTV(object):
    """Create a Kalman Time-Varying object.
    The methods of the object can predict, update, sample and smooth the
    mean and variance of the Kalman Filter. This method is useful if one 
    wants to track an object with streaming observations.

    Parameters
    ----------
    muState_past : ndarray(nState)
        Mean estimate for state at time n-1 given observations from 
        times [0...n-1]; :math:`\mu_{n-1|n-1}`. 
    varState_past : ndarray(nState, nState)
        Covariance of estimate for state at time n-1 given observations from
        times [0...n-1]; :math:`\Sigma_{n-1|n-1}`.
    muState_pred : ndarray(nState)
        Mean estimate for state at time n given observations from 
        times [0...n-1]; :math:`\mu_{n|n-1}`. 
    varState_pred : ndarray(nState, nState)
        Covariance of estimate for state at time n given observations from
        times [0...n-1]; :math:`\Sigma_{n|n-1}`.
    muState_filt : ndarray(nState)
        Mean estimate for state at time n given observations from 
        times [0...n]; :math:`\mu_{n|n}`. 
    varState_filt : ndarray(nState, nState)
        Covariance of estimate for state at time n given observations from
        times [0...n]; :math:`\Sigma_{n|n}`.
    muState_next : ndarray(nState)
        Mean estimate for state at time n+1 given observations from 
        times [0...N]; :math:`\mu_{n+1|N}`. 
    varState_next : ndarray(nState, nState)
        Covariance of estimate for state at time n+1 given observations from
        times [0...N]; :math:`\Sigma_{n+1|N}`.
    muState : ndarray(nState)
        Transition_offsets defining the solution prior; :math:`c_n`.
    wgtState : ndarray(nState, nState)
        Transition matrix defining the solution prior; :math:`T_n`.
    varState : ndarray(nState, nState)
        Variance matrix defining the solution prior; :math:`R_n`.
    xMeas : ndarray(nMeas)
        Measure at time n+1; :math:`y_{n+1}`.
    muMeas : ndarray(nMeas)
        Transition_offsets defining the measure prior; :math:`d_n`.
    wgtMeas : ndarray(nMeas, nMeas)
        Transition matrix defining the measure prior; :math:`W_n`.
    varMeas : ndarray(nMeas, nMeas)
        Variance matrix defining the measure prior; :math:`H_n`.
    """
    def __init__(self, nMeas, nState):
        self._nMeas = nMeas
        self._nState = nState
    
    def solveV(self, V, B):
        """Computes X = V^{-1}B where V is a variance matrix."""
        L, low = sp.linalg.cho_factor(V)
        return sp.linalg.cho_solve((L, low), B)
    
    def predict(self, 
                muState_past,
                varState_past,
                muState,
                wgtState,
                varState):
        """
        Perform one prediction step of the Kalman filter.
        Calculates :math:`\\theta_{n|n-1}` from :math:`\\theta_{n-1|n-1}`.
        """
        
        muState_pred = wgtState.dot(muState_past) + muState
        varState_pred = np.linalg.multi_dot([wgtState, varState_past, wgtState.T]) + varState

        return (muState_pred, varState_pred)

    def update(self,
               muState_pred,
               varState_pred,
               xMeas,
               muMeas,
               wgtMeas,
               varMeas):
        """
        Perform one update step of the Kalman filter.
        Calculates :math:`\\theta_{n|n}` from :math:`\\theta_{n|n-1}`.
        """
        muMeas_pred = wgtMeas.dot(muState_pred) + muMeas 
        varMeasState_pred = wgtMeas.dot(varState_pred)
        varMeasMeas_pred = np.linalg.multi_dot([wgtMeas, varState_pred, wgtMeas.T]) + varMeas
        varStateMeas_pred = varState_pred.dot(wgtMeas.T)
        
        varState_temp = self.solveV(varMeasMeas_pred, varStateMeas_pred.T).T
        muState_filt = muState_pred + varState_temp.dot(xMeas - muMeas_pred)
        varState_filt = varState_pred - varState_temp.dot(varMeasState_pred)

        return (muState_filt, varState_filt)
        
    def filter(self,
               muState_past,
               varState_past,
               muState,
               wgtState,
               varState,
               xMeas,
               muMeas,
               wgtMeas,
               varMeas):
        """
        Perform one step of the Kalman filter.
        Combines :func:`KalmanTV.predict` and :func:`KalmanTV.update` steps to get :math:`\\theta_{n|n}` from :math:`\\theta_{n-1|n-1}`.
        """
        muState_pred, varState_pred = self.predict(muState_past = muState_past, 
                                                   varState_past = varState_past,
                                                   muState = muState,
                                                   wgtState = wgtState,
                                                   varState = varState)

        muState_filt, varState_filt = self.update(muState_pred = muState_pred,
                                                  varState_pred = varState_pred,
                                                  xMeas = xMeas,
                                                  muMeas = muMeas,
                                                  wgtMeas = wgtMeas,
                                                  varMeas = varMeas)
        
        return (muState_pred, varState_pred, muState_filt, varState_filt)
    
    def smooth_mv(self,
                  muState_next,
                  varState_next,
                  muState_filt,
                  varState_filt,
                  muState_pred,
                  varState_pred,
                  wgtState):
        """
        Perform one step of the Kalman mean/variance smoother.
        Calculates :math:`\\theta_{n|N}` from :math:`\\theta_{n+1|N}`, :math:`\\theta_{n|n}`, and :math:`\\theta_{n+1|n}`.
        """
        varState_temp = varState_filt.dot(wgtState.T)
        varState_temp_tilde = self.solveV(varState_pred, varState_temp.T).T
        muState_smooth = muState_filt + varState_temp_tilde.dot(muState_next - muState_pred)
        varState_smooth = varState_filt + np.linalg.multi_dot([varState_temp_tilde, (varState_next - varState_pred), varState_temp_tilde.T])

        return (muState_smooth, varState_smooth)

    def smooth_sim(self,
                   xState_next,
                   muState_filt,
                   varState_filt,
                   muState_pred,
                   varState_pred,
                   wgtState):
        """
        Perform one step of the Kalman sampling smoother.
        Calculates a draw :math:`x_{n|N}` from :math:`x_{n+1|N}`, :math:`\\theta_{n|n}`, and :math:`\\theta_{n+1|n}`. 
        """
        varState_temp = varState_filt.dot(wgtState.T)
        varState_temp_tilde = self.solveV(varState_pred, varState_temp.T).T
        muState_sim = muState_filt + varState_temp_tilde.dot(xState_next - muState_pred)
        varState_sim = varState_filt - varState_temp_tilde.dot(varState_temp.T)
        xState_smooth = np.random.multivariate_normal(muState_sim, varState_sim)

        return xState_smooth
    
    def smooth(self,
               xState_next,
               muState_next,
               varState_next,
               muState_filt,
               varState_filt,
               muState_pred,
               varState_pred,
               wgtState):
        """
        Perfrom one step of both Kalman mean/variance and sampling smoothers.
        Combines :func:`KalmanTV.smooth_mv` and :func:`KalmanTV.smooth_sim` steps to get :math:`x_{n|N}` and 
        :math:`\\theta_{n|N}` from :math:`\\theta_{n+1|N}`, :math:`\\theta_{n|n}`, and :math:`\\theta_{n+1|n}`.
        """
        muState_smooth, varState_smooth = self.smooth_mv(muState_next = muState_next,
                                                         varState_next = varState_next,
                                                         muState_filt = muState_filt,
                                                         varState_filt = varState_filt,
                                                         muState_pred = muState_pred,
                                                         varState_pred = varState_pred,
                                                         wgtState = wgtState)
        
        xState_smooth = self.smooth_sim(xState_next = xState_next,
                                        muState_filt = muState_filt,
                                        varState_filt = varState_filt,
                                        muState_pred = muState_pred,
                                        varState_pred = varState_pred,
                                        wgtState = wgtState)
        
        return (muState_smooth, varState_smooth, xState_smooth)
    