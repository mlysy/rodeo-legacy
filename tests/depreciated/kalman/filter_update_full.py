"""
.. module:: filter_update_full

Kalman filter from pykalman package.

"""

from pykalman import standard as pks
from math import sqrt

def filter_update_full(filtered_state_mean, filtered_state_covariance,
                       observation, transition_matrix,
                       transition_offset, transition_covariance,
                       observation_matrix, observation_offset,
                       observation_covariance):
    """Update a Kalman Filter state estimate.
    Perform a one-step update to estimate the state at time :math:`t+1`
    give an observation at time :math:`t+1` and the previous estimate for
    time :math:`t` given observations from times :math:`[0...t]`.  This
    method is useful if one wants to track an object with streaming
    observations.

    Exactly the same as `pykalman.standard.KalmanFilter.filter_update`,
    except outputs everything needed to use `pykalman.standard._smooth`.

    Parameters
    ----------
    filtered_state_mean : [n_dim_state] array
        mean estimate for state at time t given observations from times
        [0...t].
    filtered_state_covariance : [n_dim_state, n_dim_state] array
        covariance of estimate for state at time t given observations from
        times [0...t]
    observation : [n_dim_obs] array
        observation from time t+1.
    transition_matrix : [n_dim_state, n_dim_state] array
        state transition matrix from time t to t+1.
    transition_offset : [n_dim_state] array
        state offset for transition from time t to t+1.
    transition_covariance : [n_dim_state, n_dim_state] array
        state transition covariance from time t to t+1.
    observation_matrix : [n_dim_obs, n_dim_state] array
        observation matrix at time t+1.
    observation_offset : [n_dim_obs] array
        observation offset at time t+1.
    observation_covariance : [n_dim_obs, n_dim_obs] array
        observation covariance at time t+1.

    Returns
    -------
    predicted_state_mean : [n_dim_state] array
        mean of hidden state at time t+1 given observations from 
        times [0...t+1]
    predicted_state_covariance : [n_dim_state, n_dim_state] array
        covariance of hidden state at time t+1 given observations from 
        times [0...t+1]
    kalman_gain : [n_dim_state, n_dim_obs] array
        Kalman gain matrix for time t+1
    filtered_state_mean : [n_dim_state] array
        mean of hidden state at time t+1 given observations from 
        times [0...t+1]
    filtered_state_covariance : [n_dim_state, n_dim_state] array
        covariance of hidden state at time t+1 given observations from 
        times [0...t+1]
    """
    predicted_state_mean, predicted_state_covariance = (
        pks._filter_predict(
            transition_matrix, transition_covariance,
            transition_offset, filtered_state_mean,
            filtered_state_covariance
        )
    )
    (kalman_gain, filtered_state_mean,
     filtered_state_covariance) = (
         pks._filter_correct(
             observation_matrix, observation_covariance,
             observation_offset, predicted_state_mean,
             predicted_state_covariance, observation
         )
     )
    return (predicted_state_mean, predicted_state_covariance,
            kalman_gain, filtered_state_mean,
            filtered_state_covariance)
            