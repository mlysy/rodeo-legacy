# can we define a KalmanFilter with multiple observation_covariance?
from pykalman import KalmanFilter
from scipy import stats
import scipy as sc
# import pykalman
import inspect

# inspect.getmembers(pykalman.standard._smooth)
def rand_mat(n, p = None):
    if p is not None:
        X = sc.stats.multivariate_normal.rvs(size = n, cov = sc.eye(p))
    else:
        X = sc.stats.multivariate_normal.rvs(size = 2*n, cov = sc.eye(n))
        X = X.T.dot(X)
    return X

def rand_mats(reps, n, p = None):
    return [rand_mat(n, p) for ii in range(reps)]

def repeat(x, n):
    return [x for ii in range(n)]

n_dim_obs = 2
n_dim_state = 3
n_obs = 3
transition_matrices = rand_mats(n_obs, n_dim_state)
transition_offsets = rand_mats(n_obs, 1, n_dim_state)
transition_covariance = rand_mat(n_dim_state)
observation_matrices = rand_mats(n_obs, n_dim_obs, n_dim_state)
observation_offsets = rand_mats(n_obs, 1, n_dim_obs)
observation_covariance = rand_mats(n_obs, n_dim_obs)
initial_state_mean = rand_mat(1, n_dim_state)
initial_state_covariance = rand_mat(n_dim_state)
measurements = rand_mats(n_obs, 1, n_dim_obs)

                
kf = KalmanFilter(n_dim_state = n_dim_state,
                  n_dim_obs = n_dim_obs,
                  transition_matrices = transition_matrices,
                  transition_offsets = transition_offsets,
                  transition_covariance = transition_covariance,
                  observation_matrices = observation_matrices,
                  observation_offsets = observation_offsets,
                  observation_covariance = observation_covariance,
                  initial_state_mean = initial_state_mean,
                  initial_state_covariance = initial_state_covariance)
kf0 = KalmanFilter(n_dim_state = n_dim_state,
                  n_dim_obs = n_dim_obs,
                  transition_matrices = transition_matrices,
                  transition_offsets = transition_offsets,
                  transition_covariance = transition_covariance,
                  observation_matrices = observation_matrices,
                  observation_offsets = observation_offsets,
                  observation_covariance = observation_covariance[1],
                  initial_state_mean = initial_state_mean,
                  initial_state_covariance = initial_state_covariance)

kf.filter(measurements)
kf0.filter(measurements)

kf.smooth(measurements)

