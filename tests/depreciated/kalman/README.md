# Kalman Folder Function Structure
This README provides the structure of the contents of this folder and a brief description of its usage.

    Kalman
    ├── _mou_car.py
    |   └── _mou_car                # Calculates parameters: Sigma_tilde, Q for the mOU CAR(p) process.
    |
    ├── cov_car.py
    |   └── cov_car                 # Computes the covariance function for the CAR(p) process.
    |
    ├── higher_mvncond.py
    |   └── higher_mvncond          # Computes the transition matrix and the variance matrix 
    |                                 in Y_{n+1} ~ p(Y_{n+1} | Y_n).
    |
    ├── kalman_initial_draw.py
    |   └── kalman_initial_draw     # Computes the initial draw X_L for the kalman process 
    |                                 given the initial value x_L.
    |
    ├── kalman_ode_higher.py
    |   └── kalman_ode_higher       # Approximates the solution to an univariate ordinary
    |                                 differential equations (ODEs) using Kalman filter and smoother.
    |
    ├── kalman_solver.py
    |   └── kalman_solver           # Provides a probabilistic solver for univariate ordinary 
    |                                 differential equations (ODEs) of the form w'x_t = f(x_t, t) and x_L = a.
    |
    ├── KalmanTV.py
    |   └── KalmanTV(object)        # Creates a Kalman Time-Varying object.
    |       ├── predict             # Perform one prediction step of the Kalman filter.
    |       ├── update              # Perform one update step of the Kalman filter.
    |       ├── filter              # Perform one step of the Kalman filter; combines predict and update.
    |       ├── smooth_mv           # Perform one step of the Kalman mean/variance smoother.
    |       ├── smooth_sim          # Perform one step of the Kalman sampling smoother. 
    |       └── smooth              # Perform one step of both Kalman mean/variance and sampling smoothers.
    |                                 
    └── var_car.py
        └── var_car                 # Computes the variance function for the CAR(p) process.
                                  