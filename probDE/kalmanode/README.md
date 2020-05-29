# kalmanode Folder Function Structure
This README provides the structure of the contents of this folder and a brief description of its usage.

    kalmanode
    ├── KalmanODE.pyx
    |   ├── kalman_ode              # Probabilistic ODE solver based on the Kalman filter and smoother.
    |   └── KalmanODE               # A class containing the probabilistic ODE solver using Kalman filter 
    |                                 and smoother.
    |
    ├── KalmanTV.h
    |   └── KalmanTV                # A class containing the Kalman filtering and smoothering algorithms
    |                                 written in C++ using Eigen.
    |
    ├── KalmanTVODE.h
    |   └── KalmanTVODE             # A class inheriting from KalmanTV which implements the Kalman algorithms
    |                                 for the use case of ODES written in C++ using Eigen.
    |                               
    └── KalmanTVODE.pxd
        └── KalmanTVODE             # Cython wrapper of the C++ class for Python interface.
                                  