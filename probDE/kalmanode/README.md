# kalmanode Folder Function Structure
This README provides the structure of the contents of this folder and a brief description of its usage.

    kalmanode
    └── KalmanODE.pyx
        ├── forecast                # Forecast the observed state from the current state.
        ├── forecast_sch            # Forecast the observed state from the current state via the Schobert method.
        ├── kalman_ode              # Probabilistic ODE solver based on the Kalman filter and smoother.
        └── KalmanODE               # A class containing the probabilistic ODE solver using Kalman filter 
                                      and smoother.