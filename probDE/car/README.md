# car Folder Function Structure
This README provides the structure of the contents of this folder and a brief description of its usage.

    car
    ├── car_cov.py
    |   └── car_cov                 # Computes the covariance function for the CAR(p) process.
    |
    ├── car_init.py
    |   ├── root_gen                # Creates p CAR model roots.
    |   ├── zero_pad                # Pad x0 with p-len(x0) 0s at the end of x0.
    |   ├── car_initial_draw        # Computes the initial draw for the kalman process.
    |   ├── car_state               # Calculate the state transition matrix and variance matrix used in the 
    |   |                             model in Kalman solver.
    |   └── car_init                # Initialize the Kalman parameters via the CAR(p) process.
    |
    ├── car_mou.py
    |   └── car_mou                 # Calculates parameters for the mOU CAR(p) process.
    |                               
    └── car_var.py
        └── car_var                 # Computes the variance function for the CAR(p) process.
                                  