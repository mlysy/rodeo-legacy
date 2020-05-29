# utils Folder Function Structure
This README provides the structure of the contents of this folder and a brief description of its usage.

    utils
    └── utils.py
        ├── mvncond                 # Calculates A, b, and V such that y[~icond] | y[icond] ~ N(A*y[icond] + b, V).
        ├── solveV                  # Computes X = V^{-1}B where V is a variance matrix.
        ├── indep_init              # Initializes the necessary parameters in KalmanODE.
        ├── norm_sim                # Simulates from x ~ N(mu, V).
        └── rand_mat                # Simulate a nxp random matrix from N(0, 1).
