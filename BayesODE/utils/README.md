# utils Folder Function Structure
This README provides the structure of the contents of this folder and a brief description of its usage.

    utils
    └── utils.py
        ├── mvcond                  # Calculates A, b, and V such that y[~icond] | y[icond] ~ N(A*y[icond] + b, V).
        ├── solveV                  # Computes X = V^{-1}B where V is a variance matrix.
        ├── root_gen                # Creates p CAR model roots.
        └── zero_pad                # Pad x0 with 0s at the end of x0 so that x0 is of size p.
