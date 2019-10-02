from distutils.core import setup

setup(
    name='BayesODE',
    version='0.1',
    author='Martin Lysy & Mohan Wu',
    author_email='mhwu@edu.uwaterloo.ca',
    packages=['BayesODE', 'BayesODE/Bayesian', 'BayesODE/Kalman', 'BayesODE/Kalman/Old', 'BayesODE/Kalman/pykalman', 'BayesODE/utils', 'BayesODE/Tests']
)
