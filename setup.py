from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='BayesODE',
    version='0.0.1',
    author='Martin Lysy & Mohan Wu',
    author_email='mhwu@edu.uwaterloo.ca',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mlysy/probDE",
    packages=find_packages(),
    install_requires=['numpy', 'scipy', 'matplotlib']
    #packages=['BayesODE', 'BayesODE/Bayesian', 'BayesODE/Kalman', 'BayesODE/Kalman/Old', 'BayesODE/Kalman/pykalman', 'BayesODE/utils', 'BayesODE/Tests']
)
