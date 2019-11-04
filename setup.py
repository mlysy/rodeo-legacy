from setuptools import setup, find_packages
from os import path

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as fh:
    long_description = fh.read()

setup(
    name='probDE',
    version='0.0.1',
    author='Mohan Wu, Martin Lysy',
    author_email='mlysy@uwaterloo.ca',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mlysy/probDE",
    packages=find_packages(),
    install_requires=['numpy', 'scipy', 'matplotlib']
    #packages=['probDE', 'probDE/Bayesian', 'probDE/Kalman', 'probDE/Kalman/Old', 'probDE/Kalman/pykalman', 'probDE/utils', 'probDE/Tests']
)
