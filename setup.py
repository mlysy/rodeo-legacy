from setuptools import setup, find_packages, Extension
import numpy as np
import scipy as sp
# from setuptools import setup, find_packages
from os import path

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as fh:
    long_description = fh.read()

# compile with cython if it's installed
try:
    from Cython.Distutils import build_ext
    USE_CYTHON = True
except ImportError:
    USE_CYTHON = False

cmdclass = {}
if USE_CYTHON:
    # extensions = cythonize(extensions)
    cmdclass.update({'build_ext': build_ext})

# extension modules
# cpp_modules = ['kalmantv', 'mat_mult', 'kalman_ode_higher']
# ext = '.pyx' if USE_CYTHON else 'cpp'
# # c_modules = ['mat_mult', 'kalman_ode_higher']
# ext_modules = [Extension("kalmantv.cython",
#                          ["cython/{}".format(mod)+ext for mod in cpp_modules],
#                          include_dirs=[
#                              np.get_include(),
#                              sp.get_include(),
#                              "cython/eigen-3.3.7"],
#                          language='c++')]

# c/cpp modules
ext_c = '.pyx' if USE_CYTHON else '.c'
ext_cpp = '.pyx' if USE_CYTHON else 'cpp'
ext_modules = [Extension("probDE.cython.kalmantv",
                         ["cython/kalmantv"+ext_cpp],
                         include_dirs=[
                             np.get_include(),
                             "cython/eigen-3.3.7"],
                         extra_compile_args=["-O3"],
                         language='c++'),
               Extension("probDE.cython.KalmanTest.kalmantest",
                         ["cython/KalmanTest/kalmantest"+ext_cpp],
                         include_dirs=[
                             np.get_include(),
                             "cython/eigen-3.3.7"],
                         extra_compile_args=["-O3"],
                         language='c++'),
               Extension("probDE.cython.mat_mult",
                         ["cython/mat_mult"+ext_c],
                         include_dirs=[
                             np.get_include(),
                             sp.get_include()],
                         extra_compile_args=["-O3"],
                         language='c'),
               Extension("probDE.cython.kalman_ode_higher",
                         ["cython/kalman_ode_higher"+ext_c],
                         include_dirs=[
                             np.get_include()
                         ],
                         extra_compile_args=["-O3"],
                         language='c'),
               Extension("probDE.cython.kalman_ode_solve_cy",
                         ["cython/kalman_ode_solve_cy"+ext_c],
                         include_dirs=[
                             np.get_include()
                         ],
                         extra_compile_args=["-O3"],
                         language='c'),
               Extension("probDE.cython.kalman_ode_offline_cy",
                         ["cython/kalman_ode_offline_cy"+ext_c],
                         include_dirs=[
                             np.get_include()
                         ],
                         extra_compile_args=["-O3"],
                         language='c')]


setup(
    name='probDE',
    version='0.0.1',
    author='Mohan Wu, Martin Lysy',
    author_email='mlysy@uwaterloo.ca',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mlysy/probDE",
    packages=find_packages(),

    # cython
    cmdclass=cmdclass,
    ext_modules=ext_modules,

    install_requires=['numpy', 'scipy', 'matplotlib'],
    setup_requires=['setuptools>=38'],

    # install_requires=['numpy', 'scipy', 'matplotlib']
    # packages=['probDE', 'probDE/Bayesian', 'probDE/Kalman', 'probDE/Kalman/Old', 'probDE/Kalman/pykalman', 'probDE/utils', 'probDE/Tests']
)
