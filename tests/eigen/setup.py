import platform
import numpy as np
import scipy as sp
from os import path
from setuptools import setup, find_packages, Extension

# eigen path
#eigen_path = "/Users/mlysy/Documents/proj/buqDE/kalmantv/eigen-3.3.7"
eigen_path = "eigen-3.3.7"
# compiler options
if platform.system() != "Windows":
    extra_compile_args = ["-O3", "-ffast-math",
                          "-mtune=native", "-march=native", "-fopenmp"]
else:
    extra_compile_args = ["-O2", "/openmp"]
# remove numpy depreciation warnings as documented here:
# http://docs.cython.org/en/latest/src/userguide/source_files_and_compilation.html#configuring-the-c-build
disable_numpy_warnings = [("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]


# compile with cython if it's installed
try:
    from Cython.Distutils import build_ext
    USE_CYTHON = True
except ImportError:
    USE_CYTHON = False

cmdclass = {}
if USE_CYTHON:
    # extensions = cythonize(extensions)
    cmdclass.update({"build_ext": build_ext})


# c/cpp modules
ext_c = ".pyx" if USE_CYTHON else ".c"
ext_cpp = ".pyx" if USE_CYTHON else "cpp"
ext_modules = [Extension("rodeo.tests.ode_functions",
                         ["ode_functions"+ext_c],
                         extra_compile_args=extra_compile_args,
                         define_macros=disable_numpy_warnings,
                         language="c"),
               Extension("rodeo.tests.ode_functions_ctuple",
                         ["ode_functions_ctuple"+ext_c],
                         extra_compile_args=extra_compile_args,
                         define_macros=disable_numpy_warnings,
                         language="c")]

setup(
    name="rodeo_test",
    version="0.0.2",
    author="Mohan Wu, Martin Lysy",
    author_email="mlysy@uwaterloo.ca",
    description="Eigen backed KalmanODE for testing",
    url="https://github.com/mlysy/rodeo",
    packages=[],

    # cython
    cmdclass=cmdclass,
    ext_modules=ext_modules,

    install_requires=["numpy", "scipy"],
    setup_requires=["setuptools>=38"]
)
