from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy

USE_FLOAT = 1  # use float (1) or double (0)

setup(
    name = 'solvers',
    ext_modules = cythonize([Extension(
        'solvers',
        ['solvers.pyx',
         'solvers/Loss.cpp',
         'solvers/SGD.cpp',
         'solvers/MISO.cpp',
         'solvers/SAGA.cpp'],
        language='c++',
        include_dirs=[numpy.get_include(), 'solvers'],
        extra_compile_args=['-std=c++11', '-fopenmp'],
        extra_link_args=['-std=c++11', '-fopenmp', '-lglog'],
        define_macros=[('USE_FLOAT', USE_FLOAT)],
        )],
        compile_time_env={'USE_FLOAT': USE_FLOAT})
)
