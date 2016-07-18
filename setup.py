from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

USE_FLOAT = 0  # use float (1) or double (0)

setup(
    name = 'solvers',
    ext_modules = cythonize([Extension(
        'solvers',
        ['solvers.pyx',
         'solvers/Loss.cpp',
         'solvers/SGD.cpp',
         'solvers/MISO.cpp'],
        language='c++',
        extra_compile_args=['-std=c++11'],
        extra_link_args=['-std=c++11'],
        define_macros=[('USE_FLOAT', USE_FLOAT)],
        )],
        compile_time_env={'USE_FLOAT': USE_FLOAT})
)
