from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

setup(
    name = 'solvers',
    ext_modules = cythonize([Extension(
        'solvers',
        ['solvers.pyx',
         'solvers/SGD.cpp',
         'solvers/MISO.cpp'],
        language='c++',
        extra_compile_args=['-std=c++11'],
        extra_link_args=['-std=c++11'],
        )])
)
