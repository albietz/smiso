# distutils: include_dirs = /scratch/clear/abietti/local/include

import numpy as np
cimport numpy as np

from cpython.ref cimport PyObject, Py_INCREF
from cython.operator cimport dereference as deref
from libcpp.string cimport string
from libc.stdint cimport int64_t
from libc.stdint cimport uintptr_t

np.import_array()

ctypedef float Double

cdef extern from "numpy/arrayobject.h":
    void PyArray_SetBaseObject(np.ndarray, PyObject*)

cdef extern from "solvers/Solver.h" namespace "solvers":
    void iterateBlock[SolverT](SolverT& solver,
                               const size_t blockSize,
                               const Double* const XData,
                               const Double* const yData,
                               const int64_t* const idxData) nogil

    void iterateBlockIndexed[SolverT](SolverT& solver,
                                      const size_t dataSize,
                                      const Double* const XData,
                                      const Double* const yData,
                                      const size_t blockSize,
                                      const int64_t* const idxData) nogil

cdef extern from "solvers/SGD.h" namespace "solvers":
    cdef cppclass SGD:
        SGD(size_t dim, Double lr, Double lmbda, string loss)
        void startDecay()
        size_t t()
        size_t nfeatures()
        Double* wdata()

cdef class SGDSolver:
    cdef SGD* solver

    def __init__(self, dim, lr=0.1, lmbda=0., loss="logistic"):
        self.solver = new SGD(dim, lr, lmbda, loss)

    def __dealloc__(self):
        del self.solver

    property nfeatures:
        def __get__(self):
            return self.solver.nfeatures()

    property w:
        def __get__(self):
            cdef np.npy_intp shape[1]
            shape[0] = self.nfeatures
            cdef np.ndarray[Double, ndim=1] arr = \
                np.PyArray_SimpleNewFromData(1, shape, np.NPY_FLOAT32,
                                             self.solver.wdata())
            Py_INCREF(self)
            PyArray_SetBaseObject(arr, <PyObject*>self)
            return arr

    def start_decay(self):
        self.solver.startDecay()

    def iterate(self,
                Double[:,:] X not None,
                Double[:] y not None,
                int64_t[:] idx not None):
        iterateBlock[SGD](deref(self.solver),
                          X.shape[0],
                          &X[0,0],
                          &y[0],
                          &idx[0])

    def iterate_indexed(self,
                        Double[:,:] X not None,
                        Double[:] y not None,
                        int64_t[:] idx not None):
        iterateBlockIndexed[SGD](deref(self.solver),
                                 X.shape[0],
                                 &X[0,0],
                                 &y[0],
                                 idx.shape[0],
                                 &idx[0])

cdef extern from "solvers/MISO.h" namespace "solvers":
    cdef cppclass MISO:
        MISO(size_t dim, size_t n, Double lmbda, string loss)
        void startDecay()
        size_t t()
        size_t nfeatures()
        size_t nexamples()
        Double* wdata()

cdef class MISOSolver:
    cdef MISO* solver

    def __init__(self, dim, n, lmbda=0.1, loss="logistic"):
        self.solver = new MISO(dim, n, lmbda, loss)

    def __dealloc__(self):
        del self.solver

    property nfeatures:
        def __get__(self):
            return self.solver.nfeatures()

    property w:
        def __get__(self):
            cdef np.npy_intp shape[1]
            shape[0] = self.nfeatures
            cdef np.ndarray[Double, ndim=1] arr = \
                np.PyArray_SimpleNewFromData(1, shape, np.NPY_FLOAT32,
                                             self.solver.wdata())
            Py_INCREF(self)
            PyArray_SetBaseObject(arr, <PyObject*>self)
            return arr

    def start_decay(self):
        self.solver.startDecay()

    def iterate(self,
                Double[:,:] X not None,
                Double[:] y not None,
                int64_t[:] idx not None):
        iterateBlock[MISO](deref(self.solver),
                           X.shape[0],
                           &X[0,0],
                           &y[0],
                           &idx[0])

    def iterate_indexed(self,
                        Double[:,:] X not None,
                        Double[:] y not None,
                        int64_t[:] idx not None):
        iterateBlockIndexed[MISO](deref(self.solver),
                                  X.shape[0],
                                  &X[0,0],
                                  &y[0],
                                  idx.shape[0],
                                  &idx[0])
