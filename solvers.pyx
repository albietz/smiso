# distutils: include_dirs = /scratch/clear/abietti/local/include

import numpy as np
cimport numpy as np

from cpython.ref cimport PyObject, Py_INCREF
from cython.operator cimport dereference as deref
from libc.stdint cimport int32_t, int64_t
from libcpp.string cimport string
from libcpp cimport bool

np.import_array()

IF USE_FLOAT:
    ctypedef float Double
    npDOUBLE = np.NPY_FLOAT32
    dtype = np.float32
ELSE:
    ctypedef double Double
    npDOUBLE = np.NPY_FLOAT64
    dtype = np.float64


cdef extern from "numpy/arrayobject.h":
    void PyArray_SetBaseObject(np.ndarray, PyObject*)

cdef extern from "solvers/common.h" namespace "solvers":
    void _center "solvers::center"(
        Double* const XData, const size_t rows, const size_t cols)
    void _normalize "solvers::normalize"(
        Double* const XData, const size_t rows, const size_t cols)

def center(Double[:,::1] X not None):
    _center(&X[0,0], X.shape[0], X.shape[1])

def normalize(Double[:,::1] X not None):
    _normalize(&X[0,0], X.shape[0], X.shape[1])

cdef extern from "solvers/Solver.h" namespace "solvers":
    void iterateBlock "solvers::Solver::iterateBlock"[SolverT](
            SolverT& solver,
            const size_t blockSize,
            const Double* const XData,
            const Double* const yData,
            const int64_t* const idxData) nogil

    void iterateBlockIndexed "solvers::Solver::iterateBlockIndexed"[SolverT](
            SolverT& solver,
            const size_t dataSize,
            const Double* const XData,
            const Double* const yData,
            const size_t blockSize,
            const int64_t* const idxData) nogil

    cdef cppclass OneVsRest[SolverT]:
        OneVsRest(size_t nclasses, ...)
        size_t nclasses()
        void startDecay()
        void decay(Double mult)
        void iterateBlock(...)
        void iterateBlockIndexed(...)
        void predict(const size_t sz,
                     int32_t* const out,
                     const Double* const XData)

cdef extern from "solvers/Loss.h" namespace "solvers":
    void setGradSigma "solvers::Loss::setGradSigma"(
            const Double gradSigma)
    Double gradSigma "solvers::Loss::gradSigma"()

cdef extern from "solvers/SGD.h" namespace "solvers":
    cdef cppclass _SGD "solvers::SGD":
        _SGD(size_t dim, Double lr, Double lmbda, string loss)
        void startDecay()
        void decay(Double mult)
        size_t t()
        size_t nfeatures()
        Double* wdata()


def set_grad_sigma(Double gradSigma):
    setGradSigma(gradSigma)

def grad_sigma():
    return gradSigma()

cdef class SGD:
    cdef _SGD* solver

    def __cinit__(self, size_t dim, Double lr=0.1,
                  Double lmbda=0., string loss="logistic"):
        self.solver = new _SGD(dim, lr, lmbda, loss)

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
                np.PyArray_SimpleNewFromData(1, shape, npDOUBLE,
                                             self.solver.wdata())
            Py_INCREF(self)
            PyArray_SetBaseObject(arr, <PyObject*>self)
            return arr

    def start_decay(self):
        self.solver.startDecay()

    def decay(self, Double multiplier=0.5):
        self.solver.decay(multiplier)

    def iterate(self,
                Double[:,::1] X not None,
                Double[::1] y not None,
                int64_t[::1] idx not None):
        iterateBlock[_SGD](deref(self.solver),
                           X.shape[0],
                           &X[0,0],
                           &y[0],
                           &idx[0])

    def iterate_indexed(self,
                        Double[:,::1] X not None,
                        Double[::1] y not None,
                        int64_t[::1] idx not None):
        iterateBlockIndexed[_SGD](deref(self.solver),
                                  X.shape[0],
                                  &X[0,0],
                                  &y[0],
                                  idx.shape[0],
                                  &idx[0])

cdef class SGDOneVsRest:
    cdef OneVsRest[_SGD]* solver

    def __cinit__(self, size_t nclasses, size_t dim, Double lr=0.1,
                  Double lmbda=0., string loss="logistic"):
        self.solver = new OneVsRest[_SGD](nclasses, dim, lr, lmbda, loss)

    def __dealloc__(self):
        del self.solver

    def start_decay(self):
        self.solver.startDecay()

    def decay(self, Double multiplier=0.5):
        self.solver.decay(multiplier)

    def iterate(self,
                Double[:,::1] X not None,
                int32_t[::1] y not None,
                int64_t[::1] idx not None):
        self.solver.iterateBlock(X.shape[0], &X[0,0], &y[0], &idx[0])

    def iterate_indexed(self,
                        Double[:,::1] X not None,
                        int32_t[::1] y not None,
                        int64_t[::1] idx not None):
        self.solver.iterateBlockIndexed(
                X.shape[0], &X[0,0], &y[0], idx.shape[0], &idx[0])

    def predict(self, Double[:,::1] X not None):
        preds = np.empty(X.shape[0], dtype=np.int32)
        cdef int32_t[:] out = preds
        self.solver.predict(out.shape[0], &out[0], &X[0,0])
        return preds


cdef extern from "solvers/MISO.h" namespace "solvers":
    cdef cppclass _MISO "solvers::MISO":
        _MISO(size_t dim, size_t n, Double lmbda, string loss, bool computeLB)
        void startDecay()
        void decay(Double mult)
        size_t t()
        size_t nfeatures()
        size_t nexamples()
        Double* wdata()
        Double lowerBound()

cdef class MISO:
    cdef _MISO* solver

    def __cinit__(self, size_t dim, size_t n,
                  Double lmbda=0.1, string loss="logistic", bool compute_lb=False):
        self.solver = new _MISO(dim, n, lmbda, loss, compute_lb)

    def __dealloc__(self):
        del self.solver

    property nfeatures:
        def __get__(self):
            return self.solver.nfeatures()

    property nexamples:
        def __get__(self):
            return self.solver.nexamples()

    property w:
        def __get__(self):
            cdef np.npy_intp shape[1]
            shape[0] = self.nfeatures
            cdef np.ndarray[Double, ndim=1] arr = \
                np.PyArray_SimpleNewFromData(1, shape, npDOUBLE,
                                             self.solver.wdata())
            Py_INCREF(self)
            PyArray_SetBaseObject(arr, <PyObject*>self)
            return arr

    def start_decay(self):
        self.solver.startDecay()

    def decay(self, Double multiplier=0.5):
        self.solver.decay(multiplier)

    def lower_bound(self):
        return self.solver.lowerBound()

    def iterate(self,
                Double[:,::1] X not None,
                Double[::1] y not None,
                int64_t[::1] idx not None):
        iterateBlock[_MISO](deref(self.solver),
                            X.shape[0],
                            &X[0,0],
                            &y[0],
                            &idx[0])

    def iterate_indexed(self,
                        Double[:,::1] X not None,
                        Double[::1] y not None,
                        int64_t[::1] idx not None):
        iterateBlockIndexed[_MISO](deref(self.solver),
                                   X.shape[0],
                                   &X[0,0],
                                   &y[0],
                                   idx.shape[0],
                                   &idx[0])


cdef class MISOOneVsRest:
    cdef OneVsRest[_MISO]* solver

    def __cinit__(self, size_t nclasses, size_t dim, size_t n,
                  Double lmbda=0.1, string loss="logistic", bool compute_lb=False):
        self.solver = new OneVsRest[_MISO](nclasses, dim, n, lmbda, loss, compute_lb)

    def __dealloc__(self):
        del self.solver

    def start_decay(self):
        self.solver.startDecay()

    def decay(self, Double multiplier=0.5):
        self.solver.decay(multiplier)

    def iterate(self,
                Double[:,::1] X not None,
                int32_t[::1] y not None,
                int64_t[::1] idx not None):
        self.solver.iterateBlock(X.shape[0], &X[0,0], &y[0], &idx[0])

    def iterate_indexed(self,
                        Double[:,::1] X not None,
                        int32_t[::1] y not None,
                        int64_t[::1] idx not None):
        self.solver.iterateBlockIndexed(
                X.shape[0], &X[0,0], &y[0], idx.shape[0], &idx[0])

    def predict(self, Double[:,::1] X not None):
        preds = np.empty(X.shape[0], dtype=np.int32)
        cdef int32_t[:] out = preds
        self.solver.predict(out.shape[0], &out[0], &X[0,0])
        return preds
