#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include "cblas_alt_template.h"
#include "linalg.h"

#include "common.h"

#include "ctypes_utils.h"


#define MAKE_INIT_NAME(x) init ## x (void)
#define MODNAME_INIT(s) MAKE_INIT_NAME(s)

#define STR_VALUE(arg)      #arg
#define FUNCTION_NAME(name) STR_VALUE(name)

#define MODNAME_STR FUNCTION_NAME(MODNAME)

/*
   Get the include directories within python using

   import distutils.sysconfig
   print distutils.sysconfig.get_python_inc()
   import numpy as np 
   print np.get_include()

   gcc  -fPIC -shared -g -Wall -O3 \
   -I /usr/include/python2.7 -I /usr/lib64/python2.7/site-packages/numpy/core/include \
   mymath.c -o mymath.so 

*/

extern "C" {
    static PyObject * pyckn_whitening(PyObject *, PyObject *);
    static PyObject * pyckn_encode(PyObject *, PyObject *, PyObject *keywds);
    // static void init_ckn_st(void);
}


static PyObject * pyckn_encode_layer(PyObject *self, PyObject *args, PyObject *keywds) {
    PyArrayObject* W=NULL;
    PyArrayObject* b=NULL;
    PyArrayObject* Wfilt=NULL;
    PyArrayObject* mu=NULL;
    PyArrayObject* W2=NULL;
    PyArrayObject* W3=NULL;
    PyArrayObject* pyinmap=NULL;
    int threads=-1;
    int verbose = 0;
    Layer<float> l;
    Map<float> inmap;
    l.new_subsampling = false;

    /* parse inputs */
    static char *kwlist[] = {
        "inmap", 
        "num_layer",
        "npatch",
        "nfilters",
        "subsampling",
        "zero_padding",
        "type_layer",
        "type_kernel",
        "type_learning",
        "type_regul",
        "sigma",
        "W",
        "b",
        "Wfilt",
        "mu",
        "W2",
        "W3",
        "verbose", 
        "threads", NULL};

    const char* format =  "O!i(iii)i(iii)iiiiifO!O!|O!O!O!O!ii";
    if (!PyArg_ParseTupleAndKeywords(args, keywds, format, kwlist,

                &PyArray_Type, &pyinmap,
                
                &l.num_layer,
                
                &l.npatch[0],&l.npatch[1],&l.npatch[2],
                
                &l.nfilters,
                
                &l.subsampling[0],&l.subsampling[1],&l.subsampling[2],
                
                &l.zero_padding,
                &l.type_layer,
                &l.type_kernel,
                &l.type_learning,
                &l.type_regul,
                
                &l.sigma,
                
                &PyArray_Type, &W,
                &PyArray_Type, &b,
                
                &PyArray_Type, &Wfilt,
                &PyArray_Type, &mu,
                &PyArray_Type, &W2,
                &PyArray_Type, &W3,
                
                &verbose,
                &threads))
        return NULL;

    if (!npyToMatrix(W, l.W)) return NULL;
    if (!npyToVector(b, l.b)) return NULL;
    if (!npyToMatrix(Wfilt, l.Wfilt)) return NULL;
    if (!npyToVector(mu, l.mu)) return NULL;
    if (!npyToMatrix(W2, l.W2)) return NULL;
    if (!npyToMatrix(W3, l.W3)) return NULL;

    if (!npyToMap<float>(pyinmap, inmap)) return NULL;
    Map<float>* outmap = new Map<float>();

    if (threads == -1) {
        threads=1;
#ifdef _OPENMP
        threads =  MIN(MAX_THREADS,omp_get_num_procs());
#endif
    } 
    threads=init_omp(threads);

    /* actual computation */

    encode_layer(inmap, *outmap, l, verbose);
    PyObject* PyOutmap = (PyObject*)wrapMap(outmap);
    Py_XDECREF(PyOutmap);

    return PyOutmap;
}

static PyObject * pyckn_centering(PyObject *self, PyObject *args) {
    PyArrayObject * PyX;
    int threads=-1;
    int channels = 1;

    /* parse inputs */

    if (!PyArg_ParseTuple(args, "O!|ii",
                &PyArray_Type, &PyX, &channels, &threads))
        return NULL;

    if (threads == -1) {
        threads=1;
#ifdef _OPENMP
        threads =  MIN(MAX_THREADS,omp_get_num_procs());
#endif
    } 
    threads=init_omp(threads);

    Matrix<float>* X = new Matrix<float>();
    if (!npyToMatrix(PyX, *X)) return NULL;

    centering(*X, channels);
   
    Py_RETURN_NONE;
}

static PyObject * pyckn_whitening(PyObject *self, PyObject *args) {
    PyArrayObject * PyX;
    int threads=-1;

    /* parse inputs */

    if (!PyArg_ParseTuple(args, "O!|i",
                &PyArray_Type, &PyX, &threads))
        return NULL;

    if (threads == -1) {
        threads=1;
#ifdef _OPENMP
        threads =  MIN(MAX_THREADS,omp_get_num_procs());
#endif
    } 
    threads=init_omp(threads);

    Matrix<float>* X = new Matrix<float>();
    if (!npyToMatrix(PyX, *X)) return NULL;

    int m = X->m();
    Matrix<float>* Wfilt = new Matrix<float>(m, m);
    Vector<float>* mu = new Vector<float>(m);

    whitening(*X, *Wfilt, *mu);
   
    PyArrayObject* PyWfilt = wrapMatrix(Wfilt);
    PyArrayObject* PyMu = wrapVector(mu);

    PyObject* result = Py_BuildValue("OO", PyWfilt, PyMu);
    Py_XDECREF(PyWfilt);
    Py_XDECREF(PyMu);

    return result;
}

static PyObject * pyckn_optimize_sgd (PyObject *self, PyObject *args, PyObject *keywds) {
    PyArrayObject* PyX=NULL;
    PyArrayObject* PyWfilt1=NULL;
    PyArrayObject* PyW=NULL;
    PyArrayObject* PyB=NULL;
    PyObject* PyIndval1=NULL;
    PyObject* PyIndval2=NULL;
    float sigma;
    int nfilters;
    int max_iter_outer = 300;
    int minibatch = 1000;
   
    // reset defaults from mex_optimize_ckn.cpp globals 
    second_order_heuristic = true;
    preconditioning_heuristic = true;
    regularization_heuristic = false;
    balancing_heuristic = false;
    reduce_schedule = 50;
    iter_per_batch = 1000;
    thrs=4.0;
    
    int threads = -1;
    int device = 0;

    static char *kwlist[] = {"X", "indval1", "indval2", "W", "b", "Wfilt1", "sigma", "nfilters", "max_iter_outer", "second_order_heuristic", "preconditioning_heuristic", "regularization_heuristic", "balancing_heuristic", "reduce_schedule", "iter_per_batch", "thrs", "threads", "device", NULL};

    const char* format =  "O!O!O!O!O!O!fi|iiiiiiiifii";
    if (!PyArg_ParseTupleAndKeywords(args, keywds, format, kwlist,
        &PyArray_Type, &PyX, 
        &PyArray_Type, &PyIndval1, 
        &PyArray_Type, &PyIndval2, 
        &PyArray_Type, &PyW, 
        &PyArray_Type, &PyB, 
        &PyArray_Type, &PyWfilt1, 
        &sigma, 
        &nfilters, 
        &max_iter_outer, 
        &second_order_heuristic, 
        &preconditioning_heuristic, 
        &regularization_heuristic, 
        &balancing_heuristic, 
        &reduce_schedule, 
        &iter_per_batch, 
        &thrs, 
        &threads, 
        &device))
            return NULL;


    Matrix<float>* X = new Matrix<float>();
    if (!npyToMatrix(PyX, *X)) return NULL;
    Matrix<float>* W = new Matrix<float>();
    if (!npyToMatrix(PyW, *W)) return NULL;
    Vector<float>* b = new Vector<float>();
    if (!npyToVector(PyB, *b)) return NULL;
    Matrix<float>* Wfilt1 = new Matrix<float>();
    if (!npyToMatrix(PyWfilt1, *Wfilt1)) return NULL;

    Vector<float>* objval = new Vector<float>();


    std::vector<int> indval1;
    std::vector<int> indval2;
    if (!sequenceToVector(PyIndval1, indval1))
        return NULL;
    if (!sequenceToVector(PyIndval2, indval2))
        return NULL;

#ifdef CUDA
    cudaStat = cudaSetDevice(device);
#endif

    const float scal = compute_scaling_factor<float>(*X, indval1, indval2, sigma);
    optimize_ckn_sgd(*X, indval1, indval2, *W, *b, *Wfilt1, sigma, scal, minibatch, max_iter_outer, *objval);

    PyArrayObject* PyObjval = wrapVector(objval);
    PyObject* result = Py_BuildValue("OOO", PyW, PyB, PyObjval);
    Py_XDECREF(PyObjval);
    return result;
}

static PyMethodDef method_list[] = {
    {"whitening",  pyckn_whitening, METH_VARARGS, "Whiten array rows"},
    {"encode_layer",  (PyCFunction)pyckn_encode_layer,METH_VARARGS | METH_KEYWORDS, "Encode a single input with a single layer"},
    {"optimize_sgd",  (PyCFunction)pyckn_optimize_sgd,METH_VARARGS | METH_KEYWORDS, "Train CKN layer by SGD"},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};

static struct PyModuleDef cknmodule = {
   PyModuleDef_HEAD_INIT,
   "_ckn_st",   /* name of module */
   NULL, /* module documentation, may be NULL */
   -1,       /* size of per-interpreter state of the module,
                or -1 if the module keeps state in global variables. */
   method_list,
   NULL//, NULL, NULL, NULL
};



PyMODINIT_FUNC
PyInit__ckn_cuda(void) {

  PyObject* m;
  m = PyModule_Create(&cknmodule);
  assert_py_obj(m!=NULL, "failed to create ckn module object");

  // initialize wrapper classes
  MatrixWrapperType.tp_new = PyType_GenericNew;
  VectorWrapperType.tp_new = PyType_GenericNew;
  MapWrapperType.tp_new = PyType_GenericNew;
  assert_py_obj(PyType_Ready(&MapWrapperType) < 0,
      "Map wrapper type failed to initialize");
  assert_py_obj(PyType_Ready(&MatrixWrapperType) < 0,
      "Matrix wrapper type failed to initialize");
  assert_py_obj(PyType_Ready(&VectorWrapperType) < 0,
      "Vector wrapper type failed to initialize");

  /* required, otherwise numpy functions do not work */
  import_array();

  Py_INCREF(&MatrixWrapperType);
  Py_INCREF(&MapWrapperType);
  Py_INCREF(&VectorWrapperType);
  PyModule_AddObject(m, "MyDealloc_Type_Mat", (PyObject *)&MatrixWrapperType);
  PyModule_AddObject(m, "MyDealloc_Type_Map", (PyObject *)&MapWrapperType);
  PyModule_AddObject(m, "MyDealloc_Type_Vec", (PyObject *)&VectorWrapperType);

  return m;
}
PyMODINIT_FUNC PyInit_ckn_st(void) {
    
    // initialize wrapper classes
    _MatrixWrapperType.tp_new = PyType_GenericNew;
    if (PyType_Ready(&_MatrixWrapperType) < 0)
        return; // FIXME Error handling
    _VectorWrapperType.tp_new = PyType_GenericNew;
    if (PyType_Ready(&_VectorWrapperType) < 0)
        return; // FIXME Error handling
    _MapWrapperType.tp_new = PyType_GenericNew;
    if (PyType_Ready(&_MapWrapperType) < 0)
        return; // FIXME Error handling

    PyObject* m = Py_InitModule3("_ckn_st", method_list,
            "Training and application of Convolutional Kernel Networks");
    /* required, otherwise numpy functions do not work */
    import_array();
    Py_INCREF(&_MatrixWrapperType);
    Py_INCREF(&_MapWrapperType);
    Py_INCREF(&_VectorWrapperType);
    PyModule_AddObject(m, "MatrixWrapperType", (PyObject *)&_MatrixWrapperType);
    PyModule_AddObject(m, "MapWrapperType", (PyObject *)&_MapWrapperType);
    PyModule_AddObject(m, "VectorWrapperType", (PyObject *)&_VectorWrapperType);
}
