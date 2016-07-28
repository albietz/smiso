#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include "cblas_alt_template.h"

#include "linalg.h"
#include "common.h"
#include "approx_kernel.h"

#ifdef CUDA
#include "common_cudnn.h"
#endif

#include "ctypes_utils.h"

#include <iostream>
using namespace std;

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

// extern "C" {
//   // static PyObject * pyckn_whitening(PyObject *, PyObject *);
//   static PyObject * pyckn_encode_cudnn(PyObject *, PyObject *, PyObject *);
//   // static void init_ckn_st(void);
// }

// enum LayerType {
//   raw_patchmap = 0,
//   centering = 1,
//   center_whiten = 2,
//   center_shift = 3,
//   gradient_2d = 4,
//   center_local_whiten = 5
// };


int parse_layer(Layer<float> &l, PyObject* l_py, int idx_layer) {
  if (!PyDict_Check(l_py)) 
    throw "Layer object should be a dictionary object.";

  PyArrayObject* W=(PyArrayObject*)PyDict_GetItemString(l_py, "W");
  PyArrayObject* b=(PyArrayObject*)PyDict_GetItemString(l_py, "b");
  // PyArrayObject* Wfilt=(PyArrayObject*)PyDict_GetItemString(l_py, "Wfilt");
  PyArrayObject* W2=(PyArrayObject*)PyDict_GetItemString(l_py, "W2");

  string layer_str = "layer_" + to_string(idx_layer);

  // assert_py_throw(npyToMatrix(W, l.W), "Failed to read W from dictionary");
  if (!npyToMatrix(W, l.W, layer_str+".W"))
    return 0;
  if (!npyToVector(b, l.b, layer_str+".b"))
    return 0;
  // if (!npyToMatrix(Wfilt, l.Wfilt, layer_str+".Wfilt"))
  //   return 0;
  if (!npyToMatrix(W2, l.W2, layer_str+".W2"))
    return 0;

  PyObject* npatch = PyDict_GetItemString(l_py, "npatch");
  assert_py_throw(PyLong_Check(npatch), "npatch is not an int object");
  l.npatch = PyLong_AsLong(npatch);
  assert_py_throw(l.npatch > 0, "npatch must be positive");

  PyObject* type_layer = PyDict_GetItemString(l_py, "type_layer");
  assert_py_throw(PyLong_Check(type_layer), "type_layer is not an int object");
  l.type_layer = static_cast<typelayer_t>(PyLong_AsLong(type_layer));
  assert_py_throw(l.type_layer == 0,
      "type_layer must be 0 since other layer types are not supported yet.");

  PyObject* subsampling = PyDict_GetItemString(l_py, "subsampling");
  assert_py_throw(PyLong_Check(subsampling), "subsampling is not an int object");
  l.subsampling = PyLong_AsLong(subsampling);
  assert_py_throw(l.subsampling > 0, "subsampling must be positive");

  PyObject* zero_padding = PyDict_GetItemString(l_py, "zero_padding");
  assert_py_throw(PyBool_Check(zero_padding),
      "zero_padding is not a boolean object");
  l.zero_padding = zero_padding == Py_True;

  PyObject* new_subsampling = PyDict_GetItemString(l_py, "new_subsampling");
  assert_py_throw(PyBool_Check(new_subsampling),
      "new_subsampling is not a boolean object");
  l.new_subsampling = new_subsampling == Py_True;

  l.num_layer = idx_layer;
  l.stride = 1;
  l.type_kernel = 0;
  l.type_learning = 1;
  l.type_regul = 0;
  l.pooling_mode = POOL_GAUSSIAN_FILTER;
  
  return 1;
}



static PyObject * pyckn_nystrom_multiprojection(
    PyObject *self,
    PyObject *args,
    PyObject *keywds) {
  // {{{ the objects we will need

  PyArrayObject* pyX=NULL;

  const int type_regul(0);
  const int type_kernel(0);
  float sigma(-1);
  float lambda2(-1);
  int nfilters(-1);
  int num_subspaces(-1);

  int iter_kmeans(10);
  int iter_opt(0);
  bool compute_cov(false);

  int cuda_device(-1);
  int threads(-1);
  int verbose(0);

  // }}} 

  // {{{ parse inputs
  static char *kwlist[] = {
    "X", 
    "sigma",
    "lambda2",
    "nfilters",
    "num_subspaces",
    "iter_kmeans",
    "iter_opt",
    "compute_cov",
    "threads",
    "cuda_device",
    NULL};

  const char* format =  "Offii|iipiii";
  if (!PyArg_ParseTupleAndKeywords(args, keywds, format, kwlist,
        &pyX,

        &sigma,
        &lambda2,
        &nfilters,
        &num_subspaces,

        &iter_kmeans,
        &iter_opt,
        &compute_cov,

        &cuda_device,
        &threads,
        &verbose))
          return NULL;

  // }}}

  if (cuda_device >= 0) {
    cudaStat = cudaSetDevice(cuda_device);
    cusolver_status = cusolverDnCreate(&cusolver_handle);
    cublas_stat = cublasCreate(&handle);
    cout << "GPU Version (device" << cuda_device << ")" << endl;
  }


  Matrix<float> X;
  if (!npyToMatrix<float>((PyArrayObject*)pyX, X, "training examples"))
    return NULL;
  if (num_subspaces <= 1) {
    PyErr_SetString(PyExc_TypeError, "num_subspaces must be greater or equal to 2");
    return NULL;
  }

  const int m = X.m();
  Matrix<float>* subspace_centroids = new Matrix<float>(m, num_subspaces);
  Matrix<float>* W[num_subspaces];
  Vector<float>* b[num_subspaces];
  Matrix<float>* W2[num_subspaces];
  Matrix<float>* W3[num_subspaces];
  const int W3dim = compute_cov ? m : 1;

  for (int idx_subspace=0; idx_subspace<num_subspaces; ++idx_subspace) {
    W[idx_subspace] = new Matrix<float>(m, nfilters);
    b[idx_subspace] = new Vector<float>(nfilters);
    W2[idx_subspace] = new Matrix<float>(nfilters, nfilters);
    W3[idx_subspace] = new Matrix<float>(W3dim, W3dim);
  }

  try{
  /* actual computation */

  // if (cuda_device >= 0)
    // nystrom_ckn_cuda_multi(X, W, b, W2, W3,type_regul,sigma,threads,iter_opt,iter_kmeans,lambda2);
  // else
    nystrom_ckn_multiprojection(X,*subspace_centroids,nfilters,W,b,W2,W3,type_regul,type_kernel,sigma,threads,iter_opt,iter_kmeans,lambda2,compute_cov);
  
    cout << "Training finished" << endl;
  } catch (const char* error_msg) {
    PyErr_SetString(PyExc_TypeError, error_msg);
    return NULL;
  }

  PyObject* PyWlist = PyList_New(num_subspaces);
  PyObject* Pyblist = PyList_New(num_subspaces);
  PyObject* PyW2list = PyList_New(num_subspaces);
  PyObject* PyW3list = PyList_New(num_subspaces);

  for (int idx_subspace=0; idx_subspace<num_subspaces; ++idx_subspace) {
    cout << "wrapping matrices for subspace " << idx_subspace+1 << endl;
    PyObject* PyW =  (PyObject*)wrapMatrix(W[idx_subspace]);
    PyObject* Pyb =  (PyObject*)wrapVector(b[idx_subspace]);
    PyObject* PyW2 = (PyObject*)wrapMatrix(W2[idx_subspace]);
    PyObject* PyW3 = (PyObject*)wrapMatrix(W3[idx_subspace]);

    if (PyList_SetItem(PyWlist,  idx_subspace, PyW)  != 0)  return NULL;
    if (PyList_SetItem(Pyblist,  idx_subspace, Pyb)  != 0)  return NULL;
    if (PyList_SetItem(PyW2list, idx_subspace, PyW2) != 0) return NULL;
    if (PyList_SetItem(PyW3list, idx_subspace, PyW3) != 0) return NULL;
  }

  PyObject* PySubspaceCentroids = (PyObject*)wrapMatrix(subspace_centroids);
  return Py_BuildValue("OOOOO", PySubspaceCentroids, PyWlist, Pyblist, PyW2list, PyW3list);
}



static PyObject * pyckn_nystrom(
    PyObject *self,
    PyObject *args,
    PyObject *keywds) {
  // {{{ the objects we will need

  PyArrayObject* pyX=NULL;

  const int type_regul(0);
  const int type_kernel(0);
  float sigma(-1);
  float lambda2(-1);
  int nfilters(-1);

  int iter_kmeans(10);
  int iter_opt(0);
  bool compute_cov(false);

  int cuda_device(-1);
  int threads(-1);
  int verbose(0);

  // }}} 

  // {{{ parse inputs
  static char *kwlist[] = {
    "X", 
    "sigma",
    "lambda2",
    "nfilters",
    "iter_kmeans",
    "iter_opt",
    "compute_cov",
    "threads",
    "cuda_device",
    NULL};

  const char* format =  "Offi|iipiii";
  if (!PyArg_ParseTupleAndKeywords(args, keywds, format, kwlist,
        &pyX,

        &sigma,
        &lambda2,
        &nfilters,

        &iter_kmeans,
        &iter_opt,
        &compute_cov,

        &cuda_device,
        &threads,
        &verbose))
          return NULL;

  // }}}

  if (cuda_device >= 0) {
    cudaStat = cudaSetDevice(cuda_device);
    cusolver_status = cusolverDnCreate(&cusolver_handle);
    cublas_stat = cublasCreate(&handle);
    cout << "GPU Version (device" << cuda_device << ")" << endl;
  }


  Matrix<float> X;
  if (!npyToMatrix<float>((PyArrayObject*)pyX, X, "training examples"))
    return NULL;
  const int m = X.m();
  Matrix<float>* W = new Matrix<float>(m, nfilters);
  Vector<float>* b = new Vector<float>(nfilters);
  Matrix<float>* W2 = new Matrix<float>(nfilters, nfilters);
  const int W3dim = compute_cov ? m : 1;
  Matrix<float>* W3 = new Matrix<float>(W3dim, W3dim);

  cerr << "Inputs:" << endl;
  X.print_dims("X");
  W->print_dims("W");
  b->print_dim("b");
  W2->print_dims("W2");
  W3->print_dims("W3");

    /* actual computation */
  try{

  if (cuda_device >= 0)
    nystrom_ckn_cuda(X,*W,*b,*W2,*W3,type_regul,sigma,threads,iter_opt,iter_kmeans,lambda2);
  else
    nystrom_ckn(X,*W,*b,*W2,*W3,type_regul,type_kernel,sigma,threads,iter_opt,iter_kmeans,lambda2,compute_cov);
  
  } catch (const char* error_msg) {
    PyErr_SetString(PyExc_TypeError, error_msg);
    return NULL;
  }

  PyObject* PyW = (PyObject*)wrapMatrix(W);
  PyObject* Pyb = (PyObject*)wrapVector(b);
  PyObject* PyW2 = (PyObject*)wrapMatrix(W2);
  PyObject* PyW3 = (PyObject*)wrapMatrix(W3);

  return Py_BuildValue("OOOO", PyW, Pyb, PyW2, PyW3);
}



static PyObject * pyckn_compute_sigma(
    PyObject *self,
    PyObject *args,
    PyObject *keywds) {
  // {{{ the objects we will need

  PyArrayObject* pyX=NULL;
  float quantile=0.1;
  int verbose = 0;
  Matrix<float> X;

  // }}} 

  // {{{ parse inputs
  static char *kwlist[] = {
    "X", 
    "quantile",

    // from here on optional
    "verbose", 
    NULL};

  const char* format =  "Of|i";
  if (!PyArg_ParseTupleAndKeywords(args, keywds, format, kwlist,
        &pyX,
        &quantile,

        &verbose))
          return NULL;
  // }}}

  float sigma=-1;
  try {
    if (!npyToMatrix<float>((PyArrayObject*)pyX, X, "training samples"))
      return NULL;

    /* actual computation */
    if (X.n() == 0) {
      PyErr_SetString(PyExc_TypeError, "Cannot compute sigma: matrix X has 0 columns!");
      return NULL;
    }
    sigma = compute_sigma(X, quantile);
    if (sigma <= 0) {
      PyErr_SetString(PyExc_TypeError, ("Invalid result for sigma: "+to_string(sigma)).c_str());
      return NULL;
    }
  } catch (const char* error_msg) {
    PyErr_SetString(PyExc_TypeError, error_msg);
    return NULL;
  }

  return Py_BuildValue("f", sigma);
}



static PyObject * pyckn_encode_cpu(
    PyObject *self,
    PyObject *args,
    PyObject *keywds) {
  // {{{ the objects we will need

  PyArrayObject* pyinmap=NULL;

  int verbose = 0;
  Map<float> inmap;
  int threads(-1);

  PyObject* py_layers = NULL;
  // }}} 

  // {{{ parse inputs
  static char *kwlist[] = {
    "inmap", 
    "layers",

    // from here on optional
    "threads",
    "verbose", 
    NULL};

  const char* format =  "OO|ii";
  if (!PyArg_ParseTupleAndKeywords(args, keywds, format, kwlist,
        &pyinmap,
        &py_layers,
        
        &threads,
        &verbose))
          return NULL;
  // }}}

  // {{{ parse layer objects
  assert_py_obj(PySequence_Check(py_layers), "Expected sequence of Layers.") 

  const int nlayers = PySequence_Length(py_layers);
  Layer<float> layers[nlayers];

  // The cudnn version encodes the whole network and
  // returns a matrix of feature vectors
  Matrix<float>* psi = new Matrix<float>();
  try {
    for (int idx_layer=0; idx_layer<nlayers; ++idx_layer) {
      if (!parse_layer(layers[idx_layer], PySequence_GetItem(py_layers, idx_layer), idx_layer+1))
        return NULL;
    }

    // }}}


    if (!npyToMap<float>(pyinmap, inmap, "input map"))
      return NULL;

    if (threads <= 0) {
      threads=1;
#ifdef _OPENMP
      threads =  MIN(MAX_THREADS,omp_get_num_procs());
#endif
    } 
    threads=init_omp(threads);

    /* actual computation */

    bool compute_covs = false;
    bool normalize = false;
    Map<float> map_zero;
    inmap.refSubMapZ(0,map_zero);
    Map<float> map_zero_out;
    encode_ckn_map(map_zero,&(layers[0]),nlayers,map_zero_out,verbose);
    const INTM ndesc=map_zero_out.x()*map_zero_out.y()*map_zero_out.z();
    if (verbose) {
      PRINT_I(map_zero_out.x());
      PRINT_I(map_zero_out.y());
      PRINT_I(map_zero_out.z());
    }
    psi->resize(ndesc,inmap.z());
    encode_ckn(inmap,&(layers[0]),nlayers,*psi,compute_covs,normalize);
  } catch (const char* error_msg) {
    PyErr_SetString(PyExc_TypeError, error_msg);
    return NULL;
  }

  PyObject* PyOutmap = (PyObject*)wrapMatrix(psi);

  return PyOutmap;
}



static PyObject * pyckn_encode_cudnn(
    PyObject *self,
    PyObject *args,
    PyObject *keywds) {
  // {{{ the objects we will need

  PyArrayObject* pyinmap=NULL;

  int verbose = 0;
  Map<float> inmap;
  int cuda_device(0);

  PyObject* py_layers = NULL;
  int batch_size = 256;
  // }}} 

  // {{{ parse inputs
  static char *kwlist[] = {
    "inmap", 
    "layers",
    "cuda_device",

    // from here on optional
    "batch_size",
    "verbose", 
    NULL};

  PRINT_I(verbose);

  const char* format =  "OOi|ii";
  if (!PyArg_ParseTupleAndKeywords(args, keywds, format, kwlist,
        &pyinmap,
        &py_layers,
        &cuda_device,
        &batch_size,
        
        &verbose))
          return NULL;
  // }}}

  // The cudnn version encodes the whole network and
  // returns a matrix of feature vectors
  Matrix<float>* psi = new Matrix<float>();

  // {{{ parse layer objects
  assert_py_obj(PySequence_Check(py_layers), "Expected sequence of Layers.") 

  const int nlayers = PySequence_Length(py_layers);
  Layer<float> layers[nlayers];
  try {
    for (int idx_layer=0; idx_layer<nlayers; ++idx_layer) {
      if (verbose)
        cout << "parsing layer " << idx_layer+1 << endl;
      if (!parse_layer(layers[idx_layer], PySequence_GetItem(py_layers, idx_layer), idx_layer+1))
        return NULL;
    }

    // }}}


    if (!npyToMap<float>(pyinmap, inmap, "input map"))
      return NULL;

    /* actual computation */

    init_cuda(cuda_device, true, true);
    encode_ckn_cudnn(inmap, layers, nlayers, psi[0], batch_size, verbose);
    destroy_cuda(true, true);
  } catch (const char* error_msg) {
    PyErr_SetString(PyExc_TypeError, error_msg);
    return NULL;
  }

  PyObject* PyOutmap = (PyObject*)wrapMatrix(psi);

  return PyOutmap;
}



static PyMethodDef method_list[] = {
  // {"whitening",  pyckn_whitening, METH_VARARGS, "Whiten array rows"},
  {"encode_cudnn",  (PyCFunction)pyckn_encode_cudnn, METH_VARARGS | METH_KEYWORDS, "Encode a collection of inputs using CUDNN"},
  {"encode_cpu",  (PyCFunction)pyckn_encode_cpu, METH_VARARGS | METH_KEYWORDS, "Encode a collection of inputs on the CPU"},
  {"train_layer",  (PyCFunction)pyckn_nystrom, METH_VARARGS | METH_KEYWORDS, "Train a CKN layer using the Nystrom method."},
  {"train_layer_multiprojection",  (PyCFunction)pyckn_nystrom_multiprojection, METH_VARARGS | METH_KEYWORDS, "Train a CKN layer with multiple subspace projection."},
  {"compute_sigma",  (PyCFunction)pyckn_compute_sigma, METH_VARARGS | METH_KEYWORDS, "Compute sigma for a given training set and quantile."},
  // {"optimize_sgd",  (PyCFunction)pyckn_optimize_sgd,METH_VARARGS | METH_KEYWORDS, "Train CKN layer by SGD"},
  {NULL, NULL, 0, NULL}        /* Sentinel */
};

static struct PyModuleDef cknmodule = {
   PyModuleDef_HEAD_INIT,
   "_ckn_cuda",   /* name of module */
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
  assert_py_obj(PyType_Ready(&MapWrapperType) >= 0,
      "Map wrapper type failed to initialize");
  assert_py_obj(PyType_Ready(&MatrixWrapperType) >= 0,
      "Matrix wrapper type failed to initialize");
  assert_py_obj(PyType_Ready(&VectorWrapperType) >= 0,
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
