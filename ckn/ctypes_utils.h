#include <Python.h>
#include <numpy/arrayobject.h>
#include "linalg.h"

/**
 * A lot of this can probably be done with generics.
 * Since I am only starting to learn this stuff I'll put that off to later.
 * My "straight-forward" attemps didn;t work at least...
 */


// check for a condition, and fail with an exception strin set if it is false
#define assert_py_obj(condition, error) if (! (condition) ) { \
    PyErr_SetString(PyExc_TypeError, (error)); \
    return NULL; \
  }

// the same macro, but for cases where the calling method returns an integer
#define assert_py_int(condition, error) if (! (condition) ) { \
    PyErr_SetString(PyExc_TypeError, (error)); \
    return 0; \
  }

// and another version that throws the error message as a const char* instead
#define assert_py_throw(condition, error) if (! (condition) ) { \
    throw (error); \
  }



template <typename T> inline string getTypeName();
template <> inline string getTypeName<float>() { return "float32"; };
template <> inline string getTypeName<double>() { return "float64"; };

template <typename T> inline int getTypeObject();
template <> inline int getTypeObject<float>() { return NPY_FLOAT32; };
template <> inline int getTypeObject<double>() { return NPY_FLOAT64; };


// these structs hold define the python type objects for Vector, Matrix and Map
// they only hold pointers to the actual C++ objects
// this way, the data does not get deallocated immediately when objects leave
// the scope
template <typename T> struct VectorWrapper {
    PyObject_HEAD;
    Vector<T> *obj;
};

template <typename T> struct MatrixWrapper {
    PyObject_HEAD;
    Matrix<T> *obj;
};

template <typename T> struct MapWrapper {
    PyObject_HEAD;
    Map<T> *obj;
};


// these are the deallocation methods for she structs defined above
// they'll be linked to the destructor hooks of the python objects further down
template <typename T>
static void _delete_cpp_mat(MatrixWrapper<T>* self){
    if (self && self->obj) {
        delete self->obj;
    }
    Py_TYPE(self)->tp_free((PyObject*)self);
}

template <typename T>
static void _delete_cpp_vec(VectorWrapper<T>* self){
    if (self && self->obj) {
        delete self->obj;
    Py_TYPE(self)->tp_free((PyObject*)self);
    }
}

template <typename T>
static void _delete_cpp_map(MapWrapper<T>* self){
    if (self && self->obj) {
        delete self->obj;
    Py_TYPE(self)->tp_free((PyObject*)self);
    }
}


static PyTypeObject MatrixWrapperType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "ckn_cuda.MatrixWrapper", /*tp_name*/
    sizeof(MatrixWrapper<float>), /*tp_basicsize*/ // FIXME: does this break if using double?
    0, /*tp_itemsize*/
    (destructor)_delete_cpp_mat<float>, /*tp_dealloc*/ // FIXME: does this break if using double?
    0, /*tp_print*/
    0, /*tp_getattr*/
    0, /*tp_setattr*/
    0, /*tp_compare*/
    0, /*tp_repr*/
    0, /*tp_as_number*/
    0, /*tp_as_sequence*/
    0, /*tp_as_mapping*/
    0, /*tp_hash */
    0, /*tp_call*/
    0, /*tp_str*/
    0, /*tp_getattro*/
    0, /*tp_setattro*/
    0, /*tp_as_buffer*/
    Py_TPFLAGS_DEFAULT, /*tp_flags*/
    "Internal deallocator object for the Matrix class", /* tp_doc */
};

static PyTypeObject VectorWrapperType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "ckn_cuda.VectorWrapper", /*tp_name*/
    sizeof(VectorWrapper<float>), /*tp_basicsize*/ // FIXME: does this break if using double?
    0, /*tp_itemsize*/
    (destructor)_delete_cpp_vec<float>, /*tp_dealloc*/ // FIXME: does this break if using double?
    0, /*tp_print*/
    0, /*tp_getattr*/
    0, /*tp_setattr*/
    0, /*tp_compare*/
    0, /*tp_repr*/
    0, /*tp_as_number*/
    0, /*tp_as_sequence*/
    0, /*tp_as_mapping*/
    0, /*tp_hash */
    0, /*tp_call*/
    0, /*tp_str*/
    0, /*tp_getattro*/
    0, /*tp_setattro*/
    0, /*tp_as_buffer*/
    Py_TPFLAGS_DEFAULT, /*tp_flags*/
    "Internal deallocator object for the Vector class", /* tp_doc */
};

static PyTypeObject MapWrapperType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "ckn_cuda.MapWrapper", /*tp_name*/
    sizeof(MapWrapper<float>), /*tp_basicsize*/ // FIXME: does this break if using double?
    0, /*tp_itemsize*/
    (destructor)_delete_cpp_map<float>, /*tp_dealloc*/ //FIXME does this break if using double?
    0, /*tp_print*/
    0, /*tp_getattr*/
    0, /*tp_setattr*/
    0, /*tp_compare*/
    0, /*tp_repr*/
    0, /*tp_as_number*/
    0, /*tp_as_sequence*/
    0, /*tp_as_mapping*/
    0, /*tp_hash */
    0, /*tp_call*/
    0, /*tp_str*/
    0, /*tp_getattro*/
    0, /*tp_setattro*/
    0, /*tp_as_buffer*/
    Py_TPFLAGS_DEFAULT, /*tp_flags*/
    "Internal deallocator object for the Map class", /* tp_doc */
};

template <typename T>
inline PyArrayObject* copyMatrix(Matrix<T>* obj) {
    cout << "matrix data: " << obj->rawX() << endl;
    int nd=2;
    cout << "n: " << obj->n() << " m: " << obj->m() << endl;
    npy_intp dims[2]={obj->n(), obj->m()};
    PyArrayObject* arr=NULL;
    arr = (PyArrayObject*)PyArray_EMPTY(nd, dims, getTypeObject<T>(), 0);
    Matrix<T> copymat((T*)PyArray_DATA(arr), dims[1], dims[0]);
    cout << "numpy array data: " << PyArray_DATA(arr) << endl;
    if (arr == NULL) goto fail;
    copymat.copy(*obj);
    return arr;
fail:
    delete obj; // FIXME Error Handling!?
    cout << "FAIL in copyMatrix" << endl;
    Py_XDECREF(arr);
    return NULL;
}


template <typename T>
inline PyArrayObject* wrapMatrix(Matrix<T>* obj) {
    int nd=2;
    npy_intp dims[2]={obj->n(), obj->m()};
    PyObject* newobj=NULL;
    PyArrayObject* arr=NULL;
    void *mymem = (void*)(obj->rawX());
    arr = (PyArrayObject*)PyArray_SimpleNewFromData(nd, dims, getTypeObject<T>(), mymem);

    npy_intp* strides = PyArray_STRIDES(arr);
    for (int idx=0; idx<PyArray_NDIM(arr); ++idx)

    if (arr == NULL) goto fail;
    newobj = (PyObject*)PyObject_New(MatrixWrapper<T>, &MatrixWrapperType);
    if (newobj == NULL) goto fail;
    ((MatrixWrapper<T> *)newobj)->obj = obj;
    PyArray_SetBaseObject((PyArrayObject*)arr, newobj);
    return arr;
fail:
    delete obj; // FIXME Error Handling!?
    cout << "FAIL in wrapMatrix" << endl;
    Py_XDECREF(arr);
    return NULL;
}

template <typename T>
inline PyArrayObject* wrapVector(Vector<T>* obj) {
    int nd=1;
    npy_intp dims[1]={obj->n()};
    PyObject* newobj=NULL;
    PyArrayObject* arr=NULL;
    void *mymem = (void*)(obj->rawX());
    arr = (PyArrayObject*)PyArray_SimpleNewFromData(nd, dims, getTypeObject<T>(), mymem);
    if (arr == NULL) goto fail;
    newobj = (PyObject*)PyObject_New(VectorWrapper<T>, &VectorWrapperType);
    if (newobj == NULL) goto fail;
    ((VectorWrapper<T> *)newobj)->obj = obj;
    PyArray_SetBaseObject((PyArrayObject*)arr, newobj);
    return arr;
fail:
    delete obj; // FIXME Error Handling!?
    Py_XDECREF(arr);
    return NULL;
}

template <typename T>
inline PyArrayObject* wrapMap(Map<T>* obj) {
    int nd=3;
    npy_intp dims[3]={obj->z(), obj->y(), obj->x()};
    PyObject* newobj=NULL;
    PyArrayObject* arr=NULL;
    void *mymem = (void*)(obj->rawX());
    arr = (PyArrayObject*)PyArray_SimpleNewFromData(nd, dims, getTypeObject<T>(), mymem);
    if (arr == NULL) goto fail;
    newobj = (PyObject*)PyObject_New(MapWrapper<T>, &MapWrapperType);
    if (newobj == NULL) goto fail;
    ((MapWrapper<T> *)newobj)->obj = obj;
    PyArray_SetBaseObject((PyArrayObject*)arr, newobj);
    return arr;
fail:
    delete obj; // FIXME Error Handling!?
    Py_XDECREF(arr);
    return NULL;
}

template <typename T> 
static int npyToMatrix(PyArrayObject* array, Matrix<T>& matrix, string obj_name) {
    if (array==NULL) {
        return 1;
    }
    if(!(PyArray_NDIM(array) == 2 &&
                PyArray_TYPE(array) == getTypeObject<T>() &&
                (PyArray_FLAGS(array) & NPY_ARRAY_C_CONTIGUOUS))) {
        PyErr_SetString(PyExc_TypeError, (obj_name + " should be c-contiguous 2D "+getTypeName<T>()+" array").c_str());
        return 0;
    }
    
    T *rawX =  reinterpret_cast<T*>(PyArray_DATA(array));
    const npy_intp *shape = PyArray_DIMS(array);
    npy_intp n = shape[0];
    npy_intp m = shape[1];
    
    matrix.setData(rawX, m, n);
    return 1;
}

template <typename T>
static int npyToVector(PyArrayObject* array, Vector<T>& matrix, string obj_name) {
    if (array==NULL) {
        return 1;
    }
    T *rawX =  reinterpret_cast<T*>(PyArray_DATA(array));
    const npy_intp *shape = PyArray_DIMS(array);
    npy_intp n = shape[0];
    
    if(!(PyArray_NDIM(array) == 1 &&
                PyArray_TYPE(array) == getTypeObject<T>() &&
                (PyArray_FLAGS(array) & NPY_ARRAY_C_CONTIGUOUS))) {
        PyErr_SetString(PyExc_TypeError, (obj_name + " should be c-contiguous 1D "+getTypeName<T>()+" array").c_str());
        return 0;
    }
    matrix.setData(rawX, n);
    return 1;
}

template <typename T>
static int npyToMap(PyArrayObject* array, Map<T>& matrix, string obj_name) {
    if (array==NULL) {
        return 1;
    }
    T *rawX =  reinterpret_cast<T*>(PyArray_DATA(array));
    const npy_intp *shape = PyArray_DIMS(array);
    npy_intp z = shape[0];
    npy_intp y = shape[1];
    npy_intp x = shape[2];
    
    if(PyArray_NDIM(array) != 3) {
        PyErr_SetString(PyExc_TypeError, (obj_name + " should be 3D array.").c_str());
        return 0;
    }
    if (PyArray_TYPE(array) != getTypeObject<T>()) {
        PyErr_SetString(PyExc_TypeError, (obj_name + " has wrong data type.").c_str());
        return 0;
    }
        if (!(PyArray_FLAGS(array) & NPY_ARRAY_C_CONTIGUOUS)) {
        PyErr_SetString(PyExc_TypeError, (obj_name + " is not contiguous.").c_str());
        return 0;
    }
    matrix.setData(rawX, x, y, z);
    return 1;
}

template <typename T>
static int sequenceToVector(PyObject* seq, std::vector<T>& res) {
    if (!PySequence_Check(seq)) {
        PyErr_SetString(PyExc_TypeError, "input should be a sequence");
        return 0;
    }

    int n = PySequence_Size(seq);
    res.resize(n);

    for (int i=0; i<n; ++i) {
        PyObject* elem_i = PySequence_GetItem(seq, i);
        // FIXME this should be possible to do for neasrly arbitrary types
        if (!PyLong_Check(elem_i)) {
            PyErr_SetString(PyExc_TypeError, "Expeted integer elements only!");
            return 0;
        }
        res[i] = PyLong_AsLong(elem_i);
    }
    return 1;
};
