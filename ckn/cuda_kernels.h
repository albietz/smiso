#ifndef CUDA_KERNELS_H
#define CUDA_KERNELS_H
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#ifdef CUDNN
#include <cudnn.h>
#endif

template <typename T>
void cuda_exp(const int n, T* y);

template <typename T>
void cuda_add_exp(const int n, T* y, T b);


template <typename T>
void cuda_inv(const int n, T* y);

template <typename T>
void cuda_inv_sqrt(const int n, T* y);

template <typename T>
void cuda_sqrt(const int n, T* y);

template <typename T>
void cuda_inv_thrs(const int n, T* x, T* y, const T thrs);

template <typename T>
void cuda_inv_sqrt_thrs(const int n, T* x, T* y, T* y2, const T thrs);

template <typename T>
void cuda_inv_sqrt_add(const int n, T* x, const T a);

template <typename T>
void cuda_thrsmax(const int n, T* y, const T nu);

template <typename T>
void cuda_set(const int n, T* y, const T nu);

template <typename T>
void cuda_sqr(const int n, T* y);

template <typename T>
void cuda_sqr(const int n, T*X, T* y);

template <typename T>
void cuda_custom(const int n, T* y, const T thrs);

template <typename T>
void cuda_custom2(const int n, T* y, const T thrs);

template <typename T>
void cuda_custom3(const int n, T* x, T* y, const T scal);

template <typename T>
void cuda_custom4(const int n, T* x1, T* x2, T* x3, T* x4);

template <typename T>
void cuda_custom5(const int n, T* x1, T* x2);

template <typename T>
void cuda_custom6(const int n, T* x1, T* x2);

template <typename T>
void cuda_custom7(const int n, T* x1, T* x2);

template <typename T>
void cuda_mult(const int n, T* y, T* x);

template <typename T>
void cuda_custom_relu(const int n, T* z, T* y, T* x);


#endif
