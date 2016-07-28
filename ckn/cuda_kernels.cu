#include "cuda_kernels.h"
#define MIN(a,b) (((a) > (b)) ? (b) : (a))

// Inspired from caffee implementation

#define CUDA_KERNEL_LOOP(i, n) \
   for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
         i < (n); \
         i += blockDim.x * gridDim.x)

#if __CUDA_ARCH__ >= 200
#define CUDA_NUM_THREADS 1024
#else
#define CUDA_NUM_THREADS 512
#endif

#define MAX_BLOCKS 65535
#define MAX_SIZE MAX_BLOCKS*CUDA_NUM_THREADS 

inline int GET_BLOCKS(const int N) {
   return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
};

template <typename T>
__global__ void kernel_exp(const int n, T* data);

template <>
__global__ void kernel_exp(const int n, double* data) {
   CUDA_KERNEL_LOOP(index, n) {
      data[index] = exp(data[index]);
   }
};

template <>
__global__ void kernel_exp(const int n, float* data) {
   CUDA_KERNEL_LOOP(index, n) {
      data[index] = expf(data[index]);
   }
};
template <typename T>
__global__ void kernel_add_exp(const int n, T* data, T b);

template <>
__global__ void kernel_add_exp(const int n, double* data, double b) {
   CUDA_KERNEL_LOOP(index, n) {
      data[index] = exp(data[index]+b);
   }
};

template <>
__global__ void kernel_add_exp(const int n, float* data, float b) {
   CUDA_KERNEL_LOOP(index, n) {
      data[index] = expf(data[index]+b);
   }
};

template <typename T>
__global__ void kernel_thrsmax(const int n, T* data, const T nu);

template <>
__global__ void kernel_thrsmax(const int n, double* data, const double nu) {
   CUDA_KERNEL_LOOP(index, n) {
      data[index] = fmax(data[index],nu);
   }
};

template <>
__global__ void kernel_thrsmax(const int n, float* data, const float nu) {
   CUDA_KERNEL_LOOP(index, n) {
      data[index] = fmaxf(data[index],nu);
   }
};

template <typename T>
__global__ void kernel_set(const int n, T* data, const T nu) {
   CUDA_KERNEL_LOOP(index, n) {
      data[index] = nu;
   }
}

template <typename T>
__global__ void kernel_inv(const int n, T* data) {
   CUDA_KERNEL_LOOP(index, n) {
      data[index] = T(1.0)/(data[index]);
   }
}

template <typename T>
__global__ void kernel_mult(const int n, T* data, T* data2) {
   CUDA_KERNEL_LOOP(index, n) {
      data[index] = (data[index]) * (data2[index]);
   }
}



template <typename T>
__global__ void kernel_sqr(const int n, T* data) {
   CUDA_KERNEL_LOOP(index, n) {
      data[index] = data[index]*data[index];
   }
}

template <typename T>
__global__ void kernel_sqr(const int n, T*in, T* out) {
   CUDA_KERNEL_LOOP(index, n) {
      out[index] = in[index]*in[index];
   }
}

template <typename T>
__global__ void kernel_inv_thrs(const int n, T* data, T* out, const T thrs);

template <>
__global__ void kernel_inv_thrs(const int n, double* data, double* out, const double thrs) {
   CUDA_KERNEL_LOOP(index, n) {
      out[index] = double(1.0)/fmax((data[index]),thrs);
   }
};

template <>
__global__ void kernel_inv_thrs(const int n, float* data, float* out, const float thrs) {
   CUDA_KERNEL_LOOP(index, n) {
      out[index] = float(1.0)/fmaxf((data[index]),thrs);
   }
};

template <typename T>
__global__ void kernel_inv_sqrt_thrs(const int n, T* data, T* out, T* out2, const T thrs);

template <>
__global__ void kernel_inv_sqrt_thrs(const int n, double* data, double* out, double* out2, const double thrs) {
   CUDA_KERNEL_LOOP(index, n) {
      out[index] = (sqrt(data[index]));
      out2[index] = double(1.0)/fmax((out[index]),thrs);
   }
};

template <>
__global__ void kernel_inv_sqrt_thrs(const int n, float* data, float* out, float* out2, const float thrs) {
   CUDA_KERNEL_LOOP(index, n) {
      out[index] = (sqrtf(data[index]));
      out2[index] = float(1.0)/fmaxf((out[index]),thrs);
   }
};


template <typename T>
__global__ void kernel_inv_sqrt_add(const int n, T* data, const T thrs);

template <>
__global__ void kernel_inv_sqrt_add(const int n, double* data, const double thrs) {
   CUDA_KERNEL_LOOP(index, n) {
      data[index] = double(1.0)/(sqrt(sqrt(data[index]+thrs)));
   }
};

template <>
__global__ void kernel_inv_sqrt_add(const int n, float* data, const float thrs) {
   CUDA_KERNEL_LOOP(index, n) {
      data[index] = float(1.0)/sqrtf(sqrtf(data[index]+thrs));
   }
};





template <typename T>
__global__ void kernel_custom(const int n, T* data, const T thrs);

template <>
__global__ void kernel_custom(const int n, double* data, const double thrs) {
   CUDA_KERNEL_LOOP(index, n) {
      data[index] = thrs/fmax(sqrt(data[index]),thrs) - 1.0;
   }
};

template <>
__global__ void kernel_custom(const int n, float* data, const float thrs) {
   CUDA_KERNEL_LOOP(index, n) {
      data[index] = thrs/fmaxf(sqrtf(data[index]),thrs) - float(1.0);
   }
};

template <typename T>
__global__ void kernel_inv_sqrt(const int n, T* data);

template <>
__global__ void kernel_inv_sqrt(const int n, double* data) {
   CUDA_KERNEL_LOOP(index, n) {
      data[index] = double(1.0)/(sqrt(data[index]));
   }
};

template <>
__global__ void kernel_inv_sqrt(const int n, float* data) {
   CUDA_KERNEL_LOOP(index, n) {
      data[index] = float(1.0)/(sqrtf(data[index]));
   }
};

template <typename T>
__global__ void kernel_sqrt(const int n, T* data);

template <>
__global__ void kernel_sqrt(const int n, double* data) {
   CUDA_KERNEL_LOOP(index, n) {
      data[index] = (sqrt(data[index]));
   }
};

template <>
__global__ void kernel_sqrt(const int n, float* data) {
   CUDA_KERNEL_LOOP(index, n) {
      data[index] = (sqrtf(data[index]));
   }
};

template <typename T>
__global__ void kernel_custom2(const int n, T* data, const T thrs);

template <>
__global__ void kernel_custom2(const int n, double* data, const double thrs) {
   CUDA_KERNEL_LOOP(index, n) {
      data[index] = thrs/fmax(sqrt(data[index]),thrs);
   }
};

template <>
__global__ void kernel_custom2(const int n, float* data, const float thrs) {
   CUDA_KERNEL_LOOP(index, n) {
      data[index] = thrs/fmaxf(sqrtf(data[index]),thrs);
   }
};

template <typename T>
__global__ void kernel_custom3(const int n, T* data, T* data2, const T scal);

template <>
__global__ void kernel_custom3(const int n, double* data, double* data2, const double scal) {
   CUDA_KERNEL_LOOP(index, n) {
      data[index]=scal*data2[index]*fmax(1.0-data[index]*data2[index],0);
   }
};

template <>
__global__ void kernel_custom3(const int n, float* data, float* data2, const float scal) {
   CUDA_KERNEL_LOOP(index, n) {
      data[index]=scal*data2[index]*fmaxf(1.0f-data[index]*data2[index],0.0f);
   }
};

template <typename T>
__global__ void kernel_custom4(const int n, T* data, T* data2, T* data3, T* data4) {
   CUDA_KERNEL_LOOP(index, n) {
      data[index]=data3[index]*data4[index]-data[index]*data2[index];
   }
};


template <typename T>
__global__ void kernel_custom5(const int n, T* data, T* data2) {
   CUDA_KERNEL_LOOP(index, n) {
      data[index]=data[index]*data2[index]*data2[index];
   }
};

template <typename T>
__global__ void kernel_custom6(const int n, T* data, T* data2) {
   CUDA_KERNEL_LOOP(index, n) {
      data[index]=fmax(T(1.0)-data[index]*data2[index],T(0));
      data[index]=data[index]*data[index];
   }
};

template <typename T>
__global__ void kernel_custom7(const int n, T* data, T* data2) {
   CUDA_KERNEL_LOOP(index, n) {
      data[index]=data[index]-data2[index];
      data[index]=T(0.5)*data[index]*data[index];
   }
};

template <typename T>
__global__ void kernel_custom_relu(const int n, T* x, T* y, T* z);

template <>
__global__ void kernel_custom_relu(const int n, double* x, double* y, double* z) {
   CUDA_KERNEL_LOOP(index, n) {
      double sign=y[index] > 0 & z[index] > 0;
      y[index]=y[index]*sign;
      z[index]=z[index]*sign;
      x[index]=y[index]*z[index];
   }
};

template <>
__global__ void kernel_custom_relu(const int n, float* x, float* y, float* z) {
   CUDA_KERNEL_LOOP(index, n) {
      float sign=y[index] > 0 & z[index] > 0;
      y[index]=y[index]*sign;
      z[index]=z[index]*sign;
      x[index]=y[index]*z[index];
   }
};



template <typename T>
void cuda_set(const int n, T* data, const T nu) {
   for (int i=0; i<n; i+=MAX_SIZE) {
      const int current_size=MIN(n-i,MAX_SIZE);
      kernel_set<T><<<GET_BLOCKS(current_size), CUDA_NUM_THREADS>>>(current_size, data+i,nu);
   }
};

template <typename T>
void cuda_thrsmax(const int n, T* data, const T nu) {
   for (int i=0; i<n; i+=MAX_SIZE) {
      const int current_size=MIN(n-i,MAX_SIZE);
      kernel_thrsmax<T><<<GET_BLOCKS(current_size), CUDA_NUM_THREADS>>>(current_size, data+i,nu);
   }
};

template <typename T>
void cuda_exp(const int n, T* data) {
   for (int i=0; i<n; i+=MAX_SIZE) {
      const int current_size=MIN(n-i,MAX_SIZE);
      kernel_exp<T><<<GET_BLOCKS(current_size), CUDA_NUM_THREADS>>>(current_size, data+i);
   }
};

template <typename T>
void cuda_add_exp(const int n, T* data, T b) {
   for (int i=0; i<n; i+=MAX_SIZE) {
      const int current_size=MIN(n-i,MAX_SIZE);
      kernel_add_exp<T><<<GET_BLOCKS(current_size), CUDA_NUM_THREADS>>>(current_size, data+i, b);
   }
};


template <typename T>
void cuda_sqr(const int n, T* data) {
   for (int i=0; i<n; i+=MAX_SIZE) {
      const int current_size=MIN(n-i,MAX_SIZE);
      kernel_sqr<T><<<GET_BLOCKS(current_size), CUDA_NUM_THREADS>>>(current_size, data+i);
   }
};

template <typename T>
void cuda_sqr(const int n, T* datain, T* dataout) {
   for (int i=0; i<n; i+=MAX_SIZE) {
      const int current_size=MIN(n-i,MAX_SIZE);
      kernel_sqr<T><<<GET_BLOCKS(current_size), CUDA_NUM_THREADS>>>(current_size, datain+i,dataout+i);
   }
};


template <typename T>
void cuda_mult(const int n, T* data, T* data2) {
   for (int i=0; i<n; i+=MAX_SIZE) {
      const int current_size=MIN(n-i,MAX_SIZE);
      kernel_mult<T><<<GET_BLOCKS(current_size), CUDA_NUM_THREADS>>>(current_size, data+i, data2+i);
   }
};

template <typename T>
void cuda_inv_thrs(const int n, T* data, T* out, T thrs) {
   for (int i=0; i<n; i+=MAX_SIZE) {
      const int current_size=MIN(n-i,MAX_SIZE);
      kernel_inv_thrs<T><<<GET_BLOCKS(current_size), CUDA_NUM_THREADS>>>(current_size, data+i,out+i,thrs);
   }
};

template <typename T>
void cuda_inv_sqrt_thrs(const int n, T* data, T* out, T* out2, T thrs) {
   for (int i=0; i<n; i+=MAX_SIZE) {
      const int current_size=MIN(n-i,MAX_SIZE);
      kernel_inv_sqrt_thrs<T><<<GET_BLOCKS(current_size), CUDA_NUM_THREADS>>>(current_size, data+i,out+i,out2+i,thrs);
   }
};

template <typename T>
void cuda_inv_sqrt_add(const int n, T* data, T thrs) {
   for (int i=0; i<n; i+=MAX_SIZE) {
      const int current_size=MIN(n-i,MAX_SIZE);
      kernel_inv_sqrt_add<T><<<GET_BLOCKS(current_size), CUDA_NUM_THREADS>>>(current_size, data+i,thrs);
   }
};


template <typename T>
void cuda_inv(const int n, T* data) {
   for (int i=0; i<n; i+=MAX_SIZE) {
      const int current_size=MIN(n-i,MAX_SIZE);
      kernel_inv<T><<<GET_BLOCKS(current_size), CUDA_NUM_THREADS>>>(current_size, data+i);
   }
};

template <typename T>
void cuda_inv_sqrt(const int n, T* data) {
   for (int i=0; i<n; i+=MAX_SIZE) {
      const int current_size=MIN(n-i,MAX_SIZE);
      kernel_inv_sqrt<T><<<GET_BLOCKS(current_size), CUDA_NUM_THREADS>>>(current_size, data+i);
   }
};

template <typename T>
void cuda_sqrt(const int n, T* data) {
   for (int i=0; i<n; i+=MAX_SIZE) {
      const int current_size=MIN(n-i,MAX_SIZE);
      kernel_sqrt<T><<<GET_BLOCKS(current_size), CUDA_NUM_THREADS>>>(current_size, data+i);
   }
};

template <typename T>
void cuda_custom(const int n, T* data, const T thrs) {
   for (int i=0; i<n; i+=MAX_SIZE) {
      const int current_size=MIN(n-i,MAX_SIZE);
      kernel_custom<T><<<GET_BLOCKS(current_size), CUDA_NUM_THREADS>>>(current_size, data+i, thrs);
   }
};

template <typename T>
void cuda_custom2(const int n, T* data, const T thrs) {
   for (int i=0; i<n; i+=MAX_SIZE) {
      const int current_size=MIN(n-i,MAX_SIZE);
      kernel_custom2<T><<<GET_BLOCKS(current_size), CUDA_NUM_THREADS>>>(current_size, data+i, thrs);
   }
};

template <typename T>
void cuda_custom3(const int n, T* data, T* data2, const T scal) {
   for (int i=0; i<n; i+=MAX_SIZE) {
      const int current_size=MIN(n-i,MAX_SIZE);
      kernel_custom3<T><<<GET_BLOCKS(current_size), CUDA_NUM_THREADS>>>(current_size, data+i, data2+i, scal);
   }
};

template <typename T>
void cuda_custom4(const int n, T* data, T* data2, T* data3, T* data4) {
   for (int i=0; i<n; i+=MAX_SIZE) {
      const int current_size=MIN(n-i,MAX_SIZE);
      kernel_custom4<T><<<GET_BLOCKS(current_size), CUDA_NUM_THREADS>>>(current_size, data+i, data2+i, data3+i, data4+i);
   }
};

template <typename T>
void cuda_custom5(const int n, T* data, T* data2) {
   for (int i=0; i<n; i+=MAX_SIZE) {
      const int current_size=MIN(n-i,MAX_SIZE);
      kernel_custom5<T><<<GET_BLOCKS(current_size), CUDA_NUM_THREADS>>>(current_size, data+i, data2+i);
   }
};

template <typename T>
void cuda_custom6(const int n, T* data, T* data2) {
   for (int i=0; i<n; i+=MAX_SIZE) {
      const int current_size=MIN(n-i,MAX_SIZE);
      kernel_custom6<T><<<GET_BLOCKS(current_size), CUDA_NUM_THREADS>>>(current_size, data+i, data2+i);
   }
};

template <typename T>
void cuda_custom7(const int n, T* data, T* data2) {
   for (int i=0; i<n; i+=MAX_SIZE) {
      const int current_size=MIN(n-i,MAX_SIZE);
      kernel_custom7<T><<<GET_BLOCKS(current_size), CUDA_NUM_THREADS>>>(current_size, data+i, data2+i);
   }
};


template <typename T>
void cuda_custom_relu(const int n, T* x, T* y, T* z) {
   for (int i=0; i<n; i+=MAX_SIZE) {
      const int current_size=MIN(n-i,MAX_SIZE);
      kernel_custom_relu<T><<<GET_BLOCKS(current_size), CUDA_NUM_THREADS>>>(current_size, x+i, y+i, z+i);
   }
};


template void cuda_set (int, double*,double);
template void cuda_set (int, float*,float);

template void cuda_thrsmax (int, double*,double);
template void cuda_thrsmax (int, float*,float);

template void cuda_exp (int, double*);
template void cuda_exp (int, float*);

template void cuda_add_exp (int, double*, double);
template void cuda_add_exp (int, float*, float);

template void cuda_inv (int, double*);
template void cuda_inv (int, float*);

template void cuda_inv_thrs (int, double*,double*, double);
template void cuda_inv_thrs (int, float*,float*, float);

template void cuda_inv_sqrt_thrs (int, double*,double*, double*, double);
template void cuda_inv_sqrt_thrs (int, float*,float*, float*, float);

template void cuda_inv_sqrt (int, double*);
template void cuda_inv_sqrt (int, float*);

template void cuda_inv_sqrt_add (int, double*,double);
template void cuda_inv_sqrt_add (int, float*,float);

template void cuda_sqrt (int, double*);
template void cuda_sqrt (int, float*);

template void cuda_sqr (int, double*);
template void cuda_sqr (int, float*);
template void cuda_sqr (int, double*, double*);
template void cuda_sqr (int, float*, float*);

template void cuda_custom (int, double*,double);
template void cuda_custom (int, float*,float);

template void cuda_custom2 (int, double*,double);
template void cuda_custom2 (int, float*,float);

template void cuda_custom3 (int, double*,double*,double);
template void cuda_custom3 (int, float*,float*,float);

template void cuda_custom4 (int, double*,double*,double*, double*);
template void cuda_custom4 (int, float*,float*, float*,float*);

template void cuda_custom5 (int, double*,double*);
template void cuda_custom5 (int, float*,float*);

template void cuda_custom6 (int, double*,double*);
template void cuda_custom6 (int, float*,float*);

template void cuda_custom7 (int, double*,double*);
template void cuda_custom7 (int, float*,float*);

template void cuda_mult (int, double*, double*);
template void cuda_mult (int, float*, float*);

template void cuda_custom_relu (int, double*, double*, double*);
template void cuda_custom_relu (int, float*, float*, float*);
