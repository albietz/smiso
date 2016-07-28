#ifndef COMMON_CUDNN_H
#define COMMON_CUDNN_H
#include <cudnn.h>
#include "common.h"

#define CHECK_NULL(x) if ((x) == NULL) \
                               cout << #x << " == NULL!" << endl;
/// a few generic functions for initializing cuda and for debugging

void init_cuda(const int device, const bool cudnn = false, const bool cusolver = false) {
   cout << "Use Device " << device << endl;
   Timer time;
   time.start();
   checkCudaErrors(cudaSetDevice(device));
   checkCudaErrors(cublasCreate(&handle));
   if (cusolver)
      checkCusolver(cusolverDnCreate(&cusolver_handle));
   if (cudnn)
      checkCUDNN(cudnnCreate(&cudnn_handle));
   time.stop();
   cout << "Time for initialization of CUDA: " << endl;
   time.printElapsed();
};

void destroy_cuda(const bool cudnn = false, const bool cusolver = false) {
   checkCudaErrors(cublasDestroy(handle));
   if (cusolver)
      checkCusolver(cusolverDnDestroy(cusolver_handle));
   if (cudnn)
      checkCUDNN(cudnnDestroy(cudnn_handle));
};

void print_convolution3d(cudnnConvolutionDescriptor_t& conv) {
   cudnnConvolutionMode_t mode;
   cudnnDataType_t dataType;
   int pad[3];
   int u[3];
   int up[3];
   int arraysize;
   checkCUDNN(cudnnGetConvolutionNdDescriptor(conv,3,&arraysize,pad,u,up,&mode,&dataType));
   cout << "*** Print convolution characteristics ***" << endl;
   cout << "Num dims " << arraysize << endl;
   cout << "Pad " << pad[0] << " " << pad[1] << " "  << pad[2] << endl;
   cout << "U " << u[0] << " " << u[1] << " "  << u[2] << endl;
   cout << "Up " << up[0] << " " << up[1] << " "  << up[2] << endl;
   cout << "size conv: " << arraysize << endl;
   cout << "Format/Type: " << mode << " x " << dataType << endl;
};

void print_filter3d(cudnnFilterDescriptor_t& filter) {
   cudnnDataType_t dataType;
   cudnnTensorFormat_t format;
   int dims[5];
   int num_dims;
   checkCUDNN(cudnnGetFilterNdDescriptor(filter,5,&dataType,&format,&num_dims,dims));
   cout << "*** Print filter characteristics ***" << endl;
   cout << "Num dims " << num_dims << endl;
   cout << "Size " << dims[0] << "  " << dims[1] << "  " << dims[2]<< "  " << dims[3]<< "  " << dims[4]<< endl;
   cout << "Format/Type: " << format << " x " << dataType << endl;
};

void print_tensor3d(cudnnTensorDescriptor_t& tensor) {
   cudnnDataType_t dataType;
   int dims[5];
   int strides[5];
   int num_dims;
   checkCUDNN(cudnnGetTensorNdDescriptor(tensor,5,&dataType,&num_dims,dims,strides));
   cout << "*** Print tensor characteristics ***" << endl;
   cout << "Num dims " << num_dims << endl;
   cout << "Size " << dims[0] << "  " << dims[1] << "  " << dims[2]<< "  " << dims[3]<< "  " << dims[4]<< endl;
   cout << "Strides " << strides[0] << "  " << strides[1] << "  " << strides[2]<< "  " << strides[3]<< "  " << strides[4]<< endl;
   cout << "Format/Type: " << dataType << endl;
};

void print_convolution(cudnnConvolutionDescriptor_t& conv) {
   cudnnConvolutionMode_t mode;
   int padx, pady, ux, uy, upx, upy;
   checkCUDNN(cudnnGetConvolution2dDescriptor(conv,&padx,&pady,&ux,&uy,&upx,&upy,&mode));
   cout << "*** Print convolution characteristics ***" << endl;
   cout << "Size " << padx << " " << pady << " "  << ux << "  " << uy << " " << upx<< " " << upy << endl;
};

void print_filter(cudnnFilterDescriptor_t& filter) {
   cudnnDataType_t dataType;
   cudnnTensorFormat_t format;
   int k,c,h,w;
   checkCUDNN(cudnnGetFilter4dDescriptor(filter,&dataType,&format,&k,&c,&h,&w));
   cout << "*** Print filter characteristics ***" << endl;
   cout << "Input channels " << c << endl;
   cout << "Output channels " << k << endl;
   cout << "Size " << w << " x " << h << endl;
   cout << "Format/Type: " << format << " x " << dataType << endl;
};

void print_tensor(cudnnTensorDescriptor_t& tensor) {
   cudnnDataType_t dataType;
   int n,c,h,w;
   int ns,cs,hs,ws;
   checkCUDNN(cudnnGetTensor4dDescriptor(tensor,&dataType,&n,&c,&h,&w,&ns,&cs,&hs,&ws));
   cout << "*** Print tensor characteristics ***" << endl;
   cout << "Size " << n << " " << c << " "  << h << " x " << w << endl;
   cout << "Strides " << ns << " " << cs << " "  << hs << " " << ws << endl;
   cout << "Format/Type: " << dataType << endl;
};

/// the abstract network layer class

#define USING_NETWORK_LAYER \
   using NetworkLayer<T>::_hi; \
   using NetworkLayer<T>::_ho; \
   using NetworkLayer<T>::_wi; \
   using NetworkLayer<T>::_wo; \
   using NetworkLayer<T>::_ci; \
   using NetworkLayer<T>::_co; \
   using NetworkLayer<T>::_n; \
   using NetworkLayer<T>::_pinput; \
   using NetworkLayer<T>::_poutput; \
   using NetworkLayer<T>::_input; \
   using NetworkLayer<T>::_output; \
   using NetworkLayer<T>::_pdelta_input; \
   using NetworkLayer<T>::_pdelta_output; \
   using NetworkLayer<T>::_workspace_size; \
   using NetworkLayer<T>::_pworkspace; \
   using NetworkLayer<T>::_verbose; \

template <typename T> class NetworkLayer {
   public:
      NetworkLayer(const int verbose=0) {
         _workspace_size=0; 
         _allocate_input=false;
         _allocate_output=false;
         _allocate_delta=false;
         checkCUDNN(cudnnCreateTensorDescriptor(&_input));
         checkCUDNN(cudnnCreateTensorDescriptor(&_output));
         _verbose=verbose;
      };
      virtual ~NetworkLayer() {  
         checkCUDNN(cudnnDestroyTensorDescriptor(_input));
         checkCUDNN(cudnnDestroyTensorDescriptor(_output));
         if (_workspace_size > 0)
            checkCudaErrors(cudaFree(_pworkspace));    
         if (_allocate_output)
            checkCudaErrors(cudaFree(_poutput));
         if (_allocate_input)
            checkCudaErrors(cudaFree(_pinput));
         if (_allocate_delta) 
            checkCudaErrors(cudaFree(_pdelta_output));
      };

      void resize_workspace(const size_t size) {
         if (size > _workspace_size) {
            if (_workspace_size > 0) 
               checkCudaErrors(cudaFree(_pworkspace));    
            _workspace_size=size;
            checkCudaErrors(cudaMalloc(&_pworkspace,_workspace_size));    
         }
      };
      
      void allocate_delta() {
         _allocate_delta=true;
         checkCudaErrors(cudaMalloc(&_pdelta_output,sizeof(T)*_ho*_wo*_co*_n));
      };

      void upload_input(T* input, const int nim) {
         checkCudaErrors(cudaMemcpyAsync(_pinput,input,sizeof(T)*_wi*_hi*_ci*nim,cudaMemcpyHostToDevice));
      };

      virtual void set_input(const int h, const int w, const int c, const int n) {
         _wi=w; _hi=h; _ci=c; _n=n;
         if (_verbose)
            cout << "Input dimensions: " << _ci << " x " <<
               _wi << " x " << _hi << " x " << _n << endl;
         checkCUDNN(cudnnSetTensor4dDescriptor(_input, CUDNN_TENSOR_NHWC,
                  CUDNN_DATA_FLOAT, _n, _ci, _hi, _wi));
         this->allocate_input();
      };

      virtual void set_input(NetworkLayer<T>& input) {
         _wi=input.w(); _hi=input.h(); _ci=input.p(); _n=input.n();
         if (_verbose)
            cout << "Input dimensions: " << _ci << " x " <<
               _wi << " x " << _hi << " x " << _n << endl;
         checkCUDNN(cudnnSetTensor4dDescriptor(_input, CUDNN_TENSOR_NHWC,
                  CUDNN_DATA_FLOAT, _n, _ci, _hi, _wi));
         _pinput=input.output();
      };

      virtual void set_output(const int h, const int w, const int c) {
         _wo=w; _ho=h; _co=c;
         if (_verbose)
            cout << "Output dimensions: " << _co << " x " <<
               _wo << " x " << _ho << " x " << _n << endl;
         PRINT_I(_n);
         PRINT_I(_co);
         PRINT_I(_ho);
         PRINT_I(_wo);
         checkCUDNN(cudnnSetTensor4dDescriptor(_output, CUDNN_TENSOR_NHWC,
                  CUDNN_DATA_FLOAT, _n, _co, _ho, _wo));
         this->allocate_output();
      };

      virtual void initialize_backward(T* delta = nullptr) = 0;
      virtual void backward(CudaMatrix<T>* grad = nullptr) = 0;
      virtual void forward() = 0;
      
      int h() const { return _ho; };
      int w() const { return _wo; };
      int p() const { return _co; };
      int n() const { return _n; };
      int size_map() const { return _ho*_wo*_co; };
      T* output() const { return _poutput; };
      T* delta() const { return _pdelta_output; };
      size_t workspace_size() const { return _workspace_size; };

   protected:
      int _hi;
      int _ho;
      int _wi;
      int _wo;
      int _ci;
      int _co;
      int _n;
      
      cudnnTensorDescriptor_t _input, _output;
      T* _pinput;
      T* _poutput;
      T* _pdelta_input;
      T* _pdelta_output;
      
      void* _pworkspace;
      size_t _workspace_size;

      int _verbose;

   private:
      void allocate_output() {
         _allocate_output=true;
         checkCudaErrors(cudaMalloc(&_poutput,sizeof(T)*_ho*_wo*_co*_n));
      };
      void allocate_input() {
         _allocate_input=true;
         checkCudaErrors(cudaMalloc(&_pinput,sizeof(T)*_hi*_wi*_ci*_n));
      };

      bool _allocate_input;
      bool _allocate_output;
      bool _allocate_delta;
};
 

template <typename T> class ConvLayer : public NetworkLayer<T> {
   public:
      USING_NETWORK_LAYER;

      ConvLayer(const int verbose=0)
      : NetworkLayer<T>(verbose) { _initialized_backward=false; _initialized_forward=false; };
      virtual ~ConvLayer() { 
         if (_initialized_forward) {
            checkCUDNN(cudnnDestroyTensorDescriptor(_norms));
            checkCUDNN(cudnnDestroyTensorDescriptor(_ones));
            checkCUDNN(cudnnDestroyFilterDescriptor(_filters));
            checkCUDNN(cudnnDestroyFilterDescriptor(_filters_ones));
            checkCUDNN(cudnnDestroyConvolutionDescriptor(_conv));
            checkCUDNN(cudnnDestroyConvolutionDescriptor(_conv_ones));
            checkCudaErrors(cudaFree(_pW));    
            checkCudaErrors(cudaFree(_ptmp));
            checkCudaErrors(cudaFree(_ptmp2));
            checkCudaErrors(cudaFree(_pnorms));
            checkCudaErrors(cudaFree(_pinv_norms));
            checkCudaErrors(cudaFree(_pones));
         }
         if (_initialized_backward) {
            checkCUDNN(cudnnDestroyFilterDescriptor(_filters_ones2));
            checkCUDNN(cudnnDestroyConvolutionDescriptor(_conv_ones2));
            checkCUDNN(cudnnDestroyTensorDescriptor(_scals));
         }
      };

      virtual void set_input(const size_t h, const size_t w, const size_t c, const size_t n, const T alpha) {
         _alpha=alpha;
         cout << "Convolutional layer" << endl;
         NetworkLayer<T>::set_input(h,w,c,n);
      };

      virtual void set_input(NetworkLayer<T>& input, const T alpha) {
         _alpha=alpha;
         cout << "Convolutional layer" << endl;
         NetworkLayer<T>::set_input(input);
      };

      void set_filter(const size_t e, const size_t p, const bool zeropad) {
         _initialized_forward=true;
         _zero_padding = zeropad;
         _e=e;
         NetworkLayer<T>::set_output(zeropad ? _hi : _hi - e +1,zeropad ? _wi : _wi - e +1,p);

         /// create descriptors 
         checkCUDNN(cudnnCreateTensorDescriptor(&_norms));
         checkCUDNN(cudnnCreateTensorDescriptor(&_ones));
         checkCUDNN(cudnnCreateFilterDescriptor(&_filters));
         checkCUDNN(cudnnCreateFilterDescriptor(&_filters_ones));
         checkCUDNN(cudnnCreateConvolutionDescriptor(&_conv));
         checkCUDNN(cudnnCreateConvolutionDescriptor(&_conv_ones));
         // Set tensor sizes
         checkCUDNN(cudnnSetTensor4dDescriptor(_norms, CUDNN_TENSOR_NHWC,
                  CUDNN_DATA_FLOAT, _n, 1, _ho, _wo));
         checkCUDNN(cudnnSetTensor4dDescriptor(_ones, CUDNN_TENSOR_NHWC,
                  CUDNN_DATA_FLOAT, 1, _ci, _e, _e));
         // set filters sizes
         checkCUDNN(cudnnSetFilter4dDescriptor(_filters, CUDNN_DATA_FLOAT,
                  CUDNN_TENSOR_NHWC, _co, _ci, _e, _e));
         checkCUDNN(cudnnSetFilter4dDescriptor(_filters_ones, CUDNN_DATA_FLOAT,
                  CUDNN_TENSOR_NHWC, 1, _ci, _e, _e));
         // set convolution parameters
         if (_zero_padding) {
            const int pad=e/2; // will only work with odd filters TODO
            checkCUDNN(cudnnSetConvolution2dDescriptor(_conv,
                     pad, pad, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION));
            checkCUDNN(cudnnSetConvolution2dDescriptor(_conv_ones,
                     pad, pad, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION));
         } else {
            checkCUDNN(cudnnSetConvolution2dDescriptor(_conv,
                     0, 0, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION));
            checkCUDNN(cudnnSetConvolution2dDescriptor(_conv_ones,
                     0, 0, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION));
         }
         /// Set up convolution algorithm
         checkCUDNN(cudnnGetConvolutionForwardAlgorithm(cudnn_handle, _input,
                  _filters, _conv, _output,
                  CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &_conv_algo));
         _conv_algo=CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
         checkCUDNN(cudnnGetConvolutionForwardAlgorithm(cudnn_handle, _input,
                  _filters_ones, _conv_ones, _norms,
                  CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &_conv_ones_algo));
         _conv_ones_algo=CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;

         /// Compute workspace size
         size_t size, size2 = 0;
         checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnn_handle,
                  _input, _filters, _conv, _output, _conv_algo, &size));
         checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnn_handle,
                  _input, _filters_ones, _conv_ones, _norms,
                  _conv_ones_algo, &size2));

         /// allocate workspace 
         this->resize_workspace(MAX(size,size2));
         checkCudaErrors(cudaMalloc(&_pones,sizeof(T)*_ci*_e*_e));
         float param_one = 1.0f;
         checkCUDNN(cudnnSetTensor(cudnn_handle,_ones,_pones,&param_one));
         checkCudaErrors(cudaMalloc(&_ptmp,sizeof(T)*MAX(_hi*_wi*_ci,_ho*_wo*_co)*_n));
         checkCudaErrors(cudaMalloc(&_ptmp2,sizeof(T)*_ho*_wo*_co*_n));
         checkCudaErrors(cudaMalloc(&_pW,sizeof(T)*_ci*_e*_e*_co));
         checkCudaErrors(cudaMalloc(&_pnorms,sizeof(T)*_ho*_wo*_n));
         checkCudaErrors(cudaMalloc(&_pinv_norms,sizeof(T)*_ho*_wo*_n));
      };

      void set_data(T* W, T* b) {
         _b=*b;
         checkCudaErrors(cudaMemcpyAsync(_pW,W,sizeof(T)*_ci*_e*_e*_co,cudaMemcpyHostToDevice));
      }

      virtual void initialize_backward(T* delta = nullptr) {
         _initialized_backward=true;
         _pdelta_input=delta;
         this->allocate_delta();
         checkCUDNN(cudnnGetConvolutionBackwardDataAlgorithm(cudnn_handle, 
                  _filters,_output,_conv,_input,
                  CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST,0,&_conv_backward_data_algo));
         PRINT_I(_conv_backward_data_algo);
         checkCUDNN(cudnnGetConvolutionBackwardFilterAlgorithm(cudnn_handle, 
                  _input,_output,_conv,_filters,
                  CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST,0,&_conv_backward_filter_algo));
         PRINT_I(_conv_backward_filter_algo);

         checkCUDNN(cudnnCreateTensorDescriptor(&_scals));
         checkCUDNN(cudnnSetTensor4dDescriptor(_scals, CUDNN_TENSOR_NHWC,
                  CUDNN_DATA_FLOAT, _n, 1, _hi, _wi));
         checkCUDNN(cudnnCreateFilterDescriptor(&_filters_ones2));
         checkCUDNN(cudnnSetFilter4dDescriptor(_filters_ones2, CUDNN_DATA_FLOAT,
                  CUDNN_TENSOR_NHWC, 1, 1, _e, _e));
         checkCUDNN(cudnnCreateConvolutionDescriptor(&_conv_ones2));
         if (_zero_padding) {
            const int pad=_e/2; 
            checkCUDNN(cudnnSetConvolution2dDescriptor(_conv_ones2,
                     pad, pad, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION));
         } else {
            checkCUDNN(cudnnSetConvolution2dDescriptor(_conv_ones2,
                     _e-1, _e-1, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION));
         }
         _conv_ones2_algo=CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;

         /// resize workspace if necessary
         size_t size;
         checkCUDNN(cudnnGetConvolutionBackwardFilterWorkspaceSize(cudnn_handle,
                  _input,_output,_conv,_filters,_conv_backward_filter_algo,&size));
         if (_pdelta_input != nullptr) {
            size_t size2=0;
            checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnn_handle,
                     _norms, _filters_ones2, _conv_ones2, _scals,
                     _conv_ones2_algo, &size2));
            size=MAX(size,size2);
            checkCUDNN(cudnnGetConvolutionBackwardDataWorkspaceSize(cudnn_handle,
                     _filters,_output,_conv,_input,_conv_backward_data_algo,&size2));
            size=MAX(size,size2);
         }
         this->resize_workspace(size);
         _vec_ones.resize(_co);
         _vec_ones.set(T(1.0));
      };

      virtual void forward() {
         /// compute the square of the elements
         float alpha = 1.0f, beta = 0.0f;
         cuda_sqr(_wi*_hi*_ci*_n,_pinput,_ptmp);

         /// compute the squared norms
         checkCUDNN(cudnnConvolutionForward(cudnn_handle, &alpha, _input,
                  _ptmp, _filters_ones, _pones, _conv_ones, _conv_ones_algo, 
                  _pworkspace, _workspace_size, &beta, _norms,_pnorms));

         /// compute the norms
         cuda_sqrt(_ho*_wo*_n,_pnorms);

         /// compute the inv norms
         cuda_inv_thrs(_ho*_wo*_n,_pnorms,_pinv_norms,T(EPS_NORM));

         /// perform the convolution
         checkCUDNN(cudnnConvolutionForward(cudnn_handle, &alpha, _input,
                  _pinput, _filters, _pW, _conv, _conv_algo,
                  _pworkspace, _workspace_size, &beta, _output,_ptmp));

         /// normalize
         checkCudaErrors(cublasSdgmm(handle,CUBLAS_SIDE_RIGHT,_co,_ho*_wo*_n,_ptmp,_co,_pinv_norms,1,_ptmp2,_co));

         /// take exponential - b
         cuda_add_exp(_co*_ho*_wo*_n,_ptmp2,_b);

         /// multiply back by the norm
         checkCudaErrors(cublasSdgmm(handle,CUBLAS_SIDE_RIGHT,_co,_ho*_wo*_n,_ptmp2,_co,_pnorms,1,_poutput,_co));
      };

      void backward(CudaMatrix<T>* grad = nullptr) {
         /// compute g(U)
         /// note that here, poutput = ptmp2*diag(pnorms),   size output
         /// note that here, ptmp = output of raw convolution, size output
         cuda_mult(_wo*_ho*_co*_n,_ptmp2,_pdelta_output); // ptmp2 contains B_j/alpha
         CudaMatrix<T> tmp2_mat(_ptmp2,_co,_ho*_wo*_n);
         float beta = 1.0f;
         float scal = _alpha/_n;
         checkCUDNN(cudnnConvolutionBackwardFilter(cudnn_handle, &scal, _input,
                  _pinput, _output, _ptmp2, _conv, _conv_backward_filter_algo,
                  _pworkspace, _workspace_size, &beta, _filters, grad->rawX()));
         
         if (_pdelta_input != nullptr) {
            /// compute h2(U)
            cuda_custom4(_wo*_ho*_co*_n,_ptmp,_ptmp2,_poutput,_pdelta_output); // ptmp contains the right variable
            CudaMatrix<T> tmp_mat(_ptmp,_co,_ho*_wo*_n);
            tmp_mat.multTrans(_vec_ones,_vec_tmp);
            cuda_custom5(_ho*_wo*_n,_vec_tmp.rawX(),_pinv_norms);
            T beta_zero = 0.0f;

            checkCUDNN(cudnnConvolutionForward(cudnn_handle, &beta, _norms,
                     _vec_tmp.rawX(), _filters_ones2, _pones, _conv_ones2, _conv_ones2_algo,
                     _pworkspace, _workspace_size, &beta_zero, _scals,_ptmp));
            checkCudaErrors(cublasSdgmm(handle,CUBLAS_SIDE_RIGHT,_ci,_hi*_wi*_n,_pinput,_ci,_ptmp,1,_pdelta_input,_ci));
            /// compute h1(U) and add the result to h2(U),   note that pW = alpha pZ
            checkCUDNN(cudnnConvolutionBackwardData(cudnn_handle, &beta, 
                     _filters,_pW,_output,_ptmp2,_conv,_conv_backward_data_algo,
                     _pworkspace, _workspace_size,&beta,_input,_pdelta_input));
         }
      };

      int size_filter() const { return _e*_e*_ci; };
      T* filter() const { return _pW; };

   private:
      /// for forward pass only
      bool _initialized_forward;
      int _e;
      bool _zero_padding;
      cudnnTensorDescriptor_t _norms, _ones;
      cudnnFilterDescriptor_t _filters, _filters_ones;
      cudnnConvolutionDescriptor_t _conv, _conv_ones;
      cudnnConvolutionFwdAlgo_t _conv_algo, _conv_ones_algo;
      T _b;
      T* _pW;
      T* _ptmp;
      T* _ptmp2;
      T* _pnorms;
      T* _pinv_norms;
      T* _pones;
      /// for backward pass only
      bool _initialized_backward;
      cudnnConvolutionBwdDataAlgo_t _conv_backward_data_algo;
      cudnnConvolutionBwdFilterAlgo_t _conv_backward_filter_algo;
      cudnnFilterDescriptor_t _filters_ones2;
      cudnnConvolutionDescriptor_t _conv_ones2;
      cudnnConvolutionFwdAlgo_t _conv_ones2_algo;
      cudnnTensorDescriptor_t _scals;
      T _alpha;
      CudaVector<T> _vec_ones, _vec_tmp;
};

template <typename T> class PoolLayer : public NetworkLayer<T> {
   public:
      PoolLayer(const int verbose=0)
      : NetworkLayer<T>(verbose) {  };
      virtual ~PoolLayer() { };

      virtual void set_input(NetworkLayer<T>& input, const int sub) = 0;

   protected:
      int _sub;
};

template <typename T> class GaussianPoolLayer : public PoolLayer<T> {
   public:
      USING_NETWORK_LAYER;
      using PoolLayer<T>::_sub; 

      GaussianPoolLayer(const int verbose=0)
      : PoolLayer<T>(verbose) { _initialized_forward=false; };
      virtual ~GaussianPoolLayer() { 
         if (_initialized_forward) {
            checkCudaErrors(cudaFree(_pfilt));
            checkCUDNN(cudnnDestroyTensorDescriptor(_input_conv));
            checkCUDNN(cudnnDestroyTensorDescriptor(_output_conv));
            checkCUDNN(cudnnDestroyFilterDescriptor(_filters));
            checkCUDNN(cudnnDestroyConvolutionDescriptor(_conv));
            checkCUDNN(cudnnDestroyTensorDescriptor(_output_nchw));
            checkCUDNN(cudnnDestroyTensorDescriptor(_input_nchw));
            checkCudaErrors(cudaFree(_ptmp_input));
            checkCudaErrors(cudaFree(_ptmp_output));
            if (_one_dim_conv) {
               checkCUDNN(cudnnDestroyTensorDescriptor(_tmp_conv));
               checkCUDNN(cudnnDestroyFilterDescriptor(_filters2));
               checkCUDNN(cudnnDestroyConvolutionDescriptor(_conv2));
               checkCudaErrors(cudaFree(_ptmp_conv));
            } 
         }
      };

      virtual void set_input(NetworkLayer<T>& input, const int sub) { 
         cout << "Subsampling layer" << endl;
         NetworkLayer<T>::set_input(input);
         _sub=sub;
         _one_dim_conv=sub > 3;
         NetworkLayer<T>::set_output(ceil((double(_hi))/sub),ceil((double(_hi))/sub),_ci);
         _initialized_forward=true;

         if (_ho != _wo)
            cout << "Non-square images are not supported at the moment" << endl;
         const T sigma = _sub/sqr<T>(T(2.0));
         const INTM h2 = ((_ho-1)*_sub+1);
         const bool even = (_hi-h2) % 2 == 1;
         const INTM s2 = even ? 2*_sub : 2*_sub+1;
         const INTM h3 = h2 + s2 -1;
         const INTM pad = (h3-_hi)/2;
         T* filt = new T[s2];
         if (even) {
            for(int ii=-_sub; ii<_sub; ++ii){
               const T ind=ii+T(0.5);
               filt[ii+_sub] = exp(-(1.0/(2*sigma*sigma))*ind*ind);
            }
         } else {
            for(int ii=-_sub; ii<=_sub; ++ii){
               filt[ii+_sub] = exp(-(1.0/(2*sigma*sigma))*ii*ii);
            }
         }
         if (_one_dim_conv) {
            T sum=0;
            for(int ii=0; ii<s2; ++ii) 
               sum+=filt[ii];
            for(int ii=0; ii<s2; ++ii) 
               filt[ii] /= sum;
            checkCudaErrors(cudaMalloc(&_pfilt,sizeof(T)*s2));
            checkCudaErrors(cudaMemcpy(_pfilt,filt,sizeof(T)*s2,cudaMemcpyHostToDevice));
            checkCudaErrors(cudaDeviceSynchronize());
         } else {
            T* filt2 = new T[s2*s2];
            T sum=0;
            for(int ii=0; ii<s2; ++ii) {
               for(int jj=0; jj<s2; ++jj) {
                  filt2[ii*s2+jj]=filt[ii]*filt[jj];
                  sum+=filt2[ii*s2+jj];
               }
            }
            for(int ii=0; ii<s2*s2; ++ii) 
               filt2[ii] /= sum;
            checkCudaErrors(cudaMalloc(&_pfilt,sizeof(T)*s2*s2));
            checkCudaErrors(cudaMemcpy(_pfilt,filt2,sizeof(T)*s2*s2,cudaMemcpyHostToDevice));
            checkCudaErrors(cudaDeviceSynchronize());
            delete[](filt2);
         }
         delete[](filt);

         checkCUDNN(cudnnCreateTensorDescriptor(&_input_conv));
         checkCUDNN(cudnnCreateTensorDescriptor(&_output_conv));
         checkCUDNN(cudnnSetTensor4dDescriptor(_input_conv, CUDNN_TENSOR_NCHW, 
                  CUDNN_DATA_FLOAT, _n*_ci, 1, _hi, _wi));
         checkCUDNN(cudnnSetTensor4dDescriptor(_output_conv, CUDNN_TENSOR_NCHW,
                  CUDNN_DATA_FLOAT, _n*_ci, 1, _ho, _wo));
         checkCudaErrors(cudaMalloc(&_ptmp_input,sizeof(T)*_hi*_wi*_ci*_n));
         checkCudaErrors(cudaMalloc(&_ptmp_output,sizeof(T)*_ho*_wo*_ci*_n));
         checkCUDNN(cudnnCreateTensorDescriptor(&_input_nchw));
         checkCUDNN(cudnnSetTensor4dDescriptor(_input_nchw, CUDNN_TENSOR_NCHW,
                  CUDNN_DATA_FLOAT, _n, _ci, _hi, _wi));
         checkCUDNN(cudnnCreateTensorDescriptor(&_output_nchw));
         checkCUDNN(cudnnSetTensor4dDescriptor(_output_nchw, CUDNN_TENSOR_NCHW,
                  CUDNN_DATA_FLOAT, _n, _ci, _ho, _wo));
         checkCUDNN(cudnnCreateConvolutionDescriptor(&_conv));
         checkCUDNN(cudnnCreateFilterDescriptor(&_filters));

         size_t size=0;
         if (_one_dim_conv) {
            checkCUDNN(cudnnSetConvolution2dDescriptor(_conv,
                     pad, 0, _sub, 1, 1, 1, CUDNN_CROSS_CORRELATION));
            checkCUDNN(cudnnCreateConvolutionDescriptor(&_conv2));
            checkCUDNN(cudnnSetConvolution2dDescriptor(_conv2,
                     0, pad, 1, _sub, 1, 1, CUDNN_CROSS_CORRELATION));
            _conv_algo=CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
            _conv_algo2=CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
            checkCUDNN(cudnnSetFilter4dDescriptor(_filters, CUDNN_DATA_FLOAT,
                     CUDNN_TENSOR_NCHW, 1, 1, s2, 1));
            checkCUDNN(cudnnCreateFilterDescriptor(&_filters2));
            checkCUDNN(cudnnSetFilter4dDescriptor(_filters2, CUDNN_DATA_FLOAT,
                     CUDNN_TENSOR_NCHW, 1, 1, 1, s2));
            checkCUDNN(cudnnCreateTensorDescriptor(&_tmp_conv));
            checkCUDNN(cudnnSetTensor4dDescriptor(_tmp_conv, CUDNN_TENSOR_NCHW,
                     CUDNN_DATA_FLOAT, _n*_ci, 1, _ho, _wi));
            checkCudaErrors(cudaMalloc(&_ptmp_conv,sizeof(T)*_ho*_wi*_ci*_n));
            checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnn_handle,
                     _input_conv, _filters, _conv, _tmp_conv, _conv_algo, &size));
            size_t size2=0;
            checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnn_handle,
                     _tmp_conv, _filters2, _conv2, _output_conv, _conv_algo2, &size2));
            size=MAX(size,size2);
         } else {
            checkCUDNN(cudnnSetFilter4dDescriptor(_filters, CUDNN_DATA_FLOAT,
                     CUDNN_TENSOR_NCHW, 1, 1, s2, s2));
            checkCUDNN(cudnnSetConvolution2dDescriptor(_conv,
                     pad, pad, _sub, _sub, 1, 1, CUDNN_CROSS_CORRELATION));
            _conv_algo=CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
            checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnn_handle,
                     _input_conv, _filters, _conv, _output_conv, _conv_algo, &_workspace_size));
         }
         this->resize_workspace(size);
      };

      virtual void initialize_backward(T* delta = nullptr) {
         _pdelta_input=delta;
         this->allocate_delta();
         size_t size=0;
         if (_one_dim_conv) {
            if (_ci*_n >= 65536) {
               _conv_backward_algo = CUDNN_CONVOLUTION_BWD_DATA_ALGO_1;
               _conv_backward_algo2 = CUDNN_CONVOLUTION_BWD_DATA_ALGO_1;
            } else {
               _conv_backward_algo = CUDNN_CONVOLUTION_BWD_DATA_ALGO_0;
               _conv_backward_algo2 = CUDNN_CONVOLUTION_BWD_DATA_ALGO_0;
            }
            checkCUDNN(cudnnGetConvolutionBackwardDataWorkspaceSize(cudnn_handle,
                     _filters,_tmp_conv,_conv,_input_conv,_conv_backward_algo,&size));
            size_t size2;
            checkCUDNN(cudnnGetConvolutionBackwardDataWorkspaceSize(cudnn_handle,
                     _filters2,_output_conv,_conv2,_tmp_conv,_conv_backward_algo2,&size2));
            size=MAX(size,size2);
         } else {
            checkCUDNN(cudnnGetConvolutionBackwardDataAlgorithm(cudnn_handle, 
                     _filters,_output_conv,_conv,_input_conv,
                     CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST,0,&_conv_backward_algo));
            if (_ci*_n >= 65536)
               _conv_backward_algo = CUDNN_CONVOLUTION_BWD_DATA_ALGO_1;
            PRINT_I(_ci*_n);
            PRINT_I(_conv_backward_algo);
            checkCUDNN(cudnnGetConvolutionBackwardDataWorkspaceSize(cudnn_handle,
                     _filters,_output_conv,_conv,_input_conv,_conv_backward_algo,&size));
         }
         this->resize_workspace(size);
      };

      virtual void forward() {
         float alpha = 1.0f, beta = 0.0f;
         /// change NHWC to NCHW
         checkCUDNN(cudnnTransformTensor(cudnn_handle,&alpha,_input,_pinput,&beta,_input_nchw,_ptmp_input));
         /// convolution spatial
         if (_one_dim_conv) {
            checkCUDNN(cudnnConvolutionForward(cudnn_handle, &alpha, _input_conv,
                     _ptmp_input, _filters, _pfilt, _conv, _conv_algo, 
                     _pworkspace, _workspace_size, &beta, _tmp_conv,_ptmp_conv));
            checkCUDNN(cudnnConvolutionForward(cudnn_handle, &alpha, _tmp_conv,
                     _ptmp_conv, _filters2, _pfilt, _conv2, _conv_algo2, 
                     _pworkspace, _workspace_size, &beta, _output_conv,_ptmp_output));
         } else {
            checkCUDNN(cudnnConvolutionForward(cudnn_handle, &alpha, _input_conv,
                     _ptmp_input, _filters, _pfilt, _conv, _conv_algo, 
                     _pworkspace, _workspace_size, &beta, _output_conv,_ptmp_output));
         }
         /// change back to NHWC
         checkCUDNN(cudnnTransformTensor(cudnn_handle,&alpha,_output_nchw,_ptmp_output,&beta,_output,_poutput));
      };

      virtual void backward(CudaMatrix<T>* grad = nullptr) {
         /// computes h(U)
         float alpha = 1.0f, beta = 0.0f;
         /// change NHWC to NCHW
         checkCUDNN(cudnnTransformTensor(cudnn_handle,&alpha,_output,_pdelta_output,&beta,_output_nchw,_ptmp_output));
         /// convolution spatial
         if (_one_dim_conv) {
            checkCUDNN(cudnnConvolutionBackwardData(cudnn_handle, &alpha,
                     _filters2, _pfilt, _output_conv,_ptmp_output, _conv2,
                     _conv_backward_algo2, _pworkspace,_workspace_size,
                     &beta,_tmp_conv,_ptmp_conv));
            checkCUDNN(cudnnConvolutionBackwardData(cudnn_handle, &alpha,
                     _filters, _pfilt, _tmp_conv,_ptmp_conv, _conv,
                     _conv_backward_algo, _pworkspace,_workspace_size,
                     &beta,_input_conv,_ptmp_input));
         } else {
            checkCUDNN(cudnnConvolutionBackwardData(cudnn_handle, &alpha,
                     _filters, _pfilt, _output_conv,_ptmp_output, _conv,
                     _conv_backward_algo, _pworkspace,_workspace_size,
                     &beta,_input_conv,_ptmp_input));
         }
         /// change back to NHWC
         checkCUDNN(cudnnTransformTensor(cudnn_handle,&alpha,_input_nchw,_ptmp_input,&beta,_input,_pdelta_input));
      };

   private:
      bool _initialized_forward;
      T* _ptmp_input;
      T* _ptmp_output;
      T* _pfilt;
      cudnnTensorDescriptor_t _input_nchw, _input_conv, _output_nchw, _output_conv;
      cudnnFilterDescriptor_t _filters;
      cudnnConvolutionDescriptor_t _conv;
      cudnnConvolutionFwdAlgo_t _conv_algo;

      /// variables for the for backward pass only
      cudnnConvolutionBwdDataAlgo_t _conv_backward_algo;

      /// variables for one_dimensional_filtering
      bool _one_dim_conv;
      cudnnFilterDescriptor_t _filters2;
      cudnnConvolutionDescriptor_t _conv2;
      cudnnConvolutionFwdAlgo_t _conv_algo2;
      cudnnTensorDescriptor_t _tmp_conv;
      cudnnConvolutionBwdDataAlgo_t _conv_backward_algo2;
      T* _ptmp_conv;
};


template <typename T> class SimplePoolLayer : public PoolLayer<T> {
   public:
      USING_NETWORK_LAYER;
      using PoolLayer<T>::_sub; 

      SimplePoolLayer(const int verbose=0)
      : PoolLayer<T>(verbose) { _initialized_forward=false; };
      virtual ~SimplePoolLayer() { 
         if (_initialized_forward) {
            checkCUDNN(cudnnDestroyPoolingDescriptor(_pooler));
         }
      };

      virtual void set_input(NetworkLayer<T>& input, const int sub) { 
         cout << "Simple Subsampling layer" << endl;
         NetworkLayer<T>::set_input(input);
         _sub=sub;
         NetworkLayer<T>::set_output(ceil((double(_hi))/sub),ceil((double(_hi))/sub),_ci);
         _initialized_forward=true;
         const INTM pady = ceil((_wo*_sub-_wi)/T(2.0));
         const INTM padx = ceil((_ho*_sub-_hi)/T(2.0));
         checkCUDNN(cudnnCreatePoolingDescriptor(&_pooler));
         checkCUDNN(cudnnSetPooling2dDescriptor(_pooler,static_cast<cudnnPoolingMode_t>(POOL_AVERAGE),CUDNN_PROPAGATE_NAN,_sub,_sub,padx,pady,_sub,_sub));
      };

      virtual void initialize_backward(T* delta = nullptr) {
         _pdelta_input=delta;
         this->allocate_delta();
      };

      virtual void forward() {
         float alpha = 1.0f, beta = 0.0f;
         checkCUDNN(cudnnPoolingForward(cudnn_handle,_pooler,&alpha,_input,_pinput,&beta,_output,_poutput));
      };

      virtual void backward(CudaMatrix<T>* grad = nullptr) {
         /// computes h(U)
         float alpha = 1.0f, beta = 0.0f;
         checkCUDNN(cudnnPoolingBackward(cudnn_handle,_pooler,&alpha,_output,_poutput,_output,_pdelta_output,_input,_pinput,&beta,_input,_pdelta_input));
      };
      
   private:
      bool _initialized_forward;
      cudnnPoolingDescriptor_t _pooler;
};

template <typename T> class DummyPoolLayer : public PoolLayer<T> {
   public:
      USING_NETWORK_LAYER;
      using PoolLayer<T>::_sub; 

      DummyPoolLayer(const int verbose=0)
      : PoolLayer<T>(verbose) {  };
      virtual ~DummyPoolLayer() { };

      virtual void set_input(NetworkLayer<T>& input, const int sub) { 
         cout << "No Subsampling layer" << endl;
         NetworkLayer<T>::set_input(input);
         _wo=_wi; _ho=_hi; _co=_ci; _sub=sub;
         _poutput=_pinput;
      };

      virtual void initialize_backward(T* delta) {
         _pdelta_input=delta;
         _pdelta_output=delta;
      };

      virtual void forward() { };

      virtual void backward(CudaMatrix<T>* grad = nullptr) { };
};


template <typename T> class MultLayer : public NetworkLayer<T> {
   public:
      USING_NETWORK_LAYER;

      MultLayer(const int verbose=0)
      : NetworkLayer<T>(verbose) {  };
      virtual ~MultLayer() { };

      virtual void set_input(NetworkLayer<T>& input, const T alpha) { 
         cout << "Mult layer" << endl;
         _alpha=alpha;
         NetworkLayer<T>::set_input(input);
         NetworkLayer<T>::set_output(_hi,_wi,_ci);
      };

      virtual void forward() {
         CudaMatrix<T> input(_pinput,_ci,_hi*_wi*_n);
         CudaMatrix<T> output(_poutput,_ci,_hi*_wi*_n);
         _W2.mult(input,output);
      };

      virtual void backward(CudaMatrix<T>* grad = nullptr) {
         /// computes h(U)
         CudaMatrix<T> delta_input(_pdelta_input,_ci,_hi*_wi*_n);
         CudaMatrix<T> delta_output(_pdelta_output,_ci,_hi*_wi*_n);
         _W2.mult(delta_output,delta_input);

         /// computes g(U)
         CudaMatrix<T> output(_poutput,_ci,_hi*_wi*_n);
         output.mult(delta_input,_C,false,true);
         _W3.mult(_C,_C2);
         _C2.mult(_W3,_C);
         _C.addTranspose(_C2);
         _C2.mult_elementWise(_W4);
         _W1.mult(_C2,*grad,false,false,-T(0.5)/_n); // W1 = alpha Z
      };

      virtual void initialize_backward(T* delta) {
         _pdelta_input=delta;
         this->allocate_delta();
      };

      void set_data_backward(const Matrix<T>& W3, const Matrix<T>& W4) {
         _W3.setMatrix(W3);
         _W4.setMatrix(W4);
      };

      void upload_W2(const Matrix<T>& W2) {
         _W2.setMatrix(W2);
      };

      virtual void set_data(T* pW, const int m) {
         _W1.setData(pW,m,_ci);
      };

      void recompute_W234(const T lambda2) {
         _W1.mult(_W1,_W4,true,false,T(1.0)/_alpha); // W1 = alpha Z
         _W4.add_exp(-_alpha); 
         CudaMatrix<T> U, tmp;
         CudaVector<T> S;
         _W4.eigSym(U,S);
         S.inv_sqrt_add(lambda2);
         tmp.copy(U);
         tmp.multDiagRight(S);
         tmp.mult(tmp,_W2,false,true);
         tmp.mult(U,_W3,false,true);
      };

      virtual void do_gradient_step(CudaMatrix<T>& grad, const T eta) {
         /// gradient step
         _W1.add(grad,-eta);
         /// normalization
         _W1.sqr(grad);
         if (_ones.n()==0) {
            _ones.resize(_W1.n());
            _ones.set(T(1.0));
         }
         grad.multTrans(_ones,_tmp);
         _tmp.inv_sqrt();
         _W1.multDiagRight(_tmp);
      };

      void get_parameters(Layer<T>& layer) {
         _W1.getMatrix(layer.W);
         _W2.getMatrix(layer.W2);
      };

   private:
      CudaMatrix<T> _W1, _W2, _W3, _W4;
      CudaMatrix<T> _C, _C2;
      CudaVector<T> _ones, _tmp;
      T _alpha;
};

#define USING_PREDICTION_LAYER \
   using PredictionLayer<T>::_psi; \
using PredictionLayer<T>::_gamma; \
using PredictionLayer<T>::_delta; \
using PredictionLayer<T>::_ones; \
using PredictionLayer<T>::_ones2; \
using PredictionLayer<T>::_W; \
using PredictionLayer<T>::_Y; \
using PredictionLayer<T>::_b; \
using PredictionLayer<T>::_is_bias; 

template <typename T> class PredictionLayer {
   public:
      PredictionLayer(const int verbose=0)
         : _is_bias(true) {  };
      virtual ~PredictionLayer() { };

      virtual void set_input(NetworkLayer<T>& input) {
         const int m = input.size_map();
         const int n = input.n();
         _ones.resize(n);
         _ones.set(T(1.0));
         _ones2.resize(m);
         _ones2.set(T(1.0));
         _psi.setData(input.output(),m,n);
      };

      virtual void initialize_backward(T* delta) {
         _delta.setData(delta,_psi.m(),_psi.n());
      };

      virtual void initialize_forward(const Matrix<T>& W, const Vector<T>& b) { 
         _W.setMatrix(W);
         if (_is_bias)
            _b.setVector(b);
      };

      virtual void forward() {
         _W.mult(_psi,_gamma,true,false);
         if (_is_bias)
            _gamma.rank1Update(_b,_ones);
      };

      virtual void upload_labels(T* Y, const int nim) {
         _Y.resize(_W.n(),_psi.n());
         checkCudaErrors(cudaMemcpyAsync(_Y.rawX(),Y,sizeof(T)*_W.n()*nim,cudaMemcpyHostToDevice));
      };

      void do_gradient_step(CudaMatrix<T>& gradW, CudaVector<T>& gradb, const T eta, const T lambda) {
         _W.scal(T(1.0)-eta*lambda);
         _W.add(gradW,-eta);
         if (_is_bias) _b.add(gradb,-eta);
      };

      void get_parameters(Matrix<T>& W, Vector<T>& b) {
         _W.getMatrix(W);
         if (_is_bias) _b.getVector(b);
      };

      bool is_bias() const { return _is_bias; };


      virtual void backward(CudaMatrix<T>& gradW, CudaVector<T>& gradb) = 0;  
      virtual T loss() = 0; 
      // TODO, set labels

   protected:
      CudaMatrix<T> _psi, _gamma, _delta, _W, _Y;
      CudaVector<T> _ones, _ones2, _b;
      bool _is_bias;
      int _verbose;
};


template <typename T> class SqHingeLossLayer : public PredictionLayer<T> {
   public:
      USING_PREDICTION_LAYER; 

      SqHingeLossLayer(const bool is_bias, const int verbose=0)
      : PredictionLayer<T>(verbose) { _is_bias=is_bias; };
      virtual ~SqHingeLossLayer() { };

      virtual void backward(CudaMatrix<T>& gradW, CudaVector<T>& gradb) {
         cuda_custom3(_gamma.m()*_gamma.n(),_gamma.rawX(),_Y.rawX(),T(-2.0/_Y.m()));
         _psi.mult(_gamma,gradW,false,true,T(1.0)/_psi.n());
         if (_is_bias)
            _gamma.mult(_ones,gradb,T(1.0)/_psi.n());
         gradW.multTrans(_ones2,_tmp);
         gradW.rank1Update(_ones2,_tmp,-T(1.0)/_psi.m());
         _W.mult(_gamma,_delta); /// we assume W is centered
      };

      virtual T loss() {
         cuda_custom6(_gamma.m()*_gamma.n(),_gamma.rawX(),_Y.rawX());
         return _gamma.asum()/(_Y.m()*_Y.n());
      };

   private:
      CudaVector<T> _tmp;
      CudaVector<T> _gamma2;
};

template <typename T> class SquareLossLayer : public PredictionLayer<T> {
   public:
      USING_PREDICTION_LAYER; 

      SquareLossLayer(const int verbose=0)
      :PredictionLayer<T>(verbose) {  };
      SquareLossLayer(const bool is_bias, const int verbose=0)
      :PredictionLayer<T>(verbose) { _is_bias=is_bias; };
      virtual ~SquareLossLayer() { };

      virtual void backward(CudaMatrix<T>& gradW, CudaVector<T>& gradb) {
         _gamma.add(_Y,-T(1.0));
         _psi.mult(_gamma,gradW,false,true,T(1.0)/(_Y.m()*_psi.n()));
         if (_is_bias)
            _gamma.mult(_ones,gradb,T(1.0)/(_Y.m()*_psi.n()));
         _W.mult(_gamma,_delta);
      };

      virtual T loss() {
         cuda_custom7(_gamma.m()*_gamma.n(),_gamma.rawX(),_Y.rawX());
         return _gamma.asum()/(_Y.m()*_Y.n());
      };
};

template <typename T> class SquareLossConv1x1Layer : public SquareLossLayer<T> {
   public:
      USING_PREDICTION_LAYER; 

      SquareLossConv1x1Layer(const int nchannels, const int verbose=0)
      :SquareLossLayer<T>(verbose) { 
         _is_bias=false; 
         _nchannels=nchannels;
      };
      virtual ~SquareLossConv1x1Layer() { };

      virtual void forward() {
         const int m = _psi.m();
         const int n = _psi.n();
         _sizemap = m/_nchannels;
         _gamma.resize(_sizemap,n);
         CudaMatrix<T> psi(_psi.rawX(),_nchannels,n*_sizemap);
         CudaVector<T> gamma_vec(_gamma.rawX(),_sizemap*n);
         CudaVector<T> W_vec(_W.rawX(),_sizemap*n);
         psi.multTrans(W_vec,gamma_vec);
      };

      virtual void backward(CudaMatrix<T>& gradW, CudaVector<T>& gradb) {
         _gamma.add(_Y,-T(1.0));
         const int m = _psi.m();
         const int n = _psi.n();
         gradW.resize(_nchannels,1);
         CudaMatrix<T> psi(_psi.rawX(),_nchannels,n*_sizemap);
         CudaVector<T> gamma_vec(_gamma.rawX(),_sizemap*n);
         CudaVector<T> gradW_vec(gradW.rawX(),_nchannels);
         CudaVector<T> W_vec(_W.rawX(),_nchannels);
         psi.mult(gamma_vec,gradW_vec,T(1.0)/(_sizemap*n));
         CudaMatrix<T> delta(_delta.rawX(),_nchannels,n*_sizemap);
         delta.rank1Update(W_vec,gamma_vec);
      };

      virtual void upload_labels(T* Y, const int nim) {
         _Y.resize(_sizemap,_psi.n());
         checkCudaErrors(cudaMemcpyAsync(_Y.rawX(),Y,sizeof(T)*_sizemap*nim,cudaMemcpyHostToDevice));
      };

   private:
      int _nchannels;
      int _sizemap;
};




template <typename T> class Network {
   public:
      Network(Layer<T> layers[],
            const int nlayers,
            const int hi,
            const int wi,
            const int ci,
            const int n,
            const bool recompute_W234 = false,
            const T lambda2 = 0,
            const loss_t loss = SQLOSS,
            const bool is_bias = true,
            const int verbose=0) {
         _nlayers=nlayers;
         _loss=loss;
         _conv_layers = new ConvLayer<T>[nlayers];
         _pool_layers = new PoolLayer<T>*[nlayers];
         _mult_layers = new MultLayer<T>[nlayers];
         _gradients = new CudaMatrix<T>[nlayers];
         _gradients_acc = new CudaMatrix<T>[nlayers];
         _lambda2=lambda2;
         _upload_W234 = !recompute_W234;
         for (int ii=0; ii<nlayers; ++ii) {
            Layer<T>& layer = layers[ii];
            const T alpha = T(1.0)/(layers[ii].sigma*layers[ii].sigma);
            const int p = layer.W.n();
            const int sub = layer.subsampling;

            /// set up the convolutional layer
            if (ii==0) {
               _conv_layers[ii].set_input(hi,wi,ci,n,alpha);
            } else {
               _conv_layers[ii].set_input(_mult_layers[ii-1],alpha);
            } 
            _conv_layers[ii].set_filter(layer.npatch,p,layer.zero_padding);
            _conv_layers[ii].set_data(layer.W.rawX(),layer.b.rawX());

            /// set up the pooling layer
            if (sub==1) {
               _pool_layers[ii]=new DummyPoolLayer<T>();
            } else if (layers[ii].pooling_mode==POOL_GAUSSIAN_FILTER) {
               _pool_layers[ii]=new GaussianPoolLayer<T>();
            } else {
               _pool_layers[ii]=new SimplePoolLayer<T>();
            }
            _pool_layers[ii]->set_input(_conv_layers[ii],sub);

            /// set up the projection layer
            _mult_layers[ii].set_input(*_pool_layers[ii],alpha);
            _mult_layers[ii].set_data(_conv_layers[ii].filter(),_conv_layers[ii].size_filter());
            if (recompute_W234) {
               _mult_layers[ii].recompute_W234(_lambda2);
            } else {
               _mult_layers[ii].upload_W2(layer.W2);
            }
         }

         /// set up the prediction layer
         if (loss == SQLOSS) {
            _prediction_layer = new SquareLossLayer<T>(is_bias);
         } else if (loss == SQLOSS_CONV) {
            _prediction_layer = new SquareLossConv1x1Layer<T>(_mult_layers[_nlayers-1].p());
         } else {
            _prediction_layer = new SqHingeLossLayer<T>(is_bias);
         }
         _prediction_layer->set_input(_mult_layers[_nlayers-1]);
      };

      virtual ~Network() {
         delete[](_conv_layers);
         for (int ii=0; ii<_nlayers; ++ii)
            delete(_pool_layers[ii]);
         delete[](_pool_layers);
         delete[](_mult_layers);
         delete[](_gradients);
         delete[](_gradients_acc);
         delete(_prediction_layer);
      };

      void forward(T* input_data, const int nim) {
         _conv_layers[0].upload_input(input_data,nim);
         for (int jj=0; jj<_nlayers; ++jj) {
            _conv_layers[jj].forward();
            _pool_layers[jj]->forward();
            _mult_layers[jj].forward();
         }
      };

      void get_output(T* output, const int nim) {
         checkCudaErrors(cudaMemcpy(output,_mult_layers[_nlayers-1].output(),sizeof(T)*_mult_layers[_nlayers-1].size_map()*nim,cudaMemcpyDeviceToHost));
      };

      void backward(T* Y, const int nim) {
         _prediction_layer->upload_labels(Y,nim);
         _prediction_layer->backward(_gradW,_gradb);
         // TODO: stop backward if gradient is zero?
         for (int jj=_nlayers-1; jj>= 0; --jj) {
            _mult_layers[jj].backward(&_gradients[jj]);
            _pool_layers[jj]->backward();
            _conv_layers[jj].backward(&_gradients[jj]);
         }
      };
      
      void initialize_forward(const Matrix<T>& W, const Vector<T>& b) { 
         _prediction_layer->initialize_forward(W,b);
      };
   
      void initialize_backward(Layer<T> layers[]) { 
         if (_upload_W234)
            for (int jj=0; jj<_nlayers; ++jj) 
               _mult_layers[jj].set_data_backward(layers[jj].W3,layers[jj].W4);
         for (int jj=0; jj<_nlayers; ++jj)  {
            if (jj==0) {
               _conv_layers[0].initialize_backward();
            } else {
               _conv_layers[jj].initialize_backward(_mult_layers[jj-1].delta());
            }
            _pool_layers[jj]->initialize_backward(_conv_layers[jj].delta());
            _mult_layers[jj].initialize_backward(_pool_layers[jj]->delta());
         };
         _prediction_layer->initialize_backward(_mult_layers[_nlayers-1].delta());
      };

      void get_gradients(Matrix<T>& gradW, Vector<T>& gradb, Layer<T> layers[]) {
         for (int jj=0; jj<_nlayers; ++jj) _gradients[jj].getMatrix(layers[jj].gradW);
         _gradW.getMatrix(gradW);
         if (_prediction_layer->is_bias())
            _gradb.getVector(gradb);
      };

      void forward_predictions() {
         _prediction_layer->forward();
      };

      void do_gradient_step(const T eta, const T momentum, const T lambda) {
         if (_gradW_acc.m()==0) {
            _gradW_acc.copy(_gradW);
            if (_prediction_layer->is_bias())
               _gradb_acc.copy(_gradb);
            for (int ii=0; ii<_nlayers; ++ii) 
               _gradients_acc[ii].copy(_gradients[ii]);
         } else {
            _gradW_acc.scal(momentum);
            _gradW_acc.add(_gradW,T(1.0)-momentum);
            _gradW.copy(_gradW_acc);
            if (_prediction_layer->is_bias()) {
               _gradb_acc.scal(momentum);
               _gradb_acc.add(_gradb,T(1.0)-momentum);
               _gradb.copy(_gradb_acc);
            }
            for (int ii=0; ii<_nlayers; ++ii) {
               _gradients_acc[ii].scal(momentum);
               _gradients_acc[ii].add(_gradients[ii],T(1.0)-momentum);
               _gradients[ii].copy(_gradients_acc[ii]);
            }
         }
         _prediction_layer->do_gradient_step(_gradW,_gradb,eta,lambda);
         for (int ii = 0; ii<_nlayers; ++ii)
            _mult_layers[ii].do_gradient_step(_gradients[ii],eta); 
      };

      void recompute_W234() {
         for (int ii = 0; ii<_nlayers; ++ii)
            _mult_layers[ii].recompute_W234(_lambda2); 
      }

      void get_parameters(Layer<T> layers[], Matrix<T>& W, Vector<T>& b) {
         for (int ii = 0; ii<_nlayers; ++ii)
            _mult_layers[ii].get_parameters(layers[ii]);
         _prediction_layer->get_parameters(W,b);
      };

      int output_size() const { return _mult_layers[_nlayers-1].size_map(); };
      T* output() { return _mult_layers[_nlayers-1].output(); };

   private:
      int _nlayers;
      ConvLayer<T>* _conv_layers;
      PoolLayer<T>** _pool_layers;
      MultLayer<T>* _mult_layers;
      PredictionLayer<T>* _prediction_layer;
      CudaMatrix<T>* _gradients;
      CudaMatrix<T> _gradW;
      CudaVector<T> _gradb; 
      CudaMatrix<T>* _gradients_acc;
      CudaMatrix<T> _gradW_acc;
      CudaVector<T> _gradb_acc; // TODO: add momentum variables, + function do_gradient_step
      T _lambda2;
      bool _upload_W234;
      loss_t _loss;
};

template <typename Tin, typename T>
inline void encode_ckn_cudnn(const Map<Tin>& maps, Layer<T> layers[], const int nlayers, Matrix<T>& psi, const int batch_size, const int verbose=0) {
   Timer time, time2, time3;
   checkCudaErrors(cudaDeviceSynchronize());
   time.start();
   const int n = maps.z();
   const int nchannels=maps.y()/maps.x();
   int batchsize=MIN(batch_size,n);
   Network<T> network(layers,nlayers,maps.x(),maps.x(),nchannels,batchsize);
   psi.resize(network.output_size(),n);
   T* input_data = new T[maps.x()*maps.y()*batchsize];

   for (int ii=0; ii<n; ii+=batchsize) {
      /// get input data
      const int nim = MIN(n,ii+batchsize) - ii;
      Tin* input_in = maps.rawX()+maps.x()*maps.y()*ii;
      convert_image_data_map_switch<Tin,T>(input_in,input_data,maps.x()*maps.y()/nchannels,nchannels,nim); 

      /// forward pass
      checkCudaErrors(cudaDeviceSynchronize());
      time2.start();
      network.forward(input_data,nim);
      checkCudaErrors(cudaDeviceSynchronize());
      time2.stop();
      
      /// get back the data on cpu 
      time3.start();
      network.get_output(psi.rawX()+ii*psi.m(),nim);
      checkCudaErrors(cudaDeviceSynchronize());
      time3.stop();
   }

   delete[](input_data);
   cout << "Time doing computations, including data upload" << endl;
   time2.printElapsed();
   cout << "Time downloading output" << endl;
   time3.printElapsed();
   cout << "Total time" << endl;
   time.printElapsed();
};

template <typename Tin, typename T>
inline void ckn_backprop_cudnn(const Map<Tin>& maps, Layer<T> layers[], const int nlayers, Matrix<T>& Y, const Matrix<T>& W, const Vector<T>& b, Matrix<T>& gradW, const T lambda2, const loss_t loss) {
   Timer time, time2, time3, time4, time5, time6;
   time6.start();
   const int n = maps.z();
   PRINT_I(n)
   const int nchannels=maps.y()/maps.x();
   Vector<T> gradb;
   /// various initializations
   time.start();
   Network<T> network(layers,nlayers,maps.x(),maps.x(),nchannels,n,false,lambda2,loss); // TODO do recompute W234 
   network.initialize_backward(layers);
   network.initialize_forward(W,b);

   /// upload the data on gpu
   T* input_data = new T[maps.x()*maps.y()*n];
   convert_image_data_map_switch<Tin,T>(maps.rawX(),input_data,maps.x()*maps.y()/nchannels,nchannels,n); 
   checkCudaErrors(cudaDeviceSynchronize());
   time.stop();
   
   // forward pass
   time2.start();
   network.forward(input_data,n);
   checkCudaErrors(cudaDeviceSynchronize());
   time2.stop();
   
   /// computing predictions for classification with squared hinge loss
   time3.start();
   network.forward_predictions();
   checkCudaErrors(cudaDeviceSynchronize());
   time3.stop();

   /// backward pass
   time4.start();
   network.backward(Y.rawX(),n);
   checkCudaErrors(cudaDeviceSynchronize());
   time4.stop();
   
   /// download the gradients on cpu
   time5.start();
   network.get_gradients(gradW,gradb,layers);
   checkCudaErrors(cudaDeviceSynchronize());
   time5.stop();

   cout << "Time for initialization" << endl;
   time.printElapsed();
   cout << "Time for forward pass, including data upload" << endl;
   time2.printElapsed();
   cout << "Time for predictions" << endl;
   time3.printElapsed();
   cout << "Time for backward pass" << endl;
   time4.printElapsed();
   cout << "Time for gradient download" << endl;
   time5.printElapsed();
   cout << "Total time" << endl;
   time6.stop();
   time6.printElapsed();

   delete[](input_data);
};

/// TODO: put parameters sgd in a struct
/// add: validation set

/// sgd solver with mometum and fixed step size
template <typename Tin, typename T>
inline void sgd_solver(const Map<Tin>& maps, Layer<T> layers[], const int nlayers, Matrix<T>& Y, Matrix<T>& W, Vector<T>& b, const T lambda2, const T lambda, const loss_t loss, const int epochs, const int batch_size, const T momentum, const T eta) {
   const int n = maps.z();
   const int nchannels=maps.y()/maps.x();
   const int iter_per_epochs = n/batch_size;
   const int size_map=maps.x()*maps.y();
   Vector<int> per;
   const bool is_bias = b.n() > 0;
   const int nlabels = Y.m();

   /// initialization
   T* input_data = new T[maps.x()*maps.y()*batch_size];
   T* input_labels = new T[nlabels*batch_size];
   Network<T> network(layers,nlayers,maps.x(),maps.x(),nchannels,batch_size,true,lambda2,loss,is_bias);
   network.initialize_backward(layers);
   network.initialize_forward(W,b);

   for (int ii=0; ii<epochs; ++ii) {
      per.randperm(n);
      for (int jj=0; jj<iter_per_epochs; ++jj) {
         /// get the data and perform the forward pass
         for (int kk=0; kk<batch_size; ++kk)
            convert_image_data_map_switch<Tin,T>(maps.rawX()+size_map*per[jj*batch_size+kk],
                  input_data+kk*size_map,size_map/nchannels,nchannels,1);
         network.forward(input_data,batch_size);
         network.forward_predictions();
         
         /// perfor the backward pass
         for (int kk=0; kk<batch_size; ++kk)
            memcpy(input_labels+kk*nlabels,Y.rawX()+per[jj*batch_size+kk]*nlabels,sizeof(T)*nlabels);
         network.backward(input_labels,batch_size);

         /// do gradient step
         network.do_gradient_step(eta,momentum,lambda);
         network.recompute_W234();
      }
   }

   // get the network parameters
   network.get_parameters(layers,W,b);

   delete[](input_data);
   delete[](input_labels);
};


#endif
// vim: set softtabstop=3 shiftwidth=3 expandtab :
