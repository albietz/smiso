path_mkl=/opt/intel/composerxe/mkl/lib/intel64/
path_mkl=/scratch2/clear/mairal/intel/composerxe/mkl/lib/intel64/
path_mkl=/scratch2/clear/mairal/intel_beta/mkl/lib/intel64/
path_icc=/opt/intel/composerxe/
path_icc=/scratch2/clear/mairal/intel/composerxe/
pathlibiomp=/scratch2/clear/mairal/intel_beta/compilers_and_libraries_2017/linux/lib/intel64/
# path_cuda=/scratch2/clear/mairal/cuda-7.5/
path_cuda=/scratch2/clear/mairal/cuda-8.0rc/

CC:=g++
# CC:=$(path_icc)/bin/icpc
NVCC=$(path_cuda)/bin/nvcc

lib_mkl_st:=-Wl,--start-group $(path_mkl)libmkl_intel_ilp64.a $(path_mkl)libmkl_sequential.a $(path_mkl)libmkl_core.a -Wl,--end-group # -Wl,-version-script=ckn.map
lib_mkl_mt:=-Wl,--start-group $(path_mkl)libmkl_intel_ilp64.a $(path_mkl)libmkl_gnu_thread.a $(path_mkl)libmkl_core.a -Wl,--end-group
lib_cuda_static:=-Wl,--start-group $(path_cuda)/lib64/libcudart_static.a $(path_cuda)/lib64/libcublas_static.a $(path_cuda)/lib64/libcusolver_static.a $(path_cuda)/lib64/libcudnn_static.a $(path_cuda)/lib64/libculibos.a -Wl,--end-group

defines:=-DHAVE_MKL -DAXPBY -DINT_64BITS
defines_cuda:=$(defines) -DCUDA -DCUDNN -DCUDA_8
includes=-I.
flags=-O3 -fPIC -Wall -g -fopenmp -std=c++11 -lm
flags_cuda=-O3 -fPIC -Wall -g -fopenmp -std=c++11 -lm
lib_cuda=-lcudart -lcublas -lcusolver -lcudnn
python_include:=-I. -I$(shell python3 -c 'import distutils.sysconfig; print(distutils.sysconfig.get_python_inc())') -I$(shell python3 -c 'import numpy as np; print(np.get_include())')
# python_include:=-I. -I/usr/include/python3.4/ -I/home/lear/dwynen/.local/lib/python3.4/site-packages/numpy/core/include 

all: _ckn_cuda.so

_ckn_cuda.so: ckn_cuda.o cuda_kernels.o
	$(CC) -DMODNAME=_ckn_cuda ckn_cuda.o cuda_kernels.o -L$(path_cuda)/lib64 $(lib_mkl_st) $(lib_cuda) $(includes) $(flags_cuda) -shared $(defines_cuda) -lrt -o _ckn_cuda.so

cuda_kernels.o: cuda_kernels.cu
	$(NVCC) -DMODNAME=_ckn_cuda -c cuda_kernels.cu -Xcompiler -fpic -o cuda_kernels.o

ckn_cuda.o: $(wildcard *.c)  $(wildcard *.cpp)  $(wildcard *.h)  $(wildcard *.hpp)
	$(CC) -c ckn.cpp -o ckn_cuda.o $(includes) -I$(path_cuda)/include $(python_include) $(flags_cuda) $(defines_cuda)

clean:
	rm -f ckn_cuda.o cuda_kernels.o _ckn_cuda.so


# a failed attempt to make the module expose exactly what it needs to, 
# hiding all the statically linked methods
# idea taken from http://stackoverflow.com/a/2251477/393885
# tmpobj/cuos_common_barrier.o: $(path_cuda)/lib64/libculibos.a
# 	mkdir -p tmpobj
# 	cd tmpobj && ar x $(path_cuda)/lib64/libculibos.a
#
# tmpobj/activation.o: $(path_cuda)/lib64/libcudnn_static.a
# 	mkdir -p tmpobj
# 	cd tmpobj && ar x $(path_cuda)/lib64/libcudnn_static.a
#
# tmpobj/analyze.o: $(path_cuda)/lib64/libcusolver_static.a
# 	mkdir -p tmpobj
# 	cd tmpobj && ar x $(path_cuda)/lib64/libcusolver_static.a
#
# tmpobj/asum.o: $(path_cuda)/lib64/libcublas_static.a
# 	mkdir -p tmpobj
# 	cd tmpobj && ar x $(path_cuda)/lib64/libcublas_static.a
#
# tmpobj/libcudart_static.a.o: $(path_cuda)/lib64/libcudart_static.a
# 	mkdir -p tmpobj
# 	cd tmpobj && ar x $(path_cuda)/lib64/libcudart_static.a
#
# tmpobj/alloc_c_structure_lp64.o: $(path_mkl)libmkl_core.a
# 	mkdir -p tmpobj
# 	cd tmpobj && ar x $(path_mkl)libmkl_core.a
#
# tmpobj/amuxy_cpardiso_omp_seq_lp64.o: $(path_mkl)libmkl_gnu_thread.a
# 	mkdir -p tmpobj
# 	cd tmpobj && ar x $(path_mkl)libmkl_gnu_thread.a
#
# tmpobj/_allocate_internal.o: $(path_mkl)libmkl_intel_ilp64.a
# 	mkdir -p tmpobj
# 	cd tmpobj && ar x $(path_mkl)libmkl_intel_ilp64.a
#
# _ckn_cuda.so: ckn_cuda.o cuda_kernels.o \
# 	tmpobj/asum.o \
# 	tmpobj/libcudart_static.a.o \
# 	tmpobj/activation.o \
# 	tmpobj/cuos_common_barrier.o \
# 	tmpobj/analyze.o \
# 	tmpobj/alloc_c_structure_lp64.o \
# 	tmpobj/amuxy_cpardiso_omp_seq_lp64.o \
# 	tmpobj/_allocate_internal.o
# 	$(CC) -DMODNAME=_ckn_cuda ckn_cuda.o cuda_kernels.o tmpobj/*.o $(includes) $(flags_cuda) -shared $(defines_cuda) -lrt -o _ckn_cuda.so
#
