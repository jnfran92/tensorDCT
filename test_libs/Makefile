
CC = nvcc
OPT_FLAG = -O3
OPEN_MP_FALG = -Xcompiler -fopenmp
# BLAS LAPACK FFT libs
LIBS = -L/usr/lib -llapack -lblas -lfftw3 -lm 

all: test1

test_1: matmul_fft_base.cu
		$(CC) $(OPT_FLAG) $(OPEN_MP_FALG) $(LIBS) matmul_fft_base.cu -o ./build/matmul_fft_base 


