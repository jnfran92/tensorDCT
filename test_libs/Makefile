
CC = nvcc
OPT_FLAG = -O3
OPEN_MP_FALG = -Xcompiler -fopenmp

# If are using UACH server use this
LIBS = -L/usr/lib/gcc/x86_64-pc-linux-gnu/5.5.0/ -llapack -lblas -lfftw3 -lgfortran -lm 

# If are using local server uncomment this
#LIBS = -L/usr/lib -llapack -lblas -lfftw3 -lgfortran -lm -L/usr/lib/gcc/x86_64-pc-linux-gnu/5.5.0/
CUDA_LIBS = -lcublas 

all: test1

test_1: matmul_fft_base.cu
		$(CC) $(OPT_FLAG) $(OPEN_MP_FALG) $(LIBS) matmul_fft_base.cu -o ./build/t1 


test_2: fftw_1d.cu
		$(CC) $(OPT_FLAG) $(OPEN_MP_FALG) $(LIBS) fftw_1d.cu -o ./build/t2

test_3: fftw_2d.cu
		$(CC) $(OPT_FLAG) $(OPEN_MP_FALG) $(LIBS) fftw_2d.cu -o ./build/t3

test_4: matmul_cublas.cu
		$(CC) $(OPT_FLAG) $(OPEN_MP_FALG) $(LIBS) $(CUDA_LIBS) matmul_cublas.cu -o ./build/t4
