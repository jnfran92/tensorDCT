
CC = nvcc
all:
		$(CC) -O3 -Xcompiler -fopenmp matmul_1d_mini.cu -o prog -L/usr/lib -llapack -lblas -lfftw3 -lm
