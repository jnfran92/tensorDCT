
#define _USE_MATH_DEFINES
#include <math.h>
#include <fftw3.h>

//CBLAS
extern "C"
{
	#include <cblas.h>
}


#include <iostream>
#include <stdlib.h>
#include <stdio.h>

//CuBlas
#include <cublas_v2.h>

#include "omp.h"

// custom headers
#include "./include/tools.h"

/*__global__ void julia_kernel( int n, double *a, double *b, double *c  ){*/

	/*int i = threadIdx.y;*/
	/*int j = threadIdx.x;*/

	/*int gi = threadIdx.y + blockDim.y*blockIdx.y;*/
	/*int gj = threadIdx.x + blockDim.x*blockIdx.x;*/

/*}*/


int main( int argc, char**  argv  ){

	int args_needed = 1;
	if (argc < args_needed + 1 ){
		std::cout <<"Arg number error, needed: " << args_needed<< std::endl; 	
		return 0;
	}


	std::cout << "cuFFT Test - Discrete Cosine Transform" << std::endl;
	
	// CUDA Timmers
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);


	// OMP
	int ncpu = 1;
	omp_set_num_threads(ncpu);

	// Creating matrices - using two vectors
	std::cout << "PI number: "<< M_PI << std::endl;
	
	int size_m = 16;
	int size_n = 16;

	double *x_n = new double[size_m * size_n]; 
	
	double *m_line = new double[size_m];
	double *n_line = new double[size_n];

	// Fill and Print	
	fill_vector_cos(3, size_m, m_line);
	fill_vector_cos(2, size_n, n_line);
	
	print_array(m_line, size_m);
	print_array(n_line, size_n);

	// CPU - CBLAS
	cblas_dgemm(CblasRowMajor, 		// Layout
			CblasNoTrans, 		// trans a
			CblasNoTrans,		// trans b
			16,			// m
			16,			// n
			1,			// k
			1.0,			// alpha
			m_line,			// a matrix
			1,			// lda
			n_line,			// b matrix
			16,			// ldb
			0.0,			// beta
			x_n,			// c matrix
			16			// ldc
			   );	
	
	print_array(x_n, 16*16);

	// CUDA
	// CUDA Handle
	cublasHandle_t cublasHandle;
  	cudaEvent_t startcublas;
     	cudaEvent_t stopcublas;
	// cublas event create
	cudaEventCreate(&startcublas);
	cudaEventCreate(&stopcublas);
	
	cublasCreate(&cublasHandle)
	// Tensor cores enabled
	cublasSetMathMode(cublasHandle, CUBLAS_TENSOR_OP_MATH);	

	cudaEventRecord(startcublas);
	
	cublasGemmEx(
			cublasHandle, 
			CUBLAS_OP_N, 
			CUBLAS_OP_N,
			MATRIX_M, 
			MATRIX_N, 
			MATRIX_K,
		       	&alpha,
			a_fp16, 
			CUDA_R_16F, 
			MATRIX_M,						                
			b_fp16, 
			CUDA_R_16F, 
			MATRIX_K,
			&beta, 
			c_cublas, 
			CUDA_R_32F, 
			MATRIX_M,
			CUDA_R_32F, 
			CUBLAS_GEMM_DFALT_TENSOR_OP);

	cudaEventRecord(stopcublas);
	
	float cublasTime;
	cudaEventSynchronize(stopcublas);
	cudaEventElapsedTime(&cublasTime, startcublas, stopcublas);

	std::cout << "cublas took: " << cublasTime << std::endl;

	// Free data	
	delete x_n;
	delete m_line;
	delete n_line;

	// cuda free
	cudaEventDestroy(startcublas);
	cudaEventDestroy(stopcublas);

	/*cudaFree();*/


	return 0;
}





