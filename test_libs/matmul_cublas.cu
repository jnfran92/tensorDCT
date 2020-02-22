
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
	// Data
	double *m_line_d;
	double *n_line_d;
	double *x_n_d;

	// CUDA Malloc
	cudaMalloc((void **)&m_line_d, 	sizeof(double)*size_m);
	cudaMalloc((void **)&n_line_d, 	sizeof(double)*size_n);
	cudaMalloc((void **)&x_n_d, 	sizeof(double)*size_m*size_n);

	// CUDA Handle
	cublasHandle_t cublasHandle;
  	cudaEvent_t startcublas;
     	cudaEvent_t stopcublas;

	// cublas event create
	cudaEventCreate(&startcublas);
	cudaEventCreate(&stopcublas);
	
	cublasCreate(&cublasHandle);
	// Tensor cores enabled
	/*cublasSetMathMode(cublasHandle, CUBLAS_TENSOR_OP_MATH);	*/

	cudaEventRecord(startcublas);

	const double alpha = 1.0;
	const double beta = 0.0;
	const double *alpha_ptr = &alpha;
	const double *beta_ptr = &beta;

	cublasDgemm(
			cublasHandle,		// hanlde
			CUBLAS_OP_T,		// trans a 
			CUBLAS_OP_T,		// trans b
                    	16,			// m 
			16,			// n 
			1, 			// k
		        alpha_ptr, 		// alpha
	  		m_line_d,		// a matrix 
			1,			// lda
			n_line_d, 		// b matrix
			16,			// ldb
 			beta_ptr,		// beta
		        x_n_d, 			// c matrix
			16			// ldc
			);
	
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

	cudaFree(m_line_d);
	cudaFree(n_line_d);
	cudaFree(x_n_d);


	return 0;
}





