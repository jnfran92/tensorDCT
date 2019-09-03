
#define _USE_MATH_DEFINES
#include <math.h>

#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include "../utils/utils.h"
#include "../utils/matutils.h"

#include "omp.h"

#include "./include/tools.h"

__global__ void julia_kernel( int n, double *a, double *b, double *c  ){

	int i = threadIdx.y;
	int j = threadIdx.x;

	int gi = threadIdx.y + blockDim.y*blockIdx.y;
	int gj = threadIdx.x + blockDim.x*blockIdx.x;

}


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
	int size_n = 8;

	double *x_n = new double[size_m * size_n]; 
	
	double *m_line = new double[size_m];
	double *n_line = new double[size_n];

	// Fill and Print	
	fill_vector_cos(1, size_m, m_line);
	print_array(m_line, size_m);

	delete x_n;
	delete m_line;
	delete n_line;


	return 0;
}





