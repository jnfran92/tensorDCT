
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
//#include "../utils/utils.h"
//#include "../utils/matutils.h"

#include "omp.h"

// custom headers
#include "./include/tools.h"


int main( int argc, char**  argv  ){

	int args_needed = 1;
	if (argc < args_needed + 1 ){
		std::cout <<"Arg number error, needed: " << args_needed<< std::endl; 	
		return 0;
	}


	std::cout << "cuFFT Test - Discrete Cosine Transform" << std::endl;
	
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

	// Matmul m_lin * n_lin in x_n
	// c = a*b
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

	// Free data	
	delete x_n;
	delete m_line;
	delete n_line;


	return 0;
}





