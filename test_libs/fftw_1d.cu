
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
	
	// CUDA Timmers
	/*cudaEvent_t start, stop;*/
	/*cudaEventCreate(&start);*/
	/*cudaEventCreate(&stop);*/

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

	/*
	   FFTW
	*/
	// FFTW one dimension using m_line
	fftw_complex *data_in, *data_out;
	fftw_plan p;
	// allocating data
	data_in = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * size_m);	
	data_out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * size_m);
	
	// fill data
	std::cout<< "Data in FFTW" << std::endl;
	for (int i=0; i<size_m; i++){
		data_in[0][i] = m_line[i]; 	// real data
		data_in[1][i] = 0.0; 		// imaginary data
		std::cout << data_in[0][i] << " - "<< data_in[1][i] << std::endl;
	}
	
	// executing fft	
	p = fftw_plan_dft_1d(size_m, data_in, data_out, FFTW_FORWARD, FFTW_ESTIMATE);
	fftw_execute(p);

	/*fdata = fftwf_data(data, ndata, nfft);*/


	std::cout << "results " << std::endl;
	// fill data
	for (int i=0; i<size_m; i++){
		/*data_in[0][i] = m_line[i]; 	// real data*/
		/*data_in[1][i] = 0.0; 		// imaginary data*/
		std::cout << data_out[0][i] << " - "<< data_out[1][i] << std::endl;
	}


	// free data
	fftw_destroy_plan(p);
	fftw_free(data_in);
	fftw_free(data_out);
	
	delete x_n;
	delete m_line;
	delete n_line;


	return 0;
}





