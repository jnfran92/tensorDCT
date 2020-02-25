
#ifndef TOOLS_H
#define TOOLS_H

#define _USE_MATH_DEFINES
#include <math.h>
#include <iostream>

/**
 * Fill a vector os size N with cosine values: cos(2*pi*k*n/N)
 * 
 * **/
void fill_vector_cos(int k_arg, int n_arg, double* vector){
	double k = (double)k_arg;
	double n = (double)n_arg;
	for (int i=0; i<n_arg; i++){
		double val = (double)i;
		vector[i] = cos(2*M_PI*k*(val / (n-1.0)));		
	}
}

/**
 * Print any array of doubles up to limit value
 * **/
void print_array(double* data, int limit){
	std::cout << "print array"<< std::endl;
	for (int i=0; i<limit; i++){
		std::cout << data[i] << std::endl;
	}
}

/**
 * Print any array of doubles up to limit value
 * **/
void print_array(double* data, int m, int n){
	std::cout << "print array"<< std::endl;
	for (int i=0; i<m; i++){
		for (int j=0; j<n; j++){
//			std::cout << data[i] << " ";
			printf("%3.4f ", data[i] );
		}
		std::cout <<std::endl;
	}
}




#endif

