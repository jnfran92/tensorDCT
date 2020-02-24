//
// Created by Juan Francisco on 2020-02-22.
//

//CBLAS
extern "C"
{
#include <cblas.h>
}



#include <cstdio>
#include "test_libs/include/tools.h"

int main(int argv, char** argc){
    printf("Comparing DCT tensorcores , cublas and cufft\n");

    // Creating matrices - using two vectors
    std::cout << "PI number: "<< M_PI << std::endl;

    int size_m = 16;
    int size_n = 16;

    auto *x_n = new double[size_m * size_n];

    auto *m_line = new double[size_m];
    auto *n_line = new double[size_n];

    // Fill and Print
    fill_vector_cos(3, size_m, m_line);
    fill_vector_cos(2, size_n, n_line);

    print_array(m_line, size_m);
    print_array(n_line, size_n);


    // Building a 2d matrix based on m_line using CBLAS
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

    print_array(x_n , 16*16);






    return 0;
}
