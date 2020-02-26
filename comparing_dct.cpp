//
// Created by Juan Francisco on 2020-02-22.
//
#include <cstdio>
//
////CBLAS
//extern "C"
//{
//#include <cblas.h>
//}

//#include "mkl.h"
#include "./include/cuda_helper/cuda_include_helper.h"
#include "include/common_gpu_utils.h"
#include "test_libs/include/tools.h"
#include "include/dct/cublas/DctCuBlas.h"


int main(int argv, char** argc){
    printf("Comparing DCT tensorcores , cublas and cufft\n");


    // Creating matrices - using two vectors
    std::cout << "PI number: "<< M_PI << std::endl;

    int size_m = 16;
    int size_n = 16;

    int dim_y = size_m;
    int dim_x = size_n;
    DctCuBlas dctCuBlas;
    dctCuBlas = DctCuBlas(dim_y, dim_x);

//    thrust::host_vector<>
    auto *x_n_host = new double[size_m * size_n];

    auto *m_line = new double[size_m];
    auto *n_line = new double[size_n];

    // Fill and Print
    fill_vector_cos(3, size_m, m_line);
    fill_vector_cos(2, size_n, n_line);

    print_array(m_line, size_m);
    print_array(n_line, size_n);

    int M = size_m;
    int N = size_n;
    int K = 1;
    for (int i=0; i<M; i++){
        for (int j=0; j<N; j++){
            for (int k=0; k<K; k++){
                x_n_host[i*size_n + j] += m_line[i*K + k] * n_line[k*N + j];
            }
        }
    }


    print_array(x_n_host , M, N);


    // GPU copies
    int seq_size = size_m * size_n;
    thrust::device_vector<double> x_n(seq_size, 0.0);
    double *x_n_ptr = thrust::raw_pointer_cast(&x_n[0]);

    thrust::device_vector<double> x_k(seq_size, 0.0);
    double *x_k_ptr = thrust::raw_pointer_cast(&x_k[0]);

//    cudaMemcpy(x_n_ptr, x_n_host, sizeof(double)*size_m*size_n, cudaMemcpyHostToDevice);

//    print_dvector(x_n, "x_n");


    // Init DCT
    dctCuBlas.dct(x_n, x_k);
    cudaDeviceSynchronize();

    print_dvector(x_k, "x_k");


//    delete[] x_n_host;

    return 0;
}
