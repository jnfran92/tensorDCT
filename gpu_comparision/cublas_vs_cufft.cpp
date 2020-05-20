//
// Created by Juan Francisco on 5/19/20.
//

#include <cstdio>
#include <fftw3.h>
#include "../test_libs/include/tools.h"
#include "../include/dct/cublas/DctCuBlas.h"

#include "../include/common_gpu_utils.h"

#include <thrust/device_vector.h>
#include <cublas_v2.h>
#include <cufft.h>
#include <chrono>

using namespace std::chrono;


void cublas_dct(int &dim_y, int &dim_x, thrust::device_vector<double> &x_n, thrust::device_vector<double> &x_k){
    std::cout<< "cublas_dct" << std::endl;
    // DCT
    cudaEvent_t start_cublas;
    cudaEvent_t stop_cublas;
    DctCuBlas  dctCuBlas(dim_y, dim_x);

    cudaEventCreate(&start_cublas);
    cudaEventCreate(&stop_cublas);

    // Init DCT
    cudaEventRecord(start_cublas);
    dctCuBlas.dct(x_n, x_k);
    cudaEventRecord(stop_cublas);


    float cublasTime;
    cudaEventSynchronize(stop_cublas);
    cudaEventElapsedTime(&cublasTime, start_cublas, stop_cublas);
    std::cout << "cublas took[ms]: " << cublasTime << std::endl;
}


void cublas_idct(int &dim_y, int &dim_x, thrust::device_vector<double> &x_n, thrust::device_vector<double> &x_k){
    std::cout<< "cublas_idct" << std::endl;
    // DCT
    cudaEvent_t start_cublas;
    cudaEvent_t stop_cublas;
    DctCuBlas  dctCuBlas(dim_y, dim_x);

    cudaEventCreate(&start_cublas);
    cudaEventCreate(&stop_cublas);

    // Init iDCT
    cudaEventRecord(start_cublas);
    dctCuBlas.idct(x_k, x_n);
    cudaEventRecord(stop_cublas);


    float cublasTime;
    cudaEventSynchronize(stop_cublas);
    cudaEventElapsedTime(&cublasTime, start_cublas, stop_cublas);
    std::cout << "cublas took[ms]: " << cublasTime << std::endl;
}



void cufft_double_fft(int &dim_y, int &dim_x, double *x_n){
    std::cout<< "cufft_double_fft" << std::endl;
    // FFTW two dimensions using m_line.dot(n_line) = x_n
    cufftDoubleComplex *data_in, *data_out;
    cufftDoubleComplex *data_in_d, *data_out_d;

    cudaEvent_t start_cufft;
    cudaEvent_t stop_cufft;
    cudaEventCreate(&start_cufft);
    cudaEventCreate(&stop_cufft);


    data_in = new cufftDoubleComplex[dim_x*dim_y];
    data_out = new cufftDoubleComplex[dim_x*dim_y];

//    std::cout<< "Data in FFTW -------------" << std::endl;
    for (int i=0; i<dim_y; i++){
        for (int j=0; j<dim_x; j++){
            data_in[i*dim_x + j].x = x_n[i*dim_x + j]; 	// real data
            data_in[i*dim_x + j].y = 0.0; 		// imaginary data
//            std::cout << data_in[i*dim_y + j][0] << " - "<< data_in[i*dim_y + j][1] << std::endl;
        }
    }

    // cuda stuff
    cufftResult cufftResult_t;
    cufftHandle plan;
    cufftPlan2d(&plan, dim_x, dim_y, CUFFT_Z2Z);

    cudaMalloc(&data_in_d, sizeof(cufftDoubleComplex)*dim_x*dim_y);
    cudaMalloc(&data_out_d, sizeof(cufftDoubleComplex)*dim_x*dim_y);
    cudaMemcpy(data_in_d, data_in, sizeof(cufftDoubleComplex)*dim_x*dim_y, cudaMemcpyHostToDevice);


//    auto start_global = high_resolution_clock::now();
    cudaEventRecord(start_cufft);

    cufftResult_t = cufftExecZ2Z(plan, (cufftDoubleComplex *)data_in_d, (cufftDoubleComplex *)data_out_d, CUFFT_FORWARD);
    assert(cufftResult_t == CUFFT_SUCCESS);
    cudaDeviceSynchronize();

    cudaEventRecord(stop_cufft);


    float cublasTime;
    cudaEventSynchronize(stop_cufft);
    cudaEventElapsedTime(&cublasTime, start_cufft, stop_cufft);
    std::cout << "cufft took[ms]: " << cublasTime << std::endl;


    cufftDestroy(plan);
    cudaFree(data_in_d); cudaFree(data_out_d);
    delete[](data_in);
    delete[](data_out);
}




int main(int argv, char** argc){
    printf("Comparing DCT tensorcores , cublas and cufft\n");

    int size_m = 16;
    int size_n = 8;

    int dim_y = size_m;
    int dim_x = size_n;

    printf("dim_y %d   dim_x %d\n",dim_y, dim_x);

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

    thrust::device_vector<double> x_tmp(seq_size, 0.0);
    double *x_tmp_ptr = thrust::raw_pointer_cast(&x_tmp[0]);

    cudaMemcpy(x_n_ptr, x_n_host, sizeof(double)*size_m*size_n, cudaMemcpyHostToDevice);

    // Printing!
    //    print_dvector(x_n, "x_n");
    //    print_dvector(x_k, "x_k");
    //    print_dvector(x_n, "x_n");

    // cublas
    cublas_dct(dim_y, dim_x, x_n, x_k);
    print_dvector(x_k, "x_k");

    cublas_idct(dim_y, dim_x, x_n, x_tmp);
    print_dvector(x_tmp, "x_tmp");
//    print_dvector(x_tmp, "x_tmp");



    delete[] x_n_host;
    delete[] m_line;
    delete[] n_line;

    return 0;
}
