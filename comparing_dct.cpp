//
// Created by Juan Francisco on 2020-02-22.
//
#include <cstdio>
#include <fftw3.h>
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

#include <thrust/device_vector.h>
#include <cublas_v2.h>
#include <cufft.h>
#include <chrono>

using namespace std::chrono;

void cublas_dct(int &dim_y, int &dim_x, thrust::device_vector<double> &x_n, thrust::device_vector<double> &x_k){
    std::cout<< "cublas_dct" << std::endl;
    // DCT
    cudaEvent_t startcublas;
    cudaEvent_t stopcublas;
    DctCuBlas  dctCuBlas(dim_y, dim_x);

    cudaEventCreate(&startcublas);
    cudaEventCreate(&stopcublas);

    // Init DCT
    cudaEventRecord(startcublas);
    dctCuBlas.dct(x_n, x_k);
    cudaEventRecord(stopcublas);


    float cublasTime;
    cudaEventSynchronize(stopcublas);
    cudaEventElapsedTime(&cublasTime, startcublas, stopcublas);
    std::cout << "cublas took[ms]: " << cublasTime << std::endl;
}


void cublas_idct(int &dim_y, int &dim_x, thrust::device_vector<double> &x_n, thrust::device_vector<double> &x_k){
    std::cout<< "cublas_idct" << std::endl;
    // DCT
    cudaEvent_t startcublas;
    cudaEvent_t stopcublas;
    DctCuBlas  dctCuBlas(dim_y, dim_x);

    cudaEventCreate(&startcublas);
    cudaEventCreate(&stopcublas);

    // Init iDCT
    cudaEventRecord(startcublas);
    dctCuBlas.idct(x_k, x_n);
    cudaEventRecord(stopcublas);


    float cublasTime;
    cudaEventSynchronize(stopcublas);
    cudaEventElapsedTime(&cublasTime, startcublas, stopcublas);
    std::cout << "cublas took[ms]: " << cublasTime << std::endl;
}

void fftw_dct(int &dim_y, int &dim_x, double *x_n){
    std::cout<< "fftw_dct" << std::endl;
    // FFTW two dimensions using m_line.dot(n_line) = x_n
    fftw_complex *data_in, *data_out;
    fftw_plan p;
    // allocating data
    data_in = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * dim_x * dim_y);
    data_out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * dim_x * dim_y);

    // fill data
//    std::cout<< "Data in FFTW -------------" << std::endl;
    for (int i=0; i<dim_y; i++){
        for (int j=0; j<dim_x; j++){
            data_in[i*dim_x + j][0] = x_n[i*dim_x + j]; 	// real data
            data_in[i*dim_x + j][1] = 0.0; 		// imaginary data
//            std::cout << data_in[i*dim_y + j][0] << " - "<< data_in[i*dim_y + j][1] << std::endl;
        }
    }


    p = fftw_plan_dft_2d(
            dim_x,
            dim_y,
            data_in,
            data_out,
            FFTW_FORWARD,
            FFTW_ESTIMATE);


    auto start_global = high_resolution_clock::now();
    // executing fft
    fftw_execute(p);

    auto stop_global = high_resolution_clock::now();
    auto duration_t = duration_cast<milliseconds>(stop_global - start_global);
    std::cout << "fftw took[ms]: " << duration_t.count() << std::endl;
}


void cufft_dct(int &dim_y, int &dim_x, double *x_n){
    std::cout<< "cufft_dct" << std::endl;
    // FFTW two dimensions using m_line.dot(n_line) = x_n
    cufftComplex *data_in, *data_out;
    cufftComplex *data_in_d, *data_out_d;


    data_in = new cufftComplex[dim_x*dim_y];
    data_out = new cufftComplex[dim_x*dim_y];

//    std::cout<< "Data in FFTW -------------" << std::endl;
    for (int i=0; i<dim_y; i++){
        for (int j=0; j<dim_x; j++){
            data_in[i*dim_x + j].x = (float)x_n[i*dim_x + j]; 	// real data
            data_in[i*dim_x + j].y = 0.0; 		// imaginary data
//            std::cout << data_in[i*dim_y + j][0] << " - "<< data_in[i*dim_y + j][1] << std::endl;
        }
    }

    // cuda stuff
    cufftResult cufftResult_t;
    cufftHandle plan;
    cufftPlan2d(&plan, dim_x, dim_y, CUFFT_C2C);

    cudaMalloc(&data_in_d, sizeof(cufftComplex)*dim_x*dim_y);
    cudaMalloc(&data_out_d, sizeof(cufftComplex)*dim_x*dim_y);
    cudaMemcpy(data_in_d, data_in, sizeof(cufftComplex)*dim_x*dim_y, cudaMemcpyHostToDevice);


    auto start_global = high_resolution_clock::now();

    cufftResult_t = cufftExecC2C(plan, (cufftComplex *)data_in_d, (cufftComplex *)data_out_d, CUFFT_FORWARD);
    cudaDeviceSynchronize();
    assert(cufftResult_t == CUFFT_SUCCESS);


    auto stop_global = high_resolution_clock::now();
    auto duration_t = duration_cast<milliseconds>(stop_global - start_global);
    std::cout << "cuFFT took[ms]: " << duration_t.count() << std::endl;

    cufftDestroy(plan);
    cudaFree(data_in_d); cudaFree(data_out_d);
    delete[](data_in);
    delete[](data_out);
}




int main(int argv, char** argc){
    printf("Comparing DCT tensorcores , cublas and cufft\n");

    int size_m = 4096;
    int size_n = 4096;

    int dim_y = size_m;
    int dim_x = size_n;

    printf("dim_y %d   dim_x %d\n",dim_y, dim_x);

//    thrust::host_vector<>
    auto *x_n_host = new double[size_m * size_n];

    auto *m_line = new double[size_m];
    auto *n_line = new double[size_n];

    // Fill and Print
    fill_vector_cos(3, size_m, m_line);
    fill_vector_cos(2, size_n, n_line);

//    print_array(m_line, size_m);
//    print_array(n_line, size_n);

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

//    print_array(x_n_host , M, N);


    // GPU copies
    int seq_size = size_m * size_n;
    thrust::device_vector<double> x_n(seq_size, 0.0);
    double *x_n_ptr = thrust::raw_pointer_cast(&x_n[0]);

    thrust::device_vector<double> x_k(seq_size, 0.0);
    double *x_k_ptr = thrust::raw_pointer_cast(&x_k[0]);

    cudaMemcpy(x_n_ptr, x_n_host, sizeof(double)*size_m*size_n, cudaMemcpyHostToDevice);

    // Printing!
    //    print_dvector(x_n, "x_n");
    //    print_dvector(x_k, "x_k");
    //    print_dvector(x_n, "x_n");

    // cublas
    cublas_dct(dim_y, dim_x, x_n, x_k);
    cublas_idct(dim_y, dim_x, x_n, x_k);
    fftw_dct(dim_y, dim_x, x_n_host);
    cufft_dct(dim_y, dim_x, x_n_host);



    delete[] x_n_host;
    delete[] m_line;
    delete[] n_line;

    return 0;
}
