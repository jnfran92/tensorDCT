//
// Created by Juan Francisco on 2020-02-22.
//

#include "DctCuBlas.h"

/**
 *
 * iDCT Normalization functor: x_n_i*sigma,  sigma=(4/(x_size*y_size))
 * x_n_i is the result of iDCT matmul
 */
struct idct_norm_ftr{
    double sigma;
    idct_norm_ftr(double sigma) : sigma(sigma) {}
    __host__ __device__ double operator()(const double& x_n_i) const{
        return x_n_i*sigma;
    }
};


__global__ void fill_cosine_matrix_kernel(double* matrix, int N){

    int k = blockIdx.x;
    int n = blockIdx.y;

    double temp_val = cos(((M_PI*k)/(N-1))*n);
    matrix[k * N + n] = temp_val;
}

DctCuBlas::DctCuBlas(int dim_y, int dim_x) {

    int y_size = dim_y;    //m -> i
    int x_size = dim_x;     //n -> j

    c_x = thrust::device_vector<double>(x_size*x_size, 1.0);
    c_x_ptr = thrust::raw_pointer_cast(&c_x[0]);


    c_y = thrust::device_vector<double>(y_size*y_size, 1.0);
    c_y_ptr = thrust::raw_pointer_cast(&c_y[0]);


    // auxiliary vector for both DCT and iDCT
    tmp_k = thrust::device_vector<double>(x_size*y_size, 0.0);
    tmp_k_ptr = thrust::raw_pointer_cast(&tmp_k[0]);


    dim3 block_x(1, 1, 1); // fixed size
    dim3 grid_x(x_size, x_size, 1);

    fill_cosine_matrix_kernel<<<grid_x, block_x>>>(c_x_ptr, x_size);
    cudaDeviceSynchronize();


    dim3 block_y(1, 1, 1); // fixed size
    dim3 grid_y(y_size,y_size, 1);


    fill_cosine_matrix_kernel<<<grid_y, block_y>>>(c_y_ptr, y_size);
    cudaDeviceSynchronize();

    // CuBLAS creation
    cublasCreate(&cublasHandle);
}

DctCuBlas::DctCuBlas() {

}

void DctCuBlas::dct(thrust::device_vector<double> &x_n, thrust::device_vector<double> &x_k) {

    double *x_n_ptr = thrust::raw_pointer_cast(&x_n[0]);
    double *x_k_ptr = thrust::raw_pointer_cast(&x_k[0]);

    const double alpha = 1.0;
    const double beta = 0.0;
    const double *alpha_ptr = &alpha;
    const double *beta_ptr = &beta;

    cublasDgemm(
            cublasHandle,	// handle
            CUBLAS_OP_N,	// no trans a
            CUBLAS_OP_N,	// no trans b
            dim_x,			// m
            dim_y,			// n
            dim_y,			// k
            alpha_ptr, 		// alpha
            x_n_ptr,		// a matrix
            dim_x,			// lda
            c_y_ptr,		// b matrix
            dim_y,			// ldb
            beta_ptr,		// beta
            tmp_k_ptr,		// c matrix
            dim_x			// ldc
    );

    cublasDgemm(
            cublasHandle,	// handle
            CUBLAS_OP_N,	// no trans a
            CUBLAS_OP_N,	// no trans b
            dim_x,			// m
            dim_y,			// n
            dim_x,			// k
            alpha_ptr, 		// alpha
            c_x_ptr,		// a matrix
            dim_x,			// lda
            tmp_k_ptr,		// b matrix
            dim_x,			// ldb
            beta_ptr,		// beta
            x_k_ptr,		// c matrix
            dim_x		    // ldc
    );
}

void DctCuBlas::idct(thrust::device_vector<double> &x_k, thrust::device_vector<double> &x_n) {
// naive idct
    dct(x_k, x_n);
    thrust::transform(
            x_n.begin(),
            x_n.end(),
            x_n.begin(),
            idct_norm_ftr(4.0/(double(dim_x)*double(dim_y)))
    );
}
