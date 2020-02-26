//
// Created by Juan Francisco on 2020-02-22.
//

#ifndef TENSORDCT_DCTCUBLAS_H
#define TENSORDCT_DCTCUBLAS_H

#include "../../cuda_helper/cuda_include_helper.h"
//#include <thrust/device_vector.h>
//#include <cublas_v2.h>

/**
 * 2D DCT using CuBLAS
 */
class DctCuBlas {
private:
    int dim_x;
    int dim_y;

    thrust::device_vector<double> c_x;
    double *c_x_ptr;        // cosine basis in x

    thrust::device_vector<double> c_y;
    double *c_y_ptr;        // cosine basis in y

    thrust::device_vector<double> tmp_k;
    double *tmp_k_ptr;      // tmp matrix for dct and idct operations

    cublasHandle_t cublasHandle;


public:
    /**
     * Cosine Transform processor constructor for 2D, x and y dim are needed.
     * @param dim_y
     * @param dim_x
     */
    DctCuBlas(int dim_y, int dim_x);

    DctCuBlas();

    /**
     * Discrete Cosine Transform using CBLAS and matmul
     * @param x_n_ptr input matrix pointer - time/space domain
     * @param x_k_ptr output matrix pointer applied dct on it.
     */
    void dct(thrust::device_vector<double> &x_n, thrust::device_vector<double> &x_k);

    /**
     * Discrete Cosine Inverse Transform using CBLAS and matmul
     * @param x_k_ptr input matrix pointer freq domain
     * @param x_n_ptr output matrix pointer applied idct on it.
     */
    void idct(thrust::device_vector<double> &x_k, thrust::device_vector<double> &x_n);
};


#endif //TENSORDCT_DCTCUBLAS_H
