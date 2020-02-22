//
// Created by Juan Francisco on 2020-02-22.
//

#ifndef TENSORDCT_DCTCUFFT_H
#define TENSORDCT_DCTCUFFT_H

#include "../../cuda_helper/cuda_include_helper.h"


class DctCuFFT {
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

};


#endif //TENSORDCT_DCTCUFFT_H
