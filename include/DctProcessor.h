//
// Created by Juan Francisco on 2020-02-22.
//

#ifndef TENSORDCT_DCTPROCESSOR_H
#define TENSORDCT_DCTPROCESSOR_H

#include "cuda_stuff.h"

class DctProcessor {

    /**
     * Discrete Cosine Transform
     * @param x_n_ptr input matrix pointer - time/space domain
     * @param x_k_ptr output matrix pointer applied dct on it.
     */
    virtual void dct(thrust::device_vector<double> &x_n, thrust::device_vector<double> &x_k) = 0;

    /**
     * Inverse Discrete Cosine Transform
     * @param x_k_ptr input matrix pointer freq domain
     * @param x_n_ptr output matrix pointer applied idct on it.
     */
    virtual void idct(thrust::device_vector<double> &x_k, thrust::device_vector<double> &x_n) = 0;

};


#endif //TENSORDCT_DCTPROCESSOR_H
