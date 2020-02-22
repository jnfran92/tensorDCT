//
// Created by Juan Francisco on 2020-02-22.
//

#ifndef TENSORDCT_CUDA_STUFF_H
#define TENSORDCT_CUDA_STUFF_H

// CUDA Stuff
#include <cuda_runtime.h>
#include <cuda.h>
#include <device_launch_parameters.h>

// Thrust
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/sequence.h>
#include <thrust/functional.h>

#include <thrust/iterator/zip_iterator.h>

#include <thrust/for_each.h>



// CUBLAS and OTHER Libs
#include <cublas_v2.h>

#endif //TENSORDCT_CUDA_STUFF_H
