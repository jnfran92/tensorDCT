cmake_minimum_required(VERSION 3.13)
project(tensorDCT LANGUAGES CXX CUDA)

add_library(whatever STATIC
        fftw_1d.cu
        fftw_2d.cu
        matmul_cublas.cu
        matmul_fft_base.cu
        )


add_custom_target(whatever2 COMMAND make test_1)