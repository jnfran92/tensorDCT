cmake_minimum_required(VERSION 3.13)
project(tensorDCT LANGUAGES CXX)


include_directories(
        "/Users/Shared/Relocated Items/Security/Developer/NVIDIA/CUDA-9.2/include"
        "./third_party/"
        "/Users/Juan/cuda_mac_installer_tk/Developer/NVIDIA/CUDA-10.2/include"
        "/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/System/Library/Frameworks/Accelerate.framework/Versions/Current/Frameworks/vecLib.framework/Headers/"
)


add_library(whatever STATIC
        fftw_1d.cu
        fftw_2d.cu
        matmul_cublas.cu
        matmul_fft_base.cu
        ../include/common_gpu_utils.h
        ../include/cuda_helper/cuda_include_helper.h ../include/dct/cublas/DctCuBlas.cpp ../include/dct/cublas/DctCuBlas.h ../include/dct/tensorcores/DctTC.cpp ../include/dct/tensorcores/DctTC.h ../comparing_dct.cpp ../include/dct/cufft/DctCuFFT.cpp ../include/dct/cufft/DctCuFFT.h ../gpu_comparision/cublas_vs_cufft.cpp)


add_custom_target(whatever2 COMMAND make test_1)