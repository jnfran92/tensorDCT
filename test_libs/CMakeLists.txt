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
        ../include/DctProcessor.cpp
        ../include/DctProcessor.h ../include/cuda_stuff.h)


add_custom_target(whatever2 COMMAND make test_1)