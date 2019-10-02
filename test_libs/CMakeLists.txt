cmake_minimum_required(VERSION 3.13)
project(tensorDCT LANGUAGES CXX CUDA)

include(CTest)


# Request that particles be built with -std=c++11
# As this is a public compile feature anything that links to
# particles will also build with -std=c++11
target_compile_features(fftw_1d PUBLIC cxx_std_11)

# We need to explicitly state that we need all CUDA files in the
# particle library to be built with -dc as the member functions
# could be called by other libraries and executables
set_target_properties( fftw_1d PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(fftw_1d.cu)

set_property(TARGET fftw_1d_test
        PROPERTY CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(fftw_1d_test PRIVATE fftw_1d)