# Test FFT, Matmul, in 1D, 2D, 3D

Testing basic working of : `fftw`, `cblas`, `lapack`, etc libraries
Each test is verified with data generated in Python.

Test status:

- test_1 `matmul_fft_bases.cu` : Matrix multipliaction using `cblas` is working fine: vectors product to generate the Fourier Bases :white_check_mark:

- test_2 `fftw_1d`: 1D fftw compared with python :white_check_mark: 

- test_3 `fftw_2d`: 2D fftw compared with python :white_check_mark:

- test_4 `matmul_cublas.cu`: 2D matrix multiplication 

## Notes

To install FFTW:

	sudo apt-get update
	sudo apt-get install fftw-dev

it also works:

	sudo apt-get install libfftw3-dev libfftw3-doc

For `nvcc` if it gets errors(`__float128 undefined`) with FFTW then , modifiy the`fftw3.h` file and change:

	/* __float128 (quad precision) is a gcc extension on i386, x86_64, and ia64     
	   for gcc >= 4.6 (compiled in FFTW with --enable-quad-precision) */
	#if (__GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ >= 6)) \
	 && !(defined(__ICC) || defined(__INTEL_COMPILER)) \
	 && (defined(__i386__) || defined(__x86_64__) || defined(__ia64__))



with:

	/* __float128 (quad precision) is a gcc extension on i386, x86_64, and ia64     
	   for gcc >= 4.6 (compiled in FFTW with --enable-quad-precision) */
	#if (__GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ >= 6)) \
	 && !(defined(__ICC) || defined(__INTEL_COMPILER) || defined(__CUDACC__)) \
	 && (defined(__i386__) || defined(__x86_64__) || defined(__ia64__))

For Ubunut to install BLAS and LAPACK use:

	sudo apt-get install liblapack-dev -y ; sudo apt-get install liblapack3 -y ; sudo apt-get install libopenblas-base -y ; sudo apt-get install libopenblas-dev -y ;

Therefore, include library in the Makefile like:

	g++ ... -L/usr/lib -llapack -lblas

If there is a problem with shared libraries(ex: libgfortran..), see [this](https://www.cprogramming.com/tutorial/shared-libraries-linux-gcc.html).

## Reference

- https://docs.nvidia.com/cuda/cufft/index.html




