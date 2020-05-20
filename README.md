# tensorDCT
Discrete Cosine Transform optimized by using NVIDIA Tensorcore

## Content

### Script `comparing_dct.cpp`:

Results:

        Comparing DCT tensorcores , cublas and cufft
        dim_y 3000   dim_x 3000
        cublas_dct
        cublas took[ms]: 18.1094
        cublas_idct
        cublas took[ms]: 18.3419
        fftw_dct
        fftw took[ms]: 507
        cufft_float_fft
        cufft took[ms]: 1.89949
        cufft_double_fft
        cufft took[ms]: 3.19274



 
 
### About cufft_test folder

On Ubuntu, BLAS and LAPACK can be installed in one command:
	
		sudo apt-get install liblapack-dev -y ; sudo apt-get install liblapack3 -y ; sudo apt-get install libopenblas-base -y ; sudo apt-get install libopenblas-dev -y ;

Therefore, include library in the Makefile like:
	
		g++ ... -L/usr/lib -llapack -lblas
