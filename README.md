# tensorDCT
Discrete Cosine Transform optimized by using NVIDIA Tensorcore

## Info
 
### About cufft_test fodler

On Ubuntu, BLAS and LAPACK can be installed in one command:
	
		sudo apt-get install liblapack-dev -y ; sudo apt-get install liblapack3 -y ; sudo apt-get install libopenblas-base -y ; sudo apt-get install libopenblas-dev -y ;

Therefore, include library in the Makefile like:
	
		g++ ... -L/usr/lib -llapack -lblas
