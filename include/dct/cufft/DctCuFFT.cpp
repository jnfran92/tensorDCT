//
// Created by Juan Francisco on 2020-02-22.
//

#include "DctCuFFT.h"

DctCuFFT::DctCuFFT(int dim_y, int dim_x) {

}

DctCuFFT::DctCuFFT() {

}

void DctCuFFT::dct(thrust::device_vector<double> &x_n, thrust::device_vector<double> &x_k) {

}

void DctCuFFT::idct(thrust::device_vector<double> &x_k, thrust::device_vector<double> &x_n) {


//    // FFTW two dimensions using m_line.dot(n_line) = x_n
//    fftw_complex *data_in, *data_out;
//    fftw_plan p;
//    // allocating data
//    data_in = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * size_m * size_n);
//    data_out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * size_m * size_n);
//
//    // fill data
//    std::cout<< "Data in FFTW -------------" << std::endl;
//    for (int i=0; i<size_m; i++){
//        for (int j=0; j<size_n; j++){
//            data_in[i*size_n + j][0] = x_n[i*size_n + j]; 	// real data
//            data_in[i*size_n + j][1] = 0.0; 		// imaginary data
//            std::cout << data_in[i*size_n + j][0] << " - "<< data_in[i*size_n + j][1] << std::endl;
//        }
//    }
//
//    p = fftw_plan_dft_2d(
//            size_m,
//            size_n,
//            data_in,
//            data_out,
//            FFTW_FORWARD,
//            FFTW_ESTIMATE);
//
//    // executing fft
//    fftw_execute(p);
//
//    std::cout << "results----- " << std::endl;
//    // fill data


}
