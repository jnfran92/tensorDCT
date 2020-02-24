//
// Created by Juan Francisco on 2019-12-11.
//

#ifndef BOOSTED_ARD_MAIN_H
#define BOOSTED_ARD_MAIN_H



/**
 * Gaussian functor applied for each element
 */
struct gaussian_ftr{
    double A;
    double mu;
    double sigma;
    gaussian_ftr(double A, double mu, double sigma) : A(A), mu(mu), sigma(sigma) {}
    __host__ __device__ double operator()(const double& x) const{
        return A*exp(-pow(x - mu, 2.0)/(2.0*pow(sigma, 2.0)));
    }
};


/**
 * Display each element of a vector
 */
struct display_vector_ftr{
    __host__ __device__ void operator()(float x){
//        std::cout << x << "  ";
        printf("%f ", x);
    }
};


/**
 * Print for host Thrust vectors
 * @param vector
 * @param description
 */
void print_dvector(thrust::host_vector<double> &vector, const std::string &description){
    std::cout << "Printing: " << description << "--- --- --- --- --- --- --- --- --- "<<std::endl;
    thrust::for_each(vector.begin(), vector.end(), display_vector_ftr());
    std::cout << std::endl;
}


/**
 * Print for device Thrust vectors
 * @param vector
 * @param description
 */
void print_dvector(thrust::device_vector<double> &vector, const std::string &description){
    thrust::host_vector<double> temp = vector;
    print_dvector(temp, description);
}



/**
 * Multiply each element of fdtd_matrix by the lambda value, for building x and y axis matrices.
 */
struct fdtd_kernel_builder_ftr{
    double lambda;
    explicit fdtd_kernel_builder_ftr(double lambda) : lambda(lambda) {}
    __host__ __device__ double operator()(const double& fdtd_value) const{
        return fdtd_value*lambda;
    }

};



#endif //BOOSTED_ARD_MAIN_H
