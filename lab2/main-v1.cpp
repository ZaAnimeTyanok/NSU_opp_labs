#include <iostream>
#include <vector>
#include <cmath>
#include <omp.h>

static const int N = 900;
static const double eps = 1e-4;
static const double tau = 1e-4;

static double start_time = 0;
static int iter = 0;


double vector_norm(const std::vector<double>& v){
    double sum = 0;

    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < v.size(); i++){
        sum += v[i] * v[i];
    }

    return std::sqrt(sum);
}

std::vector<double> get_Ax(const std::vector<double>& x_vector){
    std::vector<double> Ax_vector(N, 0.0);

    #pragma omp parallel for
    for (int i = 0; i < N; i++){
        for (int j = 0; j < N; j++){
            Ax_vector[i] += x_vector[j] * (1.0 + static_cast<double>(i == j));       
        }
    }

    return Ax_vector;
}

std::vector<double> find_x_vector(std::vector<double>& b_vector){
    iter = 0;

    const double b_norm = vector_norm(b_vector);
    
    std::vector<double> x_vector(N, 0.0);

    while (true){
        std::vector<double> Ax_vector = get_Ax(x_vector);
        std::vector<double> r_vector(N);
        

        double r_norm_sq = 0.0;
        
        #pragma omp parallel for reduction(+:r_norm_sq)
        for (int i = 0; i < N; i++){
            r_vector[i] = Ax_vector[i] - b_vector[i];
            r_norm_sq += r_vector[i] * r_vector[i];
        }

        double r_norm = std::sqrt(r_norm_sq);

        double res = r_norm / b_norm;

        if (iter % 1000 == 0) {
            double now_time = omp_get_wtime();    
            std::cout << "Iterations " << iter << ", residual = " << res << ", time: " << now_time - start_time <<std::endl;
        }

        if (res < eps){
            break;   
        }
        
        #pragma omp parallel for
        for (int i = 0; i < N; i++) {
            x_vector[i] -= tau * r_vector[i];
        }

        iter++;
    }

    return x_vector;
}

void simple_test(){
    std::vector<double> b_vector(N, N + 1.0);
    std::vector<double> x_vector = find_x_vector(b_vector);

    double max_error = 0;

    #pragma omp parallel for reduction(max:max_error)
    for (int i = 0; i < N; i++){
        max_error = std::max(max_error, std::fabs(x_vector[i] - 1.0));
    }
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "LAB2 V1 SIMPLE TEST" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Threads: " << omp_get_max_threads() << std::endl;
    std::cout << "Iterations: " << iter << std::endl;
    std::cout << "Max error: " << max_error << std::endl;
    
    iter = 0;
}

void not_simple_test(){
    std::vector<double> u(N);

    for (int i = 0; i < N; i++) {
        u[i] = std::sin(2.0 * M_PI * i / N);
    }

    std::vector<double> b_vector = get_Ax(u);
    std::vector<double> x_vector = find_x_vector(b_vector);

    double max_error = 0;
    
    #pragma omp parallel for reduction(max:max_error)
    for (int i = 0; i < N; i++){
        max_error = std::max(max_error, std::fabs(x_vector[i] - u[i]));
    }

    std::cout << "\n========================================" << std::endl;
    std::cout << "LAB2 V1 NOT SIMPLE TEST" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Threads: " << omp_get_max_threads() << std::endl;
    std::cout << "Matrix size (N) = " << N << std::endl;
    std::cout << "Iterations: " << iter << std::endl;
    std::cout << "Max error: " << max_error << std::endl;

    iter = 0;
}

int main(){
    start_time = omp_get_wtime();
    
    //simple_test();
    not_simple_test();
    
    double end_time = omp_get_wtime();
    
    std::cout << "Total time: " << end_time - start_time << " seconds" << std::endl;
    std::cout << "========================================" << std::endl;
    
    return 0;
}