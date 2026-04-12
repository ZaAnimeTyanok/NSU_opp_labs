#include <iostream>
#include <vector>
#include <cmath>
#include <mpi.h>

using namespace std;

#define EXCHANGE_AX_DATA_TAG 1
#define STOP_SIGNAL_TAG 2
#define CONTINUE_SIGNAL_TAG 3
#define EXCHANGE_X_DATA_TAG 4

class ProcessManager {
private:
    int n;
    int mpi_rank;
    int mpi_size;
    int row_count;
    int start_row;
    
    double tau = 1e-3;
    double eps = 1e-5;

    int final_iterations = 0;
    double final_time = 0.0;
    
    std::vector<int> all_start_rows;
    std::vector<int> all_row_counts;
    
    std::vector<double> local_x;
    std::vector<double> local_b; 
    std::vector<double> local_Ax;     
    
    double vector_norm(const std::vector<double>& v) {
        double sum = 0;
        for (double val : v) {
            sum += val * val;
        }
        return std::sqrt(sum);
    }
    
    std::vector<double> gather_full_vector(const std::vector<double>& local_part) {
        std::vector<double> full_vector(n);
        
        if (mpi_rank == 0) {
            std::copy(local_part.begin(), local_part.end(), full_vector.begin() + start_row);
        
            for (int src = 1; src < mpi_size; src++) {
                MPI_Recv(&full_vector[all_start_rows[src]], all_row_counts[src], MPI_DOUBLE,
                         src, EXCHANGE_X_DATA_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }

            for (int dest = 1; dest < mpi_size; dest++) {
                MPI_Send(full_vector.data(), n, MPI_DOUBLE, dest, EXCHANGE_X_DATA_TAG, MPI_COMM_WORLD);
            }
            
        } else {
            MPI_Send(local_part.data(), row_count, MPI_DOUBLE, 0, EXCHANGE_X_DATA_TAG, MPI_COMM_WORLD);
            
            MPI_Recv(full_vector.data(), n, MPI_DOUBLE, 0, EXCHANGE_X_DATA_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        
        return full_vector;
    }
    
    std::vector<double> compute_local_Ax(const std::vector<double>& full_x) {
        std::vector<double> result(row_count, 0.0);
        
        for (int i = 0; i < row_count; i++) {
            int global_row = start_row + i;
            
            for (int j = 0; j < n; j++) {
                result[i] += full_x[j] * (1 + (int)(j == global_row));
            }
        }

        return result;
    }
    
    void run() {
        std::vector<double> full_b = gather_full_vector(local_b);
        double b_norm = vector_norm(full_b);
        
        double start_time = MPI_Wtime();
        int iter = 0;
        
        MPI_Status status;
        
        while (true) {
            std::vector<double> full_x = gather_full_vector(local_x);
            
            local_Ax = compute_local_Ax(full_x);
            
            if (mpi_rank == 0) {
                std::vector<double> full_Ax(n);
                std::copy(local_Ax.begin(), local_Ax.end(), full_Ax.begin() + start_row);
                
                for (int src = 1; src < mpi_size; src++) {
                    MPI_Recv(&full_Ax[all_start_rows[src]], all_row_counts[src], MPI_DOUBLE,
                             src, EXCHANGE_AX_DATA_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                }
                
                std::vector<double> r_vector(n);
                for (int i = 0; i < n; i++) {
                    r_vector[i] = full_Ax[i] - full_b[i];
                }
                
                double r_norm = vector_norm(r_vector);
                
                if (iter % 100 == 0) {
                    std::cout << "Iter " << iter << ", residual = " << r_norm / b_norm << std::endl;
                }
                
                if (r_norm / b_norm < eps) {
                    for (int dest = 1; dest < mpi_size; dest++) {
                        MPI_Send(NULL, 0, MPI_INT, dest, STOP_SIGNAL_TAG, MPI_COMM_WORLD);
                    }
                    break;
                }
                
                for (int i = 0; i < row_count; i++) {
                    int global_idx = start_row + i;
                    local_x[i] -= tau * r_vector[global_idx];
                }
            
                
            } else {
                MPI_Send(local_Ax.data(), row_count, MPI_DOUBLE, 0, EXCHANGE_AX_DATA_TAG, MPI_COMM_WORLD);
                
                MPI_Probe(0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
                
                if (status.MPI_TAG == STOP_SIGNAL_TAG) {
                    MPI_Recv(NULL, 0, MPI_INT, 0, STOP_SIGNAL_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    break;
                }
            }
            
            iter++;
        }
        
        double end_time = MPI_Wtime();
        final_iterations = iter;
        final_time = end_time - start_time;
    }
    
public:
    explicit ProcessManager(int& _argc, char**& _argv, int N) : n(N) {
        MPI_Init(&_argc, &_argv);
        MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
        MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
        
        int rows_per_proc = N / mpi_size;
        int remainder = N % mpi_size;
        
        if (mpi_rank == 0) {
            all_start_rows.resize(mpi_size);
            all_row_counts.resize(mpi_size);
            
            int offset = 0;
            for (int i = 0; i < mpi_size; i++) {
                all_start_rows[i] = offset;
                all_row_counts[i] = rows_per_proc + (int)(i < remainder);
                offset += all_row_counts[i];
            }
            
            row_count = all_row_counts[0];
            start_row = all_start_rows[0];
        } 
        
        else {
            row_count = rows_per_proc + (int)(mpi_rank < remainder);
            int start = 0;
            for (int i = 0; i < mpi_rank; i++) {
                start += rows_per_proc + (int)(i < remainder);
            }
            start_row = start;
        }
        
        local_x = std::vector<double>(row_count, 0.0);
        local_b = std::vector<double>(row_count, 0.0);
    }
    
    ~ProcessManager() {
        MPI_Finalize();
    }
    
    void run_simple_test() {
        for (int i = 0; i < row_count; i++) {
            local_b[i] = n + 1.0;
        }
        
        run();
        
        if (mpi_rank == 0) {
            std::vector<double> full_x = gather_full_vector(local_x);
            
            std::cout << "\n========================================" << std::endl;
            std::cout << "VARIANT 2: Vectors distributed" << std::endl;
            std::cout << "========================================" << std::endl;
            std::cout << "Converged in " << final_iterations << " iterations" << std::endl;
            std::cout << "Time: " << final_time << " seconds" << std::endl;
            
            double max_error = 0.0;
            for (int i = 0; i < n; i++) {
                max_error = std::max(max_error, std::fabs(full_x[i] - 1.0));
            }
            
            std::cout << "Max error: " << max_error << std::endl;
            std::cout << "========================================" << std::endl;
        }
    }
    
    void run_not_simple_test() {
        std::vector<double> u(n);
        for (int i = 0; i < n; i++) {
            u[i] = std::sin(2.0 * M_PI * i / n);
        }
        
        std::vector<double> full_u = u;
        std::vector<double> local_b_temp(row_count, 0.0);
        for (int i = 0; i < row_count; i++) {
            int global_row = start_row + i;
            for (int j = 0; j < n; j++) {
                local_b_temp[i] += full_u[j] * (1 + (int)(j == global_row));
            }
        }
        

        local_b = local_b_temp;
        
        run();
        
        if (mpi_rank == 0) {
            std::vector<double> full_x = gather_full_vector(local_x);
            
            std::cout << "\n========================================" << std::endl;
            std::cout << "VARIANT 2: NOT SIMPLE TEST (arbitrary solution)" << std::endl;
            std::cout << "========================================" << std::endl;
            std::cout << "Converged in " << final_iterations << " iterations" << std::endl;
            std::cout << "Time: " << final_time << " seconds" << std::endl;
            
            double max_error = 0.0;
            for (int i = 0; i < n; i++) {
                max_error = std::max(max_error, std::fabs(full_x[i] - u[i]));
            }
            
            std::cout << "Max error: " << max_error << std::endl;
            std::cout << "========================================" << std::endl;
        }
    }
};

int main(int argc, char** argv) {
    try {
        int N = 1000;
        ProcessManager pm(argc, argv, N);
        
        pm.run_simple_test();
        pm.run_not_simple_test();
        
    } 

    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
