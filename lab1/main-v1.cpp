#include <iostream>
#include <vector>
#include <cmath>
#include <mpi.h>

#define EXCHANGE_AX_DATA_TAG 1
#define STOP_SIGNAL_TAG 2
#define CONTINUE_SIGNAL_TAG 3

class ProcessManager {
private:
    int n;
    int mpi_rank;
    int mpi_size;
    int row_count;
    int start_row;
    
    double tau = 1e-4;
    double eps = 1e-4;

    int final_iterations = 0;
    double final_time = 0.0;
    
    std::vector<int> all_start_rows;
    std::vector<int> all_row_counts;
    std::vector<double> x_vector;
    std::vector<double> b_vector;
    
    double vector_norm(const std::vector<double>& v) {
        double sum = 0;
        for (double val : v) {
            sum += val * val;
        }
        return std::sqrt(sum);
    }
    
    std::vector<double> get_local_Ax() {
        std::vector<double> local_Ax(row_count, 0.0);
        
        for (int i = 0; i < row_count; i++) {
            int global_A_row = start_row + i;
            for (int j = 0; j < n; j++) {
                local_Ax[i] += x_vector[j] * (1.0 + static_cast<double>(j == global_A_row));
            }
        }

        return local_Ax;
    }
    
    void run() {
        const double b_norm = vector_norm(b_vector);
        double start_time = MPI_Wtime();
        int iter = 0;
        
        MPI_Status status;

        while (true) {
            std::vector<double> local_Ax = get_local_Ax();
            
            if (mpi_rank == 0) {
                std::vector<double> full_Ax(n);
                std::copy(local_Ax.begin(), local_Ax.end(), full_Ax.begin() + start_row);
                
                for (int src = 1; src < mpi_size; src++) {
                    MPI_Recv(&full_Ax[all_start_rows[src]], all_row_counts[src], MPI_DOUBLE,
                             src, EXCHANGE_AX_DATA_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                }
                
                std::vector<double> r_vector(n);
                
                for (int i = 0; i < n; i++) {
                    r_vector[i] = full_Ax[i] - b_vector[i];
                }
                
                double r_norm = vector_norm(r_vector);
                double res = r_norm / b_norm;

                if (iter % 1000 == 0) {
                    std::cout << "Iter " << iter << ", residual = " << res << ", time " << MPI_Wtime() - start_time << std::endl;
                }
                
                if (res < eps) {
                    for (int dest = 1; dest < mpi_size; dest++) {
                        MPI_Send(NULL, 0, MPI_INT, dest, STOP_SIGNAL_TAG, MPI_COMM_WORLD);
                    }
                    
                    break;
                }

                for (int i = 0; i < n; i++) {
                    x_vector[i] -= tau * r_vector[i];
                }
                
                for (int dest = 1; dest < mpi_size; dest++) {
                    MPI_Send(x_vector.data(), n, MPI_DOUBLE, dest, CONTINUE_SIGNAL_TAG, MPI_COMM_WORLD);
                }
                
            } else {
                MPI_Send(local_Ax.data(), row_count, MPI_DOUBLE, 0, EXCHANGE_AX_DATA_TAG, MPI_COMM_WORLD);
                
                MPI_Probe(0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
                
                if (status.MPI_TAG == STOP_SIGNAL_TAG) {
                    MPI_Recv(NULL, 0, MPI_INT, 0, STOP_SIGNAL_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    break;
                } else if (status.MPI_TAG == CONTINUE_SIGNAL_TAG) {
                    MPI_Recv(x_vector.data(), n, MPI_DOUBLE, 0, CONTINUE_SIGNAL_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
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
        
        x_vector = std::vector<double>(n, 0.0);
        b_vector = std::vector<double>(n, 0.0);
    }
    
    ProcessManager(const ProcessManager&) = delete;
    ProcessManager& operator=(const ProcessManager&) = delete;
    
    ~ProcessManager() {
        MPI_Finalize();
    }
    
    void run_simple_test() {
        b_vector = std::vector<double>(n, n + 1.0);
        x_vector = std::vector<double>(n, 0.0);
        
        run();

        if (mpi_rank == 0) {
            double max_error = 0.0;
            for (int i = 0; i < n; i++) {
                max_error = std::max(max_error, std::fabs(x_vector[i] - 1.0));
            }

            std::cout << "\n========================================" << std::endl;
            std::cout << "LAB1 V1 SIMPLE TEST" << std::endl;
            std::cout << "========================================" << std::endl;
            std::cout << "Process count:  " << mpi_size << std::endl;
            std::cout << "Matrix size (N) = " << n << std::endl;
            std::cout << "Iterations: " << final_iterations<< std::endl;
            std::cout << "Max error: " << max_error << std::endl;
            std::cout << "Total time: " << final_time << " seconds" << std::endl;
            std::cout << "========================================" << std::endl;
        }
    }
    
    void run_not_simple_test() {
        std::vector<double> u(n);
        
        for (int i = 0; i < n; i++) {
            u[i] = std::sin(2.0 * M_PI * i / n);
        }
    
        x_vector = u;

        std::vector<double> local_b = get_local_Ax();

        if (mpi_rank == 0) {
            std::vector<double> full_b(n);
            std::copy(local_b.begin(), local_b.end(), full_b.begin() + start_row);
        
            for (int src = 1; src < mpi_size; src++) {
                MPI_Recv(&full_b[all_start_rows[src]], all_row_counts[src], MPI_DOUBLE,
                        src, EXCHANGE_AX_DATA_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        
            b_vector = full_b;

            for (int dest = 1; dest < mpi_size; dest++) {
                MPI_Send(b_vector.data(), n, MPI_DOUBLE, dest, CONTINUE_SIGNAL_TAG, MPI_COMM_WORLD);
            }
        
        }
        
        else {
            MPI_Send(local_b.data(), row_count, MPI_DOUBLE, 0, EXCHANGE_AX_DATA_TAG, MPI_COMM_WORLD);
            MPI_Recv(b_vector.data(), n, MPI_DOUBLE, 0, CONTINUE_SIGNAL_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    
        x_vector = std::vector<double>(n, 0.0);
    
        run();
    
        if (mpi_rank == 0) {
            std::cout << "\n========================================" << std::endl;
            std::cout << "LAB1 V1 NOT SIMPLE TEST" << std::endl;
            std::cout << "========================================" << std::endl;
            std::cout << "Process count:  " << mpi_size << std::endl;
            std::cout << "Matrix size (N) = " << n << std::endl;
            std::cout << "Iterations: " << final_iterations<< std::endl;
        
            double max_error = 0.0;
                for (int i = 0; i < n; i++) {
                    max_error = std::max(max_error, std::fabs(x_vector[i] - u[i]));
                }
            
            std::cout << "Max error: " << max_error << std::endl;
            std::cout << "Total time: " << final_time << " seconds" << std::endl;
            std::cout << "========================================" << std::endl;
        }
    }   
};

int main(int argc, char** argv) {
    try {
        int N = 900;
        ProcessManager pm(argc, argv, N);
        //pm.run_simple_test();
        pm.run_not_simple_test();
    } 
    
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
