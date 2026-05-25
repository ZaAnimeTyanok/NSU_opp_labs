#include <iostream>
#include <vector>
#include <cmath>
#include <mpi.h>

using namespace std;

#define ROUND_SHARING_TAG 1

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
    
    int next;
    int prev;
    
    std::vector<double> local_x;
    std::vector<double> local_b; 
    std::vector<double> local_Ax;     
    
    double vector_norm(const std::vector<double>& local_v) {
        double global_sum;
        double local_sum = 0.0;

        for (double x : local_v){
            local_sum += x * x;
        }

        MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        
        return std::sqrt(global_sum);
    }
    
    void round_sharing(std::vector<double>& local_v, int new_size){
        if (mpi_size <= 1){
            return;
        } 

        std::vector<double> new_v(new_size);

        if (mpi_rank %2 == 0){
            MPI_Send(local_v.data(), local_v.size(), MPI_DOUBLE, next, ROUND_SHARING_TAG, MPI_COMM_WORLD);
            MPI_Recv(new_v.data(), new_size, MPI_DOUBLE, prev, ROUND_SHARING_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        else {
            MPI_Recv(new_v.data(), new_size, MPI_DOUBLE, prev, ROUND_SHARING_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Send(local_v.data(), local_v.size(), MPI_DOUBLE, next, ROUND_SHARING_TAG, MPI_COMM_WORLD);
        }

        local_v = new_v;
    }

    void calculate_local_Ax() {
        local_Ax = std::vector<double>(row_count, 0.0);
        int offset = start_row;

        for (int step = 0; step < mpi_size; step++){
            for (int i = 0; i < row_count; i++) {
                int global_row = start_row + i;

                for (int j = 0; j < (int)local_x.size(); j++) {
                    local_Ax[i] += local_x[j] * (1.0 + static_cast<double>(j + offset == global_row));
                }
            }

            if (mpi_size > 1){
                int prev_size = all_row_counts[(prev - step + mpi_size) % mpi_size];
                offset = (offset + prev_size) % n;
                
                round_sharing(local_x, prev_size);
            }
        }
    }
    
    void run() {
        double start_time = MPI_Wtime();
        int iter = 0;

        const double b_norm = vector_norm(local_b);

        while (true) {
            calculate_local_Ax();
            
            std::vector<double> local_r(row_count);
                
            for (int i = 0; i < row_count; i++){
                local_r[i] = local_Ax[i] - local_b[i];
            }

            double r_norm = vector_norm(local_r);
            double res = r_norm / b_norm;
            
            if (iter % 1000 == 0 && mpi_rank == 0) {
                std::cout << "Iterations " << iter << ", residual = " << res << ", time " << MPI_Wtime() - start_time << std::endl;
            }

            if (res < eps){
                break;
            }
            
            for (int i = 0; i < row_count; i++) {
                local_x[i] -= tau * local_r[i];
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
        
        all_start_rows.resize(mpi_size);
        all_row_counts.resize(mpi_size);
            
        int offset = 0;
        for (int i = 0; i < mpi_size; i++) {
            all_start_rows[i] = offset;
            all_row_counts[i] = rows_per_proc + (int)(i < remainder);
            offset += all_row_counts[i];
        }
            
        row_count = all_row_counts[mpi_rank];
        start_row = all_start_rows[mpi_rank];
        
        next = (mpi_rank + 1) % mpi_size;
        prev = (mpi_rank == 0) ? mpi_size - 1 : mpi_rank - 1;

        local_x = std::vector<double>(row_count, 0.0);
        local_b = std::vector<double>(row_count, 0.0);
    }
    
    ~ProcessManager() {
        MPI_Finalize();
    }
    
    void run_simple_test() {
        local_b = std::vector<double>(row_count, n + 1);
        local_x = std::vector<double>(row_count, 0.0);
        
        run();

        
        double max_error = 0.0;
        for (int i = 0; i < row_count; i++) {
            max_error = std::max(max_error, std::fabs(local_x[i] - 1.0));
        }

        double global_max_error;
        MPI_Allreduce(&max_error, &global_max_error, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

        if (mpi_rank == 0) {
            std::cout << "\n========================================" << std::endl;
            std::cout << "LAB1 V2 SIMPLE TEST" << std::endl;
            std::cout << "========================================" << std::endl;
            std::cout << "Process count:  " << mpi_size << std::endl;
            std::cout << "Matrix size (N) = " << n << std::endl;
            std::cout << "Iterations: " << final_iterations<< std::endl;
            std::cout << "Max error: " << global_max_error << std::endl;
            std::cout << "Total time: " << final_time << " seconds" << std::endl;
            std::cout << "========================================" << std::endl;
        }
    }
    
    void run_not_simple_test() {
        local_x.resize(row_count);
        local_b.resize(row_count);

        std::vector<double> u(row_count);

        for (int i = 0; i < row_count; i++) {
            u[i] = std::sin(2.0 * M_PI * (start_row + i) / n);
        }

        local_x = u;
        calculate_local_Ax();
        
        local_b = local_Ax;
        local_Ax.clear();

        local_x.assign(row_count, 0.0);

        run();

        double max_error = 0.0;
        for (int i = 0; i < row_count; i++) {
            max_error = std::max(max_error, std::fabs(local_x[i] - u[i]));
        }

        double global_max_error;
        MPI_Allreduce(&max_error, &global_max_error, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

        if (mpi_rank == 0) {
            std::cout << "\n========================================" << std::endl;
            std::cout << "LAB1 V2 NOT SIMPLE TEST" << std::endl;
            std::cout << "========================================" << std::endl;
            std::cout << "Process count:  " << mpi_size << std::endl;
            std::cout << "Matrix size (N) = " << n << std::endl;
            std::cout << "Iterations: " << final_iterations<< std::endl;
            std::cout << "Max error: " << global_max_error << std::endl;
            std::cout << "Total time: " << final_time << " seconds" << std::endl;
            std::cout << "========================================" << std::endl;
        }
    }
};

int main(int argc, char** argv) {
    try {
        int N = 900;
        ProcessManager pm(argc, argv, N);
        
        // pm.run_simple_test();
        pm.run_not_simple_test();
        
    } 

    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
