#include <iostream>
#include <vector>
#include <mpi.h>
#include <cstdlib>
#include <ctime>
#include <fstream>

#define GRID_DIM 2
#define ZERO_ROOT 0

static int p1 = 2;
static int p2 = 2;
static int n1 = 200;
static int n2 = 200;
static int n3 = 200;
static int test_n1 = 2;
static int test_n2 = 3;
static int test_n3 = 4;
static unsigned int rand_seed = 42;
static bool test_mode = false;

bool read_config() {
    std::ifstream f("config.txt");
    if (!f.is_open()) return false;
    std::string key;
    
    while (f >> key) {
        if (key == "p1") f >> p1;
        else if (key == "p2") f >> p2;
        else if (key == "n1") f >> n1;
        else if (key == "n2") f >> n2;
        else if (key == "n3") f >> n3;
        else if (key == "test_n1") f >> test_n1;
        else if (key == "test_n2") f >> test_n2;
        else if (key == "test_n3") f >> test_n3;
        else if (key == "rand_seed") f >> rand_seed;
    }

    return true;
}

void parse_argv(int argc, char** argv, int rank){
    for (int i = 0; i < argc; i++) {
        if (strcmp(argv[i], "--config") == 0 || strcmp(argv[i], "-c") == 0) {
            if (!read_config() && rank == 0){
                std::cout << "Can not open 'config.txt', using default data" << std::endl;
            }
            break;
        }
    }
    for (int i = 0; i < argc; i++) {
        if (strcmp(argv[i], "--test") == 0 || strcmp(argv[i], "-t") == 0) {
            test_mode = true;
            n1 = test_n1;
            n2 = test_n2;
            n3 = test_n3;
            break;
        }
    }
    for (int i = 0; i < argc; i++) {
        if (strcmp(argv[i], "--random") == 0 || strcmp(argv[i], "-r") == 0) {
            rand_seed = time(nullptr);
            break;
        }
    }
}


std::vector<double> generate_matrix(int rows, int cols) {
    std::vector<double> matrix(rows * cols);

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            matrix[i * cols + j] = static_cast<double>(rand() % 10 + 1);
        }    
    }

    return matrix;
}

std::vector<double> transpose_matrix(const std::vector<double>& mat, int rows, int cols) {
    std::vector<double> transposed(rows * cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            transposed[j * rows + i] = mat[i * cols + j];
        }
    }

    return transposed;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    parse_argv(argc, argv, rank);
    srand(rand_seed);
    
    if (p1 * p2 != size) {
        if (rank == 0) {
            std::cerr << "Error: " << p1 << "x" << p2 << " = " << p1*p2 
                      << " but total processes = " << size << std::endl;
        }

        MPI_Finalize();
        return 1;
    }
    
    int dims[2] = {p1, p2};  
    int periods[2] = {0, 0};
    int reorder = 1;          
    
    MPI_Comm grid;
    MPI_Cart_create(MPI_COMM_WORLD, GRID_DIM, dims, periods, reorder, &grid);
    
    int coords[2];
    MPI_Cart_coords(grid, rank, GRID_DIM, coords);

    int my_row = coords[0];
    int my_col = coords[1];
    
    MPI_Comm row_comm;
    MPI_Comm_split(grid, my_row, my_col, &row_comm);
    
    MPI_Comm col_comm;
    MPI_Comm_split(grid, my_col, my_row, &col_comm);
    
    int local_rows_A = n1 / p1;
    if (my_row < n1 % p1){
        local_rows_A++;
    }

    int local_cols_B = n3 / p2;
    if (my_col < n3 % p2){
        local_cols_B++;
    }
    
    std::vector<double> local_A(local_rows_A * n2);
    std::vector<double> local_B(n2 * local_cols_B); //actually the transposed submatrix B
    std::vector<double> local_C(local_rows_A * local_cols_B, 0.0);


    MPI_Barrier(grid);
    if (my_col == 0 && my_row == 0) std::cout << "Generating A and B matrices and sending to processes..." << std::endl;
    double total_start = MPI_Wtime();
    std::vector<double> A;
    std::vector<double> B;

    //generating A matrix and sending to processes
    if (my_col == 0) {
        if (my_row == 0) {
            A = generate_matrix(n1, n2);

            std::vector<int> send_counts_A(p1);
            std::vector<int> offsets_A(p1);

            int offset = 0;
        
            for (int i = 0; i < p1; i++) {        
                int rows = n1 / p1 + (i < n1 % p1 ? 1 : 0);    
                send_counts_A[i] = rows * n2; 
                offsets_A[i] = offset;
            
                offset += send_counts_A[i];
            }        

            MPI_Scatterv(A.data(), send_counts_A.data(), offsets_A.data(), MPI_DOUBLE,
                        local_A.data(), local_A.size(), MPI_DOUBLE,
                        ZERO_ROOT, col_comm);
        }

        else {
            MPI_Scatterv(nullptr, nullptr, nullptr, MPI_DOUBLE,
                        local_A.data(), local_A.size(), MPI_DOUBLE,
                        ZERO_ROOT, col_comm);

        }
    }
    
    MPI_Bcast(local_A.data(), local_A.size(), MPI_DOUBLE, ZERO_ROOT, row_comm);


    //generating B matrix and sending to processes
    if (my_row == 0) {
        if (my_col == 0) {
            B = generate_matrix(n2, n3);
            std::vector<double> B_transposed = transpose_matrix(B, n2, n3);

            std::vector<int> send_counts_B(p2);
            std::vector<int> offsets_B(p2);

            int offset = 0;
        
            for (int i = 0; i < p2; i++) {        
                int cols = n3 / p2 + (i < n3 % p2 ? 1 : 0);    
                send_counts_B[i] = n2 * cols; 
                offsets_B[i] = offset;
            
                offset += send_counts_B[i];
            }     
        
            MPI_Scatterv(B_transposed.data(), send_counts_B.data(), offsets_B.data(), MPI_DOUBLE,
                        local_B.data(), local_B.size(), MPI_DOUBLE,
                        ZERO_ROOT, row_comm);
        }

        else {
            MPI_Scatterv(nullptr, nullptr, nullptr, MPI_DOUBLE,
                        local_B.data(), local_B.size(), MPI_DOUBLE,
                        ZERO_ROOT, row_comm);
        }
    }

    MPI_Bcast(local_B.data(), local_B.size(), MPI_DOUBLE, ZERO_ROOT, col_comm);

    double first_segment_end = MPI_Wtime();
    if (my_col == 0 && my_row == 0) std::cout << "Matrices sent, completed in " << first_segment_end - total_start << " seconds" <<std::endl;

    //calculating local_C
    if (my_col == 0 && my_row == 0) std::cout << "\nCalculating C..." << std::endl;
    MPI_Barrier(grid);
    double mul_start = MPI_Wtime();

    for (int i = 0; i < local_rows_A; i++){
        for (int k = 0; k < n2; k++){
            double a_ik = local_A[i * n2 + k];
            
            for (int j = 0; j < local_cols_B; j++){
                local_C[i * local_cols_B + j] += a_ik * local_B[j * n2 + k];
            }
        }
    }

    MPI_Barrier(grid);
    double mul_end = MPI_Wtime();
    if (my_col == 0 && my_row == 0) std::cout << "C calculated, completed in  " << mul_end - mul_start << " seconds" << std::endl;
    

    // building C matrix
    if (my_col == 0 && my_row == 0) std::cout << "\nBuilding C matrix..." << std::endl;
    double bld_start = MPI_Wtime();
    
    std::vector<double> C;
    
    if (my_row == 0 && my_col == 0) {
        std::vector<int> recv_counts(size);
        std::vector<int> offsets_C(size);
        
        int offset = 0;
        
        for (int i = 0; i < p1; i++) {
            for (int j = 0; j < p2; j++) {
                int rank_in_grid;
                int src_coords[2] = {i, j};
                MPI_Cart_rank(grid, src_coords, &rank_in_grid);
                
                int rows = n1 / p1 + (i < n1 % p1 ? 1 : 0);
                int cols = n3 / p2 + (j < n3 % p2 ? 1 : 0);
                
                recv_counts[rank_in_grid] = rows * cols;
                offsets_C[rank_in_grid] = offset;
                offset += recv_counts[rank_in_grid];
            }
        }
        
        C = std::vector<double>(n1 * n3, 0.0);
        MPI_Gatherv(local_C.data(), local_C.size(), MPI_DOUBLE,
                C.data(), recv_counts.data(), offsets_C.data(), MPI_DOUBLE,
                ZERO_ROOT, grid);
    }
    
    else {
        MPI_Gatherv(local_C.data(), local_C.size(), MPI_DOUBLE,
                nullptr, nullptr, nullptr, MPI_DOUBLE,
                ZERO_ROOT, grid);
    }

    MPI_Barrier(grid);
    double total_end = MPI_Wtime();

    if (my_row == 0 && my_col == 0) {
        std::cout << "Completed. C built in " << total_end - bld_start << " seconds"<<std::endl;
        std::cout << "\n========================================" << std::endl;
        std::cout << "LAB3" << std::endl;
        std::cout << "========================================" << std::endl;
        std::cout << "Matrix sizes: " << n1 << "x" << n2 << " * " << n2 << "x" << n3 << std::endl;
        std::cout << "Grid: " << p1 << "x" << p2 << std::endl;
        std::cout << "Multiplication time: " << mul_end - mul_start << " seconds" << std::endl;
        std::cout << "Total time (with communication): " << total_end - total_start << " seconds" << std::endl;
        std::cout << "========================================" << std::endl;

        if (test_mode) { 
            std::cout << "Random seed: " << rand_seed << std::endl;
            
            std::cout << "\nA matrix:" << std::endl;
            for (int i = 0; i < n1; i++){
                for (int j = 0; j < n2; j++){
                    std::cout << A[i * n2 + j] << " ";
                }   
                std::cout << std::endl;
            }

            std::cout << "\nB matrix:" << std::endl;
            for (int i = 0; i < n2; i++){
                for (int j = 0; j < n3; j++){
                    std::cout << B[i * n3 + j] << " ";
                }
                std::cout << std::endl;
            }

            std::cout << "\nC matrix:" << std::endl;
            for (int i = 0; i < n1; i++){
                for (int j = 0; j < n3; j++){
                    std::cout << C[i * n3 + j] << " ";
                }
                std::cout << std::endl;
            }
            std::cout << "========================================" << std::endl;
        }
    }

    MPI_Comm_free(&row_comm);
    MPI_Comm_free(&col_comm);
    MPI_Comm_free(&grid);
    
    MPI_Finalize();
    return 0;
}