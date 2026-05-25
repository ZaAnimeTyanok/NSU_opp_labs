#include <iostream>
#include <vector>
#include <cmath>
#include <mpi.h>
#include <fstream>
#include <cstdlib>

static double X0 = -1.0;
static double Y0 = -1.0;
static double Z0 = -1.0;
static double Dx = 2.0;
static double Dy = 2.0; 
static double Dz = 2.0;

static double a = 1e5;
static double eps = 1e-12;

static int Nx = 200;
static int Ny = 200;
static int Nz = 200;

bool read_config() {
    std::ifstream f("config.txt");
    if (!f.is_open()) return false;
    
    std::string key;
    while (f >> key) {
        if (key == "X0") f >> X0;
        else if (key == "Y0") f >> Y0;
        else if (key == "Z0") f >> Z0;
        else if (key == "Dx") f >> Dx;
        else if (key == "Dy") f >> Dy;
        else if (key == "Dz") f >> Dz;
        else if (key == "a") f >> a;
        else if (key == "eps") f >> eps;
        else if (key == "Nx") f >> Nx;
        else if (key == "Ny") f >> Ny;
        else if (key == "Nz") f >> Nz;
    }

    return true;
}

inline int idx(int i, int j, int k) {
    return ((i * Ny) + j) * Nz + k;
}

double exact_phi(double x, double y, double z) {
    return x*x + y*y + z*z;
}

double rho_func(double x, double y, double z) {
    return 6.0 - a * exact_phi(x, y, z);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    for (int i = 0; i < argc; i++) {
        if (strcmp(argv[i], "--config") == 0 || strcmp(argv[i], "-c") == 0) {
            if (!read_config() && rank == 0){
                std::cerr << "Can not open 'config.txt', using default data" << std::endl;
            }
            break;
        }
    }

    if (Nx < size) {
        if (rank == 0) {
            std::cerr << "Error: Nx < number of processes" << std::endl;
        }

        MPI_Finalize();
        
        return 1;
    }
    
    int local_nx = Nx / size;
    int rem = Nx % size;
    if (rank < rem) local_nx++;
    
    int start_x = 0;
    for (int i = 0; i < rank; i++) {
        start_x += Nx / size + (i < rem ? 1 : 0);
    }
    
    double hx = Dx / (Nx - 1);
    double hy = Dy / (Ny - 1);
    double hz = Dz / (Nz - 1);
    
    double inv_hx2 = 1.0 / (hx * hx);
    double inv_hy2 = 1.0 / (hy * hy);
    double inv_hz2 = 1.0 / (hz * hz);
    double coeff = 1.0 / (2.0 * inv_hx2 + 2.0 * inv_hy2 + 2.0 * inv_hz2 + a);
    
    int local_nx_ghost = local_nx + 2;
    std::vector<double> phi(local_nx_ghost * Ny * Nz, 0.0);
    std::vector<double> phi_new(local_nx_ghost * Ny * Nz, 0.0);
    std::vector<double> rho(local_nx_ghost * Ny * Nz, 0.0);
    
    //border and rho init
    for (int i = 0; i < local_nx_ghost; i++) {
        int global_i = start_x + i - 1;
        double x = X0 + global_i * hx;

        for (int j = 0; j < Ny; j++) {
            double y = Y0 + j * hy;

            for (int k = 0; k < Nz; k++) {
                double z = Z0 + k * hz;
                int ind = idx(i, j, k);
                rho[ind] = rho_func(x, y, z);
                
                if (global_i == 0 || global_i == Nx-1 || j == 0 || j == Ny-1 || k == 0 || k == Nz-1) {
                    phi[ind] = exact_phi(x, y, z);
                    phi_new[ind] = phi[ind];
                }
            }
        }
    }

    int prev = (rank == 0) ? MPI_PROC_NULL : rank - 1;
    int next = (rank == size-1) ? MPI_PROC_NULL : rank + 1;
    
    int layer_size = Ny * Nz;
    std::vector<double> send_left(layer_size), send_right(layer_size);
    std::vector<double> recv_left(layer_size), recv_right(layer_size);
    
    double start_time = MPI_Wtime();
    int iter = 0;
    
    while (true) {
        MPI_Request reqs[4];
        MPI_Irecv(recv_left.data(), layer_size, MPI_DOUBLE, prev, 0, MPI_COMM_WORLD, &reqs[0]);
        MPI_Irecv(recv_right.data(), layer_size, MPI_DOUBLE, next, 1, MPI_COMM_WORLD, &reqs[1]);
        
        int left_real = 1;
        int right_real = local_nx;
        
        std::copy(phi.begin() + idx(left_real, 0, 0),
                  phi.begin() + idx(left_real + 1, 0, 0),
                  send_left.data());

        std::copy(phi.begin() + idx(right_real, 0, 0),
                  phi.begin() + idx(right_real + 1, 0, 0),
                  send_right.data());
        
        MPI_Isend(send_left.data(), layer_size, MPI_DOUBLE, prev, 1, MPI_COMM_WORLD, &reqs[2]);
        MPI_Isend(send_right.data(), layer_size, MPI_DOUBLE, next, 0, MPI_COMM_WORLD, &reqs[3]);
        
        //calculating nodes that do not require ghost layers
        double local_max_diff = 0.0;
        for (int i = 2; i <= local_nx - 1; i++) {
            int global_i = start_x + i - 1;
            
            for (int j = 1; j < Ny-1; j++) {
                for (int k = 1; k < Nz-1; k++) {
                    int c = idx(i, j, k);
                    
                    double new_val = coeff * (
                        (phi[idx(i+1, j, k)] + phi[idx(i-1, j, k)]) * inv_hx2 +
                        (phi[idx(i, j+1, k)] + phi[idx(i, j-1, k)]) * inv_hy2 +
                        (phi[idx(i, j, k+1)] + phi[idx(i, j, k-1)]) * inv_hz2 -
                        rho[c]
                    );
                    
                    phi_new[c] = new_val;
                    double diff = fabs(new_val - phi[c]);
                    
                    if (diff > local_max_diff) local_max_diff = diff;
                }
            }
        }
        
        MPI_Waitall(4, reqs, MPI_STATUSES_IGNORE);
        
        // ghost layers copying
        std::copy(recv_left.begin(), recv_left.end(), phi.begin() + idx(0, 0, 0));
        std::copy(recv_right.begin(), recv_right.end(), phi.begin() + idx(local_nx_ghost-1, 0, 0));
        
        //calculating nodes that require ghost layers
        std::vector<int> borders = (local_nx == 1) ? std::vector({1}) : std::vector({1, local_nx}); 
        for (int i : borders) {
            if (i == 1 && start_x == 0) continue; //global left border
            if (i == local_nx && start_x+local_nx-1 == Nx-1) continue; //global right border
            
            for (int j = 1; j < Ny-1; j++) {
                for (int k = 1; k < Nz-1; k++) {
                    int c = idx(i, j, k);
                    
                    double new_val = coeff * (
                        (phi[idx(i+1, j, k)] + phi[idx(i-1, j, k)]) * inv_hx2 +
                        (phi[idx(i, j+1, k)] + phi[idx(i, j-1, k)]) * inv_hy2 +
                        (phi[idx(i, j, k+1)] + phi[idx(i, j, k-1)]) * inv_hz2 -
                        rho[c]
                    );

                    phi_new[c] = new_val;
                    double diff = fabs(new_val - phi[c]);

                    if (diff > local_max_diff) local_max_diff = diff;
                }
            }
        }

        double global_max_diff;
        MPI_Allreduce(&local_max_diff, &global_max_diff, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        
        phi.swap(phi_new);
        iter++;
        
        if (rank == 0 && iter % 50 == 0) {
            std::cout << "Iter " << iter << ", max_diff = " << global_max_diff << std::endl;
        }
        
        if (global_max_diff < eps || iter > 20000) break;
    }
    

    std::vector<double> phi_full;

    if (rank == 0) {
        phi_full.resize(Nx * Ny * Nz);
    
        std::vector<int> recv_counts(size);
        std::vector<int> offsets(size);
    
        int offset = 0;

        for (int i = 0; i < size; i++) {
            int rows = Nx / size + (i < Nx % size ? 1 : 0);
            recv_counts[i] = rows * Ny * Nz;
            offsets[i] = offset;
            offset += recv_counts[i];
        }
    
        MPI_Gatherv(phi.data() + idx(1, 0, 0), local_nx * Ny * Nz, MPI_DOUBLE,
                phi_full.data(), recv_counts.data(), offsets.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);
    }

    else {
        MPI_Gatherv(phi.data() + idx(1, 0, 0), local_nx * Ny * Nz, MPI_DOUBLE,
                nullptr, nullptr, nullptr, MPI_DOUBLE,
                0, MPI_COMM_WORLD);
    }
    
    
    double end_time = MPI_Wtime();
    
    if (rank == 0) {
        double max_error = 0.0;
        
        for (int i = 0; i < Nx; i++) {
            double x = X0 + i * hx;
    
            for (int j = 0; j < Ny; j++) {
                double y = Y0 + j * hy;
                for (int k = 0; k < Nz; k++) {
                    double z = Z0 + k * hz;
                    double exact = exact_phi(x, y, z);
                    double numeric = phi_full[idx(i, j, k)];
                    double err = fabs(numeric - exact);
                    if (err > max_error) max_error = err;
                }
            }
        }

        std::cout << "\n========================================" << std::endl;
        std::cout << "LAB4" << std::endl;
        std::cout << "========================================" << std::endl;
        std::cout << "Grid: " << Nx << " x " << Ny << " x " << Nz << std::endl;
        std::cout << "Processes: " << size << std::endl;
        std::cout << "Iterations: " << iter << std::endl;
        std::cout << "Max error from exact solution: " << max_error << std::endl;
        std::cout << "Time: " << end_time - start_time << " seconds" << std::endl;
        std::cout << "========================================" << std::endl;
    }
    
    MPI_Finalize();
    return 0;
}