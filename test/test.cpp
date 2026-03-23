#include <iostream>
#include <vector>
#include <iomanip>
#include <omp.h>
#include <random>

using Matrix1D = std::vector<double>;

void consecutive_lu_decomposition(const Matrix1D& A, Matrix1D& L, Matrix1D& U, int n)
{
    for (int i = 0; i < n; i++) {
        U[i * n + i] = 1.0;
    }

    for (int k = 0; k < n; k++) {

        for (int i = k; i < n; i++) {
            double sum = 0.0;
            for (int j = 0; j < k; j++) {
                sum += L[i * n + j] * U[j * n + k];
            }

            L[i * n + k] = A[i * n + k] - sum;
        }

        for (int i = k + 1; i < n; i++) {
            double sum = 0.0;
            for (int j = 0; j < k; j++) {
                sum += L[k * n + j] * U[j * n + i];
            }
            U[k * n + i] = (A[k * n + i] - sum) / L[k * n + k];
        }
    }

}

void task_lu_decomposition(const Matrix1D& A, Matrix1D& L, Matrix1D& U, int n)
{
    for (int i = 0; i < n; i++) {
        U[i * n + i] = 1.0;
    }

    #pragma omp parallel num_threads(4)
    {
        #pragma omp single
        {
            for (int k = 0; k < n; k++) {

                for (int i = k; i < n; i++) {
                    #pragma omp task firstprivate(i,k) shared(A, L, U)
                    {
                        double sum = 0.0;
                        for (int j = 0; j < k; j++) {
                            sum += L[i * n + j] * U[j * n + k];
                        }

                        L[i * n + k] = A[i * n + k] - sum;
                    }
                }
                #pragma omp taskwait


                for (int i = k + 1; i < n; i++) {
                    #pragma omp task firstprivate(i,k) shared(A, L, U)
                    {
                        double sum = 0.0;
                        for (int j = 0; j < k; j++) {
                            sum += L[k * n + j] * U[j * n + i];
                        }

                        U[k * n + i] = (A[k * n + i] - sum) / L[k * n + k];
                    }
                }

                #pragma omp taskwait
            }
        }
    }
}
////
///
///
void parallel_lu_decomposition(const Matrix1D& A, Matrix1D& L, Matrix1D& U, int n) {
    for (int i = 0; i < n; i++) {
        U[i * n + i] = 1.0;
    }
    omp_lock_t lock;
    omp_init_lock(&lock);

    for (int k = 0; k < n; k++) {

        #pragma omp parallel for num_threads(4)
        for (int i = k; i < n; i++) {
            double sum = 0.0;
            for (int j = 0; j < k; j++) {
                sum += L[i * n + j] * U[j * n + k];
            }
            L[i * n + k] = A[i * n + k] - sum;
        }

        #pragma omp parallel for num_threads(4)
        for (int i = k + 1; i < n; i++) {
            double sum = 0.0;
            for (int j = 0; j < k; j++) {
                sum += L[k * n + j] * U[j * n + i];
            }

            U[k * n + i] = (A[k * n + i] - sum) / L[k * n + k];
        }
    }

    omp_destroy_lock(&lock);
}

void print_matrix(const Matrix1D& M, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            std::cout << std::setw(8) << std::setprecision(4) << M[i * n + j] << " ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}

bool verify_lu(const Matrix1D& A, const Matrix1D& L, const Matrix1D& U, int n) {
    const double epsilon = 1e-6;
    bool is_correct = true;

    #pragma omp parallel for num_threads(4) shared(is_correct)
    for (int i = 0; i < n; i++) {
        if (!is_correct) continue;

        for (int k = 0; k < n; k++)
        {
            double sum = 0.0;
            for (int j = 0; j < n; j++) {
                sum += L[i * n + j] * U[j * n + k];
            }

            if (std::abs(sum - A[i * n + k]) > epsilon) {
                #pragma omp critical
                {
                    if (is_correct)
                    {
                        std::cout << "Validation failed at [" << i << "][" << k << "]: "
                                  << "Expected " << A[i * n + k] << ", but got " << sum << "\n";
                        is_correct = false;
                    }
                }
            }
        }
    }
    return is_correct;
}


void reset_matrices(Matrix1D& L, Matrix1D& U)
{
    std::fill(L.begin(), L.end(), 0.0);
    std::fill(U.begin(), U.end(), 0.0);
}


int main() {


    std::vector<int> sizes = {29000};


    for (int size : sizes)
    {

        printf("\nsize of square matrix: %d \n", size);
        int n = size;
        Matrix1D A(n * n), L(n * n, 0.0), U(n * n, 0.0);

        unsigned int base_seed = std::random_device{}();



        #pragma omp parallel
        {
            std::mt19937 rng(base_seed + omp_get_thread_num());
            std::uniform_int_distribution<int> distribution(1, 100 );
            #pragma omp for
            for (int i = 0; i < n * n; i++) A[i] = distribution(rng);
        }


        // print_matrix(A,n);


        double start_time;
        double end_time;


        printf("Beginning of parallel LU decomposition\n");
        reset_matrices(L, U);
        start_time = omp_get_wtime();
        parallel_lu_decomposition(A, L, U, n);
        end_time = omp_get_wtime();
        printf("Time: %f\n",    end_time - start_time);
        // if (verify_lu(A,L,U,n))
        // {
        //     printf("Matrix Verification OK\n");
        // }
        // else
        // {
        //     printf("Matrix Verification FAILED\n");
        // }

        // std::cout << "Matrix L:\n";
        // // print_matrix(L, n);
        //
        // std::cout << "Matrix U:\n";
        // // print_matrix(U, n);

        printf("Beginning of task parallel LU decomposition\n");
        reset_matrices(L, U);
        start_time = omp_get_wtime();
        task_lu_decomposition(A,L,U,n);
        end_time = omp_get_wtime();
        printf("Time: %f\n",    end_time - start_time);
        // if (verify_lu(A,L,U,n))
        // {
        //     printf("Matrix Verification OK\n");
        // }
        // else
        // {
        //     printf("Matrix Verification FAILED\n");
        // }

        //
        printf("Beginning of consecutive LU decomposition\n");
        reset_matrices(L, U);
        start_time = omp_get_wtime();
        consecutive_lu_decomposition(A,L,U,n);
        end_time = omp_get_wtime();
        printf("Time: %f\n",    end_time - start_time);
        // if (verify_lu(A,L,U,n))
        // {
        //     printf("Matrix Verification OK\n");
        // }
        // else
        // {
        //     printf("Matrix Verification FAILED\n");
        // }
    }
    return 0;
}