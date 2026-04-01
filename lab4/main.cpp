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


    // Swaps two rows in matrix A and updates the permutation vector P.
    // This is used for partial pivoting to improve numerical stability.
    void swap_rows(Matrix1D& A,std::vector<int>& P,int n, int k, int max_row_index)
    {
        if (k == max_row_index) return; // No need to swap if the max element is already on the diagonal

        // Parallelize the row swapping. Each thread swaps a portion of the row elements.
        #pragma omp for schedule(static)
        for (int j = 0; j < n; j++) {
            std::swap(A[k * n + j], A[max_row_index * n + j]);
        }

        // Only one thread should update the permutation vector to avoid race conditions
        #pragma omp single
        std::swap(P[k], P[max_row_index]);
    }

    // Performs LU decomposition on a vertical panel (block) of the matrix using the SAXPY approach.
    // Also applies partial pivoting to find the max element in the current column.
    void SAXPY_parallel_lu_dec(Matrix1D& A, std::vector<int>& P,int n, int current_B, int k)
    {
        double global_max;
        int global_max_indx;

        // Parallel region for processing the block column-by-column
        #pragma omp parallel
        for (int step = k; step < k + current_B; step++)
        {
            // Reset the global maximum for the current column. Only one thread does this.
            #pragma omp single
            {
                global_max = -1.0;
                global_max_indx = step;
            }

            // Локальні змінні для кожного потоку (Local variables for each thread)
            double local_max = -1.0;
            int local_max_indx = step;

            // Distribute the search for the max element in the column across threads.
            // 'nowait' allows threads to proceed to the critical section immediately without waiting for others.
            #pragma omp for nowait
            for (int i = step; i < n; i++)
            {
                double current_abs = std::abs(A[i * n + step]);
                if (current_abs > local_max)
                {
                    local_max = current_abs;
                    local_max_indx = i;
                }
            }

        // Safely update the global maximum using a critical section
#pragma omp critical
        {
            if (local_max > global_max)
            {
                 global_max  = local_max;
                 global_max_indx = local_max_indx;
            }
        }

        // Ensure all threads have finished finding the maximum before swapping rows
        #pragma omp barrier

        // Swap the current row with the row having the maximum element (Pivoting)
        swap_rows(A,P,n,step,global_max_indx);

        // Compute the multipliers for the current column of L
#pragma omp for
        for (int j = step + 1; j < k + current_B; j++) {
            A[step * n + j] = A[step * n + j] / A[step * n + step];
        }

        // Update the remaining submatrix in the panel (SAXPY operation: A = A - X * Y)
#pragma omp for
        for (int i = step + 1; i < n; i++) {  // row
            for (int j = step + 1; j < k + current_B; j++) {             // col
                A[i * n + j] -= A[i * n + step] * A[step * n + j];
            }
        }

    }
}

// Performs Blocked LU Decomposition, which improves cache utilization for large matrices.
void blocked_lu_dec(Matrix1D& A, Matrix1D& L, Matrix1D& U, std::vector<int>& P,int n)
{
    int B = 64; // Розмір блоку (Block size. Chosen to fit well in CPU cache)

    // Iterate over the matrix in blocks of size B
    for (int k = 0; k < n; k += B)
    {
        // Handle edge cases where the remaining matrix size is less than the block size
        int current_B = std::min(B, n - k);

        // 1. Panel Factorization: Decompose the current vertical block (panel)
        SAXPY_parallel_lu_dec(A,P,n, current_B, k);

// Parallel region for updating the rest of the matrix
#pragma omp parallel
        {

// 2. Update the right panel (U elements)
#pragma omp for // U
        for (int j = k + current_B; j < n; j += B) // Start of next vertical block in right panel (U12)
        {
            int j_limit = std::min(j + B, n); // щоб не вийти за межі матриці, якщо N не ділиться на B націло. (Prevent out-of-bounds access)

            for (int jj = j; jj < j_limit; jj++)  // Внутрішній цикл по стовпцях (Inner loop over columns)
            {
                for (int i = 0; i < current_B; i++) // Цикл по рядках (Loop over rows within the current block)
                {
                    for (int m = 0; m < i; m++) // Обчислює суму добутків. Віднімає вплив усіх уже обчислених вище елементів (L) від поточного значення. (Subtract already computed L elements)
                    {
                        A[(k + i) * n + jj] = A[(k + i) * n + jj] - (A[(k + i) * n + (k + m)] * A[(k + m) * n + jj]); // SAXPY
                    }
                    // Divide by the diagonal element to get the final U value
                    A[(k + i) * n + jj] /= A[(k + i) * n + (k + i)];

                }
            }
        }

// 3. Update the trailing submatrix (Schur complement)
// 'collapse(2)' combines the two outer loops into one larger iteration space for better load balancing among threads
#pragma omp for collapse(2)
        for (int i = k + current_B; i < n; i += B) {
            for (int j = k + current_B; j < n; j += B) {
                int i_limit = std::min(i + B, n);
                int j_limit = std::min(j + B, n);

                // Perform block matrix multiplication to update the submatrix
                for (int row = i; row < i_limit; row++)
                {
                    for (int col = j; col < j_limit; col++)
                    {
                        double sum = 0.0;
                        for (int m = 0; m < current_B; m++)
                        {
                            sum += A[row * n + (k + m)] * A[(k + m) * n + col];
                        }

                        A[row * n + col] -= sum;
                    }
                }

            }
        }
    }
    }

// 4. Extract L and U from the combined matrix A
// In-place LU decomposition stores both L and U in A. This step separates them into distinct matrices.
#pragma omp parallel for collapse(2)
    for (int row = 0; row < n; row++)
    {
        for (int col = 0; col < n; col++)
        {
            if (row == col) // Діагональ (Diagonal elements)
            {
                U[row * n + col] = 1; // U has 1s on the diagonal
                L[row * n + col] = A[row * n + col];
            }
            else if (row > col) // Lower triangular part (L)
            {
                L[row * n + col] = A[row * n + col];
                U[row * n + col] = 0.0;
            }else // Upper triangular part (U)
            {
                U[row * n + col] = A[row * n + col];
                L[row * n + col] = 0.0;
            }
        }
    }
}


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


bool verify_SAXPY(const Matrix1D& A, std::vector<int> P, const Matrix1D& L, const Matrix1D& U, int n)
{
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

            if (std::abs(sum - A[P[i] * n + k]) > epsilon) {
#pragma omp critical
                {
                    if (is_correct)
                    {
                        std::cout << "Validation failed at [" << i << "][" << k << "]: "
                                  << "Expected " << A[P[i] * n + k] << ", but got " << sum << "\n";
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



bool verify_b_SAXPY(const Matrix1D& A_orig, const std::vector<int>& P,
                  const Matrix1D& L, const Matrix1D& U, int n)
{
    const double epsilon = 1e-5;
    bool is_correct = true;

#pragma omp parallel for shared(is_correct) schedule(dynamic)
    for (int i = 0; i < n; i++) {
        if (!is_correct) continue;

        for (int k = 0; k < n; k++) {
            double sum = 0.0;

            int limit = std::min(i, k);
            for (int j = 0; j <= limit; j++) {
                sum += L[i * n + j] * U[j * n + k];
            }


            if (std::abs(sum - A_orig[P[i] * n + k]) > epsilon) {
#pragma omp critical
                {
                    if (is_correct) {
                        std::cout << "Validation failed at row " << i << " (orig row " << P[i]
                                  << "), col " << k << ":\n"
                                  << "Expected: " << A_orig[P[i] * n + k]
                                  << ", Result: " << sum << "\n";
                        is_correct = false;
                    }
                }
            }
        }
    }
    return is_correct;
}


int main() {


    std::vector<int> sizes = {1000};



    for (int size : sizes)
    {

        printf("\nsize of square matrix: %d \n", size);
        int n = size;
        Matrix1D A(n * n), L(n * n, 0.0), U(n * n, 0.0), COPY(n*n);
        std::vector<int> P (n);
        for (int i = 0; i < n; i++)
        {
            P[i] = i;
        }


        unsigned int base_seed = std::random_device{}();



        #pragma omp parallel
        {
            std::mt19937 rng(base_seed + omp_get_thread_num());
            std::uniform_int_distribution<int> distribution(-10, 10 );
            #pragma omp for
            for (int i = 0; i < n * n; i++) A[i] = distribution(rng);
        }


        // print_matrix(A,n);


        double start_time;
        double end_time;
        COPY = A;

        // printf("Beginning of parallel LU decomposition\n");
        // reset_matrices(L, U);
        // start_time = omp_get_wtime();
        // parallel_lu_decomposition(A, L, U, n);
        // end_time = omp_get_wtime();
        // printf("Time: %f\n",    end_time - start_time);
        // if (verify_lu(A,L,U,n))
        // {
        //     printf("Matrix Verification OK\n");
        // }
        // else
        // {
        //     printf("Matrix Verification FAILED\n");
        // }

        // std::cout << "Matrix L:\n";
        // print_matrix(L, n);
        //
        // std::cout << "Matrix U:\n";
        // print_matrix(U, n);

        // printf("Beginning of task parallel LU decomposition\n");
        // reset_matrices(L, U);
        // start_time = omp_get_wtime();
        // task_lu_decomposition(A,L,U,n);
        // end_time = omp_get_wtime();
        // printf("Time: %f\n",    end_time - start_time);
        // if (verify_lu(A,L,U,n))
        // {
        //     printf("Matrix Verification OK\n");
        // }
        // else
        // {
        //     printf("Matrix Verification FAILED\n");
        // }
        // std::cout << "Matrix L:\n";
        // print_matrix(L, n);
        //
        // std::cout << "Matrix U:\n";
        // print_matrix(U, n);


        printf("\nBeginning of blocked LU Dec\n");
        reset_matrices(L, U);
        start_time = omp_get_wtime();
        blocked_lu_dec(A,L,U,P,n);
        end_time = omp_get_wtime();
        printf("Time: %f\n",    end_time - start_time);
        if (verify_b_SAXPY(COPY, P, L, U, n))
        {
            printf("Matrix Verification OK\n");
        }else
        {
            printf("Matrix Verification FAILED\n");
        }

        // printf("Beginning of SAXPY_parallel_lu_dec\n");
        // reset_matrices(L, U);
        // start_time = omp_get_wtime();
        // SAXPY_parallel_lu_dec(COPY,L,U,P,n);
        // end_time = omp_get_wtime();
        // printf("Time: %f\n",    end_time - start_time);
        // if (verify_SAXPY(A,P,L,U,n))
        // {
        //     printf("Matrix Verification OK\n");
        // }
        // else
        // {
        //     printf("Matrix Verification FAILED\n");
        // }

        // std::cout << "Matrix L:\n";
        // print_matrix(L, n);
        // std::cout << "Matrix U:\n";
        // print_matrix(U, n);




        //
        // printf("Beginning of consecutive LU decomposition\n");
        // reset_matrices(L, U);
        // start_time = omp_get_wtime();
        // consecutive_lu_decomposition(A,L,U,n);
        // end_time = omp_get_wtime();
        // printf("Time: %f\n",    end_time - start_time);
        // if (verify_lu(A,L,U,n))
        // {
        //     printf("Matrix Verification OK\n");
        // }
        // else
        // {
        //     printf("Matrix Verification FAILED\n");
        // }
        // std::cout << "Matrix L:\n";
        // print_matrix(L, n);
        //
        // std::cout << "Matrix U:\n";
        // print_matrix(U, n);

    }
    return 0;
}