#include <iostream>
#include <vector>
#include <iomanip>
#include <fstream>
#include <omp.h>
#include <random>
#include <sys/resource.h>
#include <algorithm>

#include <linux/perf_event.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <cstring>

using Matrix1D = std::vector<double>;

void print_resource_metrics(const std::string& label, const struct rusage& start, const struct rusage& end) {
    double user_time = (end.ru_utime.tv_sec - start.ru_utime.tv_sec) +
                       (end.ru_utime.tv_usec - start.ru_utime.tv_usec) / 1000000.0;
    double sys_time  = (end.ru_stime.tv_sec - start.ru_stime.tv_sec) +
                       (end.ru_stime.tv_usec - start.ru_stime.tv_usec) / 1000000.0;

    // В OS Linux ru_maxrss повертається у кілобайтах
    long peak_ram_mb = end.ru_maxrss / 1024;

    std::cout << std::left << std::setw(20) << label
              << " | CPU User: " << std::fixed << std::setprecision(4) << user_time << " s"
              << " | CPU Sys: " << sys_time << " s"
              << " | Peak RAM: " << peak_ram_mb << " MB\n";
}

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



class OmpCacheMissCounter {
    std::vector<int> fds;
public:
    void start() {
        int num_threads = omp_get_max_threads();
        fds.assign(num_threads, -1);

#pragma omp parallel
        {
            int tid = omp_get_thread_num();
            struct perf_event_attr pe;
            std::memset(&pe, 0, sizeof(struct perf_event_attr));
            pe.type = PERF_TYPE_HARDWARE;
            pe.size = sizeof(struct perf_event_attr);
            pe.config = PERF_COUNT_HW_CACHE_MISSES;
            pe.disabled = 1;
            pe.exclude_kernel = 1;
            pe.exclude_hv = 1;

            // Відкриття лічильника для поточного потоку
            fds[tid] = syscall(__NR_perf_event_open, &pe, 0, -1, -1, 0);
            if (fds[tid] != -1) {
                ioctl(fds[tid], PERF_EVENT_IOC_RESET, 0);
                ioctl(fds[tid], PERF_EVENT_IOC_ENABLE, 0);
            }
        }
    }

    long long stop_and_read() {
        long long total_misses = 0;

#pragma omp parallel reduction(+:total_misses)
        {
            int tid = omp_get_thread_num();
            if (fds[tid] != -1) {
                long long count = 0;
                ioctl(fds[tid], PERF_EVENT_IOC_DISABLE, 0);
                if (read(fds[tid], &count, sizeof(long long)) > 0) {
                    total_misses += count;
                }
                close(fds[tid]);
            }
        }
        fds.clear();
        return total_misses;
    }
};



int main() {
    OmpCacheMissCounter cache_counter;
    // Відкриваємо файл для запису (overwrite)
    std::ofstream outfile("benchmark_results.txt");
    if (!outfile.is_open()) {
        std::cerr << "Error opening file!" << std::endl;
        return 1;
    }

    #pragma omp parallel
    { /* Warm-up threads */ }

    std::vector<int> sizes = {100, 500, 1000, 2500, 5000};
    const int NUM_RUNS = 5;

    for (int size : sizes) {
        int n = size;
        outfile << "======================================================\n";
        outfile << "SIZE: " << n << "x" << n << "\n";
        outfile << "======================================================\n";
        std::cout << "Processing size: " << n << "..." << std::endl; // Статус у консоль

        Matrix1D A_orig(n * n);
        std::vector<int> P_orig(n);
        for (int i = 0; i < n; i++) P_orig[i] = i;

        // Генерація даних (один раз для обох алгоритмів)
        unsigned int base_seed = 42;
        #pragma omp parallel
        {
            std::mt19937 rng(base_seed + omp_get_thread_num());
            std::uniform_int_distribution<int> distribution(-10, 10);
            #pragma omp for
            for (int i = 0; i < n * n; i++) A_orig[i] = distribution(rng);
        }

        double blocked_min_wtime = 0;
        double parallel_min_wtime = 0;

        // --- Блок 1: Blocked LU ---
        {
            double total_cpu_user = 0, total_cpu_sys = 0;
            std::vector<double> w_times;
            long peak_rss = 0;
            bool is_valid = false;
            long long total_cache_misses = 0;

            for (int r = 0; r < NUM_RUNS; r++) {
                Matrix1D A_work = A_orig;
                std::vector<int> P_work = P_orig;
                Matrix1D L(n * n, 0.0), U(n * n, 0.0);

                struct rusage r_start, r_end;
                getrusage(RUSAGE_SELF, &r_start);
                cache_counter.start();
                double start_wtime = omp_get_wtime();

                blocked_lu_dec(A_work, L, U, P_work, n);

                long long run_misses = cache_counter.stop_and_read();
                total_cache_misses += run_misses;
                if (r == 0) is_valid = verify_b_SAXPY(A_orig, P_work, L, U, n);

                double end_wtime = omp_get_wtime();
                getrusage(RUSAGE_SELF, &r_end);

                w_times.push_back(end_wtime - start_wtime);
                total_cpu_user += (r_end.ru_utime.tv_sec - r_start.ru_utime.tv_sec) + (r_end.ru_utime.tv_usec - r_start.ru_utime.tv_usec) / 1000000.0;
                total_cpu_sys += (r_end.ru_stime.tv_sec - r_start.ru_stime.tv_sec) + (r_end.ru_stime.tv_usec - r_start.ru_stime.tv_usec) / 1000000.0;
                peak_rss = std::max(peak_rss, r_end.ru_maxrss);

            }
            double min_t = *std::min_element(w_times.begin(), w_times.end());
            double max_t = *std::max_element(w_times.begin(), w_times.end());
            double avg_t = std::accumulate(w_times.begin(), w_times.end(), 0.0) / NUM_RUNS;

            outfile << "Block LU dec statistic:\n";
            outfile << " Min: " << std::fixed << std::setprecision(5) << min_t << " s\n";
            outfile << " Max: " << max_t << " s\n";
            outfile << " Avg: " << avg_t << " s\n";
            outfile << " Valid: " << (is_valid ? "OK" : "FAIL") << "\n\n";

            outfile << "---------------------------\n\n";

            blocked_min_wtime = *std::min_element(w_times.begin(), w_times.end());
            outfile << "Block res usage:\n";
            outfile << "  Min Wall Time: " << blocked_min_wtime << " s\n";
            outfile << "  Avg CPU User:  " << total_cpu_user / NUM_RUNS << " s\n";
            outfile << "  Avg CPU Sys:   " << total_cpu_sys / NUM_RUNS << " s\n";
            outfile << "  Peak RAM:      " << peak_rss / 1024 << " MB\n\n";
            outfile << "  Avg Cache Miss:  " << total_cache_misses / NUM_RUNS << " misses\n\n";

            outfile << "\n\n######################################################\n\n";
        }

        // --- Блок 2: Parallel LU ---
        {
            long long total_cache_misses = 0;
            double total_cpu_user = 0, total_cpu_sys = 0;
            std::vector<double> w_times;
            long peak_rss = 0;
            bool is_valid = false;

            for (int r = 0; r < NUM_RUNS; r++) {
                Matrix1D L(n * n, 0.0), U(n * n, 0.0);

                struct rusage r_start, r_end;
                getrusage(RUSAGE_SELF, &r_start);
                double start_wtime = omp_get_wtime();
                cache_counter.start();

                parallel_lu_decomposition(A_orig, L, U, n);
                long long run_misses = cache_counter.stop_and_read();
                total_cache_misses += run_misses;

                double end_wtime = omp_get_wtime();

                getrusage(RUSAGE_SELF, &r_end);

                w_times.push_back(end_wtime - start_wtime);
                if (r == 0) is_valid = verify_lu(A_orig, L, U, n);
                total_cpu_user += (r_end.ru_utime.tv_sec - r_start.ru_utime.tv_sec) + (r_end.ru_utime.tv_usec - r_start.ru_utime.tv_usec) / 1000000.0;
                total_cpu_sys += (r_end.ru_stime.tv_sec - r_start.ru_stime.tv_sec) + (r_end.ru_stime.tv_usec - r_start.ru_stime.tv_usec) / 1000000.0;
                peak_rss = std::max(peak_rss, r_end.ru_maxrss);

            }

            double min_t = *std::min_element(w_times.begin(), w_times.end());
            double max_t = *std::max_element(w_times.begin(), w_times.end());
            double avg_t = std::accumulate(w_times.begin(), w_times.end(), 0.0) / NUM_RUNS;

            outfile << "Parallel statistic:\n";
            outfile << "  Min: " << std::fixed << std::setprecision(5) << min_t << " s\n";
            outfile << "  Max: " << max_t << " s\n";
            outfile << "  Avg: " << avg_t << " s\n";
            outfile << "  Valid: " << (is_valid ? "OK" : "FAIL") << "\n\n";

            outfile << "---------------------------\n\n";

            parallel_min_wtime = *std::min_element(w_times.begin(), w_times.end());
            outfile << "Parallel res usage:\n";
            outfile << "  Min Wall Time: " << parallel_min_wtime << " s\n";
            outfile << "  Avg CPU User:  " << total_cpu_user / NUM_RUNS << " s\n";
            outfile << "  Avg CPU Sys:   " << total_cpu_sys / NUM_RUNS << " s\n";
            outfile << "  Peak RAM:      " << peak_rss / 1024 << " MB\n\n";
            outfile << "  Avg Cache Miss:  " << total_cache_misses / NUM_RUNS << " misses\n\n";
        }

        outfile << "Efficiency Gain (Parallel/Block): " << (parallel_min_wtime / blocked_min_wtime) << "x\n\n";
    }

    outfile.close();
    std::cout << "Done! Results saved to benchmark_results.txt" << std::endl;
    return 0;
}



//
// int main() {        Мертвий код. Залишений для того-щоб швидко використовувати вже прописані запуски.
//
//     #pragma omp parallel
//     { } // Прогрів потоків
//
//     std::vector<int> sizes = {100,500,1000,2500};
//     const int NUM_RUNS = 5; // Кількість запусків для статистичної вибірки
//
//
//     for (int size : sizes)
//     {
//
//         printf("\nsize of square matrix: %d \n", size);
//         int n = size;
//         Matrix1D A(n * n), L(n * n, 0.0), U(n * n, 0.0), COPY(n*n);
//         std::vector<int> P (n);
//         for (int i = 0; i < n; i++)
//         {
//             P[i] = i;
//         }
//
//
//         unsigned int base_seed = std::random_device{}();
//
//
//
//         #pragma omp parallel
//         {
//             std::mt19937 rng(base_seed + omp_get_thread_num());
//             std::uniform_int_distribution<int> distribution(-10, 10 );
//             #pragma omp for
//             for (int i = 0; i < n * n; i++) A[i] = distribution(rng);
//         }
//
//
//         // print_matrix(A,n);
//
//
//         double start_time;
//         double end_time;
//         COPY = A;
//
//         // printf("Beginning of parallel LU decomposition\n");
//         // reset_matrices(L, U);
//         // start_time = omp_get_wtime();
//         // parallel_lu_decomposition(A, L, U, n);
//         // end_time = omp_get_wtime();
//         // printf("Time: %f\n",    end_time - start_time);
//         // if (verify_lu(A,L,U,n))
//         // {
//         //     printf("Matrix Verification OK\n");
//         // }
//         // else
//         // {
//         //     printf("Matrix Verification FAILED\n");
//         // }
//
//         // std::cout << "Matrix L:\n";
//         // print_matrix(L, n);
//         //
//         // std::cout << "Matrix U:\n";
//         // print_matrix(U, n);
//
//         // printf("Beginning of task parallel LU decomposition\n");
//         // reset_matrices(L, U);
//         // start_time = omp_get_wtime();
//         // task_lu_decomposition(A,L,U,n);
//         // end_time = omp_get_wtime();
//         // printf("Time: %f\n",    end_time - start_time);
//         // if (verify_lu(A,L,U,n))
//         // {
//         //     printf("Matrix Verification OK\n");
//         // }
//         // else
//         // {
//         //     printf("Matrix Verification FAILED\n");
//         // }
//         // std::cout << "Matrix L:\n";
//         // print_matrix(L, n);
//         //
//         // std::cout << "Matrix U:\n";
//         // print_matrix(U, n);
//
//
//         printf("\nBeginning of blocked LU Dec\n");
//         reset_matrices(L, U);
//         start_time = omp_get_wtime();
//         blocked_lu_dec(A,L,U,P,n);
//         end_time = omp_get_wtime();
//         printf("Time: %f\n",    end_time - start_time);
//         if (verify_b_SAXPY(COPY, P, L, U, n))
//         {
//             printf("Matrix Verification OK\n");
//         }else
//         {
//             printf("Matrix Verification FAILED\n");
//         }
//
//         // printf("Beginning of SAXPY_parallel_lu_dec\n");
//         // reset_matrices(L, U);
//         // start_time = omp_get_wtime();
//         // SAXPY_parallel_lu_dec(COPY,L,U,P,n);
//         // end_time = omp_get_wtime();
//         // printf("Time: %f\n",    end_time - start_time);
//         // if (verify_SAXPY(A,P,L,U,n))
//         // {
//         //     printf("Matrix Verification OK\n");
//         // }
//         // else
//         // {
//         //     printf("Matrix Verification FAILED\n");
//         // }
//
//         // std::cout << "Matrix L:\n";
//         // print_matrix(L, n);
//         // std::cout << "Matrix U:\n";
//         // print_matrix(U, n);
//
//
//
//
//         //
//         // printf("Beginning of consecutive LU decomposition\n");
//         // reset_matrices(L, U);
//         // start_time = omp_get_wtime();
//         // consecutive_lu_decomposition(A,L,U,n);
//         // end_time = omp_get_wtime();
//         // printf("Time: %f\n",    end_time - start_time);
//         // if (verify_lu(A,L,U,n))
//         // {
//         //     printf("Matrix Verification OK\n");
//         // }
//         // else
//         // {
//         //     printf("Matrix Verification FAILED\n");
//         // }
//         // std::cout << "Matrix L:\n";
//         // print_matrix(L, n);
//         //
//         // std::cout << "Matrix U:\n";
//         // print_matrix(U, n);
//
//     }
//     return 0;
// }