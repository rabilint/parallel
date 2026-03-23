#include <iostream>
#include <vector>
#include <omp.h>
#include <random>

int main()
{
    std::vector<int> sizes = {1000, 5000, 10000};
    std::vector<int> thread_counts = {1,2,3,4};

    for (int size : sizes)
    {
        int M = size, N = size;
        std::vector<int> matrix(M*N);
        unsigned int base_seed = std::random_device{}();
        #pragma omp parallel
        {
            std::mt19937 rng(base_seed + omp_get_thread_num());
            std::uniform_int_distribution<int> distribution(0, M - 1);

            #pragma omp for
            for (int i = 0; i < M * N; i++)
            {
                matrix[i] = distribution(rng);
            }
        }
        std::cout << "Data generation complete. Starting benchmark...\n with " << size << " size of matrix " << std::endl;

        for (int threads : thread_counts)
        {
            omp_set_num_threads(threads);
            long long sum_of_matrix = 0;

            double start_time = omp_get_wtime();

            #pragma omp parallel for reduction(+:sum_of_matrix)
            for (int i = 0; i < M; i++)
            {
                for (int j = 0; j < N; j++)
                {
                    sum_of_matrix += matrix[i * N + j];
                }
            }

            double end_time = omp_get_wtime();
            printf("Size: %5dx%5d | Threads: %d | Time: %f s | Sum: %lld\n",
                   M, N, threads, end_time - start_time, sum_of_matrix);
        }
    }


    std::cout << "\n--- Schedule Demonstration ---\n";
    omp_set_num_threads(4);
    const int iter_count = 15;

    std::cout << "\nSchedule: STATIC, chunk size 5\n";
    #pragma omp parallel for schedule(static, 5)
    for (int i = 0; i < iter_count; i++)
    {
        printf("Thread[%d]: calculation of the iteration number %d\n", omp_get_thread_num(), i);
    }

    std::cout << "\nSchedule: DYNAMIC, chunk size 3\n";
    #pragma omp parallel for schedule(dynamic, 3)
    for (int i = 0; i < iter_count; i++)
    {
        printf("Thread[%d]: calculation of the iteration number %d\n", omp_get_thread_num(), i);
    }

    std::cout << "\nSchedule: GUIDED, chunk size 2\n";
    #pragma omp parallel for schedule(guided, 2) ordered
    for (int i = 0; i < iter_count; i++)
    {
        #pragma omp ordered
        {
            printf("Thread[%d]: calculation of the iteration number %d\n", omp_get_thread_num(), i);
        }
    }


    return 0;
}