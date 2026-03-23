#include <complex>
#include <cstdio>
#include <iostream>
#include <omp.h>
#include <unistd.h>

double calculate_with_locks(double x, int N, int num_threads)
{
    double sum = 0;
    omp_lock_t lock;
    omp_init_lock(&lock);

    #pragma omp parallel num_threads(num_threads)
    {
        double local_sum = 0;
        #pragma omp for
        for (int n = 1; n <= N; n++) {
            double sign = (n % 2 == 0) ? -1.0 : 1.0;
            local_sum += sign * std::pow(x, n) / n;
        }

        omp_set_lock(&lock);
        sum += local_sum;
        omp_unset_lock(&lock);
    }

    omp_init_lock(&lock);
    return sum;
}


double calculate_with_explicit_sync(double x, int N, int num_threads)
{
    double sum = 0.0;

    #pragma omp parallel num_threads(num_threads)
    {
        double local_sum = 0;
        #pragma omp for
        for (int n = 1; n <= N; n++) {
            double sign = (n % 2 == 0) ? -1.0 : 1.0;
            local_sum += sign * std::pow(x, n) / n;
        }

        #pragma omp atomic
        sum += local_sum;
    }

    return sum;
}
int main()
{
    double x = 0.4;
    double epsilon = 1e-6;
    int threads = omp_get_max_threads();

    int N = 1;
    while (std::pow(std::abs(x), N + 1) / (N + 1) >= epsilon) {
        N++;
    }

    std::cout << "Required iterations (N) for precision " << epsilon << ": " << N << "\n";

    std::cout << "Target ln(1 + " << x << ") = " << std::log(1 + x) << "\n\n";

    double sum_locks = calculate_with_locks(x, N, threads);
    std::cout << "[Locks] Result: " << sum_locks << "\n";

    double sum_sync = calculate_with_explicit_sync(x, N, threads);
    std::cout << "[Explicit Sync] Result: " << sum_sync << "\n";

}