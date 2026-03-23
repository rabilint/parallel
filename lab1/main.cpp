#include <cstdio>
#include <omp.h>

int test = 0;
int privateVar = 3;
int main() {
#pragma omp parallel shared(test) firstprivate(privateVar) num_threads(3)
    {
        printf("I am %d thread from %d threads! Available procs: %d\n",
               omp_get_thread_num(), omp_get_num_threads(), omp_get_num_procs());

#pragma omp sections
        {
#pragma omp section
            {
#pragma omp atomic
                test++;

                privateVar++;
                printf("[Section 1] Executed by thread %d. privateVar = %d\n", omp_get_thread_num(), privateVar);
            }

#pragma omp section
{
#pragma omp atomic
    test++;

    privateVar += 10;
    printf("[Section 2] Executed by thread %d. privateVar = %d\n", omp_get_thread_num(), privateVar);
}

        }
    }
    printf("Global test val after parallel region: %d\n", test);


    return 0;
}
