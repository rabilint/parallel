#include <stdio.h>
#include <omp.h>

int main() {
#pragma omp parallel
    {
        printf("Thread %d says hello!\n", omp_get_thread_num());
    }
    return 0;
}