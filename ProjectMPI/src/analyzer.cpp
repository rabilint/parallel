#include "../include/analyzer.h"
#include <cstdio>
#include <mpi.h>

constexpr int LOG_LINE_LENGTH = 31;

void process_logs(char* current_buffer, int current_bytes_read)
{
    if (current_bytes_read <= 0) return;

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int my_category_color = rank % 3;
    int target_status;

    switch (my_category_color) {
    case 0: target_status = 200; break;
    case 1: target_status = 404; break;
    case 2: target_status = 500; break;
    default: target_status = 0;
    }

    long long local_match_count = 0;
    const char* ptr = current_buffer;
    const char* end = current_buffer + current_bytes_read;

    // Гарантуємо, що ми не вийдемо за межі буфера, якщо останній рядок обрізаний
    while (ptr + LOG_LINE_LENGTH <= end) {

        // 1. Надшвидкий парсинг статусу (байти 27, 28, 29)
        int st = (ptr[27] - '0') * 100 +
                 (ptr[28] - '0') * 10 +
                 (ptr[29] - '0');

        if (st == target_status) {
            uint8_t ip1 = (ptr[0] - '0') * 100 + (ptr[1] - '0') * 10 + (ptr[2] - '0');
            uint8_t ip2 = (ptr[4] - '0') * 100 + (ptr[5] - '0') * 10 + (ptr[6] - '0');
            uint8_t ip3 = (ptr[8] - '0') * 100 + (ptr[9] - '0') * 10 + (ptr[10] - '0');
            uint8_t ip4 = (ptr[12] - '0') * 100 + (ptr[13] - '0') * 10 + (ptr[14] - '0');

            local_match_count++;

        }

        ptr += LOG_LINE_LENGTH;
    }

    if (local_match_count > 0) {
        printf("[Rank %d] Processed batch: Found %lld events with status %d\n",
               rank, local_match_count, target_status);
    }
}