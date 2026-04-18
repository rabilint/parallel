//
// Created by rabilint on 13.04.26.
//

#include "../include/generator.h"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <mpi.h>

// Structure of a log entry:
// 019.062.001.243,1700002301,200
// [IP-address],[Timestamp],[HTTP-status]

/**
 * Generates a mock HTTP server log file with random data.
 *
 * @param filename The path where the generated logs will be saved
 * @param total_lines The total number of log lines to generate
 */
void generate_logs_to_file(const char* filename, long long total_lines) {
    FILE *file = fopen(filename, "w");
    if (!file) {
        perror("Error creating file");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Allocate an enlarged buffer for file I/O optimization handling massive write operations
    char io_buffer[BUFFER_SIZE];
    setvbuf(file, io_buffer, _IOFBF, BUFFER_SIZE);

    srand((unsigned)time(NULL));

    // Base Unix Timestamp (needs to be exactly 10 digits long, e.g., Unix Epoch for 2023)
    long base_timestamp = 1700000000;

    for (long long i = 0; i < total_lines; i++) {
        // 1. Generate IP address (each octet ranging from 0 to 255)
        int ip1 = rand() % 256;
        int ip2 = rand() % 256;
        int ip3 = rand() % 256;
        int ip4 = rand() % 256;

        // 2. Generate HTTP status with a weighted probability distribution
        int rand_val = rand() % 100;
        int status;
        if (rand_val < 80) {
            status = 200;      // 80% successful requests
        } else if (rand_val < 95) {
            status = 404;      // 15% errors (possible vulnerability scanning)
        } else {
            status = 500;      // 5% critical server errors
        }

        long current_timestamp = base_timestamp + i + (rand() % 10000);

        fprintf(file, "%03d.%03d.%03d.%03d,%ld,%03d\n",
                ip1, ip2, ip3, ip4, current_timestamp, status);
    }

    fclose(file);
    printf("Generation complete. Wrote %lld lines to %s\n", total_lines, filename);
}
