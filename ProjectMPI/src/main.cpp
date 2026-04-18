#include <mpi.h>
#include <vector>
#include <algorithm>
#include <iostream>
#include <string>
#include "../include/analyzer.h"
#include "../include/generator.h" // Generator header

int main(int argc, char* argv[]) {
    // Initialize MPI environment
    MPI_Init(&argc, &argv);
    int rank, np;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &np);

    // CLI Arguments Parsing
    bool run_gen = false;
    bool run_seq = false;
    long long gen_lines = 1000000;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-gen" && i + 1 < argc) {
            run_gen = true;
            gen_lines = std::stoll(argv[++i]);
        } else if (arg == "-seq") {
            run_seq = true;
        }
    }

    // 1. Data Generation (Master process only)
    if (run_gen && rank == 0) {
        printf("[Rank 0] Generating file (%lld lines)...\n", gen_lines);
        generate_logs_to_file("server_logs.csv", gen_lines);
        MPI_Finalize();
        return 0;
    }
    MPI_Barrier(MPI_COMM_WORLD); // Wait for the generation to complete

    double start_time = MPI_Wtime();

    // 2. Sequential Processing (For performance comparison)
    if (run_seq) {
        if (rank == 0) {
            printf("[Rank 0] Starting sequential processing...\n");
            process_logs_sequential("server_logs.csv");
            printf("Sequential processing time: %f seconds\n", MPI_Wtime() - start_time);
        }
        MPI_Finalize();
        return 0;
    }

    // 3. Parallel Processing
    MPI_File fh;
    if (MPI_File_open(MPI_COMM_WORLD, "server_logs.csv", MPI_MODE_RDONLY, MPI_INFO_NULL, &fh) != MPI_SUCCESS) {
        if (rank == 0) fprintf(stderr, "Error opening file\n");
        MPI_Finalize(); return 1;
    }

    // Requirement 5: Create communicators via MPI_Group (instead of MPI_Comm_split)
    MPI_Group world_group, new_group;
    MPI_Comm_group(MPI_COMM_WORLD, &world_group);

    // Assign a color based on the rank.
    // This color defines which group the process belongs to.
    int my_color = rank % 3;
    std::vector<int> group_ranks;
    for (int i = 0; i < np; ++i) {
        if (i % 3 == my_color) group_ranks.push_back(i);
    }

    // Include the calculated ranks to form a new group
    MPI_Group_incl(world_group, group_ranks.size(), group_ranks.data(), &new_group);

    // Create a new communicator for the new group
    MPI_Comm new_comm;
    MPI_Comm_create(MPI_COMM_WORLD, new_group, &new_comm);

    // Free the groups as they are no longer needed
    MPI_Group_free(&world_group);
    MPI_Group_free(&new_group);

    // File Partitioning Math (Chunking)
    MPI_Offset file_size;
    MPI_File_get_size(fh, &file_size);
    long long total_lines = file_size / LOG_LINE_LENGTH;
    size_t chunk = total_lines / np, rem = total_lines % np;

    // Distribute remainder across the first 'rem' ranks
    long long my_rows = (rank < (int)rem) ? chunk + 1 : chunk;
    MPI_Offset offset = (rank < (int)rem) ? (MPI_Offset)rank * (chunk + 1) * LOG_LINE_LENGTH
                                          : (MPI_Offset)(rank * chunk + rem) * LOG_LINE_LENGTH;
    MPI_Offset end_offset = offset + (my_rows * LOG_LINE_LENGTH);

    // Initialize the analyzer with the current rank and number of processes
    init_analyzer(rank, np);

    // Double-buffering for asynchronous I/O
    std::vector<char> buf_a(64 * 1024 * 1024), buf_b(64 * 1024 * 1024);
    char *curr = buf_a.data(), *next = buf_b.data();
    MPI_Request io_req = MPI_REQUEST_NULL;
    MPI_Status st;

    int read_bytes = 0;
    size_t to_read = std::min((size_t)(end_offset - offset), buf_a.size());

    // Pre-read the first buffer
    if (to_read > 0) {
        MPI_File_iread_at(fh, offset, curr, (int)to_read, MPI_CHAR, &io_req);
        MPI_Wait(&io_req, &st);
        MPI_Get_count(&st, MPI_CHAR, &read_bytes);
        offset += read_bytes;
    }

    // Processing loop
    while (read_bytes > 0) {
        // Start reading into the next buffer asynchronously
        size_t next_to_read = std::min((size_t)(end_offset - offset), buf_a.size());
        if (next_to_read > 0) MPI_File_iread_at(fh, offset, next, (int)next_to_read, MPI_CHAR, &io_req);

        // Process the current buffer
        process_logs(curr, read_bytes, fh);

        // Wait for the next read to complete and swap buffers
        if (next_to_read > 0) {
            MPI_Wait(&io_req, &st);
            MPI_Get_count(&st, MPI_CHAR, &read_bytes);
            offset += read_bytes;
            std::swap(curr, next);
        } else read_bytes = 0;
    }

    // Finalize processing and aggregate results
    int target_status = (my_color == 0) ? 200 : (my_color == 1) ? 404 : 500;
    finalize_processing(new_comm, target_status, np);

    // Cleanup resources
    MPI_File_close(&fh);
    if (new_comm != MPI_COMM_NULL) MPI_Comm_free(&new_comm);

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        printf("\nParallel processing time: %f seconds\n", MPI_Wtime() - start_time);
    }

    MPI_Finalize();
    return 0;
}