#include "../include/analyzer.h"
#include <cstdio>
#include <cstring>
#include <mpi.h>
#include <unordered_set>
#include <algorithm>
#include <unordered_map>

// Global structures to store the aggregated log data
static std::unordered_map<uint32_t, int> ip_counts;
static std::vector<uint32_t> out_buffers[3];
static uint64_t rr_counters[3] = {0, 0, 0};
static int g_rank, g_np, g_my_color;

/**
 * Parses a 3-digit string representation of an integer into uint32_t.
 * Optimized for performance using bitwise operations.
 *
 * @param ptr Pointer to the start of the 3-digit string
 * @return The parsed integer value
 */
inline uint32_t parse_3_digits(const char* ptr) {
    uint32_t val;
    std::memcpy(&val, ptr, sizeof(val));
    val = (val & 0x00FFFFFF) - 0x00303030;
    return (val & 0xFF) * 100 + ((val >> 8) & 0xFF) * 10 + ((val >> 16) & 0xFF);
}

/**
 * Initializes the analyzer state for the current MPI process.
 *
 * @param rank Rank of the current process
 * @param np Total number of processes
 */
void init_analyzer(int rank, int np) {
    g_rank = rank;
    g_np = np;
    g_my_color = rank % 3;
    for (int i = 0; i < 3; ++i) out_buffers[i].reserve(BATCH_LIMIT);
}

/**
 * Probes and receives pending messages from other processes.
 * Used for asynchronous point-to-point communication.
 */
void receive_pending() {
    int flag;
    MPI_Status probe_status;
    MPI_Iprobe(MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &flag, &probe_status);
    while (flag) {
        int count;
        MPI_Get_count(&probe_status, MPI_UINT32_T, &count);
        std::vector<uint32_t> incoming(count);
        MPI_Recv(incoming.data(), count, MPI_UINT32_T, probe_status.MPI_SOURCE, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // Increment the counter for the received IP addresses
        for (uint32_t ip : incoming) ip_counts[ip]++;

        MPI_Iprobe(MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &flag, &probe_status);
    }
}

/**
 * Flushes the outgoing buffer for a specific color/group.
 * Sends accumulated data to a target process in a round-robin fashion.
 *
 * @param color The target group color (0, 1, or 2)
 */
void flush_buffer(int color) {
    if (out_buffers[color].empty()) return;
    int cat_size = g_np / 3 + (color < (g_np % 3) ? 1 : 0);
    int target_rank = ( (rr_counters[color] % cat_size) * 3) + color;

    MPI_Request req;
    MPI_Isend(out_buffers[color].data(), out_buffers[color].size(), MPI_UINT32_T, target_rank, 0, MPI_COMM_WORLD, &req);

    int done = 0;
    // Wait for the send to complete while continuing to receive incoming messages
    while (!done) {
        receive_pending();
        MPI_Test(&req, &done, MPI_STATUS_IGNORE);
    }

    out_buffers[color].clear();
    rr_counters[color]++;
}

/**
 * Processes a chunk of the log file residing in memory.
 * Extracts IPs and statuses, forwarding them to the appropriate groups.
 *
 * @param current_buffer Pointer to the memory buffer holding log data
 * @param current_bytes_read Amount of bytes in the current buffer
 * @param fh MPI File handle
 */
void process_logs(const char* current_buffer, int current_bytes_read, MPI_File fh) {
    const char* ptr = current_buffer;
    const char* end = current_buffer + current_bytes_read;
    int target_st = (g_my_color == 0) ? 200 : (g_my_color == 1) ? 404 : 500;
    int counter = 0;

    // Process each log line within the buffer
    while (ptr + LOG_LINE_LENGTH <= end) {
        // Periodically check for incoming messages
        if ((counter++ & 1023) == 0) receive_pending();

        int st = parse_3_digits(ptr + 27);
        uint32_t ip = (parse_3_digits(ptr) << 24) | (parse_3_digits(ptr + 4) << 16) |
                      (parse_3_digits(ptr + 8) << 8) | parse_3_digits(ptr + 12);

        if (st == target_st) {
            ip_counts[ip]++; // Store frequency locally
        } else {
            // Forward data to the corresponding target group
            int col = (st == 200) ? 0 : (st == 404) ? 1 : 2;
            out_buffers[col].push_back(ip);
            if (out_buffers[col].size() >= BATCH_LIMIT) flush_buffer(col);
        }
        ptr += LOG_LINE_LENGTH;
    }
}

/**
 * Concludes the log processing phase. Flushes all remaining buffers,
 * waits for all peers, aggregates the local data into global results,
 * and writes the final output to a file.
 *
 * @param new_comm Sub-communicator for the current process group
 * @param target_status The HTTP status code assigned to this group
 * @param np Total number of processes
 */
void finalize_processing(MPI_Comm new_comm, int target_status, const int np) {
    for (int i = 0; i < 3; ++i) flush_buffer(i);

    MPI_Request req;
    int done = 0;
    MPI_Ibarrier(MPI_COMM_WORLD, &req);
    // Continue processing incoming messages while waiting for barrier
    while (!done) {
        receive_pending();
        MPI_Test(&req, &done, MPI_STATUS_IGNORE);
    }

    // Pack local frequencies into 64-bit integers (32 bits for IP, 32 bits for count)
    std::vector<uint64_t> local_v;
    local_v.reserve(ip_counts.size());
    for (const auto& pair : ip_counts) {
        local_v.push_back( ((uint64_t)pair.first << 32) | (uint32_t)pair.second );
    }

    int l_cnt = local_v.size(), l_rank, l_np;
    MPI_Comm_rank(new_comm, &l_rank);
    MPI_Comm_size(new_comm, &l_np);

    // Gather array sizes to rank 0 within the sub-communicator
    std::vector<int> counts(l_rank == 0 ? l_np : 0);
    MPI_Gather(&l_cnt, 1, MPI_INT, counts.data(), 1, MPI_INT, 0, new_comm);

    if (l_rank == 0) {
        std::vector<int> displs(l_np);
        int total = 0;
        for (int i = 0; i < l_np; ++i) { displs[i] = total; total += counts[i]; }

        std::vector<uint64_t> global_v(total);
        MPI_Gatherv(local_v.data(), l_cnt, MPI_UINT64_T, global_v.data(), counts.data(), displs.data(), MPI_UINT64_T, 0, new_comm);

        std::unordered_map<uint32_t, int> final_map;
        int total_requests = 0;
        for (uint64_t val : global_v) {
            uint32_t ip = (uint32_t)(val >> 32);
            int count = (int)(val & 0xFFFFFFFF);
            final_map[ip] += count;
            total_requests += count;
        }

        std::vector<std::pair<uint32_t, int>> sorted_ips(final_map.begin(), final_map.end());
        std::sort(sorted_ips.begin(), sorted_ips.end(), [](const auto& a, const auto& b) {
            return a.second > b.second;
        });

        // Write statistics to output file instead of printing to console
        char filename[64];
        sprintf(filename, "output_status_%d_%d.txt", target_status, np);
        FILE* out = fopen(filename, "w");
        if (out) {
            fprintf(out, "[SUMMARY STATISTICS] Status %d\n", target_status);
            fprintf(out, "Total requests: %d\n", total_requests);
            fprintf(out, "Unique IPs: %zu\n\n", sorted_ips.size());
            fprintf(out, "[ANOMALIES]:\n");
            for (size_t i = 0; i < 10 && i < sorted_ips.size(); ++i) {
                uint32_t ip = sorted_ips[i].first;
                fprintf(out, "%zu. %d.%d.%d.%d : %d hits\n",
                       i + 1, (ip >> 24) & 0xFF, (ip >> 16) & 0xFF,
                       (ip >> 8) & 0xFF, ip & 0xFF, sorted_ips[i].second);
            }
            fclose(out);
            printf("[SUMMARY STATISTICS] Status %d\n", target_status);
            printf("Total requests: %d\n", total_requests);
            printf("Unique IPs: %zu\n\n", sorted_ips.size());
            printf("[ANOMALIES]:\n");
            for (size_t i = 0; i < 10 && i < sorted_ips.size(); ++i) {
                uint32_t ip = sorted_ips[i].first;
                printf("%zu. %d.%d.%d.%d : %d hits\n",
                       i + 1, (ip >> 24) & 0xFF, (ip >> 16) & 0xFF,
                       (ip >> 8) & 0xFF, ip & 0xFF, sorted_ips[i].second);
            }
            printf("[Rank %d] Results for status %d saved to %s\n\n", g_rank, target_status, filename);
        }
    } else {
        MPI_Gatherv(local_v.data(), l_cnt, MPI_UINT64_T, nullptr, nullptr, nullptr, MPI_UINT64_T, 0, new_comm);
    }
}

/**
 * Sequential implementation of log processing for performance comparison.
 * Reads the file linearly and computes counts natively.
 *
 * @param filename The path to the log file to be processed
 */
void process_logs_sequential(const char* filename) {
    FILE* file = fopen(filename, "r");
    if (!file) return;

    std::unordered_map<int, std::unordered_map<uint32_t, int>> db;
    char line[64];

    while (fgets(line, sizeof(line), file)) {
        if (strlen(line) < LOG_LINE_LENGTH) continue;
        int st = parse_3_digits(line + 27);
        uint32_t ip = (parse_3_digits(line) << 24) | (parse_3_digits(line + 4) << 16) |
                      (parse_3_digits(line + 8) << 8) | parse_3_digits(line + 12);
        db[st][ip]++;
    }
    fclose(file);

    for (const auto& [status, map] : db) {
        char out_name[64];
        sprintf(out_name, "seq_output_status_%d.txt", status);
        FILE* out = fopen(out_name, "w");
        if (out) {
            std::vector<std::pair<uint32_t, int>> sorted_ips(map.begin(), map.end());
            std::sort(sorted_ips.begin(), sorted_ips.end(), [](const auto& a, const auto& b) {
                return a.second > b.second;
            });
            fprintf(out, "Sequential Analysis - Status %d\n", status);
            fprintf(out, "Unique IPs: %zu\n", sorted_ips.size());
            for (size_t i = 0; i < 5 && i < sorted_ips.size(); ++i) {
                uint32_t ip = sorted_ips[i].first;
                fprintf(out, "%zu. %d.%d.%d.%d : %d hits\n",
                       i + 1, (ip >> 24) & 0xFF, (ip >> 16) & 0xFF,
                       (ip >> 8) & 0xFF, ip & 0xFF, sorted_ips[i].second);
            }

            printf("[SUMMARY STATISTICS] Status %d\n", status);
            printf("Unique IPs: %zu\n\n", sorted_ips.size());
            printf("[ANOMALIES]:\n");

            for (size_t i = 0; i < 10 && i < sorted_ips.size(); ++i) {
                uint32_t ip = sorted_ips[i].first;
                printf("  %2zu. %3d.%3d.%3d.%3d : %6d hits\n",
                       i + 1,
                       (ip >> 24) & 0xFF,
                       (ip >> 16) & 0xFF,
                       (ip >> 8)  & 0xFF,
                       ip & 0xFF,
                       sorted_ips[i].second);
            }
            printf("Seq Results for status %d saved to %s\n\n", status ,filename);
            fclose(out);
        }
    }
}