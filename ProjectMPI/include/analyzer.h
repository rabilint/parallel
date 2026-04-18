//
// Created by rabilint on 13.04.26.
//

#ifndef PROJECTMPI_ANALYZER_H
#define PROJECTMPI_ANALYZER_H

#include <mpi.h>
#include <vector>
#include <cstdint>

// Fixed length of a generated log entry (in bytes)
constexpr int LOG_LINE_LENGTH = 31;

// Maximum number of items in an outgoing buffer before it triggers a flush operation
constexpr int BATCH_LIMIT = 10000;

/**
 * Initializes the analyzer state for the current MPI process.
 */
void init_analyzer(int rank, int np);

/**
 * Processes a chunk of the log file residing in memory.
 */
void process_logs(const char* current_buffer, int current_bytes_read, MPI_File fh);

/**
 * Concludes log processing, flushes all remaining buffers, and aggregates the results.
 */
void finalize_processing(MPI_Comm new_comm, int target_status, int np);

/**
 * A baseline sequential version of log processing for performance comparison.
 */
void process_logs_sequential(const char* filename);

#endif //PROJECTMPI_ANALYZER_H
