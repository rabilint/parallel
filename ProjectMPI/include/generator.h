//
// Created by rabilint on 13.04.26.
//

#ifndef PROJECTMPI_GENERATOR_H
#define PROJECTMPI_GENERATOR_H

// Constants for generating log lines
#define LOG_LINE_LENGTH 31
#define BUFFER_SIZE 1048576

/**
 * Generates mock HTTP server logs to a specified file.
 *
 * @param filename Name of the file to be generated
 * @param total_lines The number of log entries to generate
 */
void generate_logs_to_file(const char* filename, long long total_lines);

#endif //PROJECTMPI_GENERATOR_H
