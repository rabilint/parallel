//
// Created by rabilint on 13.04.26.
//

#ifndef PROJECTMPI_ANALYZER_H
#define PROJECTMPI_ANALYZER_H
#include <cstdint>

struct LogEvent {
    uint8_t ip[4];
    long timestamp;
    int status;
};

#include <vector>
void process_logs(const char* current_buffer, int current_bytes_read);
#endif //PROJECTMPI_ANALYZER_H
