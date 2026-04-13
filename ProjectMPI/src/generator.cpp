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


//structure of log
//019.062.001.243,1700002301,200
//[IP-адреса],[Timestamp],[HTTP-статус]

void generate_logs_to_file(const char* filename, long long total_lines) {
    FILE *file = fopen(filename, "w");
    if (!file) {
        perror("Помилка створення файлу");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Встановлюємо збільшений буфер для файлу (оптимізація I/O для великих об'ємів)
    char io_buffer[BUFFER_SIZE];
    setvbuf(file, io_buffer, _IOFBF, BUFFER_SIZE);

    srand((unsigned)time(NULL));

    // Базовий час (повинен мати рівно 10 цифр, наприклад Unix Epoch для 2023 року)
    long base_timestamp = 1700000000;

    for (long long i = 0; i < total_lines; i++) {
        // 1. Генерація IP-адреси (кожен октет від 0 до 255)
        int ip1 = rand() % 256;
        int ip2 = rand() % 256;
        int ip3 = rand() % 256;
        int ip4 = rand() % 256;

        // 2. Генерація HTTP-статусу за ймовірностями
        int rand_val = rand() % 100;
        int status;
        if (rand_val < 80) {
            status = 200;      // 80% успішних запитів
        } else if (rand_val < 95) {
            status = 404;      // 15% помилок (можливе сканування)
        } else {
            status = 500;      // 5% критичних помилок сервера
        }

        long current_timestamp = base_timestamp + i + (rand() % 10000);

        fprintf(file, "%03d.%03d.%03d.%03d,%ld,%03d\n",
                ip1, ip2, ip3, ip4, current_timestamp, status);
    }

    fclose(file);
    printf("Генерацію завершено. Записано %lld рядків у файл %s\n", total_lines, filename);
}

