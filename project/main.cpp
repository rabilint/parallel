#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>
#include <omp.h>
#include <random>

/**
 * @brief [Requirement 3] Reads binary data from an existing file into RAM.
 * Validates file existence and extracts raw bytes to a continuous vector.
 * @param filename Path to the target file.
 * @return std::vector<uint8_t> containing file bytes, or empty vector on failure.
 */
std::vector<uint8_t> read_file_custom(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file.is_open()) return {};
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<uint8_t> buffer(size);
    if (file.read(reinterpret_cast<char*>(buffer.data()), size)) return buffer;
    return {};
}

/**
 * @brief Performs bitwise XOR encryption on a data block using a pseudo-random key stream.
 * Implements a pseudo-Counter (CTR) mode to eliminate data dependencies between tasks.
 * @param data Pointer to the memory block start.
 * @param size Number of bytes in the current block.
 * @param base_key The user-provided master key.
 * @param offset Global offset of the block, used as a cryptographic nonce.
 */
void encrypt_block(uint8_t* data, const size_t size, const unsigned int base_key, const size_t offset)
{
    // Thread-local Mersenne Twister PRNG instantiation.
    // Combining base_key with offset ensures deterministic, unique byte streams per block,
    // avoiding Race Conditions common with global PRNGs.
    std::mt19937 rng(base_key + offset);
    for (size_t i = 0; i < size; ++i) {
        // XOR transformation using 1 byte extracted from the 32-bit PRNG output.
        data[i] ^= static_cast<uint8_t>(rng() & 0xFF);
    }
}

/**
 * @brief [Requirement 3] Generates a deterministic in-memory dataset using PRNG.
 * Simulates heavy data loads for OpenMP performance benchmarking without I/O bottlenecks.
 */
std::vector<uint8_t> generate_random_data(size_t size_in_bytes) {
    std::vector<uint8_t> buffer(size_in_bytes);
    std::mt19937 rng(42);
    for (size_t i = 0; i < size_in_bytes; ++i) {
        buffer[i] = static_cast<uint8_t>(rng() & 0xFF);
    }
    return buffer;
}

/**
 * @brief [Requirement 3] Generates an in-memory dataset using a custom mathematical function.
 * Extremely fast generation method utilizing arithmetic progression to bypass PRNG overhead.
 */
std::vector<uint8_t> generate_custom_function_data(size_t size_in_bytes) {
    std::vector<uint8_t> buffer(size_in_bytes);
    for (size_t i = 0; i < size_in_bytes; ++i) {
        // Arbitrary math function: modular arithmetic based on array index.
        buffer[i] = static_cast<uint8_t>((i * 13) % 256);
    }
    return buffer;
}

/**
 * @brief [Requirement 4] Dumps the finalized processed binary vector to a disk file.
 */
void write_binary_file(const std::string& filename, const std::vector<uint8_t>& data)
{
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open())
    {
        std::cerr << "Unable to open file" << std::endl;
        exit(-1);
    }
    file.write(reinterpret_cast<const char*>(data.data()), data.size());
}

int main()
{
    // Disable dynamic thread adjustment to enforce strict thread counts during benchmarking.
    omp_set_dynamic(0);

    size_t key = 42;
    std::vector<size_t> data_sizes = {10 * 1024 * 1024, 100 * 1024 * 1024, 500 * 1024 * 1024, 1000 * 1024 * 1024};
    std::vector<int> threads = {1, 2, 4};

    // --- User Interface & Input Methods Selection ---
    std::cout << "Enter encryption key manually (e.g., 42): ";
    std::cin >> key;

    std::cout << "Select data generation method for benchmark:\n"
              << "1. Random Generation (Auto)\n"
              << "2. Custom Math Function\n"
              << "3. Read from existing file (Demo)\n"
              << "Choice: ";
    int choice;
    std::cin >> choice;
    std::string filename;

    if (choice == 3) {
        std::cout << "Enter filename: ";
        std::cin >> filename;
        std::vector<uint8_t> test_file = read_file_custom(filename);
        if (test_file.empty()) {
            std::cout << "File error. Falling back to Random Generation.\n";
            choice = 1;
        } else {
            std::cout << "File read successfully. Size: " << test_file.size() << " bytes.\n";
            // Override benchmarking matrix to strictly evaluate the provided file.
            data_sizes = {test_file.size()};
        }
    }

    // Default optimal task granularity (4MB) targeting L3 Cache alignment.
    const size_t default_block_size = 4194304;
    double start_time;
    double end_time;

    // [Requirement 7] Loop through different data sizes and thread counts for comparison.
    for (size_t current_size : data_sizes)
    {
        printf("\n\nSize: %.2f MB\n", static_cast<double>(current_size) / (1024 * 1024));

        // Dynamically adjust block size for very small files to prevent OpenMP task starvation.
        size_t active_block_size = default_block_size;
        if (current_size < default_block_size * 4 && current_size > 0) {
            active_block_size = current_size / 4;
            if (active_block_size == 0) active_block_size = 1;
        }

        // Initialize pristine baseline data based on user selected method.
        std::vector<uint8_t> original_data;
        if (choice == 2) {
            original_data = generate_custom_function_data(current_size);
        } else if (choice == 3) {
            original_data = read_file_custom(filename);
        } else {
            original_data = generate_random_data(current_size);
        }

        std::vector<uint8_t> working_buffer(current_size);

        for (int amount_of_threads : threads)
        {
            // Reset working buffer to prevent decryption via symmetric XOR.
            std::memcpy(working_buffer.data(), original_data.data(), current_size);

            printf("    \nStart of encrypting with omp parallel with %d threads\n", amount_of_threads);
            start_time = omp_get_wtime();

            // Spawn the OpenMP thread pool.
            #pragma omp parallel num_threads(amount_of_threads)
            {
                // Isolate a single producer thread to partition the array and generate task objects.
                #pragma omp single
                {
                    for (size_t offset = 0; offset < current_size; offset += active_block_size) {
                        // Prevent Buffer Overread on the final tail block.
                        size_t current_block = std::min(active_block_size, current_size - offset);

                        // [Requirement 5] Dispatch an independent task to the thread pool queue.
                        // firstprivate: Captures local copies of execution boundaries.
                        // shared: Grants concurrent access to the global memory array.
                        #pragma omp task firstprivate(offset, current_block) shared(working_buffer)
                        {
                            encrypt_block(working_buffer.data() + offset, current_block, key, offset);
                        }
                    }
                } // Implicit Barrier: Pool threads dequeue and execute tasks until completion.
            }

            end_time = omp_get_wtime();
            printf("    Time: %f\n", end_time - start_time);
        }

        // --- Sequential Baseline Benchmark ---
        std::memcpy(working_buffer.data(), original_data.data(), current_size);
        printf("    \nStart of encrypting with consecutive method\n");
        start_time = omp_get_wtime();

        for (size_t offset = 0; offset < current_size; offset += active_block_size)
        {
            size_t current_block = std::min(active_block_size, current_size - offset);
            encrypt_block(working_buffer.data() + offset, current_block, key, offset);
        }

        end_time = omp_get_wtime();
        printf("Time: %f\n", end_time - start_time);

        // Export the dataset to verify encryption integrity on the final iteration.
        if (current_size == data_sizes.back()) {
            write_binary_file("output_encrypted.bin", working_buffer);
        }
    }

    return 0;
}