#include <mpi.h>
#include <random>
#include <iostream>
#include <vector>
#include <algorithm>

void print_vector(const std::vector<int>& vector)
{
    for ( int element : vector )
    {
        std::cout << element << " ";
    }
}

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    const int VECTOR_SIZE = 5;

    int x = 2;


    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (world_size < 2) {
        if (world_rank == 0) {
            std::cerr << "Error: code need 2 or more proces" << std::endl;
        }
        MPI_Finalize();
        return 0;
    }

    if (world_rank == 0) {
        std::mt19937 rng(std::random_device{}());
        std::uniform_int_distribution<int> distribution(0, 20 );

        std::vector<int> vector_to_send(VECTOR_SIZE);
        std::ranges::generate( vector_to_send, [&]() { return distribution(rng); } );
        std::vector<int> vector_recived(VECTOR_SIZE);

        MPI_Send(vector_to_send.data(), static_cast<int>(vector_to_send.size()), MPI_INT, 1, 0, MPI_COMM_WORLD);

        std::cout << "Process 0 sent vector ";
        print_vector(vector_to_send);
        std::cout << " to process 1" << std::endl;

        MPI_Recv(vector_recived.data(), VECTOR_SIZE, MPI_INT, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        std::cout << "Process 0 received vector " ;
        print_vector(vector_recived);
        std::cout  << " from process 1" << std::endl;
    }
    else if (world_rank == 1) {

        std::vector<int> received_data(VECTOR_SIZE);

        MPI_Recv(received_data.data(), VECTOR_SIZE, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        std::cout << "Process 1 received vector " ;
        print_vector(received_data);
        std::cout  << " from process 0" << std::endl;
        std::ranges::for_each(received_data, [x](int &n) {  n *= x; }) ;

        MPI_Send(received_data.data(),static_cast<int>(received_data.size()), MPI_INT, 0, 0, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}
