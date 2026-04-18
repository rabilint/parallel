#include <iostream>
#include <mpi.h>
#include <string>
#include <vector>

int main(int argc, char* argv[])
{
    int rank, np;
    MPI_Comm new_comm = MPI_COMM_NULL; // Ініціалізація null

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &np);

    std::string message = "innit.";


    int color = (rank == 0 || rank == 4) ? 0 : MPI_UNDEFINED;

    MPI_Comm_split(MPI_COMM_WORLD, color, rank, &new_comm);

    int my_new_rank = -1, new_size = -1;

    if (new_comm != MPI_COMM_NULL)
    {
        MPI_Comm_rank(new_comm, &my_new_rank);
        MPI_Comm_size(new_comm, &new_size);

        int msg_size = 0;
        if (my_new_rank == 0)
        {
            message = "new message";
            msg_size = static_cast<int>(message.size());
        }

        MPI_Bcast(&msg_size, 1, MPI_INT, 0, new_comm);

        if (my_new_rank != 0) {
            message.resize(msg_size);
        }

        MPI_Bcast(&message[0], msg_size, MPI_CHAR, 0, new_comm);
    }


    MPI_Barrier(MPI_COMM_WORLD);
    std::cout << "<<Rank " << rank << "/" << np << " | New Rank: "
              << my_new_rank << "/" << new_size << " | Msg: " << message << ">>" << std::endl;


    if (new_comm != MPI_COMM_NULL)
    {
        MPI_Comm_free(&new_comm);
    }

    MPI_Finalize();
    return 0;
}