#include <iostream>
#include <mpi.h>
#include <string>

int main(int argc, char* argv[])
{
    int rank, np;
    MPI_Comm new_comm;
    MPI_Group world_group, new_group;
    int ranks_to_include[2] = {0, 4};

    // int color; Left in code for easily switching to MPI_Comm_split

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &np);

    MPI_Comm_group(MPI_COMM_WORLD, &world_group);


    std::string message = "innit.";

    // color = (rank == 0 || rank == 4) ? 0 : MPI_UNDEFINED; Left in code for easily switching to MPI_Comm_split

    MPI_Group_incl(world_group, 2, ranks_to_include, &new_group);
    MPI_Comm_create(MPI_COMM_WORLD, new_group, &new_comm);

    // MPI_Comm_split(MPI_COMM_WORLD, color, 0, &new_comm); Left in code for easily switching to MPI_Comm_split

    int my_new_rank, new_size;
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
        if (my_new_rank != 0) message.resize(msg_size);

        MPI_Bcast(message.data(), msg_size, MPI_CHAR, 0, new_comm);
    }else
    {
        my_new_rank = -1;
        new_size = -1;
    }

    std::cout << "<<MPI_COMM WORLD: "<< rank <<" from "<< np << ". New comm: "
    << my_new_rank << " from "<< new_size << ". Message = " << message << ">>" << std::endl;

    if (new_comm != MPI_COMM_NULL)
    {

        MPI_Comm_free(&new_comm);
    }
    MPI_Group_free(&new_group);
    MPI_Group_free(&world_group);
    MPI_Finalize();
    return 0;

}