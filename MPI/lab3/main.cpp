#include <mpi.h>
#include <vector>
#include <random>

using Matrix1D = std::vector<int>;
constexpr int Matrix_COLUMN = 7;
constexpr int Matrix_ROWS = 5;

void printMatrix(const Matrix1D& matrix1_d)
{
    for (int i = 0; i < Matrix_ROWS; i++)
    {
        for (int j = 0; j < Matrix_COLUMN; j++)
        {
            printf("%d ", matrix1_d[i * Matrix_COLUMN + j]);
        }
        printf("\n");
    }
}

int main(int argc, char* argv[])
{
    int np;
    int rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &np);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);


    std::vector<int> sendcounts(np);
    std::vector<int> displs(np);

    int base_rows = Matrix_ROWS / np;
    int remainder = Matrix_ROWS % np;
    int total_elements = Matrix_ROWS * Matrix_COLUMN;
    int current_displ = 0;

    for (int i = 0; i < np; ++i)
    {
        int rows_for_this_proc = base_rows + (i < remainder ? 1 : 0);
        sendcounts[i] = rows_for_this_proc * Matrix_COLUMN;
        displs[i] = current_displ;
        current_displ += sendcounts[i];
    }

    std::vector<int> matrix_block(sendcounts[rank]);
    Matrix1D matrix;
    if (rank == 0)
    {
        matrix.resize(total_elements);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dist(-100, 100);
        for (int i = 0; i < total_elements; i++)
        {
            matrix[i] = dist(gen);
        }
    }

    if (rank == 0)
    {
        printMatrix(matrix);
    }

    MPI_Scatterv(matrix.data(), sendcounts.data(), displs.data(), MPI_INT,
             matrix_block.data(), sendcounts[rank], MPI_INT, 0, MPI_COMM_WORLD);

    std::vector<int> Col_SUM(Matrix_COLUMN, 0);
    int rows_per_proc = sendcounts[rank] / Matrix_COLUMN;
    for (int i = 0; i < rows_per_proc; i++)
    {
        for (int j = 0; j < Matrix_COLUMN; j++)
        {
            Col_SUM[j] += matrix_block[i * Matrix_COLUMN + j];
        }
    }

    std::vector<int> all_sum;
    if (rank == 0)
    {
        all_sum.resize(Matrix_COLUMN);
    }



    MPI_Reduce(Col_SUM.data(), all_sum.data(), Matrix_COLUMN, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        int max_sum = all_sum[0], max_col_index = 0;
        for (int i = 0; i < all_sum.size(); i++)
        {
            if (all_sum[i] > max_sum)
            {
                max_sum = all_sum[i];
                max_col_index = i;
            }
        }
        printf("max sum of col is %d and index of col is %d\n", max_sum, max_col_index);
    }
    MPI_Finalize();
    return 0;
}
