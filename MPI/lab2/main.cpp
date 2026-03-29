#include <mpi.h>
#include <iostream>
#include <cmath>

bool check_Runge(double I2, double I, double epsilon)
{
    return (fabs(I2 - I) / 3.) < epsilon;
}

double f(double x)
{
    return pow(x, 2 ) * sin(x);
}

double calculate_integral(double a, double b, double n)
{
    double h = (b - a) / n;
    double sum = 0;
    for (int i = 0; i < n; ++i)
    {
        sum += f(a + (i * h) + (h/2)) ;
    }
    return sum * h;
}

int main ( int argc , char * argv [ ] )
{
    int np;
    int rank;
    MPI_Init(&argc ,& argv );
    MPI_Comm_size(MPI_COMM_WORLD, &np );
    MPI_Comm_rank(MPI_COMM_WORLD, &rank );
    double res = 10000;
    double res2 = 0;
    double input[4] = {0,11, 50,100};
    int keep_going = 1;


    while (keep_going == 1)
    {
        double start = input[0];
        double end = input[1];
        double total_n = input[2];
        double n2 = input[3];
        double step = (end - start) / np;
        res = calculate_integral(start + rank * step, start + (rank + 1) * step, total_n / np);
        res2 = calculate_integral(start + rank * step, start + (rank + 1) * step, n2 / np );

        if (rank != 0)
        {
            MPI_Request send_req;
            MPI_Request send_req2;
            MPI_Isend(&res, 1, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, &send_req);
            MPI_Isend(&res2, 1, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD, &send_req2);
            MPI_Wait(&send_req, MPI_STATUS_IGNORE);
            MPI_Wait(&send_req2, MPI_STATUS_IGNORE);
        }

        if (rank == 0)
        {
            MPI_Request recv_reqs[np - 1];
            MPI_Request recv2[np - 1];
            MPI_Status status[np - 1];
            MPI_Status status2[np - 1];
            double results[np - 1];
            double result2[np - 1];

            for (int i = 0; i < (np - 1); i++)
            {
                MPI_Irecv(&results[i], 1, MPI_DOUBLE, (i + 1), 1, MPI_COMM_WORLD, &recv_reqs[i]);
                MPI_Irecv(&result2[i], 1, MPI_DOUBLE, (i + 1), 2, MPI_COMM_WORLD, &recv2[i]);
            }
            MPI_Waitall(np - 1, recv_reqs, status);
            MPI_Waitall(np - 1, recv2, status2);

            for (int i = 0; i < (np - 1); i++)
            {
                res += results[i];
                res2 += result2[i];
            }
            printf("res = %f\n", res);
            printf("res2 = %f\n", res2);
            if (!check_Runge(res2, res, 0.001))
            {
                keep_going = 1;
                input[2] *= 2;
                input[3] *= 2;
            }else
            {
                keep_going = 0;
                printf("\nresult n: %d\n", static_cast<int>( total_n));
            }

        }

        MPI_Bcast(&keep_going, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&input, 4, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (keep_going == 1)
        {
            total_n = input[2];
            n2 = input[3];

        }else
        {
            break;
        }

    }


    MPI_Finalize();
    return 0;
}