#include <mpi.h>
#include <stdio.h>

#define N 10000000LL

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    char hostname[MPI_MAX_PROCESSOR_NAME];
    int hostname_length;
    MPI_Get_processor_name(hostname, &hostname_length);

    /* Distribute any remainder among the first ranks. */
    long long base = N / size;
    long long remainder = N % size;
    long long count = base + (rank < remainder ? 1 : 0);
    long long first = rank * base
                    + (rank < remainder ? rank : remainder) + 1;
    long long last = first + count - 1;

    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();

    long long local_sum = 0;
    for (long long value = first; value <= last; ++value)
        local_sum += value;

    long long total_sum = 0;
    MPI_Reduce(&local_sum, &total_sum, 1,
               MPI_LONG_LONG_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    double elapsed = MPI_Wtime() - start;
    double maximum_time = 0;
    MPI_Reduce(&elapsed, &maximum_time, 1,
               MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    /* Print after measuring so terminal output is excluded. */
    printf("Rank %d on %s: range [%lld, %lld], local sum = %lld\n",
           rank, hostname, first, last, local_sum);

    int correct = 1;
    if (rank == 0) {
        long long expected = N * (N + 1) / 2;
        correct = (total_sum == expected);

        printf("\nProcesses = %d\n", size);
        printf("Total sum = %lld\n", total_sum);
        printf("Expected  = %lld\n", expected);
        printf("Validation: %s\n", correct ? "CORRECT" : "INCORRECT");
        printf("Maximum elapsed time = %.9f seconds\n", maximum_time);
    }

    MPI_Bcast(&correct, 1, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Finalize();
    return correct ? 0 : 1;
}
