#include <mpi.h>
#include <stdint.h>
#include <stdio.h>

#define TRIALS 10000000LL

/* Each rank uses its own pseudorandom generator state. */
static double random_unit(uint64_t *state)
{
    uint64_t x = *state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *state = x;

    uint64_t result = x * UINT64_C(2685821657736338717);
    return (double)(result >> 11) * (1.0 / 9007199254740992.0);
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    char hostname[MPI_MAX_PROCESSOR_NAME];
    int hostname_length;
    MPI_Get_processor_name(hostname, &hostname_length);

    long long local_trials = TRIALS / size;
    if (rank < TRIALS % size)
        ++local_trials;

    uint64_t state = UINT64_C(123456789)
                   + UINT64_C(987654321) * (uint64_t)rank;

    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();

    long long local_inside = 0;

    for (long long i = 0; i < local_trials; ++i) {
        double x = random_unit(&state);
        double y = random_unit(&state);

        if (x * x + y * y <= 1.0)
            ++local_inside;
    }

    long long total_inside = 0;
    long long total_trials = 0;

        const int result_tag = 100;

    if (rank == 0) {
        total_inside = local_inside;

        for (int worker = 1; worker < size; ++worker) {
            long long worker_inside;

            MPI_Recv(&worker_inside, 1, MPI_LONG_LONG_INT,
                     MPI_ANY_SOURCE, result_tag,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            total_inside += worker_inside;
        }
    } else {
        MPI_Send(&local_inside, 1, MPI_LONG_LONG_INT,
                 0, result_tag, MPI_COMM_WORLD);
    }

    MPI_Reduce(&local_trials, &total_trials, 1,
               MPI_LONG_LONG_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    double elapsed = MPI_Wtime() - start;
    double maximum_time = 0.0;

    MPI_Reduce(&elapsed, &maximum_time, 1,
               MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    printf("Rank %d on %s: trials = %lld, inside = %lld\n",
           rank, hostname, local_trials, local_inside);

    int valid = 1;

    if (rank == 0) {
        double pi = 4.0 * (double)total_inside / (double)total_trials;
        double reference = 3.141592653589793;
        double error = pi - reference;
        if (error < 0.0)
            error = -error;

        valid = (total_trials == TRIALS
                 && total_inside >= 0
                 && total_inside <= total_trials);

        printf("\nProcesses = %d\n", size);
        printf("Total trials = %lld\n", total_trials);
        printf("Inside points = %lld\n", total_inside);
        printf("Estimated pi = %.10f\n", pi);
        printf("Reference pi = %.10f\n", reference);
        printf("Absolute error = %.10f\n", error);
        printf("Trial count validation: %s\n",
               valid ? "CORRECT" : "INCORRECT");
        printf("Maximum elapsed time = %.9f seconds\n", maximum_time);
    }

    MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Finalize();
    return valid ? 0 : 1;
}