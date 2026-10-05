#include <mpi.h>
#include <iostream>
#include <vector>

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size != 2) {
        if (rank == 0)
            std::cerr << "Run this program with exactly 2 processes.\n";

        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }

    constexpr int messages = 3;
    constexpr int tag = 0;
    int number = 0;
    int correct = 1;

    if (rank == 0) {
        int packed_size;
        MPI_Pack_size(1, MPI_INT, MPI_COMM_WORLD, &packed_size);

        int buffer_size =
            messages * (packed_size + MPI_BSEND_OVERHEAD);

        std::vector<char> buffer(buffer_size);
        MPI_Buffer_attach(buffer.data(), buffer_size);

        for (int i = 0; i < messages; ++i) {
            number = i * 10;

            MPI_Bsend(&number, 1, MPI_INT, 1, tag,
                      MPI_COMM_WORLD);

            std::cout << "Process 0 buffered-send "
                      << number << std::endl;
        }

        void *detached_buffer = nullptr;
        int detached_size = 0;

        MPI_Buffer_detach(&detached_buffer, &detached_size);
    } else {
        for (int i = 0; i < messages; ++i) {
            MPI_Recv(&number, 1, MPI_INT, 0, tag,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            std::cout << "Process 1 received "
                      << number << std::endl;

            if (number != i * 10)
                correct = 0;
        }
    }

    int all_correct;
    MPI_Allreduce(&correct, &all_correct, 1,
                  MPI_INT, MPI_MIN, MPI_COMM_WORLD);

    if (rank == 0)
        std::cout << "Validation: "
                  << (all_correct ? "CORRECT" : "INCORRECT")
                  << std::endl;

    MPI_Finalize();
    return all_correct ? 0 : 1;
}
