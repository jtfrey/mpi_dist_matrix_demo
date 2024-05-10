
#include "mpi_utils.h"

int
mpi_printf(
    int         rank,
    const char  *fmt,
    ...
)
{
    static bool is_inited = false;
    static int _rank, _size, _digits;
    
    int         n = 0;
    va_list     argv;

    if ( ! is_inited ) {
        int     dummy;
        
        MPI_Comm_rank(MPI_COMM_WORLD, &_rank);
        MPI_Comm_size(MPI_COMM_WORLD, &_size);
        _digits = 1, dummy = _size;
        while ( dummy >= 10 ) _digits++, dummy /= 10;
        is_inited = true;
    }
    if ( (rank == -1) || (rank == _rank) ) {
        n = printf("[MPI-%0*d:%0*d][%d] ", _digits, _rank, _digits, _size, getpid());
        va_start(argv, fmt);
        n += vprintf(fmt, argv);
        va_end(argv);
        if ( *(fmt + strlen(fmt) - 1) != '\n' ) n += printf("\n");
    }
    return n;
}
