
#include "mpi_server_thread.h"
#include "mpi_utils.h"

// Include the matrix element kernel function:
#include "me_kernel.h"

// Default matrix size
#define GLOBAL_DIM  10000LL

// CLI options:
#include <getopt.h>

static const struct option cliOptions[] = {
        { "help", no_argument, NULL, 'h' },
        { "dims", required_argument, NULL, 'd' },
        { "blocks", required_argument, NULL, 'b' },
        { "auto-grid", no_argument, NULL, 'a' },
        { "row-major", no_argument, NULL, 'r' },
        { "column-major", no_argument, NULL, 'c' },
        { "root", required_argument, NULL, '0' },
        { NULL, 0, NULL, 0 }
    };
static const char *cliOptionsStr = "hd:b:arc0:";

//

void
usage(
    const char  *exe
)
{
    printf(
            "usage:\n\n"
            "    %s {options}\n\n"
            "  options:\n\n"
            "    --help/-h                  show this information\n"
            "    --dims/-d <matrix-2d-dims> choose matrix dimensions (default " BASE_INT_FMT ")\n"
            "    --blocks/-b <block-dims>   choose the global matrix partitioning scheme (default\n"
            "                               is to use auto-grid)\n"
            "    --auto-grid/-a             automatically choose the global matrix partitioning\n"
            "                               scheme\n"
            "    --row-major/-r             use column-major storage and distribution across ranks\n"
            "    --column-major/-c          use column-major storage and distribution across ranks\n"
            "    --root/-0 #                elect the given rank id as the root server\n"
            "\n"
            "  <matrix-2d-dims> = # | #,#   given a single integer value, a square matrix of the given\n"
            "                               number of rows and columns is chosen; otherwise, the first\n"
            "                               integer in the comma-delimited pair is the row count, the\n"
            "                               second is the column count\n"
            "  <block-dims> = # | #,#       paritition the global matrix into:\n"
            "                                   # : this integer number of rows AND columns\n"
            "                                   #,# : the given integer number of rows,columns\n"
            "\n",
            exe,
            GLOBAL_DIM
        );
}

//

bool
parseDims(
    const char  *optarg,
    base_int_t  *r,
    base_int_t  *c
)
{
    long long int   lr, lc;
    char            *endptr;
    
    lr = strtoll(optarg, &endptr, 0);
    if ( lr && (endptr > optarg) ) {
        switch ( *endptr ) {
            case ',': {
                char    *val2ptr = endptr + 1;
                lc = strtol(val2ptr, &endptr, 0);
                if ( lc && (endptr > val2ptr) ) {
                    *r = lr, *c = lc;
                    return true;
                } else {
                    mpi_printf(-1, "error parsing second element of dimensions `%s`", optarg);
                }
                break;
            }
            case '\0':
                *r = *c = lr;
                return true;
            default:
                mpi_printf(-1, "invalid character `%c` in dimensions `%s`", *endptr, optarg);
                break;
        }
    } else {
        mpi_printf(-1, "error parsing first element of dimensions `%s`", optarg);
    }
    return false;
}

//

static inline base_int_t
base_int_min(
    base_int_t      i1,
    base_int_t      i2
)
{
    return (i1 > i2) ? i2 : i1;
}

//

int
main(
    int         argc,
    char*       argv[]
)
{
    int                     thread_req, thread_prov, optch;
    mpi_server_thread_t     the_server;
    mpi_server_thread_msg_t msg;
    void                    *thread_rc;
    
    int                     root_rank = 0;
    base_int_t              global_rows = GLOBAL_DIM, global_cols = GLOBAL_DIM,
                            block_rows = 0, block_cols = 0;
    bool                    is_row_major = true;
    
    thread_req = MPI_THREAD_MULTIPLE;
    MPI_Init_thread(&argc, &argv, thread_req, &thread_prov);
    if ( thread_prov != MPI_THREAD_MULTIPLE ) {
        fprintf(stderr, "ERROR:  MPI does not support MPI_THREAD_MULTIPLE\n");
        exit(1);
    }
    MPI_Comm_rank(MPI_COMM_WORLD, &the_server.dist_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &thread_req);
    
    while ( (optch = getopt_long(argc, argv, cliOptionsStr, cliOptions, NULL)) != -1 ) {
        switch ( optch ) {
        
            case 'h':
                if ( the_server.dist_rank == 0 ) usage(argv[0]);
                exit(0);
            
            case 'd':
                if ( ! parseDims(optarg, &global_rows, &global_cols) ) exit(EINVAL);
                break;
            
            case 'b':
                if ( ! parseDims(optarg, &block_rows, &block_cols) ) exit(EINVAL);
                break;
            
            case 'a':
                block_rows = block_cols = 0;
                break;
            
            case 'r':
                is_row_major = true;
                break;
            
            case 'c':
                is_row_major = false;
                break;
            
            case '0': {
                char        *endptr;
                long int    l = strtol(optarg, &endptr, 0);
                
                if ( (l >= 0) && (endptr > optarg) ) {
                    root_rank = (int)l;
                } else {
                    mpi_printf(0, "invalid root rank id `%s`", optarg);
                    exit(EINVAL);
                }
                break;
            }
            
        }
    }
    
    if ( ! mpi_server_thread_init(&the_server, 0, global_rows, global_cols, block_rows, block_cols, is_row_major, NULL) ) {
        mpi_printf(-1, "ERROR:  unable to initialize mpi_server instance");
        MPI_Finalize();
        exit(1);
    }
    
    mpi_printf(0, "");
    mpi_printf(0, "Welcome to the threaded MPI matrix element work server demo!");
    mpi_printf(0, "");
    mpi_printf(0, "A " BASE_INT_FMT "x" BASE_INT_FMT " matrix is distributed across %d ranks and matrix elements of the form", the_server.dim_global[0], the_server.dim_global[1], thread_req);
    mpi_printf(0, "");
    mpi_printf(0, "    %s", me_kernel_description);
    mpi_printf(0, "");
    mpi_printf(0, "are calculated.");
    mpi_printf(0, "");
    MPI_Barrier(MPI_COMM_WORLD);
    
    if ( ! mpi_server_thread_start(&the_server) ) {
        mpi_printf(-1, "ERROR:  unable to launch server thread");
        MPI_Finalize();
        exit(1);
    }
    
    // Proceed to request work...
    if ( the_server.dist_rank == the_server.root_rank ) {
        int             rank;
        
        mpi_printf(-1, "matrix element loop running");
        while ( true ) {
            int_pair_t  p_low, p_high, p;
            
            if ( ! mpi_assignable_work_next_unit(the_server.assignable_work, the_server.root_rank, 0, &p_low, &p_high) ) break;
            
            //
            // Produce matrix elements:
            //
            for ( p.i = p_low.i; p.i < p_high.i; p.i++ )
                for ( p.j = p_low.j; p.j < p_high.j; p.j++ )
                    mpi_server_thread_memory_write(&the_server, p, me_kernel(p));
                    
            // Notify the work unit manager that we finished this unit:
            mpi_assignable_work_complete(the_server.assignable_work, p_low, p_high);
        }
        mpi_printf(-1, "exited element loop, waiting for all work to complete");
        while ( ! mpi_assignable_work_all_completed(the_server.assignable_work) ) sleep (1);
        mpi_printf(-1, "sending shutdown message to other ranks' server threads");
        msg.msg_type = mpi_server_thread_msg_type_memory;
        msg.msg_id = mpi_server_thread_msg_id_shutdown;
        rank = 1;
        while ( rank < the_server.dist_size )
            MPI_Send(&msg, 1, mpi_get_msg_datatype(), rank++, mpi_server_thread_msg_tag, MPI_COMM_WORLD);
        mpi_printf(-1, "canceling server thread");
        mpi_server_thread_cancel(&the_server);
    } else {
        MPI_Status  status;
        int         mpi_rc;
        
        mpi_printf(-1, "matrix element loop running");
        msg.msg_type = mpi_server_thread_msg_type_work;
        msg.msg_id = mpi_server_thread_msg_id_work_request;
        mpi_rc = MPI_Send(&msg, 1, mpi_get_msg_datatype(), the_server.root_rank, mpi_server_thread_msg_tag, MPI_COMM_WORLD);
        if ( mpi_rc == MPI_SUCCESS ) {
            while ( true ) {
                int_pair_t      p;
                
                mpi_rc = MPI_Recv(&msg, 1, mpi_get_msg_datatype(), the_server.root_rank, mpi_client_thread_msg_tag, MPI_COMM_WORLD, &status);
                if ( mpi_rc != MPI_SUCCESS ) {
                    mpi_printf(-1, "MPI_Recv error %d", mpi_rc);
                }
                if ( msg.p_low.i == -1 ) break;
                
                //
                // Produce matrix elements:
                //
                for ( p.i = msg.p_low.i; p.i < msg.p_high.i; p.i++ )
                    for ( p.j = msg.p_low.j; p.j < msg.p_high.j; p.j++ )
                        mpi_server_thread_memory_write(&the_server, p, me_kernel(p));
                
                // Notify the work unit manager that we finished this unit:
                msg.msg_type = mpi_server_thread_msg_type_work;
                msg.msg_id = mpi_server_thread_msg_id_work_complete_and_allocate;
                mpi_rc = MPI_Send(&msg, 1, mpi_get_msg_datatype(), the_server.root_rank, mpi_server_thread_msg_tag, MPI_COMM_WORLD);
            }
            mpi_printf(-1, "exited element loop");
        }
    }
    mpi_server_thread_join(&the_server);
    MPI_Barrier(MPI_COMM_WORLD);
    
    //
    // Pass the ball from rank 0 on down, when a rank receives the ball it prints
    // the upper-left 10x10 chunk of its local sub-matrix:
    //
    if ( the_server.dist_rank == 0 ) {
        base_int_t  i, j;
        int         the_ball;
        
        mpi_printf(-1, "Sub-matrices in sequence by rank:\n\nRank 0:\n");
        for ( i = 0; i < base_int_min(10, the_server.dim_per_rank[0]); i++ ) {
            printf("    %8.3lf", the_server.local_sub_matrix[mpi_server_thread_index_global_to_local_offset(&the_server, int_pair_make(i, 0))]);
            for ( j = 1; j < base_int_min(10, the_server.dim_per_rank[1]); j++ )
                printf(", %8.3lf", the_server.local_sub_matrix[mpi_server_thread_index_global_to_local_offset(&the_server, int_pair_make(i, j))]);
            printf("\n");
        }
        MPI_Send(&the_ball, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
    } else {
        base_int_t  i, j;
        int         the_ball;
        
        MPI_Recv(&the_ball, 1, MPI_INT, the_server.dist_rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("\nRank %d:\n", the_server.dist_rank);
        for ( i = the_server.local_sub_matrix_row_range.start; i < the_server.local_sub_matrix_row_range.start + base_int_min(10, the_server.dim_per_rank[0]); i++ ) {
            printf("    %8.3lf", the_server.local_sub_matrix[mpi_server_thread_index_global_to_local_offset(&the_server, int_pair_make(i, the_server.local_sub_matrix_col_range.start))]);
            for ( j = the_server.local_sub_matrix_col_range.start + 1; j < the_server.local_sub_matrix_col_range.start + base_int_min(10, the_server.dim_per_rank[1]); j++ )
                printf(", %8.3lf", the_server.local_sub_matrix[mpi_server_thread_index_global_to_local_offset(&the_server, int_pair_make(i, j))]);
            printf("\n");
        }
        if ( the_server.dist_rank + 1 < the_server.dist_size )
            MPI_Send(&the_ball, 1, MPI_INT, the_server.dist_rank + 1, 0, MPI_COMM_WORLD);
    }

    mpi_printf(-1, "ready to exit");
    
    MPI_Finalize();
    return 0;
}
