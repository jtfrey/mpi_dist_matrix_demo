
#include "mpi_server_thread.h"
#include "mpi_utils.h"

// Include the matrix element kernel function:
#include "me_kernel.h"

#define GLOBAL_ROWS    10000LL  /* Global matrix dimension, rows */
#define GLOBAL_COLS    10000LL  /* Global matrix dimension, cols */
#define GRID_ROWS          0LL
#define GRID_COLS          0LL

//    /opt/openmpi/5.0.3/bin/mpirun -np 4 --map-by :OVERSUBSCRIBE  ./mpi_dist_matrix

int
main(
    int         argc,
    char*       argv[]
)
{
    int                     thread_req, thread_prov;
    mpi_server_thread_t     the_server;
    mpi_server_thread_msg_t msg;
    void                    *thread_rc;
    
    thread_req = MPI_THREAD_MULTIPLE;
    MPI_Init_thread(&argc, &argv, thread_req, &thread_prov);
    if ( thread_prov != MPI_THREAD_MULTIPLE ) {
        fprintf(stderr, "ERROR:  MPI does not support MPI_THREAD_MULTIPLE\n");
        exit(1);
    }
    MPI_Comm_rank(MPI_COMM_WORLD, &the_server.dist_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &thread_req);
    
    //if ( thread_req != EXPECTED_SIZE ) {
    //    mpi_printf(0, "ERROR:  this program must be run with %d ranks", EXPECTED_SIZE);
    //    MPI_Finalize();
    //    exit(1);
    //}
    
    if ( ! mpi_server_thread_init(&the_server, 0, GLOBAL_ROWS, GLOBAL_COLS, GRID_ROWS, GRID_COLS, true, NULL) ) {
        mpi_printf(-1, "ERROR:  unable to initialize mpi_server instance");
        MPI_Finalize();
        exit(1);
    }
    
    mpi_printf(0, "");
    mpi_printf(0, "Welcome to the threaded MPI matrix element work server demo!");
    mpi_printf(0, "");
    mpi_printf(0, "A %dx%d matrix is distributed across %d ranks and matrix elements of the form", GLOBAL_ROWS, GLOBAL_COLS, thread_req);
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
        for ( i = 0; i < 10; i++ ) {
            printf("    %8.3lf", the_server.local_sub_matrix[mpi_server_thread_index_global_to_local_offset(&the_server, int_pair_make(i, 0))]);
            for ( j = 1; j < 10; j++ )
                printf(", %8.3lf", the_server.local_sub_matrix[mpi_server_thread_index_global_to_local_offset(&the_server, int_pair_make(i, j))]);
            printf("\n");
        }
        MPI_Send(&the_ball, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
    } else {
        base_int_t  i, j;
        int         the_ball;
        
        MPI_Recv(&the_ball, 1, MPI_INT, the_server.dist_rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("\nRank %d:\n", the_server.dist_rank);
        for ( i = the_server.local_sub_matrix_row_range.start; i < the_server.local_sub_matrix_row_range.start + 10; i++ ) {
            printf("    %8.3lf", the_server.local_sub_matrix[mpi_server_thread_index_global_to_local_offset(&the_server, int_pair_make(i, the_server.local_sub_matrix_col_range.start))]);
            for ( j = the_server.local_sub_matrix_col_range.start + 1; j < the_server.local_sub_matrix_col_range.start + 10; j++ )
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
