
#include "mpi_server_thread.h"
#include "mpi_utils.h"

#define GLOBAL_ROWS     1000
#define GLOBAL_COLS     1000

//    /opt/openmpi/5.0.3/bin/mpirun -np 4 --map-by :OVERSUBSCRIBE  ./mpi_dist_matrix

int
main(
    int         argc,
    char*       argv[]
)
{
    double                  d[(GLOBAL_ROWS / 2) * (GLOBAL_COLS / 2)];
    int                     thread_req, thread_prov;
    mpi_server_t            the_server;
    mpi_assignable_work_t   *work_units;
    mpi_server_msg_t        msg;
    pthread_t               server_thread;
    void                    *thread_rc;
    
    thread_req = MPI_THREAD_MULTIPLE;
    MPI_Init_thread(&argc, &argv, thread_req, &thread_prov);
    if ( thread_prov != MPI_THREAD_MULTIPLE ) {
        fprintf(stderr, "ERROR:  MPI does not support MPI_THREAD_MULTIPLE\n");
        exit(1);
    }
    MPI_Comm_rank(MPI_COMM_WORLD, &the_server.dist_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &the_server.dist_size);
    
    if ( the_server.dist_size != 4 ) {
        mpi_printf(0, "ERROR:  this program must be run with 4 ranks");
        MPI_Finalize();
        exit(1);
    }
    
    the_server.root_rank = 0;
    
    mpi_printf(0, "");
    mpi_printf(0, "Welcome to the threaded MPI matrix element work server demo!");
    mpi_printf(0, "");
    mpi_printf(0, "A %dx%d matrix is distributed across 4 ranks and matrix elements of the form", GLOBAL_ROWS, GLOBAL_COLS);
    mpi_printf(0, "");
    mpi_printf(0, "    A_{i,j} = Sqrt[i*i + j*j]");
    mpi_printf(0, "");
    mpi_printf(0, "are calculated.");
    mpi_printf(0, "");
    MPI_Barrier(MPI_COMM_WORLD);

    the_server.roles = (the_server.dist_rank == 0) ? mpi_server_role_all : mpi_server_role_memory_mgr;
    the_server.dim_global[0] = GLOBAL_ROWS; the_server.dim_global[1] = GLOBAL_COLS;
    the_server.dim_per_rank[0] = GLOBAL_ROWS / 2; the_server.dim_per_rank[1] = GLOBAL_COLS / 2;
    the_server.dim_blocks[0] = 2; the_server.dim_blocks[1] = the_server.dist_size / 2;
    the_server.is_row_major = true;
    
    int     ri = the_server.dist_rank / the_server.dim_blocks[0], ci = the_server.dist_rank % the_server.dim_blocks[0];
    the_server.local_sub_matrix_row_range = int_range_make(ri * (GLOBAL_ROWS / 2), (ri + 1) * (GLOBAL_ROWS / 2));
    the_server.local_sub_matrix_col_range = int_range_make(ci * (GLOBAL_COLS / 2), (ci + 1) * (GLOBAL_COLS / 2));
    
    the_server.local_sub_matrix = d;
    
    work_units = mpi_assignable_work_create(&the_server);
    
    the_server.assignable_work = work_units;
        
    pthread_create(&server_thread, NULL, mpi_server_thread, (void*)&the_server);
    
    // Proceed to request work...
    if ( the_server.dist_rank == the_server.root_rank ) {
        int             rank;
        
        mpi_printf(-1, "matrix element loop running with 4 s sleep between work units");
        while ( true ) {
            int_pair_t  p_low, p_high, p;
            
            if ( ! mpi_assignable_work_next_index(the_server.assignable_work, the_server.root_rank, 0, &p_low, &p_high) ) break;
            //
            // Produce matrix elements:
            //
            for ( p.i = p_low.i; p.i < p_high.i; p.i++ ) {
                for ( p.j = p_low.j; p.j < p_high.j; p.j++ ) {
                    double  I = (double)p.i, J = (double)p.j;
                    mpi_server_memory_write(&the_server, p, sqrt(I * I + J * J)); 
                }
            }
            mpi_assignable_work_complete_index(the_server.assignable_work, p_low, p_high);
            // Slow-down the root rank a little:
            //sleep(4);
        }
        mpi_printf(-1, "exited element loop, waiting for all work to complete");
        while ( ! mpi_assignable_work_all_completed(the_server.assignable_work) ) sleep (1);
        mpi_printf(-1, "sending shutdown message to other ranks' server threads");
        msg.msg_type = mpi_server_msg_type_memory;
        msg.msg_id = mpi_server_msg_id_shutdown;
        rank = 1;
        while ( rank < the_server.dist_size )
            MPI_Send(&msg, 1, mpi_get_msg_datatype(), rank++, mpi_server_msg_tag, MPI_COMM_WORLD);
        mpi_printf(-1, "canceling server thread");
        pthread_cancel(server_thread);
    } else {
        MPI_Status  status;
        int         mpi_rc;
        
        mpi_printf(-1, "matrix element loop running with 1 s sleep between work units");
        msg.msg_type = mpi_server_msg_type_work;
        msg.msg_id = mpi_server_msg_id_work_request;
        mpi_rc = MPI_Send(&msg, 1, mpi_get_msg_datatype(), the_server.root_rank, mpi_server_msg_tag, MPI_COMM_WORLD);
        if ( mpi_rc == MPI_SUCCESS ) {
            while ( true ) {
                int_pair_t      p;
                
                mpi_rc = MPI_Recv(&msg, 1, mpi_get_msg_datatype(), the_server.root_rank, mpi_client_msg_tag, MPI_COMM_WORLD, &status);
                if ( mpi_rc != MPI_SUCCESS ) {
                    mpi_printf(-1, "MPI_Recv error %d", mpi_rc);
                }
                if ( msg.p_low.i == -1 ) break;
                //
                // Produce matrix elements:
                //
                for ( p.i = msg.p_low.i; p.i < msg.p_high.i; p.i++ ) {
                    for ( p.j = msg.p_low.j; p.j < msg.p_high.j; p.j++ ) {
                        double  I = (double)p.i, J = (double)p.j;
                        mpi_server_memory_write(&the_server, p, sqrt(I * I + J * J)); 
                    }
                }
                //sleep(1);
                msg.msg_type = mpi_server_msg_type_work;
                msg.msg_id = mpi_server_msg_id_work_complete_and_allocate;
                mpi_rc = MPI_Send(&msg, 1, mpi_get_msg_datatype(), the_server.root_rank, mpi_server_msg_tag, MPI_COMM_WORLD);
            }
            mpi_printf(-1, "exited element loop");
        }
    }
    pthread_join(server_thread, &thread_rc);
    MPI_Barrier(MPI_COMM_WORLD);
    
    if ( the_server.dist_rank == 0 ) {
        int     i, j;
        
        mpi_printf(-1, "Sub-matrices in sequence by rank:\n\nRank 0:\n");
        for ( i = 0; i < 10; i++ ) {
            printf("    %8.3lf", d[mpi_server_index_global_to_local_offset(&the_server, int_pair_make(i, 0))]);
            for ( j = 1; j < 10; j++ )
                printf(", %8.3lf", d[mpi_server_index_global_to_local_offset(&the_server, int_pair_make(i, j))]);
            printf("\n");
        }
        MPI_Send(&i, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
    } else {
        int     i, j;
        
        MPI_Recv(&i, 1, MPI_INT, the_server.dist_rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("\nRank %d:\n", the_server.dist_rank);
        for ( i = the_server.local_sub_matrix_row_range.start; i < the_server.local_sub_matrix_row_range.start + 10; i++ ) {
            printf("    %8.3lf", d[mpi_server_index_global_to_local_offset(&the_server, int_pair_make(i, the_server.local_sub_matrix_col_range.start))]);
            for ( j = the_server.local_sub_matrix_col_range.start + 1; j < the_server.local_sub_matrix_col_range.start + 10; j++ )
                printf(", %8.3lf", d[mpi_server_index_global_to_local_offset(&the_server, int_pair_make(i, j))]);
            printf("\n");
        }
        if ( the_server.dist_rank + 1 < the_server.dist_size )
            MPI_Send(&i, 1, MPI_INT, the_server.dist_rank + 1, 0, MPI_COMM_WORLD);
    }

    mpi_printf(-1, "ready to exit");
    
    MPI_Finalize();
    return 0;
}
