/*	mpi_server_thread.h
	Copyright (c) 2024, J T Frey
*/

/*!
	@header MPI distributed matrix element server
*/

#ifndef __MPI_SERVER_THREAD_H__
#define __MPI_SERVER_THREAD_H__

#include "project_config.h"
#include "int_set.h"
#include "int_pair.h"

#include "mpi.h"

/*
 * @constant mpi_server_thread_msg_tag
 *
 * MPI tag used to send/receive messages to a rank's server thread.
 */
extern const int mpi_server_thread_msg_tag;

/*
 * @constant mpi_client_msg_tag
 *
 * MPI tag used to send/receive messages to a rank's client thread.
 */
extern const int mpi_client_thread_msg_tag;

/*
 * @function mpi_get_int_pair_datatype
 *
 * Lazily registers the int_pair_t MPI datatype and returns
 * the reference to it.
 */
MPI_Datatype mpi_get_int_pair_datatype();

/*
 * @function mpi_get_msg_datatype
 *
 * Lazily registers the mpi_server_msg_t MPI datatype and
 * returns the reference to it.
 */
MPI_Datatype mpi_get_msg_datatype();

/*
 * @enum MPI distributed matrix element server, roles
 *
 * The server thread has multiple activities it can manage:
 *
 *     - work unit manager:  respond to client requests for
 *              work units to process, resolve completed
 *              work units
 *     - memory manager:  respond to client requests to
 *              write values to the rank's local sub-matrix
 *
 * The work unit role should only be handled by a single
 * rank in the runtime.
 */
enum {
    mpi_server_thread_role_work_unit_mgr = 1 << 0,
    mpi_server_thread_role_memory_mgr = 1 << 1,
    //
    mpi_server_thread_role_all = mpi_server_thread_role_work_unit_mgr | mpi_server_thread_role_memory_mgr
};

/*
 * @typedef mpi_server_thread_role_t
 *
 * The type of a MPI server role descriptor.
 */
typedef unsigned int mpi_server_thread_role_t;

/*
 * @enum MPI distributed matrix element server, message types
 *
 * The message types to which the server responds (correlating
 * with the roles it can adopt).
 */
enum {
    mpi_server_thread_msg_type_work = 1,
    mpi_server_thread_msg_type_memory = 2
};

/*
 * @enum MPI distributed matrix element server, message ids
 *
 * The actions that can be requested of the server thread.
 * Ids are in general unique per message type, with just the
 * shutdown id being implemented by all roles.
 */
enum {
    mpi_server_thread_msg_id_work_request = 0,
    mpi_server_thread_msg_id_work_allocated = 1,
    mpi_server_thread_msg_id_work_completed = 2,
    mpi_server_thread_msg_id_work_complete_and_allocate = 3,
    //
    mpi_server_thread_msg_id_memory_write = 0,
    //
    mpi_server_thread_msg_id_shutdown = 255
};

/*
 * @typedef mpi_server_thread_msg_t
 *
 * The data structure used on-the-wire for all server thread
 * messages.  Specific message ids will/will not use all of
 * the fields.
 *
 * An MPI Datatype is registered behind the scenes so that
 * the message can be easily sent/received as a single
 * transaction.
 */
typedef struct {
    int         msg_type;
    int         msg_id;
    int_pair_t  p_low, p_high;
    double      value;
} mpi_server_thread_msg_t;

/*
 * @typedef mpi_server_thread_t
 *
 */
typedef struct {
    unsigned int                flags;
    mpi_server_thread_role_t    roles;
    //
    // The sub-matrix mapping formula is such that given
    // blocks of dim_per_rank rows/cols distributed either
    // first by column (is_row_major == true) or by row:
    //
    //    blocks_major = is_row_major ? dim_blocks[0] : dim_blocks[1];
    //    blocks_minor = is_row_major ? dim_blocks[1] : dim_blocks[0];
    //    r_lo = dim_per_rank[0] * (dist_rank / blocks_minor);
    //    r_hi = r_lo + dim_per_rank[0] - 1;
    //    c_lo = dim_per_rank[1] * (dist_rank % blocks_minor);
    //    c_hi = c_lo + dim_per_rank[1] - 1;
    //
    // In the other direction, the owning rank is determined by mapping
    // indices to blocks, then blocks to rank:
    //
    //    rank = is_row_major ?
    //              ((r / dim_per_rank[0]) * dim_blocks[1] + (c / dim_per_rank[1]))
    //            : ((c / dim_per_rank[1]) * dim_blocks[0] + (r / dim_per_rank[0]));
    //
    //
    // First, the global dimension of the matrix (rows and cols):
    int                 dim_global[2];
    // The rows and cols per rank:
    int                 dim_per_rank[2];
    // The number of blocks by rows and cols:
    int                 dim_blocks[2];
    // Are blocks associated with ranks across columns
    // (is_row_major == true) or rows:
    int                 is_row_major;
    // My rank versus the number of ranks:
    int                 dist_rank, dist_size;
    // The rank that will function as root:
    int                 root_rank;
    
    // For convenience, note the local sub-matrix row and column
    // ranges:
    int_range_t         local_sub_matrix_row_range;
    int_range_t         local_sub_matrix_col_range;

    // Local sub-matrix:
    double              *local_sub_matrix;
    
    // The thread we will run in:
    pthread_t           server_thread;
    
    // For active MPI send/recv:
    bool                is_request_active;
    MPI_Request         active_request;
    
    // Assignable work (for the root rank):
    struct mpi_assignable_work *assignable_work;
} mpi_server_thread_t;

mpi_server_thread_t*
mpi_server_thread_init(
    mpi_server_thread_t *server_info,
    int root_rank,
    int global_rows, int global_cols,
    int grid_rows, int grid_cols,
    bool is_row_major,
    double *local_sub_matrix
);

bool mpi_server_thread_start(mpi_server_thread_t *server_info);
bool mpi_server_thread_cancel(mpi_server_thread_t *server_info);
bool mpi_server_thread_join(mpi_server_thread_t *server_info);

bool mpi_server_thread_index_global_to_local(mpi_server_thread_t *server_info, int_pair_t *p);

bool mpi_server_thread_index_local_to_global(mpi_server_thread_t *server_info, int_pair_t *p);

int mpi_server_thread_index_global_to_local_offset(mpi_server_thread_t *server_info, int_pair_t p);

int mpi_server_thread_index_to_rank(mpi_server_thread_t *server_info, int_pair_t p);

void mpi_server_thread_memory_write(mpi_server_thread_t *server_info, int_pair_t p, double value);

void mpi_server_thread_summary(mpi_server_thread_t *the_server, FILE *stream);

void* mpi_server_thread_thread(void *context);


typedef struct mpi_assignable_work {
    // Reference to the local server info:
    mpi_server_thread_t     *server_info;
    
    //
    // For a row-major distribution, it would be ideal to assign rows
    // lying within the first block row to the ranks holding the
    // sub-matrices for those blocks.  Likewise, for the second block row
    // only those ranks would be ideal.  In short, the work assignment
    // should be biased toward rows that will at least partially overlap
    // with the local sub-matrix.
    //
    // For column-major distribution, the difference is that the bias is
    // w.r.t. block columns and the distribution is over columns and not
    // rows.
    //
    // To that end, the assigned work unit should consist of a pair of
    // rows [lo,hi] and columns [lo,hi].  Generally-speaking, the worker
    // rank should loop over both ranges -- for a single-row work unit
    // rows [lo,hi] = [lo,lo] and there is only a single row iteration
    // involved.
    //
    // We will split the row/col indices (as determined by the row- versus
    // column major attribute of server_info) into a number of ranges
    // matching with the block-cyclic row/col count.  Initially all ranks
    // in that row/col of the grid will be allocated indices from the
    // corresponding set -- if those workers complete all the rows early
    // then they can be assigned indices from the set with the most rows
    // available.
    //
    int                 n_slots;                // e.g. server_info->dim_blocks[0]
    int_set_ref         *available_indices;     // e.g. [server_info->dim_blocks[0]]
    int_set_ref         *assigned_indices;      // e.g. [server_info->dim_blocks[0]]
    int_set_ref         *completed_indices;     // e.g. [server_info->dim_blocks[0]]
    
    // A mutex is necessary because the root rank will allocate work
    // directly versus going through the MPI protocol; we need to ensure
    // the server thread isn't allocating work to another rank while the
    // root's client thread is doing likewise:
    pthread_mutex_t     alloc_lock;
} mpi_assignable_work_t;

//

mpi_assignable_work_t* mpi_assignable_work_create(mpi_server_thread_t *server_info);

void mpi_assignable_work_destroy(mpi_assignable_work_t *work_units);

bool mpi_assignable_work_all_completed(mpi_assignable_work_t *work_units);

bool mpi_assignable_work_next_unit(mpi_assignable_work_t *work_units, int target_rank,
            int primary_slot, int_pair_t *p_low, int_pair_t *p_high);

void mpi_assignable_work_complete(mpi_assignable_work_t *work_units, int_pair_t p_low, int_pair_t p_high);

void mpi_assignable_work_summary(mpi_assignable_work_t *work_units, FILE *stream);

#endif /* __MPI_SERVER_THREAD_H__ */
