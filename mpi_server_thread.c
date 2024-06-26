
#include "mpi_server_thread.h"
#include "mpi_utils.h"

//

const int mpi_server_thread_msg_tag = 2;
const int mpi_client_thread_msg_tag = 3;

//

static int __mpi_server_thread_int_pair_type_fields = 2;
static int __mpi_server_thread_int_pair_type_counts[] = {
                    1, // 1 base int
                    1, // 1 base int
                };
static MPI_Aint __mpi_server_thread_int_pair_type_offsets[] = {
                    offsetof(int_pair_t, i),
                    offsetof(int_pair_t, j),
                };
static MPI_Datatype __mpi_server_thread_int_pair_type_types[] = {
                    MPI_BASE_INT_T,
                    MPI_BASE_INT_T
                };

MPI_Datatype
mpi_get_int_pair_datatype()
{
    static bool is_inited = false;
    static MPI_Datatype dtype;
    
    if ( ! is_inited ) {
        MPI_Type_create_struct(
                __mpi_server_thread_int_pair_type_fields,
                __mpi_server_thread_int_pair_type_counts,
                __mpi_server_thread_int_pair_type_offsets,
                __mpi_server_thread_int_pair_type_types,
                &dtype);
        MPI_Type_commit(&dtype);
        is_inited = true;
    }
    return dtype;
}

//

static int __mpi_server_thread_msg_type_fields = 5;
static int __mpi_server_thread_msg_type_counts[] = {
                    1, // 1 int
                    1, // 1 int
                    1, // 1 int_pair
                    1, // 1 int_pair
                    1, // 1 double
                };
static MPI_Aint __mpi_server_thread_msg_type_offsets[] = {
                    offsetof(mpi_server_thread_msg_t, msg_type),
                    offsetof(mpi_server_thread_msg_t, msg_id),
                    offsetof(mpi_server_thread_msg_t, p_low),
                    offsetof(mpi_server_thread_msg_t, p_high),
                    offsetof(mpi_server_thread_msg_t, value)
                };
static MPI_Datatype __mpi_server_thread_msg_type_types[] = {
                    MPI_INT,
                    MPI_INT,
                    0,          // must be filled-in later
                    0,          // must be filled-in later
                    MPI_DOUBLE
                };

MPI_Datatype
mpi_get_msg_datatype()
{
    static bool is_inited = false;
    static MPI_Datatype dtype;
    
    if ( ! is_inited ) {
        __mpi_server_thread_msg_type_types[2] = __mpi_server_thread_msg_type_types[3] = mpi_get_int_pair_datatype();
        MPI_Type_create_struct(
                __mpi_server_thread_msg_type_fields,
                __mpi_server_thread_msg_type_counts,
                __mpi_server_thread_msg_type_offsets,
                __mpi_server_thread_msg_type_types,
                &dtype);
        MPI_Type_commit(&dtype);
        is_inited = true;
    }
    return dtype;
}

//

void 
__mpi_server_thread_cleanup(
    void    *context
)
{
    mpi_server_thread_t *SERVER = (mpi_server_thread_t*)context;
    
    pthread_mutex_lock(&SERVER->request_lock);
    if ( SERVER->is_request_active ) {
        MPI_Cancel(&SERVER->active_request);
        SERVER->is_request_active = false;
    }
    pthread_mutex_unlock(&SERVER->request_lock);
}

void*
__mpi_server_thread_start(
    void    *context
)
{
    mpi_server_thread_t *SERVER = (mpi_server_thread_t*)context;
    bool                is_running = true;

    // We want to be cancellable at any time so that the root client can terminate
    // its server thread w/o MPI messaging:
    pthread_setcancelstate(PTHREAD_CANCEL_ENABLE, NULL);
    pthread_setcanceltype(PTHREAD_CANCEL_ASYNCHRONOUS, NULL);
    
    pthread_cleanup_push(__mpi_server_thread_cleanup, context);
    
    switch ( SERVER->roles ) {
        case mpi_server_thread_role_work_unit_mgr:
            mpi_printf(-1, "server thread running a work unit manager");
            break;
        case mpi_server_thread_role_memory_mgr:
            mpi_printf(-1, "server thread running a memory manager");
            break;
        case mpi_server_thread_role_all:
            mpi_printf(-1, "server thread running work unit and memory managers");
            break;
    }
    
    while ( is_running ) {
        MPI_Status              status;
        mpi_server_thread_msg_t msg, response;
        
        pthread_mutex_lock(&SERVER->request_lock);
        MPI_Irecv(&msg, 1, mpi_get_msg_datatype(), MPI_ANY_SOURCE, mpi_server_thread_msg_tag, MPI_COMM_WORLD, &SERVER->active_request);
        SERVER->is_request_active = true;
        pthread_mutex_unlock(&SERVER->request_lock);
        MPI_Wait(&SERVER->active_request, &status);
        pthread_mutex_lock(&SERVER->request_lock);
        SERVER->is_request_active = false;
        pthread_mutex_unlock(&SERVER->request_lock);
        
        switch ( msg.msg_type ) {
            case mpi_server_thread_msg_type_work: {
                switch ( msg.msg_id ) {
                    case mpi_server_thread_msg_id_shutdown:
                        is_running = false;
                        break;
                    case mpi_server_thread_msg_id_work_complete_and_allocate:
                        mpi_assignable_work_complete(SERVER->assignable_work, msg.p_low, msg.p_high);
                    case mpi_server_thread_msg_id_work_request: {
                        // The sender rank determines the primary work set we want to consult:
                        int         sender_rank = status.MPI_SOURCE;
                        int         primary_slot = (SERVER->is_row_major) ?
                                                    (sender_rank / SERVER->dim_blocks[1])
                                                  : (sender_rank / SERVER->dim_blocks[0]);
                        base_int_t  next_index;
                        
                        //  By default, no more work available, period:
                        response.msg_type = mpi_server_thread_msg_type_work;
                        response.msg_id = mpi_server_thread_msg_id_work_allocated;
                        response.p_low = response.p_high = int_pair_make(-1, -1);
                        
                        mpi_assignable_work_next_unit(SERVER->assignable_work, sender_rank, primary_slot, &response.p_low, &response.p_high);
                        MPI_Send(&response, 1, mpi_get_msg_datatype(), sender_rank, mpi_client_thread_msg_tag, MPI_COMM_WORLD);
                        break;
                    }
                    case mpi_server_thread_msg_id_work_completed: {
                        mpi_assignable_work_complete(SERVER->assignable_work, msg.p_low, msg.p_high);
                        break;
                    }
                }
                break;
            }
            case mpi_server_thread_msg_type_memory: {
                switch ( msg.msg_id ) {
                    case mpi_server_thread_msg_id_shutdown:
                        is_running = false;
                        break;
                    case mpi_server_thread_msg_id_memory_write: {
                        mpi_server_thread_memory_write(SERVER, msg.p_low, msg.value);
                        break;
                    }
                }
                break;
            }
        }
    }
    mpi_printf(-1, "exiting server thread");
    pthread_cleanup_pop(0);
    return NULL;
}

//

enum {
    mpi_server_thread_flag_was_allocated = 1 << 0,
    mpi_server_thread_flag_owns_local_sub_matrix = 1 << 1,
    mpi_server_thread_flag_is_thread_started = 1 << 2
};

mpi_server_thread_t*
mpi_server_thread_init(
    mpi_server_thread_t *server_info,
    int                 root_rank,
    base_int_t          global_rows,
    base_int_t          global_cols,
    base_int_t          grid_rows,
    base_int_t          grid_cols,
    bool                is_row_major,
    double              *local_sub_matrix
)
{
    base_int_t          r, c;
    
    // Force the MPI datatypes to get initialized now to avoid later
    // race conditions:
    mpi_get_msg_datatype();
    
    // If server_info is NULL, allocate a new one:
    if ( ! server_info ) {
        server_info = (mpi_server_thread_t*)malloc(sizeof(mpi_server_thread_t));
        if ( ! server_info ) return NULL;
        server_info->flags = mpi_server_thread_flag_was_allocated;
    } else {
        server_info->flags = 0;
    }
    
    server_info->is_request_active = false;
    pthread_mutex_init(&server_info->request_lock, NULL);
    
    // Initialize MPI comm dimensions:
    MPI_Comm_rank(MPI_COMM_WORLD, &server_info->dist_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &server_info->dist_size);
    
    // Which rank is root?
    server_info->root_rank = root_rank;
    
    // Fill-in dimensions:
    server_info->dim_global[0] = global_rows;
    server_info->dim_global[1] = global_cols;
    
    // Initial grid dimensions, may be overridden by auto-grid:
    server_info->dim_blocks[0] = grid_rows;
    server_info->dim_blocks[1] = grid_cols;
    
    // Auto grid?
    if ( ! grid_rows || ! grid_cols ) {
        if ( mpi_auto_grid_2d(server_info->dist_size, true, true, server_info->dim_global, server_info->dim_blocks) ) {
            mpi_printf(0, "auto-grid block partitioning yielded " BASE_INT_FMT " x " BASE_INT_FMT,
                    server_info->dim_blocks[0], server_info->dim_blocks[1]);
        } else {
            mpi_printf(0, "auto-grid unable to find an exact fit for %d ranks and global dims " BASE_INT_FMT " x " BASE_INT_FMT,
                    server_info->dist_size, server_info->dim_global[0], server_info->dim_global[1]);
            if ( server_info->flags & mpi_server_thread_flag_was_allocated ) free((void*)server_info);
            return NULL;
        }
    }
    
    mpi_printf(0, "block grid dimensions [" BASE_INT_FMT "," BASE_INT_FMT "]", server_info->dim_blocks[0], server_info->dim_blocks[1]);
    
    server_info->dim_per_rank[0] = global_rows / server_info->dim_blocks[0];
    server_info->dim_per_rank[1] = global_cols / server_info->dim_blocks[1];
    
    mpi_printf(0, "base sub-matrix dimensions [" BASE_INT_FMT "," BASE_INT_FMT "]", server_info->dim_per_rank[0], server_info->dim_per_rank[1]);
    
    server_info->is_row_major = is_row_major;
    
    // Assign global row/col index ranges associated with this rank:
    if ( is_row_major ) {
        r = server_info->dist_rank / server_info->dim_blocks[1];
        c = server_info->dist_rank % server_info->dim_blocks[1];
        server_info->local_sub_matrix_row_range = int_range_make(r * server_info->dim_per_rank[0], server_info->dim_per_rank[0]);
        server_info->local_sub_matrix_col_range = int_range_make(c * server_info->dim_per_rank[1], server_info->dim_per_rank[1]);
    } else {
        r = server_info->dist_rank / server_info->dim_blocks[0];
        c = server_info->dist_rank % server_info->dim_blocks[0];
        server_info->local_sub_matrix_row_range = int_range_make(r * server_info->dim_per_rank[0], server_info->dim_per_rank[0]);
        server_info->local_sub_matrix_col_range = int_range_make(c * server_info->dim_per_rank[1], server_info->dim_per_rank[1]);
    }
    
    mpi_printf(-1, "local sub-matrix indices [" BASE_INT_FMT "," BASE_INT_FMT "]..[" BASE_INT_FMT "," BASE_INT_FMT "]",
            server_info->local_sub_matrix_row_range.start, server_info->local_sub_matrix_col_range.start,
            int_range_get_end(server_info->local_sub_matrix_row_range), int_range_get_end(server_info->local_sub_matrix_col_range));
    
    // Setup the local sub-matrix storage:
    if ( ! local_sub_matrix ) {
        local_sub_matrix = (double*)malloc(sizeof(double) * server_info->dim_per_rank[0] * server_info->dim_per_rank[1]);
        if ( ! local_sub_matrix ) {
            if ( server_info->flags & mpi_server_thread_flag_was_allocated ) free((void*)server_info);
            return NULL;
        }
        mpi_printf(0, "local sub-matrix allocated");
        server_info->flags |= mpi_server_thread_flag_owns_local_sub_matrix;
    }
    server_info->local_sub_matrix = local_sub_matrix;

    // Setup the role(s) for this instance:
    if ( server_info->dist_rank == server_info->root_rank ) {
        server_info->roles = mpi_server_thread_role_all;
        server_info->assignable_work = mpi_assignable_work_create(server_info);
    } else {        
        server_info->roles = mpi_server_thread_role_memory_mgr;
        server_info->assignable_work = NULL;
    }
    
    return server_info;
}

//

void
mpi_server_thread_destroy(
    mpi_server_thread_t *server_info
)
{
    mpi_server_thread_cancel(server_info);
    
    // We own the sub-matrix, deallocate it:
    if ( server_info->local_sub_matrix && (server_info->flags & mpi_server_thread_flag_owns_local_sub_matrix) )
        free((void*)server_info->local_sub_matrix);
    
    // This was not an external instance passed-in, it was dynamically-allocated:
    if ( server_info->flags & mpi_server_thread_flag_was_allocated )
        free((void*)server_info);
}

//

bool
mpi_server_thread_start(
    mpi_server_thread_t *server_info
)
{
    if ( ! (server_info->flags & mpi_server_thread_flag_is_thread_started) ) {
        int         rc = pthread_create(
                                &server_info->server_thread,
                                NULL,
                                __mpi_server_thread_start,
                                (void*)server_info
                            );
        if ( rc != 0 ) return false;
        
        server_info->flags |= mpi_server_thread_flag_is_thread_started;
    }
    return true;
}

//

bool
mpi_server_thread_cancel(
    mpi_server_thread_t *server_info
)
{
    if ( server_info->flags & mpi_server_thread_flag_is_thread_started ) {
        int     rc = pthread_cancel(server_info->server_thread);
        
        if ( rc == 0 )
            return mpi_server_thread_join(server_info);
    }
    return false;
}

//

bool
mpi_server_thread_join(
    mpi_server_thread_t *server_info
)
{
    if ( server_info->flags & mpi_server_thread_flag_is_thread_started ) {
        int     rc = pthread_join(server_info->server_thread, NULL);
        
        if ( rc == 0 ) {
            server_info->flags &= ~mpi_server_thread_flag_is_thread_started;
            return true;
        }
    }
    return false;
}

//

bool
mpi_server_thread_index_global_to_local(
    mpi_server_thread_t *server_info,
    int_pair_t          *p
)
{
    if ( int_range_does_contain(server_info->local_sub_matrix_row_range, p->i) &&
         int_range_does_contain(server_info->local_sub_matrix_col_range, p->j) )
    {
        p->i -= server_info->local_sub_matrix_row_range.start;
        p->j -= server_info->local_sub_matrix_col_range.start;
        return true;
    }
    return false;
}

//

bool
mpi_server_thread_index_local_to_global(
    mpi_server_thread_t *server_info,
    int_pair_t          *p
)
{
    if ( ((p->i >= 0) && (p->i < server_info->local_sub_matrix_row_range.length)) &&
         ((p->j >= 0) && (p->j < server_info->local_sub_matrix_col_range.length)) )
    {
        p->i += server_info->local_sub_matrix_row_range.start;
        p->j += server_info->local_sub_matrix_col_range.start;
        return true;
    }
    return false;
}

//

base_int_t
mpi_server_thread_index_global_to_local_offset(
    mpi_server_thread_t *server_info,
    int_pair_t          p
)
{
    if ( mpi_server_thread_index_global_to_local(server_info, &p) ) {
        if ( server_info->is_row_major ) return int_pair_get_i_major_offset(p, server_info->dim_per_rank[1]);
        return int_pair_get_j_major_offset(p, server_info->dim_per_rank[0]);
    }
    return -1;
}

//

int
mpi_server_thread_index_to_rank(
    mpi_server_thread_t *server_info,
    int_pair_t          p
)
{
    if ( server_info->is_row_major ) {
        return (p.i / server_info->dim_per_rank[0]) * server_info->dim_blocks[1] +
               (p.j / server_info->dim_per_rank[1]);
    } else {
        return (p.j / server_info->dim_per_rank[1]) * server_info->dim_blocks[0] +
               (p.i / server_info->dim_per_rank[0]);
    }
}

//

void
mpi_server_thread_memory_write(
    mpi_server_thread_t *server_info,
    int_pair_t          p,
    double              value
)
{
    base_int_t          local_offset = mpi_server_thread_index_global_to_local_offset(server_info, p);
    
    if ( local_offset >= 0 ) {
        server_info->local_sub_matrix[local_offset] = value;
    } else {
        // Send to the rank that handles this sub-matrix:
        mpi_server_thread_msg_t    msg = {
                                        .msg_type = mpi_server_thread_msg_type_memory,
                                        .msg_id = mpi_server_thread_msg_id_memory_write,
                                        .p_low = p,
                                        .p_high = p,
                                        .value = value
                                    };
        MPI_Send(
            &msg, 1, mpi_get_msg_datatype(),
            mpi_server_thread_index_to_rank(server_info, p),
            mpi_server_thread_msg_tag,
            MPI_COMM_WORLD);
    }
}

//

void
mpi_server_thread_summary(
    mpi_server_thread_t *server_info,
    FILE                *stream
)
{
    fprintf(stream, "mpi_server@%p (roles=%X, dim_global=(" BASE_INT_FMT "," BASE_INT_FMT "), dim_per_rank=(" BASE_INT_FMT "," BASE_INT_FMT "),\n"
                    "               dim_blocks=(" BASE_INT_FMT "," BASE_INT_FMT "), is_row_major=%s, dist_rank=%d,\n"
                    "               dist_size=%d, local_sub_matrix_row_range=[" BASE_INT_FMT "," BASE_INT_FMT "],\n"
                    "               local_sub_matrix_col_range=[" BASE_INT_FMT "," BASE_INT_FMT "]) {\n",
                    server_info, server_info->roles, server_info->dim_global[0], server_info->dim_global[1],
                    server_info->dim_per_rank[0], server_info->dim_per_rank[1], server_info->dim_blocks[0],
                    server_info->dim_blocks[1], server_info->is_row_major ? "true" : "false",
                    server_info->dist_rank, server_info->dist_size, server_info->local_sub_matrix_row_range.start,
                    int_range_get_end(server_info->local_sub_matrix_row_range),
                    server_info->local_sub_matrix_col_range.start,
                    int_range_get_end(server_info->local_sub_matrix_col_range)
                );
    if ( server_info->roles & mpi_server_thread_role_work_unit_mgr ) {
        fprintf(stream, "    assignable_work: ");
        mpi_assignable_work_summary(server_info->assignable_work, stream);
    }
    fprintf(stream, "}\n");
}

//
////
//

mpi_assignable_work_t*
mpi_assignable_work_create(
    mpi_server_thread_t     *server_info
)
{
    mpi_assignable_work_t   *new_work;
    void                    *new_ptr;
    size_t                  work_rec_size = sizeof(mpi_assignable_work_t);
    
    // Space for the three lists of int_set_ref's for the block rows/cols:
    work_rec_size += 3 * sizeof(int_set_ref) * ((server_info->is_row_major) ? server_info->dim_blocks[0] : server_info->dim_blocks[1]);
    
    new_ptr = malloc(work_rec_size);
    if ( new_ptr ) {
        memset(new_ptr, 0, work_rec_size);
        new_work = (mpi_assignable_work_t*)new_ptr;
        new_work->server_info = server_info;
        new_work->n_slots = (server_info->is_row_major) ? server_info->dim_blocks[0] : server_info->dim_blocks[1];
        new_work->available_indices = (int_set_ref*)(new_ptr + sizeof(mpi_assignable_work_t));
        new_work->assigned_indices = new_work->available_indices + new_work->n_slots;
        new_work->completed_indices = new_work->assigned_indices + new_work->n_slots;
        pthread_mutex_init(&new_work->alloc_lock, NULL);
        
        if ( server_info->is_row_major ) {
            int         i = 0;
            base_int_t  r = 0;
            
            while ( i < new_work->n_slots ) {
                new_work->available_indices[i] = int_set_create();
                new_work->assigned_indices[i] = int_set_create();
                new_work->completed_indices[i] = int_set_create();
                
                // Push the index set for this block to the available list:
                int_set_push_range(new_work->available_indices[i++], int_range_make(r, server_info->dim_per_rank[0]));
                
                // Next chunk:
                r += server_info->dim_per_rank[0];
            }
        } else {
            int         i = 0;
            base_int_t  c = 0;
            
            while ( i < new_work->n_slots ) {
                new_work->available_indices[i] = int_set_create();
                new_work->assigned_indices[i] = int_set_create();
                new_work->completed_indices[i] = int_set_create();
                
                // Push the index set for this block to the available list:
                int_set_push_range(new_work->available_indices[i++], int_range_make(c, server_info->dim_per_rank[1]));
                
                // Next chunk:
                c += server_info->dim_per_rank[1];
            }
        }
    }
    return new_work;
}

//

void
mpi_assignable_work_destroy(
    mpi_assignable_work_t   *work_units
)
{
    int                     i = 0;
    
    while ( i < work_units->n_slots ) {
        int_set_destroy(work_units->available_indices[i]);
        int_set_destroy(work_units->assigned_indices[i]);
        int_set_destroy(work_units->completed_indices[i++]);
    }
    free((void*)work_units);
}

//

bool
mpi_assignable_work_all_completed(
    mpi_assignable_work_t   *work_units
)
{
    int                     i = 0;
    
    pthread_mutex_lock(&work_units->alloc_lock);
    while ( i < work_units->n_slots ) {
        if ( int_set_get_length(work_units->available_indices[i]) > 0 ) break;
        if ( int_set_get_length(work_units->assigned_indices[i]) > 0 ) break;
        if ( int_set_get_length(work_units->completed_indices[i]) < ((work_units->server_info->is_row_major) ? work_units->server_info->dim_per_rank[0] : work_units->server_info->dim_per_rank[0]) ) break;
        i++;
    }
    pthread_mutex_unlock(&work_units->alloc_lock);
    return (i == work_units->n_slots);
}

//

bool
mpi_assignable_work_next_unit(
    mpi_assignable_work_t   *work_units,
    int                     target_rank,
    int                     primary_slot,
    int_pair_t              *p_low,
    int_pair_t              *p_high
)
{
    base_int_t              next_index;
    bool                    rc = false;
    
    pthread_mutex_lock(&work_units->alloc_lock);

    // Try to get a row from the preferred slot:
    if ( int_set_pop_next_int(work_units->available_indices[primary_slot], &next_index) ) {
        //mpi_printf(-1, "allocated index " BASE_INT_FMT " from primary slot %d for rank %d", next_index, primary_slot, target_rank);
        int_set_push_int(work_units->assigned_indices[primary_slot], next_index);
        if ( work_units->server_info->is_row_major ) {
            p_low->i = next_index; p_high->i = next_index + 1;
            p_low->j = 0; p_high->j = work_units->server_info->dim_global[1];
        } else {
            p_low->j = next_index; p_high->j = next_index + 1;
            p_low->i = 0; p_high->i = work_units->server_info->dim_global[0];
        }        
        rc = true;
    } else {
        // Preferred slot was empty, take a work unit from the slot with the
        // most work remaining:
        int         slot_idx = 0, slot_idx_max = -1;
        base_int_t  avail_max = 0;
        
        while ( slot_idx < work_units->n_slots ) {
            if ( slot_idx != primary_slot ) {
                int l = int_set_get_length(work_units->available_indices[slot_idx]);
                
                if ( l > avail_max ) {
                    slot_idx_max = slot_idx;
                    avail_max = l;
                }
            }
            slot_idx++;
        }
        if ( slot_idx_max >= 0 ) {
            if ( int_set_pop_next_int(work_units->available_indices[slot_idx_max], &next_index) ) {
                //mpi_printf(-1, "allocated index " BASE_INT_FMT " from alternate slot %d for rank %d", next_index, slot_idx_max, target_rank);
                int_set_push_int(work_units->assigned_indices[slot_idx_max], next_index);
                if ( work_units->server_info->is_row_major ) {
                    p_low->i = next_index; p_high->i = next_index + 1;
                    p_low->j = 0; p_high->j = work_units->server_info->dim_global[1];
                } else {
                    p_low->j = next_index; p_high->j = next_index + 1;
                    p_low->i = 0; p_high->i = work_units->server_info->dim_global[0];
                }        
                rc = true;
            }
        }
    }
    pthread_mutex_unlock(&work_units->alloc_lock);
    return rc;
}

//

void
mpi_assignable_work_complete(
    mpi_assignable_work_t   *work_units,
    int_pair_t              p_low,
    int_pair_t              p_high
)
{
    pthread_mutex_lock(&work_units->alloc_lock);
    
    if ( work_units->server_info->is_row_major ) {
        base_int_t          i = p_low.i;
        
        while ( i < p_high.i ) {
            int     slot = i / work_units->server_info->dim_per_rank[0];
            
            int_set_remove_int(work_units->assigned_indices[slot], i);
            int_set_push_int(work_units->completed_indices[slot], i++);
        }
    } else {
        base_int_t          i = p_low.j;
        
        while ( i < p_high.j ) {
            int     slot = i / work_units->server_info->dim_per_rank[1];
            
            int_set_remove_int(work_units->assigned_indices[slot], i);
            int_set_push_int(work_units->completed_indices[slot], i++);
        }
    }
    
    pthread_mutex_unlock(&work_units->alloc_lock);
}

//

void
mpi_assignable_work_summary(
    mpi_assignable_work_t   *work_units,
    FILE                    *stream
)
{
    int                     i = 0;
    
    fprintf(stream, "mpi_assignable_work@%p (n_slots=%d) {\n", work_units, work_units->n_slots);
    while ( i < work_units->n_slots ) {
        fprintf(stream, "%d: available -> ", i);
        int_set_summary(work_units->available_indices[i], stream);
        fprintf(stream, "       assigned -> ");
        int_set_summary(work_units->assigned_indices[i], stream);
        fprintf(stream, "       completed -> ");
        int_set_summary(work_units->completed_indices[i], stream);
        i++;
    }
    fprintf(stream, "}\n");
}
