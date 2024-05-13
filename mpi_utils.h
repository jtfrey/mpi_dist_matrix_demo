/*	mpi_utils.h
	Copyright (c) 2024, J T Frey
*/

/*!
	@header MPI utility functions
*/

#ifndef __MPI_UTILS_H__
#define __MPI_UTILS_H__

#include "project_config.h"
#include "mpi.h"

/*
 * @function mpi_printf
 *
 * If the calling MPI process matches the given rank,
 * the fmt string and any additional arguments are
 * passed to printf().
 *
 * If the rank argument is -1, all MPI ranks will
 * call printf().
 *
 * Returns the number of characters written, just like
 * printf() would.
 */
int mpi_printf(int rank, const char *fmt, ...);

/*
 * @function mpi_auto_grid_2d
 *
 * On entry dim_global array should contain the global matrix
 * size as row count then column count.  The function attempts to
 * partition the global size into a per-rank row x column size
 * by which the global matrix could be decomposed.  The resulting
 * row count and column count are set in the out_dim_blocks array.
 *
 * The dim_global and out_dim_blocks arrays must be at least two
 * elements in size; the integer at index 0 is the row count and
 * the integer at index 1 is column count.
 *
 * The ranks argument is the number of MPI ranks among which the
 * global matrix will be distributed.
 *
 * An must_be_exact argument of true indicates that the proposed
 * row and column count must evenly divide the global matrix.  If
 * no such mapping exists, the function will return false.
 *
 * The mapping is determined by first checking if the global matrix
 * row and column counts are equal (square matrix) and ranks is an
 * integral square.  In that case, the square root of ranks is the
 * obvious choice.
 *
 * Otherwise, the goal is to choose a row and column count that:
 *
 * - are in a ratio as close to the dimensional ratio of the
 *   global matrix (r/c vs. global_r/global_c)
 * - minimize the difference between the two integers
 *   (sub-matrices are closer to being square)
 *
 * The row,col starts as ranks,1.  On each iteration, primes from
 * 2 through 23 are tested for evenly dividing the row count; the
 * lowest factor that does so multiplies the col count.  If the
 * resulting row,col (and the ratio row/col) is lower than the
 * optimal result found thus far, the optimal result is updated
 * to the current row,col counts:
 *
 *     ranks = 24
 *     dim_global = [360, 240], r/c = 1.5
 *     iter 1:  r=24, c=1,  r/c=24,       |r-c|=23,  ∆r/c = 22.5
 *     iter 2:  r=12, c=2,  r/c=6,        |r-c|=10,  ∆r/c = 4.5
 *              r=2,  c=12, r/c=0.16667,  |r-c|=10,  ∆r/c = 1.3333
 *     iter 3:  r=6,  c=4,  r/c=1.5,      |r-c|=2,   ∆r/c = 0
 *              r=4,  c=6,  r/c=0.66667,  |r-c|=2,   ∆r/c = 0.8333
 *     iter 4:  r=3,  c=8,  r/c=0.375,    |r-c|=5,   ∆r/c = 1.125
 *              r=8,  c=3,  r/c=2.66667,  |r-c|=5,   ∆r/c = 1.1667
 *     iter 5:  r=1,  c=24, r/c=0.04167,  |r-c|=23,  ∆r/c = 1.4583
 *
 * Iteration 3a produces an exact distribution (∆r/c = 0) and also
 * a minimum to the distance between r anc c -- hooray!
 *
 * An is_verbose of true will have rank 0 print the steps of the
 * minimizing search.
 *
 * The function returns true if out_dim_blocks was assigned
 * row and column count values, false otherwise.
 */
bool mpi_auto_grid_2d(int ranks, bool must_be_exact, bool is_verbose, base_int_t *dim_global, base_int_t *out_dim_blocks);

#endif /* __MPI_UTILS_H__ */
