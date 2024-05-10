/*	int_pair.h
	Copyright (c) 2024, J T Frey
*/

/*!
	@header 32-bit integer pair
	
	Representation and functionality for a pair of 32-bit
	integers, e.g. acting as a 2D index into a matrix.
*/

#ifndef __INT_PAIR_H__
#define __INT_PAIR_H__

#include "project_config.h"

/*
 * @typedef int_pair_t
 *
 * An integer pair, a'la a 2D index pair (hint hint).
 */
typedef struct {
    base_int_t  i, j;
} int_pair_t;

/*
 * @function int_pair_make
 *
 * Initialize and return a new int_pair_t with the given
 * i, j.
 */
static inline int_pair_t
int_pair_make(
    base_int_t  i,
    base_int_t  j
)
{
    int_pair_t p = { .i = i, .j = j };
    return p;
}

/*
 * @function int_pair_is_equal
 *
 * Returns true if p1 and p2 have the same values.
 */
static inline bool
int_pair_is_equal(
    int_pair_t  p1,
    int_pair_t  p2
)
{
    return ((p1.i == p2.i) && (p1.j == p2.j));
}

/*
 * @function int_pair_make_swapped
 *
 * Initialize and return an int_pair_t with the i,j
 * from p swapped (i1 = j0, j1 = i0).
 */
static inline int_pair_t
int_pair_make_swapped(
    int_pair_t  p
)
{
    int_pair_t  out_p = { .i = p.j, .j = p.i };
    return out_p;
}

/*
 * @function int_pair_get_i_major_offset
 *
 * Given the leading_dimension (e.g. of a 2D array), calculate
 * the linear offset associated with index pair p.  The i index
 * is the major index.
 */
static inline base_int_t
int_pair_get_i_major_offset(
    int_pair_t  p,
    base_int_t  leading_dimension
)
{
    return p.i * leading_dimension + p.j;
}

/*
 * @function int_pair_make_with_i_major_offset
 *
 * Initialize and return an integer pair corresponding with the
 * given linear offset and leading_dimension.  The i index is the
 * major index (e.g. row-major).
 */
static inline int_pair_t
int_pair_make_with_i_major_offset(
    base_int_t  offset,
    base_int_t  leading_dimension
)
{
    int_pair_t  p = {
                    .i = offset / leading_dimension,
                    .j = offset % leading_dimension
                };
    return p;
}

/*
 * @function int_pair_get_j_major_offset
 *
 * Given the leading_dimension (e.g. of a 2D array), calculate
 * the linear offset associated with index pair p.  The j index
 * is the major index.
 */
static inline base_int_t
int_pair_get_j_major_offset(
    int_pair_t  p,
    base_int_t  leading_dimension
)
{
    return p.j * leading_dimension + p.i;
}

/*
 * @function int_pair_make_with_j_major_offset
 *
 * Initialize and return an integer pair corresponding with the
 * given linear offset and leading_dimension.  The ij index is the
 * major index (e.g. column-major).
 */
static inline int_pair_t
int_pair_make_with_j_major_offset(
    base_int_t  offset,
    base_int_t  leading_dimension
)
{
    int_pair_t  p = {
                    .i = offset % leading_dimension,
                    .j = offset / leading_dimension
                };
    return p;
}

#endif /* __INT_PAIR_H__ */
