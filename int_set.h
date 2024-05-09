/*	int_set.h
	Copyright (c) 2024, J T Frey
*/

/*!
	@header 32-bit integer set
	
	Representation and functionality for a set of 32-bit
	integer values.  Members of the set are represented as
	integer ranges, so the implementation is optimal for sets
	that minimize gaps (which is exactly what we need for the
	matrix element server).
*/

#ifndef __INT_SET_H__
#define __INT_SET_H__

#include "project_config.h"
#include "int_range.h"

/*
 * @typedef int_set_ref
 *
 * Opaque reference to a set of 32-bit integers.
 */
typedef struct int_set * int_set_ref;

/*
 * @function int_set_create
 *
 * Create a new (empty) integer set.
 */
int_set_ref int_set_create();

/*
 * @function int_set_destroy
 *
 * Deallocate the integer set S.
 */
void int_set_destroy(int_set_ref S);

/*
 * @function int_set_get_length
 *
 * Returns the number of integers in the set S.
 */
int int_set_get_length(int_set_ref S);

/*
 * @function int_set_push_int
 *
 * Add integer i to the set S.  Returns true if
 * the set already contained i or it was successfully
 * added, false otherwise (e.g. memory allocation
 * failure).
 */
bool int_set_push_int(int_set_ref S, int i);

/*
 * @function int_set_push_range
 *
 * Add all integers in range r to the set S.  Returns
 * true if successful, false otherwise (e.g. memory allocation
 * failure).
 */
bool int_set_push_range(int_set_ref S, int_range_t r);

/*
 * @function int_set_remove_int
 *
 * Remove integer i from the set S.  Returns true if
 * successful.
 */
bool int_set_remove_int(int_set_ref S, int i);

/*
 * @function int_set_remove_range
 *
 * Remove all integers in range r from the set S.  Returns
 * true if successful.
 */
bool int_set_remove_range(int_set_ref S, int_range_t r);

/*
 * @function int_set_peek_next_int
 *
 * Set *i to the lowest integer value currently in the set
 * without removing it from set S.  Returns true if a value
 * was present and *i was set, false if the set is empty.
 */
bool int_set_peek_next_int(int_set_ref S, int *i);

/*
 * @function int_set_pop_next_int
 *
 * Set *i to the lowest integer value currently in the set
 * and remove it from set S.  Returns true if a value
 * was present and *i was set, false if the set was empty.
 */
bool int_set_pop_next_int(int_set_ref S, int *i);

/*
 * @function int_set_summary
 *
 * Write a textual summary of the set S to the given file
 * stream.
 */
void int_set_summary(int_set_ref S, FILE *stream);

#endif /* __INT_SET_H__ */
