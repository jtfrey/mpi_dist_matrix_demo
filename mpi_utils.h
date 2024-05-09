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

int mpi_printf(int rank, const char *fmt, ...);

#endif /* __MPI_UTILS_H__ */
