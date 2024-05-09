# mpi_dist_matrix_demo

Proof-of-concept code that implements a distributed matrix element generator.

Each rank runs a server thread that manages non-local sub-matrix data writes.  E.g. if rank 5 produces a matrix element whose local sub-matrix is in rank 2, rank 5 must communicate the value to rank 2.  Any element resident in the rank's own sub-matrix is written directly.

One rank (typically the root rank, 0) is additionally responsible for allocating work to all ranks.  That server thread must:

- service requests for work units (e.g. matrix rows)
- manage in-flight and completed work units (in order to know when all elements have been generated)

Each rank runs a client thread that requests work units, produces matrix elements, and writes them into the global matrix.

## Example run

```
$ mpirun -np 4 ./mpi_dist_matrix 
[MPI-0:4] 
[MPI-0:4] Welcome to the threaded MPI matrix element work server demo!
[MPI-0:4] 
[MPI-0:4] A 20x20 matrix is distributed across 4 ranks and matrix elements of the form
[MPI-0:4] 
[MPI-0:4]     A_{i,j} = Sqrt[i*i + j*j]
[MPI-0:4] 
[MPI-0:4] are calculated.
[MPI-0:4] 
[MPI-0:4] matrix element loop running with 4 s sleep between work units
[MPI-0:4] allocated index 0 from primary slot 0 for rank 0
[MPI-0:4] server thread running work unit and memory managers
[MPI-0:4] allocated index 10 from primary slot 1 for rank 2
[MPI-0:4] allocated index 11 from primary slot 1 for rank 3
[MPI-0:4] allocated index 1 from primary slot 0 for rank 1
[MPI-2:4] matrix element loop running with 1 s sleep between work units
[MPI-2:4] server thread running a memory manager
[MPI-3:4] matrix element loop running with 1 s sleep between work units
[MPI-3:4] server thread running a memory manager
[MPI-1:4] matrix element loop running with 1 s sleep between work units
[MPI-1:4] server thread running a memory manager
[MPI-0:4] allocated index 12 from primary slot 1 for rank 2
[MPI-0:4] allocated index 13 from primary slot 1 for rank 3
[MPI-0:4] allocated index 2 from primary slot 0 for rank 1
[MPI-0:4] allocated index 14 from primary slot 1 for rank 3
[MPI-0:4] allocated index 3 from primary slot 0 for rank 1
[MPI-0:4] allocated index 15 from primary slot 1 for rank 2
[MPI-0:4] allocated index 16 from primary slot 1 for rank 3
[MPI-0:4] allocated index 4 from primary slot 0 for rank 1
[MPI-0:4] allocated index 17 from primary slot 1 for rank 2
[MPI-0:4] allocated index 5 from primary slot 0 for rank 0
[MPI-0:4] allocated index 6 from primary slot 0 for rank 1
[MPI-0:4] allocated index 18 from primary slot 1 for rank 3
[MPI-0:4] allocated index 19 from primary slot 1 for rank 2
[MPI-0:4] allocated index 7 from primary slot 0 for rank 1
[MPI-0:4] allocated index 8 from alternate slot 0 for rank 2
[MPI-0:4] allocated index 9 from alternate slot 0 for rank 3
[MPI-2:4] exited element loop
[MPI-1:4] exited element loop
[MPI-3:4] exited element loop
[MPI-0:4] exited element loop, waiting for all work to complete
[MPI-0:4] sending shutdown message to other ranks' server threads
[MPI-0:4] canceling server thread
[MPI-2:4] exiting server thread
[MPI-3:4] exiting server thread
[MPI-1:4] exiting server thread
[MPI-0:4] Sub-matrices in sequence by rank:

Rank 0:
       0.000,    1.000,    2.000,    3.000,    4.000,    5.000,    6.000,    7.000,    8.000,    9.000
       1.000,    1.414,    2.236,    3.162,    4.123,    5.099,    6.083,    7.071,    8.062,    9.055
       2.000,    2.236,    2.828,    3.606,    4.472,    5.385,    6.325,    7.280,    8.246,    9.220
       3.000,    3.162,    3.606,    4.243,    5.000,    5.831,    6.708,    7.616,    8.544,    9.487
       4.000,    4.123,    4.472,    5.000,    5.657,    6.403,    7.211,    8.062,    8.944,    9.849
       5.000,    5.099,    5.385,    5.831,    6.403,    7.071,    7.810,    8.602,    9.434,   10.296
       6.000,    6.083,    6.325,    6.708,    7.211,    7.810,    8.485,    9.220,   10.000,   10.817
       7.000,    7.071,    7.280,    7.616,    8.062,    8.602,    9.220,    9.899,   10.630,   11.402
       8.000,    8.062,    8.246,    8.544,    8.944,    9.434,   10.000,   10.630,   11.314,   12.042
       9.000,    9.055,    9.220,    9.487,    9.849,   10.296,   10.817,   11.402,   12.042,   12.728
[MPI-0:4] ready to exit

Rank 1:
      10.000,   11.000,   12.000,   13.000,   14.000,   15.000,   16.000,   17.000,   18.000,   19.000
      10.050,   11.045,   12.042,   13.038,   14.036,   15.033,   16.031,   17.029,   18.028,   19.026
      10.198,   11.180,   12.166,   13.153,   14.142,   15.133,   16.125,   17.117,   18.111,   19.105
      10.440,   11.402,   12.369,   13.342,   14.318,   15.297,   16.279,   17.263,   18.248,   19.235
      10.770,   11.705,   12.649,   13.601,   14.560,   15.524,   16.492,   17.464,   18.439,   19.416
      11.180,   12.083,   13.000,   13.928,   14.866,   15.811,   16.763,   17.720,   18.682,   19.647
      11.662,   12.530,   13.416,   14.318,   15.232,   16.155,   17.088,   18.028,   18.974,   19.925
      12.207,   13.038,   13.892,   14.765,   15.652,   16.553,   17.464,   18.385,   19.313,   20.248
      12.806,   13.601,   14.422,   15.264,   16.125,   17.000,   17.889,   18.788,   19.698,   20.616
      13.454,   14.213,   15.000,   15.811,   16.643,   17.493,   18.358,   19.235,   20.125,   21.024
[MPI-1:4] ready to exit

Rank 2:
      10.000,   10.050,   10.198,   10.440,   10.770,   11.180,   11.662,   12.207,   12.806,   13.454
      11.000,   11.045,   11.180,   11.402,   11.705,   12.083,   12.530,   13.038,   13.601,   14.213
      12.000,   12.042,   12.166,   12.369,   12.649,   13.000,   13.416,   13.892,   14.422,   15.000
      13.000,   13.038,   13.153,   13.342,   13.601,   13.928,   14.318,   14.765,   15.264,   15.811
      14.000,   14.036,   14.142,   14.318,   14.560,   14.866,   15.232,   15.652,   16.125,   16.643
      15.000,   15.033,   15.133,   15.297,   15.524,   15.811,   16.155,   16.553,   17.000,   17.493
      16.000,   16.031,   16.125,   16.279,   16.492,   16.763,   17.088,   17.464,   17.889,   18.358
      17.000,   17.029,   17.117,   17.263,   17.464,   17.720,   18.028,   18.385,   18.788,   19.235
      18.000,   18.028,   18.111,   18.248,   18.439,   18.682,   18.974,   19.313,   19.698,   20.125
      19.000,   19.026,   19.105,   19.235,   19.416,   19.647,   19.925,   20.248,   20.616,   21.024
[MPI-2:4] ready to exit

Rank 3:
      14.142,   14.866,   15.620,   16.401,   17.205,   18.028,   18.868,   19.723,   20.591,   21.471
      14.866,   15.556,   16.279,   17.029,   17.804,   18.601,   19.416,   20.248,   21.095,   21.954
      15.620,   16.279,   16.971,   17.692,   18.439,   19.209,   20.000,   20.809,   21.633,   22.472
      16.401,   17.029,   17.692,   18.385,   19.105,   19.849,   20.616,   21.401,   22.204,   23.022
      17.205,   17.804,   18.439,   19.105,   19.799,   20.518,   21.260,   22.023,   22.804,   23.601
      18.028,   18.601,   19.209,   19.849,   20.518,   21.213,   21.932,   22.672,   23.431,   24.207
      18.868,   19.416,   20.000,   20.616,   21.260,   21.932,   22.627,   23.345,   24.083,   24.839
      19.723,   20.248,   20.809,   21.401,   22.023,   22.672,   23.345,   24.042,   24.759,   25.495
      20.591,   21.095,   21.633,   22.204,   22.804,   23.431,   24.083,   24.759,   25.456,   26.173
      21.471,   21.954,   22.472,   23.022,   23.601,   24.207,   24.839,   25.495,   26.173,   26.870
[MPI-3:4] ready to exit
```