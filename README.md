# mpi_dist_matrix_demo

Proof-of-concept code that implements a distributed matrix element generator.

Each rank runs a server thread that manages non-local sub-matrix data writes.  E.g. if rank 5 produces a matrix element whose local sub-matrix is in rank 2, rank 5 must communicate the value to rank 2.  Any element resident in the rank's own sub-matrix is written directly.

One rank (typically the root rank, 0) is additionally responsible for allocating work to all ranks.  That server thread must:

- service requests for work units (e.g. matrix rows)
- manage in-flight and completed work units (in order to know when all elements have been generated)

Each rank runs a client thread that requests work units, produces matrix elements, and writes them into the global matrix.

## Getting help

The program accepts a few command line arguments which are explained via the built-in help:

```
$ ./mpi_dist_matrix --help
usage:

    ./mpi_dist_matrix {options}

  options:

    --help/-h                  show this information
    --dims/-d <matrix-2d-dims> choose matrix dimensions (default 10000)
    --blocks/-b <block-dims>   choose the global matrix partitioning scheme (default
                               is to use auto-grid)
    --auto-grid/-a             automatically choose the global matrix partitioning
                               scheme
    --row-major/-r             use column-major storage and distribution across ranks
    --column-major/-c          use column-major storage and distribution across ranks
    --root/-0 #                elect the given rank id as the root server

  <matrix-2d-dims> = # | #,#   given a single integer value, a square matrix of the given
                               number of rows and columns is chosen; otherwise, the first
                               integer in the comma-delimited pair is the row count, the
                               second is the column count
  <block-dims> = # | #,#       paritition the global matrix into:
                                   # : this integer number of rows AND columns
                                   #,# : the given integer number of rows,columns
```

## Example run

```
$ mpirun -np 8 --map-by :OVERSUBSCRIBE  ./mpi_dist_matrix --dims=40,20
[MPI-1:8][7740] local sub-matrix indices [0,10]..[9,19]
[MPI-7:8][7746] local sub-matrix indices [30,10]..[39,19]
[MPI-5:8][7744] local sub-matrix indices [20,10]..[29,19]
[MPI-0:8][7739] testing: exact=1 [4,2] |r-c|=2, ∆(r/c)=0.000000, is-exact=1
[MPI-0:8][7739] testing: exact=1 [2,4] |r-c|=2, ∆(r/c)=1.500000, is-exact=1
[MPI-0:8][7739] testing: exact=1 [2,4] |r-c|=2, ∆(r/c)=1.500000, is-exact=1
[MPI-0:8][7739] testing: exact=1 [4,2] |r-c|=2, ∆(r/c)=0.000000, is-exact=1
[MPI-0:8][7739] testing: exact=1 [1,8] |r-c|=7, ∆(r/c)=1.875000, is-exact=0
[MPI-0:8][7739] testing: exact=1 [8,1] |r-c|=7, ∆(r/c)=6.000000, is-exact=1
[MPI-0:8][7739] auto-grid block partitioning yielded 4 x 2
[MPI-0:8][7739] block grid dimensions [4,2]
[MPI-0:8][7739] base sub-matrix dimensions [10,10]
[MPI-0:8][7739] local sub-matrix indices [0,0]..[9,9]
[MPI-0:8][7739] local sub-matrix allocated
[MPI-0:8][7739] 
[MPI-0:8][7739] Welcome to the threaded MPI matrix element work server demo!
[MPI-0:8][7739] 
[MPI-0:8][7739] A 40x20 matrix is distributed across 8 ranks and matrix elements of the form
[MPI-0:8][7739] 
[MPI-0:8][7739]     A_{i,j} = Sqrt[i*i + j*j]
[MPI-0:8][7739] 
[MPI-4:8][7743] local sub-matrix indices [20,0]..[29,9]
[MPI-0:8][7739] are calculated.
[MPI-0:8][7739] 
[MPI-2:8][7741] local sub-matrix indices [10,0]..[19,9]
[MPI-3:8][7742] local sub-matrix indices [10,10]..[19,19]
[MPI-6:8][7745] local sub-matrix indices [30,0]..[39,9]
[MPI-7:8][7746] matrix element loop running
[MPI-7:8][7746] server thread running a memory manager
[MPI-5:8][7744] matrix element loop running
[MPI-5:8][7744] server thread running a memory manager
[MPI-3:8][7742] matrix element loop running
[MPI-3:8][7742] server thread running a memory manager
[MPI-1:8][7740] matrix element loop running
[MPI-1:8][7740] server thread running a memory manager
[MPI-6:8][7745] matrix element loop running
[MPI-6:8][7745] server thread running a memory manager
[MPI-4:8][7743] matrix element loop running
[MPI-4:8][7743] server thread running a memory manager
[MPI-0:8][7739] matrix element loop running
[MPI-0:8][7739] server thread running work unit and memory managers
[MPI-2:8][7741] matrix element loop running
[MPI-2:8][7741] server thread running a memory manager
[MPI-7:8][7746] exited element loop
[MPI-5:8][7744] exited element loop
[MPI-6:8][7745] exited element loop
[MPI-4:8][7743] exited element loop
[MPI-0:8][7739] exited element loop, waiting for all work to complete
[MPI-1:8][7740] exited element loop
[MPI-2:8][7741] exited element loop
[MPI-3:8][7742] exited element loop
[MPI-0:8][7739] sending shutdown message to other ranks' server threads
[MPI-0:8][7739] canceling server thread
[MPI-3:8][7742] exiting server thread
[MPI-4:8][7743] exiting server thread
[MPI-2:8][7741] exiting server thread
[MPI-1:8][7740] exiting server thread
[MPI-6:8][7745] exiting server thread
[MPI-5:8][7744] exiting server thread
[MPI-7:8][7746] exiting server thread
[MPI-0:8][7739] Sub-matrices in sequence by rank:

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
[MPI-0:8][7739] ready to exit

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
[MPI-1:8][7740] ready to exit

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
[MPI-2:8][7741] ready to exit

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
[MPI-3:8][7742] ready to exit

Rank 4:
      20.000,   20.025,   20.100,   20.224,   20.396,   20.616,   20.881,   21.190,   21.541,   21.932
      21.000,   21.024,   21.095,   21.213,   21.378,   21.587,   21.840,   22.136,   22.472,   22.847
      22.000,   22.023,   22.091,   22.204,   22.361,   22.561,   22.804,   23.087,   23.409,   23.770
      23.000,   23.022,   23.087,   23.195,   23.345,   23.537,   23.770,   24.042,   24.352,   24.698
      24.000,   24.021,   24.083,   24.187,   24.331,   24.515,   24.739,   25.000,   25.298,   25.632
      25.000,   25.020,   25.080,   25.179,   25.318,   25.495,   25.710,   25.962,   26.249,   26.571
      26.000,   26.019,   26.077,   26.173,   26.306,   26.476,   26.683,   26.926,   27.203,   27.514
      27.000,   27.019,   27.074,   27.166,   27.295,   27.459,   27.659,   27.893,   28.160,   28.460
      28.000,   28.018,   28.071,   28.160,   28.284,   28.443,   28.636,   28.862,   29.120,   29.411
      29.000,   29.017,   29.069,   29.155,   29.275,   29.428,   29.614,   29.833,   30.083,   30.364
[MPI-4:8][7743] ready to exit

Rank 5:
      22.361,   22.825,   23.324,   23.854,   24.413,   25.000,   25.612,   26.249,   26.907,   27.586
      23.259,   23.707,   24.187,   24.698,   25.239,   25.807,   26.401,   27.019,   27.659,   28.320
      24.166,   24.597,   25.060,   25.554,   26.077,   26.627,   27.203,   27.803,   28.425,   29.069
      25.080,   25.495,   25.942,   26.420,   26.926,   27.459,   28.018,   28.601,   29.206,   29.833
      26.000,   26.401,   26.833,   27.295,   27.785,   28.302,   28.844,   29.411,   30.000,   30.610
      26.926,   27.313,   27.731,   28.178,   28.653,   29.155,   29.682,   30.232,   30.806,   31.401
      27.857,   28.231,   28.636,   29.069,   29.530,   30.017,   30.529,   31.064,   31.623,   32.202
      28.792,   29.155,   29.547,   29.967,   30.414,   30.887,   31.385,   31.906,   32.450,   33.015
      29.732,   30.083,   30.463,   30.871,   31.305,   31.765,   32.249,   32.757,   33.287,   33.838
      30.676,   31.016,   31.385,   31.780,   32.202,   32.650,   33.121,   33.615,   34.132,   34.670
[MPI-5:8][7744] ready to exit

Rank 6:
      30.000,   30.017,   30.067,   30.150,   30.265,   30.414,   30.594,   30.806,   31.048,   31.321
      31.000,   31.016,   31.064,   31.145,   31.257,   31.401,   31.575,   31.780,   32.016,   32.280
      32.000,   32.016,   32.062,   32.140,   32.249,   32.388,   32.558,   32.757,   32.985,   33.242
      33.000,   33.015,   33.061,   33.136,   33.242,   33.377,   33.541,   33.734,   33.956,   34.205
      34.000,   34.015,   34.059,   34.132,   34.234,   34.366,   34.525,   34.713,   34.928,   35.171
      35.000,   35.014,   35.057,   35.128,   35.228,   35.355,   35.511,   35.693,   35.903,   36.139
      36.000,   36.014,   36.056,   36.125,   36.222,   36.346,   36.497,   36.674,   36.878,   37.108
      37.000,   37.014,   37.054,   37.121,   37.216,   37.336,   37.483,   37.656,   37.855,   38.079
      38.000,   38.013,   38.053,   38.118,   38.210,   38.328,   38.471,   38.639,   38.833,   39.051
      39.000,   39.013,   39.051,   39.115,   39.205,   39.319,   39.459,   39.623,   39.812,   40.025
[MPI-6:8][7745] ready to exit

Rank 7:
      31.623,   31.953,   32.311,   32.696,   33.106,   33.541,   34.000,   34.482,   34.986,   35.511
      32.573,   32.894,   33.242,   33.615,   34.015,   34.438,   34.886,   35.355,   35.847,   36.359
      33.526,   33.838,   34.176,   34.540,   34.928,   35.341,   35.777,   36.235,   36.715,   37.216
      34.482,   34.785,   35.114,   35.468,   35.847,   36.249,   36.674,   37.121,   37.590,   38.079
      35.440,   35.735,   36.056,   36.401,   36.770,   37.162,   37.577,   38.013,   38.471,   38.949
      36.401,   36.688,   37.000,   37.336,   37.696,   38.079,   38.484,   38.910,   39.357,   39.825
      37.363,   37.643,   37.947,   38.275,   38.626,   39.000,   39.395,   39.812,   40.249,   40.706
      38.328,   38.601,   38.897,   39.217,   39.560,   39.925,   40.311,   40.719,   41.146,   41.593
      39.294,   39.560,   39.850,   40.162,   40.497,   40.853,   41.231,   41.629,   42.048,   42.485
      40.262,   40.522,   40.804,   41.110,   41.437,   41.785,   42.154,   42.544,   42.953,   43.382
[MPI-7:8][7746] ready to exit
```