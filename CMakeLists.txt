#
# CMake build configuration for tetrominotris
#
cmake_minimum_required(VERSION 3.20)

project(mpi_dist_matrix
        VERSION 0.0.1
        DESCRIPTION "Test of distributed MPI matrix element manufacturing"
        LANGUAGES C
    )
include(GNUInstallDirs)

# We need MPI and threading:
find_package(MPI REQUIRED)
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

#
# Add project info/version variables for the sake of the configure file:
#
set(MPI_DIST_MATRIX_NAME ${PROJECT_NAME})
set(MPI_DIST_MATRIX_VERSION ${mpi_dist_matrix_VERSION})

#
# Generate our common header file and have all targets in this
# directory use the binary directory on the search path:
#
configure_file(project_config.h.in project_config.h)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

#
# The program:
#
add_executable(mpi_dist_matrix mpi_utils.c int_set.c mpi_server_thread.c mpi_client_thread.c)
target_compile_options(mpi_dist_matrix PRIVATE ${MPI_C_COMPILE_FLAGS})
target_include_directories(mpi_dist_matrix PRIVATE ${MPI_C_INCLUDE_PATH})
target_link_directories(mpi_dist_matrix PRIVATE ${MPI_C_LINK_FLAGS})
target_link_libraries(mpi_dist_matrix PRIVATE m Threads::Threads ${MPI_C_LIBRARIES})

#
# Install target(s):
#
install(TARGETS mpi_dist_matrix)
