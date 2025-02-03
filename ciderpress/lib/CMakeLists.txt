

cmake_minimum_required (VERSION 3.5)
project (ciderpress)

if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RELWITHDEBINFO)
endif()

if (CMAKE_COMPILER_IS_GNUCC)
  if (UNIX AND NOT APPLE AND NOT DEFINED ENV{LDFLAGS})
    set(CMAKE_SHARED_LINKER_FLAGS "-Wl,--no-as-needed")
  endif()
endif()
set(CMAKE_C_FLAGS "-Wall ${CMAKE_C_FLAGS}")
set(CMAKE_C_FLAGS_DEBUG "-g")
set(CMAKE_C_FLAGS_RELEASE "-g -O3")
list( APPEND CMAKE_BUILD_RPATH ${CMAKE_PREFIX_PATH}/lib )
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

option(BUILD_MARCH_NATIVE "gcc flag -march=native" off)
if (BUILD_MARCH_NATIVE)
  include(CheckCCompilerFlag)
  CHECK_C_COMPILER_FLAG("-march=native" COMPILER_SUPPORTS_MARCH_NATIVE)
  if(COMPILER_SUPPORTS_MARCH_NATIVE)
    if ("${CMAKE_C_COMPILER_ID}" STREQUAL "Intel")
      message("Add CFLAGS -march=native -unroll-aggressive")
      set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -unroll-aggressive -ipo")
      set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fno-math-errno")
    else()
      message("Add CFLAGS -march=native -ftree-vectorize")
      set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=native -funroll-loops -ftree-vectorize")
      set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fno-math-errno")
    endif()
  endif()
endif()

set(CMAKE_INCLUDE_CURRENT_DIR ON)

if(EXISTS "${PROJECT_SOURCE_DIR}/cmake.arch.inc")
  include("${PROJECT_SOURCE_DIR}/cmake.arch.inc")
endif()

find_package(OpenMP)
if(OPENMP_C_FOUND)
  set(HAVE_OPENMP 1)
  message(STATUS "OpenMP_C_LIB_NAMES" ${OpenMP_C_LIB_NAMES})
  message(STATUS "OpenMP_C_LIBRARY" ${OpenMP_C_LIBRARY})
  message(STATUS "OpenMP_C_LIBRARIES" ${OpenMP_C_LIBRARIES})
else ()
  set(OpenMP_C_FLAGS " ")
endif()

# We want MKL to use the same threading as the rest of the project
set(MKL_INTERFACE lp64)
if("iomp5" IN_LIST OpenMP_C_LIB_NAMES)
  set(MKL_THREADING intel_thread)
elseif("gomp" IN_LIST OpenMP_C_LIB_NAMES)
  set(MKL_THREADING gnu_thread)
else()
  set(MKL_THREADING sequential)
endif()
find_package(MKL CONFIG REQUIRED PATHS $ENV{MKLROOT})
set(MKL_INCLUDE_DIR ${MKL_ROOT}/include)
message(STATUS "Imported MKL targets: ${MKL_IMPORTED_TARGETS}")
message(STATUS "MKL include path is: ${MKL_INCLUDE_DIR}")
if (NOT MKL_FOUND)
  message(FATAL_ERROR "MKL not found")
endif()

if (NOT BLAS_LIBRARIES)
  find_package(BLAS)
  check_function_exists(ffsll HAVE_FFS)
endif()

find_package(MPI)
if (MPI_LIBRARIES)
  set(HAVE_MPI 1)
  set(FFTW_CONFIGURE_ARGS --enable-static=no --enable-shared=yes --enable-threads --enable-mpi=yes --enable-openmp MPILIBS=-lmpi)
  message(STATUS "Found MPI: ${FFTW_CONFIGURE_ARGS}")
else ()
  set(HAVE_MPI 0)
  set(FFTW_CONFIGURE_ARGS --enable-static=no --enable-shared=yes --enable-threads --enable-openmp)
  message(STATUS "Did not find MPI: ${FFTW_CONFIGURE_ARGS}")
endif()
# include_directories(${CMAKE_PYTHON_INCLUDE_PATH})
# link_directories(${CMAKE_PYTHON_LIBRARY_PATH})

find_package(Python REQUIRED COMPONENTS Interpreter Development)

if (NOT BLAS_LIBRARIES)
  message(FATAL_ERROR "A required library with BLAS API not found.")
else()
  message(STATUS "BLAS libraries: ${BLAS_LIBRARIES}")
endif()

include_directories(${PROJECT_SOURCE_DIR})
include_directories(${PROJECT_SOURCE_DIR}/deps/include)
include_directories(${CMAKE_INSTALL_PREFIX}/include)
link_directories(${PROJECT_SOURCE_DIR}/deps/lib ${PROJECT_SOURCE_DIR}/deps/lib64)
link_directories(${CMAKE_INSTALL_PREFIX}/lib ${CMAKE_INSTALL_PREFIX}/lib64)

# See also https://gitlab.kitware.com/cmake/community/wikis/doc/cmake/RPATH-handling
set(CMAKE_SKIP_BUILD_RPATH  True)
set(CMAKE_BUILD_WITH_INSTALL_RPATH True)
set(CMAKE_INSTALL_RPATH "\$ORIGIN:\$ORIGIN/deps/lib:\$ORIGIN/deps/lib64")
message(RPATH=${CMAKE_INSTALL_RPATH})

include(ExternalProject)
option(ENABLE_FFTW "Using fftw3" ON)
option(BUILD_FFTW "Building fftw3" ON)

string(REPLACE ":" ";" _lib_path "$ENV{LD_LIBRARY_PATH}")
find_library(LIBXC_LIBRARIES NAMES xc PATHS ${_lib_path})
add_subdirectory(mod_cider)
add_subdirectory(numint_cider)
add_subdirectory(pwutil)
add_subdirectory(sbt)
add_subdirectory(xc_utils)

if(ENABLE_FFTW AND BUILD_FFTW)
    ExternalProject_Add(libfftw3
      URL https://www.fftw.org/fftw-3.3.10.tar.gz
      PREFIX ${PROJECT_BINARY_DIR}/deps
      INSTALL_DIR ${PROJECT_SOURCE_DIR}/deps
      BUILD_IN_SOURCE True
      CONFIGURE_COMMAND ./configure ${FFTW_CONFIGURE_ARGS} CXX=${CMAKE_CXX_COMPILER} CC=${CMAKE_C_COMPILER} prefix=<INSTALL_DIR>
      BUILD_COMMAND make -j4 install
    )
    add_dependencies(pwutil libfftw3)
endif()

# NOTE some other stuff from pyscf CMake file was here

if(EXISTS "${PROJECT_SOURCE_DIR}/cmake.user.inc")
  include("${PROJECT_SOURCE_DIR}/cmake.user.inc")
endif()
