

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

if (NOT BLAS_LIBRARIES)
  find_package(BLAS)
  check_function_exists(ffsll HAVE_FFS)
endif()

# find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
include_directories(/home/kylebystrom/micromamba/envs/test_cider4/include/python3.11)
link_directories(/home/kylebystrom/micromamba/envs/test_cider4/lib/python3.11)
#message(INCLUDE_DIRECTORIES="${PYTHON_INCLUDE_DIRS}")

if (NOT BLAS_LIBRARIES)
  message(FATAL_ERROR "A required library with BLAS API not found.")
else()
  message(STATUS "BLAS libraries: ${BLAS_LIBRARIES}")
endif()

find_package(OpenMP)
if(OPENMP_FOUND)
  set(HAVE_OPENMP 1)
else ()
  set(OpenMP_C_FLAGS " ")
endif()

include_directories(${PROJECT_SOURCE_DIR})
include_directories(${PROJECT_SOURCE_DIR}/deps/include)
include_directories(${CMAKE_INSTALL_PREFIX}/include)
link_directories(${PROJECT_SOURCE_DIR}/deps/lib ${PROJECT_SOURCE_DIR}/deps/lib64)
link_directories(${CMAKE_INSTALL_PREFIX}/lib ${CMAKE_INSTALL_PREFIX}/lib64)

include(ExternalProject)
option(ENABLE_FFTW "Using fftw3" ON)
option(BUILD_FFTW "Building fftw3" ON)

add_subdirectory(mod_cider)
add_subdirectory(numint_cider)
add_subdirectory(pw_utils)
add_subdirectory(sbt)

if(ENABLE_FFTW AND BUILD_FFTW)
#  if(NOT EXISTS "${PROJECT_SOURCE_DIR}/deps/include/fftw3.h")
    ExternalProject_Add(libfftw3
      URL https://www.fftw.org/fftw-3.3.10.tar.gz
      PREFIX ${PROJECT_BINARY_DIR}/deps
      INSTALL_DIR ${PROJECT_SOURCE_DIR}/deps
      BUILD_IN_SOURCE True
      CONFIGURE_COMMAND ./configure --enable-static=no --enable-shared=yes --enable-threads --enable-mpi=yes CXX=${CMAKE_CXX_COMPILER} CC=${CMAKE_C_COMPILER} prefix=<INSTALL_DIR>
      BUILD_COMMAND make -j4 install
    )
    add_dependencies(pwutil libfftw3)
#  endif()
endif()

# TODO some other stuff from pyscf CMake file was here

# TODO some other stuff from pyscf CMake file was here

if(EXISTS "${PROJECT_SOURCE_DIR}/cmake.user.inc")
  include("${PROJECT_SOURCE_DIR}/cmake.user.inc")
endif()
