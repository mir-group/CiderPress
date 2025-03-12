

cmake_minimum_required (VERSION 3.20)
project (ciderpress)
include(ExternalProject)

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

option(BUILD_LIBXC "Build libxc library. If off, libxc must be findable by cmake" on)
option(BUILD_WITH_MKL "use MKL for BLAS and FFT" off)
option(BUILD_FFTW "Build fftw3" on)
option(BUILD_MARCH_NATIVE "gcc flag -march=native" off)
option(BUILD_WITH_MPI "Use MPI if available" on)
set(ASSUME_MPI_VENDOR "openmpi")
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

if (BUILD_WITH_MPI)
  find_package(MPI)
  if (MPI_LIBRARIES)
    set(HAVE_MPI 1)
    message(STATUS "Found MPI: ${FFTW_CONFIGURE_ARGS}")
  else ()
    set(HAVE_MPI 0)
    message(STATUS "Did not find MPI: ${FFTW_CONFIGURE_ARGS}")
  endif()
else ()
  set(HAVE_MPI 0)
endif()

if (BUILD_WITH_MKL)
  # Use MKL for the BLAS and FFT libraries
  # We want MKL to use the same threading as the rest of the project
  set(MKL_INTERFACE lp64)
  if("iomp5" IN_LIST OpenMP_C_LIB_NAMES)
    set(MKL_THREADING intel_thread)
  elseif("gomp" IN_LIST OpenMP_C_LIB_NAMES)
    set(MKL_THREADING gnu_thread)
  else()
    set(MKL_THREADING sequential)
  endif()
  if (HAVE_MPI)
    message(STATUS "THESE ARE THE MPI CXX LIBRARIES ${MPI_CXX_LIBRARIES}")
    if("${MPI_CXX_LIBRARIES}" MATCHES openmpi)
      set(MKL_MPI openmpi)
    elseif("${MPI_CXX_LIBRARIES}" MATCHES intel)
      set(MKL_MPI intelmpi)
    elseif("${MPI_CXX_LIBRARIES}" MATCHES mpich)
      set(MKL_MPI mpich)
    else()
      set(MKL_MPI "${ASSUME_MPI_VENDOR}")
      message(WARNING "Unknown MPI when setting up MKL, assuming ${ASSUME_MPI_VENDOR}")
    endif()
    set(ENABLE_CDFT ON)
    set(ENABLE_BLACS ON)
  endif()
  set(MKL_LINK dynamic)
  find_package(MKL CONFIG REQUIRED PATHS $ENV{MKLROOT})
  set(MKL_INCLUDE_DIR ${MKL_ROOT}/include)
  message(STATUS "Imported MKL targets: ${MKL_IMPORTED_TARGETS}")
  message(STATUS "MKL include path is: ${MKL_INCLUDE_DIR}")
  if(NOT MKL_FOUND)
    message(FATAL_ERROR "MKL not found")
  else()
    string(APPEND CMAKE_SHARED_LINKER_FLAGS " -Wl,--no-as-needed")
  endif()
  message(STATUS "LINKER FLAGS ${CMAKE_SHARED_LINKER_FLAGS}")
  set(FFT_BACKEND 1)
else()
  if (NOT BLAS_LIBRARIES)
    find_package(BLAS)
    find_package(LAPACK)
    check_function_exists(ffsll HAVE_FFS)
  endif()

  set(FFTW_CONFIGURE_ARGS --enable-static=no --enable-shared=yes --enable-threads --enable-openmp)
  if (BUILD_MARCH_NATIVE)
    list(APPEND FFTW_CONFIGURE_ARGS --enable-avx --enable-sse2 --enable-avx2 --enable-avx512)
  endif()
  if (HAVE_MPI)
    list(APPEND FFTW_CONFIGURE_ARGS --enable-mpi=yes MPILIBS=-lmpi)
  endif()

  if (NOT BLAS_LIBRARIES)
    message(FATAL_ERROR "A required library with BLAS API not found.")
  else()
    message(STATUS "BLAS libraries: ${BLAS_LIBRARIES}")
  endif()
  set(FFT_BACKEND 2)

  include(ExternalProject)
  if(BUILD_FFTW)
    ExternalProject_Add(libfftw3
      URL https://www.fftw.org/fftw-3.3.10.tar.gz
      PREFIX ${PROJECT_BINARY_DIR}/deps
      INSTALL_DIR ${PROJECT_SOURCE_DIR}/deps
      BUILD_IN_SOURCE True
      CONFIGURE_COMMAND ./configure ${FFTW_CONFIGURE_ARGS} CXX=${CMAKE_CXX_COMPILER} CC=${CMAKE_C_COMPILER} prefix=<INSTALL_DIR>
      BUILD_COMMAND make -j4 install
    )
  endif()
endif()

find_package(Python REQUIRED COMPONENTS Interpreter Development)

include_directories(${PROJECT_SOURCE_DIR})
include_directories(${PROJECT_SOURCE_DIR}/deps/include)
include_directories(${CMAKE_INSTALL_PREFIX}/include)
link_directories(${PROJECT_SOURCE_DIR}/deps/lib ${PROJECT_SOURCE_DIR}/deps/lib64)
link_directories(${CMAKE_INSTALL_PREFIX}/lib ${CMAKE_INSTALL_PREFIX}/lib64)
list(APPEND CMAKE_PREFIX_PATH "${PROJECT_SOURCE_DIR}/deps")

# From PySCF CMakeLists.txt
# See also https://gitlab.kitware.com/cmake/community/wikis/doc/cmake/RPATH-handling
if (WIN32)
  #?
elseif (APPLE)
  set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
  set(CMAKE_INSTALL_RPATH "@loader_path;@loader_path/deps/lib;@loader_path/deps/lib64")
  set(CMAKE_BUILD_RPATH "@loader_path;@loader_path/deps/lib;@loader_path/deps/lib64")
else ()
  set(CMAKE_SKIP_BUILD_RPATH  True)
  set(CMAKE_BUILD_WITH_INSTALL_RPATH True)
  set(CMAKE_INSTALL_RPATH "\$ORIGIN:\$ORIGIN/deps/lib:\$ORIGIN/deps/lib64")
endif ()

if(BUILD_LIBXC)
else()
  find_package(Libxc REQUIRED COMPONENTS shared C)
endif()

add_subdirectory(mod_cider)
add_subdirectory(numint_cider)
add_subdirectory(pwutil)
add_subdirectory(sbt)
add_subdirectory(xc_utils)
add_subdirectory(fft_wrapper)
add_dependencies(mcider fft_wrapper)
add_dependencies(sbt fft_wrapper)
if (BUILD_FFTW AND NOT BUILD_WITH_MKL)
  add_dependencies(fft_wrapper libfftw3)
endif()

if(BUILD_LIBXC)
    ExternalProject_Add(libxc
    #GIT_REPOSITORY https://gitlab.com/libxc/libxc.git
    #GIT_TAG master
    URL https://gitlab.com/libxc/libxc/-/archive/7.0.0/libxc-7.0.0.tar.gz
    PREFIX ${PROJECT_BINARY_DIR}/deps
    INSTALL_DIR ${PROJECT_SOURCE_DIR}/deps
    CMAKE_ARGS -DCMAKE_BUILD_TYPE=RELEASE -DBUILD_SHARED_LIBS=1
            -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>
            -DCMAKE_INSTALL_LIBDIR:PATH=lib
            -DENABLE_FORTRAN=0 -DDISABLE_FXC=1 -DDISABLE_KXC=1 -DDISABLE_LXC=1
            -DCMAKE_C_CREATE_SHARED_LIBRARY=${C_LINK_TEMPLATE}
            -DENABLE_XHOST:STRING=${BUILD_MARCH_NATIVE}
            -DCMAKE_C_COMPILER:STRING=${CMAKE_C_COMPILER}
  )
  add_dependencies(xc_utils libxc)
endif()

# NOTE some other stuff from pyscf CMake file was here

if(EXISTS "${PROJECT_SOURCE_DIR}/cmake.user.inc")
  include("${PROJECT_SOURCE_DIR}/cmake.user.inc")
endif()
