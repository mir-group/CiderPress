

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
include_directories(${CMAKE_PREFIX_PATH}/include)
include_directories(${CMAKE_INSTALL_PREFIX}/include)
link_directories(${CMAKE_INSTALL_PREFIX}/lib ${CMAKE_INSTALL_PREFIX}/lib64)

# TODO some other stuff from pyscf CMake file was here

add_subdirectory(mod_cider)
add_subdirectory(numint_cider)
add_subdirectory(pw_utils)
add_subdirectory(sbt)

# TODO some other stuff from pyscf CMake file was here

if(EXISTS "${PROJECT_SOURCE_DIR}/cmake.user.inc")
  include("${PROJECT_SOURCE_DIR}/cmake.user.inc")
endif()
