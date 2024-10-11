add_library(pwutil SHARED
  grid_util.c nldf_fft_core.c ../mod_cider/sph_harm.c
)

set_target_properties(pwutil PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}
  COMPILE_FLAGS ${OpenMP_C_FLAGS}
  LINK_FLAGS ${OpenMP_C_FLAGS}
)

configure_file(
  ${PROJECT_SOURCE_DIR}/pwutil/config.h.in
  ${PROJECT_SOURCE_DIR}/pwutil/config.h
  NEWLINE_STYLE UNIX
)

target_include_directories(pwutil PRIVATE ../mod_cider/)
target_include_directories(pwutil PRIVATE ${PROJECT_SOURCE_DIR}/pwutil)
target_include_directories(pwutil PRIVATE ${PYTHON_INCLUDE_DIRS})
message(INCLUDE_DIRECTORIES="${PYTHON_INCLUDE_DIRS}")
get_property(dirs DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY INCLUDE_DIRECTORIES)
foreach (dir ${dirs})
  message(STATUS "dir='${dir}'")
endforeach()
target_link_libraries(pwutil ${PYTHON_LIBRARIES})
if (NOT ENABLE_FFTW)
  message(FATAL_ERROR "Must enable fftw to compile pwutil module")
endif()
if (HAVE_MPI)
  target_link_libraries(pwutil fftw3_mpi)
  target_sources(pwutil PRIVATE nldf_fft_mpi.c gpaw_interface.c)
else ()
  target_sources(pwutil PRIVATE nldf_fft_serial.c)
endif ()
target_link_libraries(pwutil fftw3)
#target_link_libraries(pwutil ${BLAS_LIBRARIES})
target_link_libraries(pwutil MPI::MPI_C)
