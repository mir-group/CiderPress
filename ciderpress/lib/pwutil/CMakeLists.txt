add_library(pwutil SHARED
  grid_util.c nldf_fft.c ../mod_cider/sph_harm.c
)

set_target_properties(pwutil PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}
  COMPILE_FLAGS ${OpenMP_C_FLAGS}
  LINK_FLAGS ${OpenMP_C_FLAGS})

target_include_directories(pwutil PRIVATE ../mod_cider/)
target_include_directories(pwutil PRIVATE ${PYTHON_INCLUDE_DIRS})
message(INCLUDE_DIRECTORIES="${PYTHON_INCLUDE_DIRS}")
get_property(dirs DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY INCLUDE_DIRECTORIES)
foreach(dir ${dirs})
  message(STATUS "dir='${dir}'")
endforeach()
target_link_libraries(pwutil ${PYTHON_LIBRARIES})
if(ENABLE_FFTW)
  #foreach(FFTW3_LIB fftw3_mpi fftw3 fftw3_threads fftw3_omp)
  #  add_library(${FFTW3_LIB} STATIC IMPORTED)
  #  set_target_properties(${FFTW3_LIB} PROPERTIES IMPORTED_LOCATION ${PROJECT_SOURCE_DIR}/deps/lib/lib${FFTW3_LIB}.a)
  #  set_target_properties(${FFTW3_LIB} PROPERTIES INTERFACE_INCLUDE_DIRECTORIES ${PROJECT_SOURCE_DIR}/deps/include)
  #  target_link_libraries(pwutil ${FFTW3_LIB})
  #endforeach()
  #target_link_libraries(pwutil ${PROJECT_SOURCE_DIR}/deps/lib/libfftw3_mpi.so)
  target_link_libraries(pwutil fftw3_mpi)
  target_link_libraries(pwutil fftw3)
  #target_link_libraries(pwutil ${PROJECT_SOURCE_DIR}/deps/lib/libfftw3.so)
endif()
#target_link_libraries(pwutil ${BLAS_LIBRARIES})
target_link_libraries(pwutil MPI::MPI_C)
