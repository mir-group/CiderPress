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
  target_link_libraries(pwutil fftw3_threads fftw3 fftw3_mpi)
endif()
target_link_libraries(pwutil ${BLAS_LIBRARIES})
