
add_library(numint SHARED
  nr_numint.c
)

set_target_properties(numint PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}
  COMPILE_FLAGS ${OpenMP_C_FLAGS}
  LINK_FLAGS ${OpenMP_C_FLAGS})

if (HAVE_MPI)
  target_link_libraries(numint PRIVATE MPI::MPI_C)
endif()
if (BUILD_WITH_MKL)
  target_link_libraries(numint PRIVATE MKL::MKL)
  target_include_directories(numint PRIVATE ${MKL_INCLUDE_DIR})
else()
  target_link_libraries(numint PRIVATE ${BLAS_LIBRARIES})
endif()
