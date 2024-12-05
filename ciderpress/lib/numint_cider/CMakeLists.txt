
add_library(numint SHARED
  nr_numint.c
)

set_target_properties(numint PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}
  COMPILE_FLAGS ${OpenMP_C_FLAGS}
  LINK_FLAGS ${OpenMP_C_FLAGS})

# target_link_libraries(numint ${BLAS_LIBRARIES})
target_link_libraries(numint ${MKL_IMPORTED_TARGETS})
target_include_directories(numint PRIVATE ${MKL_INCLUDE_DIR})
