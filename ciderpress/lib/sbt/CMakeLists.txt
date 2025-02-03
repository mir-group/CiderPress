
add_library(sbt SHARED
  sbt.c
)

set_target_properties(sbt PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}
  COMPILE_FLAGS ${OpenMP_C_FLAGS}
  LINK_FLAGS ${OpenMP_C_FLAGS})

# target_link_libraries(sbt ${BLAS_LIBRARIES})
target_link_libraries(sbt ${MKL_IMPORTED_TARGETS})
target_include_directories(sbt PRIVATE ${MKL_INCLUDE_DIR})
