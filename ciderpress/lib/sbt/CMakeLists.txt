
add_library(sbt SHARED
  sbt.c
)

set_target_properties(sbt PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}
  COMPILE_FLAGS ${OpenMP_C_FLAGS}
  LINK_FLAGS ${OpenMP_C_FLAGS})

# target_link_libraries(sbt ${BLAS_LIBRARIES})
target_include_directories(sbt PRIVATE ../fft_wrapper)
target_link_libraries(sbt PRIVATE fft_wrapper)
