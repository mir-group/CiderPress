
add_library(fft_wrapper SHARED
  cider_fft.c
)

set_target_properties(fft_wrapper PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}
  COMPILE_FLAGS ${OpenMP_C_FLAGS}
  LINK_FLAGS ${OpenMP_C_FLAGS})

# target_link_libraries(mcider ${BLAS_LIBRARIES})
target_link_libraries(fft_wrapper ${MKL_IMPORTED_TARGETS})
target_include_directories(fft_wrapper PRIVATE ${MKL_INCLUDE_DIR})
