
add_library(xc_utils SHARED
  libxc_baselines.c
)

set_target_properties(xc_utils PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}
  COMPILE_FLAGS ${OpenMP_C_FLAGS}
  LINK_FLAGS ${OpenMP_C_FLAGS})

target_link_libraries(xc_utils ${LIBXC_LIBRARIES})
# target_link_libraries(xc_utils ${BLAS_LIBRARIES})
target_link_libraries(xc_utils ${MKL_IMPORTED_TARGETS})
target_include_directories(xc_utils PRIVATE ${MKL_INCLUDE_DIR})
