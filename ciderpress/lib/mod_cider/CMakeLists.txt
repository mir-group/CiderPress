
add_library(mcider SHARED
  frac_lapl.c cider_coefs.c cider_grids.c spline.c sph_harm.c
  conv_interpolation.c convolutions.c fast_sdmx.c pbc_tools.c debug_numint.c
  model_utils.c
)

set_target_properties(mcider PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}
  COMPILE_FLAGS ${OpenMP_C_FLAGS}
  LINK_FLAGS ${OpenMP_C_FLAGS})

# target_link_libraries(mcider ${BLAS_LIBRARIES})
target_link_libraries(mcider ${MKL_IMPORTED_TARGETS})
target_include_directories(mcider PRIVATE ${MKL_INCLUDE_DIR})
