
add_library(mcider SHARED
  frac_lapl.c cider_coefs.c cider_grids.c spline.c sph_harm.c
  conv_interpolation.c convolutions.c fast_sdmx.c pbc_tools.c debug_numint.c
  model_utils.c
)

set_target_properties(mcider PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}
  COMPILE_FLAGS ${OpenMP_C_FLAGS}
  LINK_FLAGS ${OpenMP_C_FLAGS})

if (HAVE_MPI)
  target_link_libraries(mcider PRIVATE MPI::MPI_C)
endif()
if (BUILD_WITH_MKL)
  target_link_libraries(mcider PRIVATE MKL::MKL)
  target_include_directories(mcider PRIVATE ${MKL_INCLUDE_DIR})
else()
  target_link_libraries(mcider PRIVATE ${BLAS_LIBRARIES})
  target_link_libraries(mcider PRIVATE ${LAPACK_LIBRARIES})
endif()
target_include_directories(mcider PRIVATE ../fft_wrapper)
target_link_libraries(mcider PRIVATE fft_wrapper)
