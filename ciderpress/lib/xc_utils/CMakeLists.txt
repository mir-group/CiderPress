
add_library(xc_utils SHARED
  libxc_baselines.c
)

set_target_properties(xc_utils PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}
  COMPILE_FLAGS ${OpenMP_C_FLAGS}
  LINK_FLAGS ${OpenMP_C_FLAGS})

if (HAVE_MPI)
  target_link_libraries(xc_utils PRIVATE MPI::MPI_C)
endif()
if (BUILD_WITH_MKL)
  target_link_libraries(xc_utils PRIVATE MKL::MKL)
  target_include_directories(xc_utils PRIVATE ${MKL_INCLUDE_DIR})
else()
  target_link_libraries(xc_utils PRIVATE ${BLAS_LIBRARIES})
endif()
if(BUILD_LIBXC)
  target_link_libraries(xc_utils PRIVATE xc)
else()
  target_link_libraries(xc_utils PRIVATE Libxc::xc)
endif()
