add_library(pwutil SHARED
  grid_util.c ../mod_cider/sph_harm.c
)

set_target_properties(pwutil PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}
  COMPILE_FLAGS ${OpenMP_C_FLAGS}
  LINK_FLAGS ${OpenMP_C_FLAGS})

target_include_directories(pwutil PRIVATE ../mod_cider/)
target_link_libraries(pwutil ${BLAS_LIBRARIES})
