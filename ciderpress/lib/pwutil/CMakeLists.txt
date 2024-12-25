python_add_library(pwutil SHARED
  grid_util.c nldf_fft_core.c ../mod_cider/sph_harm.c
)

set_target_properties(pwutil PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}
  COMPILE_FLAGS ${OpenMP_C_FLAGS}
  LINK_FLAGS ${OpenMP_C_FLAGS}
)

configure_file(
  ${PROJECT_SOURCE_DIR}/pwutil/config.h.in
  ${PROJECT_SOURCE_DIR}/pwutil/config.h
  NEWLINE_STYLE UNIX
)

if (HAVE_MPI)
  target_link_libraries(pwutil PRIVATE MPI::MPI_C)
  target_sources(pwutil PRIVATE nldf_fft_mpi.c gpaw_interface.c)
else()
  target_sources(pwutil PRIVATE nldf_fft_serial.c)
endif()

target_include_directories(pwutil PRIVATE ../mod_cider/)
target_include_directories(pwutil PRIVATE ${PROJECT_SOURCE_DIR}/pwutil)
target_include_directories(pwutil PRIVATE ../fft_wrapper)
target_link_libraries(pwutil PRIVATE fft_wrapper)
