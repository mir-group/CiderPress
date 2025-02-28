
if (HAVE_MPI)
  add_library(fft_wrapper SHARED
    cider_fft.c
    cider_mpi_fft.c
  )
else()
  add_library(fft_wrapper SHARED
    cider_fft.c
  )
endif()

set_target_properties(fft_wrapper PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}
  COMPILE_FLAGS ${OpenMP_C_FLAGS}
  LINK_FLAGS ${OpenMP_C_FLAGS})

configure_file(
  ${PROJECT_SOURCE_DIR}/fft_wrapper/config.h.in
  ${PROJECT_SOURCE_DIR}/fft_wrapper/cider_fft_config.h
  NEWLINE_STYLE UNIX
)

if (HAVE_MPI)
  target_link_libraries(fft_wrapper PUBLIC MPI::MPI_C)
endif()

if (FFT_BACKEND EQUAL 1) # MKL
  target_include_directories(fft_wrapper PUBLIC ${MKL_INCLUDE_DIR})
  target_link_libraries(fft_wrapper PUBLIC MKL::MKL)
  if (HAVE_MPI)
    target_link_libraries(fft_wrapper PUBLIC MKL::mkl_cdft_core)
  endif()
else() # FFTW
  target_link_libraries(fft_wrapper PRIVATE fftw3)
  target_link_libraries(fft_wrapper PRIVATE fftw3_omp)
  if (HAVE_MPI)
    target_link_libraries(fft_wrapper PRIVATE fftw3_mpi)
  endif()
endif()
