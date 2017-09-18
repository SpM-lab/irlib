
#include_directories(${IRBASIS_LIBRARY_INCLUDE_DIRS})
list(APPEND LINK_ALL ${Boost_LIBRARIES} ${LAPACK_LIBRARIES} ${BLAS_LIBRARIES} ${GMP_LIBRARIES} ${MPFR_LIBRARIES})

set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${BLAS_LINKER_FLAGS} ${LAPACK_LINKER_FLAGS}" CACHE STRING "Linker flags for executables" FORCE)

install(DIRECTORY "include" DESTINATION "." FILES_MATCHING PATTERN "*.hpp")


if (Testing)
  #testing source files
  set(unittest_src unittest mpmath kernel)
  foreach(test ${unittest_src})
      add_gtest(${test} test)
  endforeach(test)
endif()
