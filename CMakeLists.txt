### fastupdate cmake configuration

cmake_minimum_required (VERSION 2.8.12)

# use only CXX compiler
project (IrBasis CXX)

# find additional cmake module in cmake dir
list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

# Print build type
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# Where to install?
message(STATUS "Will be installed in ${CMAKE_INSTALL_PREFIX}")
message(STATUS "Redifine CMAKE_INSTALL_PREFIX if you wish to change install location")

### Dependencies
find_package(Boost 1.55.0 REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(GSL REQUIRED)
find_package(LAPACK REQUIRED)

#include directories
set(IRBASIS_LIBRARY_INCLUDE_DIRS ${EIGEN3_INCLUDE_DIR} ${Boost_INCLUDE_DIRS} ${GSL_INCLUDE_DIRS} ${GSLCBLAS_INCLUDE_DIRS}  "ir_basis/include" "test") #rest taken care of by libraries dependencies

# Build and enable tests
#testing setup
option(Testing "Enable testing" ON)
include(EnableGtests) #defined in ./cmake
#include_directories("test")
message(${IRBASIS_LIBRARY_INCLUDE_DIRS})
include_directories(${IRBASIS_LIBRARY_INCLUDE_DIRS})
list(APPEND LINK_ALL ${GSL_LIBRARIES} ${GSLCBLAS_LIBRARIES} ${LAPACK_LIBRARIES} ${BLAS_LIBRARIES})

set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${BLAS_LINKER_FLAGS} ${LAPACK_LINKER_FLAGS}" CACHE STRING "Linker flags for executables" FORCE)

#testing source files
set(unittest_src unittest)

enable_testing(test)

foreach(test ${unittest_src})
    add_gtest(${test} test)
endforeach(test)

install(DIRECTORY "ir_basis/include" DESTINATION "." FILES_MATCHING PATTERN "*.hpp")
