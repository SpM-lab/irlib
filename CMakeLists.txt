### fastupdate cmake configuration

cmake_minimum_required (VERSION 2.8.12)

# use only CXX compiler
project (irlib CXX)

# find additional cmake module in cmake dir
list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/common/cmake)

# Print build type
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# Where to install?
message(STATUS "C++ libraries Will be installed in ${CMAKE_INSTALL_PREFIX}")
message(STATUS "Redifine CMAKE_INSTALL_PREFIX if you wish to change install location")

### Dependencies
find_package(Boost 1.55.0 REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(LAPACK REQUIRED) #For SVD

#include directories
set(IRLIB_LIBRARY_INCLUDE_DIRS ${EIGEN3_INCLUDE_DIR} ${EIGEN3_INCLUDE_DIR}/unsupported ${Boost_INCLUDE_DIRS}  "c++/include" "test") #rest taken care of by libraries dependencies
include_directories(${IRLIB_LIBRARY_INCLUDE_DIRS})

#Python bindings
option(PYTHON "Build Python bindings" ON)

# Build and enable tests
# testing setup
# enable_testing() must be called in the top-level CMakeLists.txt before any add_subdirectory() is called.
option(Testing "Enable testing" ON)

if (Testing)
  include(EnableTests) #defined in ./cmake
  enable_testing(test)
endif()

add_subdirectory(c++)

if (PYTHON)
  add_subdirectory(python)
endif()
