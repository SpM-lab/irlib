### fastupdate cmake configuration

cmake_minimum_required (VERSION 2.8.12)

# use only CXX compiler
project (irlib CXX)

# find additional cmake module in cmake dir
list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/common/cmake)

# Print build type
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# Build doc
option(BUILD_DOCUMENTATION "Build documentations" OFF)

# Where to install?
message(STATUS "C++ libraries Will be installed in ${CMAKE_INSTALL_PREFIX}")
message(STATUS "Redifine CMAKE_INSTALL_PREFIX if you wish to change install location")
message(STATUS "BUILDDOC" ${BUILD_DOCUMENTATION})

### Dependencies
#find_package(Eigen3 REQUIRED)
find_package(MPFR 2.3.1 REQUIRED)
find_package(GMP REQUIRED) #TO DO: version check (>=4.2.1) & MPIR support
if (BUILD_DOCUMENTATION)
  find_package(Doxygen REQUIRED)
endif()

#include directories
set(EIGEN3_INCLUDE_DIR "thirdparty/eigen3")
set(IRLIB_LIBRARY_INCLUDE_DIRS ${EIGEN3_INCLUDE_DIR} ${EIGEN3_INCLUDE_DIR}/unsupported ${Boost_INCLUDE_DIRS} ${MPFR_INCLUDES} ${GMP_INCLUDE_DIR} "c++/include" "thirdparty/mpfr_cxx" "test") #rest taken care of by libraries dependencies
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
