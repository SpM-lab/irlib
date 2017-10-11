#cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

# add common modules from ../common/cmake
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/common/cmake)

SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

#find_package(PythonLibs REQUIRED)
#message("PYTHON_LIBRARY:            ${PYTHON_LIBRARY}")
#message("PYTHONLIBS_VERSION_STRING: ${PYTHONLIBS_VERSION_STRING}")

#If this does not find the correct one, please use invoke cmake with -DPYTHON_EXECUTABLE=/path/to/your/python/executable.
find_package(PythonInterp REQUIRED)
message("Found Python executable: " ${PYTHON_EXECUTABLE})
message("Python version: " ${PYTHON_VERSION_STRING})

find_package(PythonLibs REQUIRED)
message("Found Python libs: " ${PYTHON_LIBRARIES})
message("Python libs version: " ${PYTHONLIBS_VERSION_STRING})
message("Python libraries: " ${PYTHON_LIBRARIES})
message("Python include dirs: " ${PYTHON_INCLUDE_DIRS})

if (NOT PYTHONLIBS_VERSION_STRING STREQUAL PYTHON_VERSION_STRING)
    message(WARNING "Version mismatch between Python interpreter and libs: " ${PYTHONLIBS_VERSION_STRING} " and " ${PYTHON_VERSION_STRING})
endif()

find_package(NumPy REQUIRED)
message("Found NumPy: " ${NUMPY_INCLUDE_DIR})
message("NumPy include dirs: " ${PYTHON_NUMPY_INCLUDE_DIR})

find_package(SWIG REQUIRED 3.0)
include(${SWIG_USE_FILE})
include(GenerateExportHeader)

include_directories(
        ${PYTHON_INCLUDE_DIRS}
        ${NUMPY_INCLUDE_DIRS}
)

option(INSTALL_INTO_USER_SITE_PACKAGES_DIRECTORY "Install python modules into a per user site-packages directory" ON)

if (INSTALL_INTO_USER_SITE_PACKAGES_DIRECTORY)
    execute_process(COMMAND ${PYTHON_EXECUTABLE}  -m site --user-site OUTPUT_VARIABLE PYTHON_SITE_PACKAGES OUTPUT_STRIP_TRAILING_WHITESPACE)
else()
    execute_process(COMMAND ${PYTHON_EXECUTABLE} -c "from __future__ import print_function; from distutils.sysconfig import get_python_lib; print(get_python_lib())" OUTPUT_VARIABLE PYTHON_SITE_PACKAGES OUTPUT_STRIP_TRAILING_WHITESPACE)
endif()
message(STATUS "Python modules will be installed into" ${PYTHON_SITE_PACKAGES} ".")

install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/__init__.py DESTINATION ${PYTHON_SITE_PACKAGES}/irlib)

set(GF_SRCS "")

set(GF_INTERFACE_FILES basis.i)

set_source_files_properties(${GF_INTERFACE_FILES} PROPERTIES CPLUSPLUS ON)
swig_add_library(basis LANGUAGE python SOURCES ${GF_INTERFACE_FILES} ${GF_SRCS})
swig_link_libraries(basis ${PYTHON_LIBRARIES} ${GMP_LIBRARIES} ${MPFR_LIBRARIES})

install(TARGETS _basis DESTINATION ${PYTHON_SITE_PACKAGES}/irlib)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/basis.py DESTINATION ${PYTHON_SITE_PACKAGES}/irlib)

set_source_files_properties(piecewise_polynomial.i PROPERTIES CPLUSPLUS ON)
swig_add_library(piecewise_polynomial LANGUAGE python SOURCES piecewise_polynomial.i ${GF_SRCS})
swig_link_libraries(piecewise_polynomial ${PYTHON_LIBRARIES} ${GMP_LIBRARIES} ${MPFR_LIBRARIES})

install(TARGETS _piecewise_polynomial DESTINATION ${PYTHON_SITE_PACKAGES}/irlib)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/piecewise_polynomial.py DESTINATION ${PYTHON_SITE_PACKAGES}/irlib)

#configure_file(${CMAKE_SOURCE_DIR}/common/cmake/sitecustomize.py ${CMAKE_CURRENT_BINARY_DIR}/sitecustomize.py @ONLY)


if (Testing)
  add_subdirectory("test")
endif()
