#cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

# add common modules from ../common/cmake
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/common/cmake)

SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

#If this does not find the correct one, please use invoke cmake with -DPYTHON_EXECUTABLE=/path/to/your/python/executable.
find_package(PythonInterp REQUIRED)
message("Found Python executable: " ${PYTHON_EXECUTABLE})
message("Python version: " ${PYTHON_VERSION_STRING})

#find_package(PythonLibs EXACT ${PYTHON_VERSION_STRING} REQUIRED)
find_package(PythonLibs REQUIRED)
message("Found Python libs: " ${PYTHON_LIBRARIES})
message("Python libs version: " ${PYTHONLIBS_VERSION_STRING})
message("Python libraries: " ${PYTHON_LIBRARIES})
message("Python include dirs: " ${PYTHON_INCLUDE_DIRS})

if (NOT PYTHONLIBS_VERSION_STRING STREQUAL PYTHON_VERSION_STRING)
    message(FATAL_ERROR "Version mismatch between Python interpreter and libs")
endif()

find_package(NumPy REQUIRED)
message("Found NumPy: " ${NUMPY_INCLUDE_DIR})
message("NumPy include dirs: " ${PYTHON_NUMPY_INCLUDE_DIR})

find_package(SWIG REQUIRED 3.0)
include(${SWIG_USE_FILE})
include(GenerateExportHeader)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_BINARY_DIR}
    ${PYTHON_INCLUDE_PATH}
    ${NUMPY_INCLUDE_DIRS}
)

execute_process(COMMAND ${PYTHON_EXECUTABLE} -c "from distutils.sysconfig import get_python_lib; print get_python_lib()" OUTPUT_VARIABLE PYTHON_SITE_PACKAGES OUTPUT_STRIP_TRAILING_WHITESPACE)

install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/__init__.py DESTINATION ${PYTHON_SITE_PACKAGES}/irlib)

set(GF_SRCS "")

set(GF_INTERFACE_FILES basis.i)

set_source_files_properties(${GF_INTERFACE_FILES} PROPERTIES CPLUSPLUS ON)
swig_add_library(basis LANGUAGE python SOURCES ${GF_INTERFACE_FILES} ${GF_SRCS})
swig_link_libraries(basis ${PYTHON_LIBRARIES} ${LAPACK_LIBRARIES} ${BLAS_LIBRARIES})

include_directories(
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/c++/include/irlib
        ${PROJECT_BINARY_DIR}
        ${PYTHON_INCLUDE_PATH}
        ${NUMPY_INCLUDE_DIRS}
)

install(TARGETS _basis DESTINATION ${PYTHON_SITE_PACKAGES}/irlib)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/basis.py DESTINATION ${PYTHON_SITE_PACKAGES}/irlib)
