irlib
======
C++ header-file only library for generating kernel "intermediate-representation" (ir) basis

# Table of Contents
- [Requirements](#requirements)
- [Installation](#installation)
- [Examples](#examples)

## Requirements
### C++11 compiler

### Lapack and Blas
For SVD

### Boost (>= 1.55.0)
Only header-file libraries are needed.

### Eigen3 (>= 3.3)
Header-file libraries for linear algebra. Eigen Tensor library in unsupported modules is required.

### Python 2.7 or 3.5
For generating Python bindings

## Installation
```
$ export BOOST_ROOT=/opt/boost_1_63
$ cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DPYTHON_EXECUTABLE=/opt/local/bin/python2.7 \
    -DPYTHON_INCLUDE_DIR=/opt/local/Library/Frameworks/Python.framework/Versions/2.7/include/python2.7 \
    -DPYTHON_LIBRARY=/opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/libpython2.7.dylib \
    -DCMAKE_CXX_FLAGS="-std=c++11" \
    -DCMAKE_INSTALL_PREFIX=/opt/irlib \
    -DEIGEN3_INCLUDE_DIR=/opt/Eigen3/include/eigen3 \
    -DCMAKE_CXX_COMPILER=mpicxx-openmpi-clang39 \
    -DCMAKE_VERBOSE_MAKEFILE=ON\
    path_to_source_file_directory
$ make && sudo make install
```

## Examples
You can find examples below the directory "examples" in the source directory.