irlib
======
C++ header only library for generating kernel "intermediate-representation" (ir) basis written by Hiroshi Shinaoka, Naoya Chikano, Junya Otsuki,
Please refer to H. Shinaoka, J. Otsuki, M. Ohzeki and K. Yoshimi, PRB 96, 035147 (2017).
The basis functions u_l(x) and v_l(y) are orthonormal on the interval [-1,1] with weight 1.
The sign of u_l(x) is chosen so that u_l(1) > 0.


# Table of Contents
- [Requirements](#requirements)
- [Installation](#installation)
- [Examples](#examples)

## Requirements
### C++11 compiler

### MPFR (>= 2.31)

### GMP (>= 4.21)

### Python 2.7 or 3.5
For building Python binding

### SWIG (>= 3.0)
For building Python binding

## Installation
See the following sample.
```
$ cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DPYTHON_EXECUTABLE=/opt/local/bin/python2.7 \
    -DPYTHON_INCLUDE_DIR=/opt/local/Library/Frameworks/Python.framework/Versions/2.7/include/python2.7 \
    -DPYTHON_LIBRARY=/opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/libpython2.7.dylib \
    -DCMAKE_CXX_FLAGS="-std=c++11" \
    -DCMAKE_INSTALL_PREFIX=/opt/irlib \
    -DTesting=ON \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DCMAKE_VERBOSE_MAKEFILE=ON \
    path_to_source_file_directory
$ make && sudo make install
```

## Examples
You can find examples below the directory "examples" in the source directory.

## License
All files except for those in the directory "thirdparty" are licensed under GPLv3 or any later version. See LICENSE.txt for more details.
