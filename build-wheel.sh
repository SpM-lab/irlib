export CMAKE_FLAGS='-DCMAKE_BUILD_TYPE=Release -DPYTHON_EXECUTABLE=/opt/local/bin/python2.7 -DPYTHON_INCLUDE_DIR=/opt/local/Library/Frameworks/Python.framework/Versions/2.7/include/python2.7 -DPYTHON_LIBRARY=/opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/libpython2.7.dylib -DCMAKE_CXX_COMPILER=mpicxx-openmpi-clang39 -DCMAKE_VERBOSE_MAKEFILE=ON'
export CXXFLAGS='-std=c++11 -O1'

python2.7 ~/ClionProjects/ir_basis/setup.py bdist_wheel
