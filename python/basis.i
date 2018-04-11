/* basis.i */
%module(package="irlib", docstring="Python bindings for intermediate representation libraries") basis
%{
#define SWIG_FILE_WITH_INIT
#include <irlib/basis.hpp>

using namespace irlib;
%}

%include "stdint.i"
%include "std_string.i"
%include "std_vector.i"
%include "../common/swig/numpy.i"
%include "../common/swig/multi_array.i"

%init %{
   import_array();
%}

%multi_array_typemaps(std::vector<int>);
%multi_array_typemaps(std::vector<long>);
%multi_array_typemaps(std::vector<double>);
%multi_array_typemaps(std::vector<std::complex<double> >); 

%multi_array_typemaps(Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic>);
%multi_array_typemaps(Eigen::Matrix<std::complex<double>,Eigen::Dynamic,Eigen::Dynamic>);

%multi_array_typemaps(Eigen::Tensor<double,2>);
%multi_array_typemaps(Eigen::Tensor<double,3>);
%multi_array_typemaps(Eigen::Tensor<double,4>);
%multi_array_typemaps(Eigen::Tensor<double,5>);
%multi_array_typemaps(Eigen::Tensor<double,6>);
%multi_array_typemaps(Eigen::Tensor<double,7>);

%multi_array_typemaps(Eigen::Tensor<std::complex<double>,2>);
%multi_array_typemaps(Eigen::Tensor<std::complex<double>,3>);
%multi_array_typemaps(Eigen::Tensor<std::complex<double>,4>);
%multi_array_typemaps(Eigen::Tensor<std::complex<double>,5>);
%multi_array_typemaps(Eigen::Tensor<std::complex<double>,6>);
%multi_array_typemaps(Eigen::Tensor<std::complex<double>,7>);

/* These ignore directives must come before including header files */
%ignore irlib::basis::ulx_mp;

/* Include header files as part of interface file */
%include <irlib/common.hpp>
%include <irlib/basis.hpp>

%pythoncode {
from mpmath import *
}

%extend irlib::basis {
    %pythoncode %{
        #def ulx_mp(self, l, x):
            #return mpf(self.ulx_str(l, str(x)))

        #def vly_mp(self, l, y):
            #return mpf(self.vly_str(l, str(y)))
    %}
}
