/* piecewise_polynomial.i */
%module(package="irlib", docstring="Python bindings for intermediate representation libraries") piecewise_polynomial
%{
#define SWIG_FILE_WITH_INIT
#include <Eigen/Core>
#include <irlib/piecewise_polynomial.hpp>
%}

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

/*
 * %feature("autodoc", "This is index_mesh .") alps::gf::index_mesh;
%feature("autodoc", "Do not call") alps::gf::index_mesh::compute_points;
 */


%include <irlib/piecewise_polynomial.hpp>

/* %template(real_piecewise_polynomial) alps::gf::piecewise_polynomial<double>; */
