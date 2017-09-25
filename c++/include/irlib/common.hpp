#pragma once

#include <complex>
#include <mpreal.h>
#include <Eigen/MPRealSupport>
#include <Eigen/CXX11/Tensor>

#include "piecewise_polynomial.hpp"
#include "detail/gauss_legendre.hpp"
#include "detail/legendre_polynomials.hpp"

namespace irlib {
    namespace statistics {
        enum statistics_type {
            BOSONIC = 0,
            FERMIONIC = 1
        };
    }

    inline bool python_runtime_check(bool b, const std::string& message) {
#ifdef SWIG
        if (!b) {
            std::runtime_error(message);
        }
#endif
        return b;
    }

    using mpfr::mpreal;
    using mpfr::abs;
    using mpfr::sqrt;
    using mpfr::pow;
    using mpfr::cosh;
    using mpfr::sinh;

    using pp_type = piecewise_polynomial<double,mpreal>;
    using IR_MPREAL = mpfr::mpreal;
}



using MatrixXmp = Eigen::Matrix<mpfr::mpreal,Eigen::Dynamic,Eigen::Dynamic>;
using MatrixXc = Eigen::Matrix<std::complex<double>,Eigen::Dynamic,Eigen::Dynamic>;

