#pragma once

#include <complex>
#include <mpreal.h>
#include <Eigen/MPRealSupport>
#include <Eigen/CXX11/Tensor>

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

    using mpfr::abs;
    using mpfr::sqrt;
    using mpfr::pow;
}


#include "detail/gauss_legendre.hpp"
#include "detail/legendre_polynomials.hpp"

using MatrixXmp = Eigen::Matrix<mpfr::mpreal,Eigen::Dynamic,Eigen::Dynamic>;
using MatrixXc = Eigen::Matrix<std::complex<double>,Eigen::Dynamic,Eigen::Dynamic>;

