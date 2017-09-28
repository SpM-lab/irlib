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

    using MPREAL = mpfr::mpreal;
    using pp_type = piecewise_polynomial<double,MPREAL>;

    using MatrixXmp = Eigen::Matrix<mpfr::mpreal,Eigen::Dynamic,Eigen::Dynamic>;
    using MatrixXc = Eigen::Matrix<std::complex<double>,Eigen::Dynamic,Eigen::Dynamic>;

    inline void ir_set_default_prec(mp_prec_t prec) {
        mpfr::mpreal::set_default_prec(prec);
    }

    inline mp_prec_t  ir_get_default_prec() {
        return mpfr::mpreal::get_default_prec();
    }

    inline mp_prec_t ir_digits2bits(mp_prec_t prec) {
        return mpfr::digits2bits(prec);
    }
}

