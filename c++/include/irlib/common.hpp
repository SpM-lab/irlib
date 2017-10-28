#pragma once

#include <complex>
#include <mpreal.h>
#include <Eigen/MPRealSupport>
#include <Eigen/CXX11/Tensor>

#include "piecewise_polynomial.hpp"
#include "detail/gauss_legendre.hpp"
#include "detail/legendre_polynomials.hpp"

namespace irlib {
    inline bool python_runtime_check(bool b, const std::string& message) {
#ifdef SWIGPYTHON
        if (!b) {
            throw std::runtime_error(message);
        }
#endif
        return b;
    }

    namespace statistics {
        enum statistics_type {
            BOSONIC = 0,
            FERMIONIC = 1
        };
    }

    using std::abs;
    using std::sqrt;
    using std::pow;
    using std::cosh;
    using std::sinh;

    using mpfr::mpreal;
    using mpfr::abs;
    using mpfr::sqrt;
    using mpfr::pow;
    using mpfr::cosh;
    using mpfr::sinh;

    namespace detail {
        template<typename S> inline S sqrt(const S& s);
        template<> inline double sqrt<double>(const double& s) {return std::sqrt(s);};
        template<> inline mpfr::mpreal sqrt<mpreal>(const mpreal& s) {return mpfr::sqrt(s);};
    }

    using pp_type = piecewise_polynomial<double,mpreal>;

    using MatrixXmp = Eigen::Matrix<mpfr::mpreal,Eigen::Dynamic,Eigen::Dynamic>;
    using MatrixXc = Eigen::Matrix<std::complex<double>,Eigen::Dynamic,Eigen::Dynamic>;

    template<typename ScalarType>
    inline void ir_set_default_prec(mp_prec_t prec);

    template<typename ScalarType>
    inline mp_prec_t ir_get_default_prec();

    template<>
    inline
    void
    ir_set_default_prec<mpfr::mpreal>(mp_prec_t prec) {
        mpfr::mpreal::set_default_prec(prec);
    }

    template<>
    inline void ir_set_default_prec<double>(mp_prec_t prec) {}

    template<>
    inline mp_prec_t ir_get_default_prec<mpfr::mpreal>() {
        return mpfr::mpreal::get_default_prec();
    }

    template<>
    inline mp_prec_t ir_get_default_prec<double>() {
        return 64;
    }

    inline mp_prec_t ir_digits2bits(mp_prec_t prec) {
        return mpfr::digits2bits(prec);
    }
}

