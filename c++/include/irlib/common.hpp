#pragma once

namespace irlib {
    namespace statistics {
        enum statistics_type {
            BOSONIC = 0,
            FERMIONIC = 1
        };
    }
}

#include <mpreal.h>
#include <Eigen/MPRealSupport>
#include <Eigen/CXX11/Tensor>

#include "detail/gauss_legendre.hpp"
#include "detail/legendre_polynomials.hpp"

using MatrixXmp = Eigen::Matrix<mpfr::mpreal,Eigen::Dynamic,Eigen::Dynamic>;
