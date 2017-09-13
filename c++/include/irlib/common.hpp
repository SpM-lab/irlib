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

using MatrixXmp = Eigen::Matrix<mpfr::mpreal,Eigen::Dynamic,Eigen::Dynamic>;
