#include "gtest.h"

#include <irlib/basis.hpp>

/*
#include <boost/math/special_functions/bessel.hpp>

void compute_Tnl_legendre(int n_matsubara, int n_legendre, Eigen::Matrix<std::complex<double>,Eigen::Dynamic, Eigen::Dynamic> &Tnl) {
  double sign_tmp = 1.0;
  Tnl.resize(n_matsubara, n_legendre);
  for (int im = 0; im < n_matsubara; ++im) {
    std::complex<double> ztmp(0.0, 1.0);
    for (int il = 0; il < n_legendre; ++il) {
      Tnl(im, il) = sign_tmp * ztmp * std::sqrt(2 * il + 1.0) * boost::math::sph_bessel(il, 0.5 * (2 * im + 1) * M_PI);
      ztmp *= std::complex<double>(0.0, 1.0);
    }
    sign_tmp *= -1;
  }
}

*/
