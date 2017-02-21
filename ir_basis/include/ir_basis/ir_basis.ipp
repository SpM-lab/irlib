#pragma once

#include <iostream>
#include <complex>
#include <cmath>
#include <vector>
#include <assert.h>

#include <boost/math/special_functions/bessel.hpp>
#include <boost/multi_array.hpp>

#include <gsl/gsl_errno.h>
#include <gsl/gsl_spline.h>

#include <Eigen/Core>
#include <Eigen/SVD>

#include <alps/gf/numerical_basis.hpp>


namespace ir {
  namespace detail {
    template<typename T>
    alps::gf::piecewise_polynomial<T, 3> construct_piecewise_polynomial_cspline(
        const std::vector<double> &x_array, const std::vector<double> &y_array) {

      const int n_points = x_array.size();
      const int n_section = n_points - 1;

      boost::multi_array<double, 2> coeff(boost::extents[n_section][4]);

      gsl_interp_accel *my_accel_ptr = gsl_interp_accel_alloc();
      gsl_spline *my_spline_ptr = gsl_spline_alloc(gsl_interp_cspline, n_points);
      gsl_spline_init(my_spline_ptr, &x_array[0], &y_array[0], n_points);

      // perform spline interpolation
      for (int s = 0; s < n_section; ++s) {
        const double dx = x_array[s + 1] - x_array[s];
        coeff[s][0] = y_array[s];
        coeff[s][1] = gsl_spline_eval_deriv(my_spline_ptr, x_array[s], my_accel_ptr);
        coeff[s][2] = 0.5 * gsl_spline_eval_deriv2(my_spline_ptr, x_array[s], my_accel_ptr);
        coeff[s][3] =
            (y_array[s + 1] - y_array[s] - coeff[s][1] * dx - coeff[s][2] * dx * dx) / (dx * dx * dx);//ugly hack
        assert(
            std::abs(
                y_array[s + 1] - y_array[s] - coeff[s][1] * dx - coeff[s][2] * dx * dx - coeff[s][3] * dx * dx * dx)
                < 1e-8
        );
      }

      gsl_spline_free(my_spline_ptr);
      gsl_interp_accel_free(my_accel_ptr);

      return alps::gf::piecewise_polynomial<T, 3>(n_section, x_array, coeff);
    };

    template<typename T>
    inline std::vector<T> linspace(T minval, T maxval, int N) {
      std::vector<T> r(N);
      for (int i = 0; i < N; ++i) {
        r[i] = i * (maxval - minval) / (N - 1) + minval;
      }
      return r;
    }
  }

  template<typename Scalar, typename Kernel>
  Basis<Scalar,Kernel>::Basis(double Lambda, int n_omega, int n_tau, double cutoff)
      : tmin_(2.5), dim_(0) {

    //DE mesh for omega
    std::vector<double> tw_vec(n_omega), w_vec(2*n_omega);
    Eigen::VectorXd weight_w(2*n_omega);

    const double twmin = -2;

    double twmax = std::asinh(std::log(omega_max)*2/M_PI);

    for (int i = 0; i < n_omega; ++i) {
      tw_vec[i] = i * (twmin - twmax)/(n_omega-1) + twmax;
      w_vec[i] = - std::exp(0.5 * M_PI * std::sinh(tw_vec[i]));
      w_vec[2*n_omega - 1 - i] = - w_vec[i];
      weight_w(2*n_omega - 1 -i) = weight_w(i) = std::sqrt(0.5*M_PI*std::cosh(tw_vec[i])*std::exp(0.5*M_PI*std::sinh(tw_vec[i])));
    }

    //DE mesh for x
    tvec_ = detail::linspace(-tmin_, tmin_, n_tau); //2.5 is a very safe option.
    Eigen::VectorXd weight_x(n_tau);
    xvec_.resize(n_tau);
    for (int i = 0; i < n_tau; ++i) {
      xvec_[i] = std::tanh(0.5*M_PI*std::sinh(tvec_[i]));
      weight_x(i) = std::sqrt(0.5*M_PI*std::cosh(tvec_[i]))/std::cosh(0.5*M_PI*std::sinh(tvec_[i])); //sqrt of the weight of DE formula
    }

    K k_obj(Lambda);
    matrix_t K(n_tau, 2*n_omega);
    for (int i = 0; i < n_tau; ++i) {
      for (int j = 0; j < 2 * n_omega; ++j) {
        K(i, j) = weight_x(i) * k_obj(xvec_[i], w_vec[j]) * weight_w(j);
      }
    }

    //Perform SVD
    Eigen::BDCSVD<matrix_t> svd(K, Eigen::ComputeFullU | Eigen::ComputeFullV);

    //Count non-zero SV
    for (int i = 1; i < svd.singularValues().rows(); ++i) {
      if (std::abs(svd.singularValues()(i)/svd.singularValues()(0)) < cutoff) {
        dim_ = i;
        break;
      }
    }

    //Rescale U and V
    U_ = svd.matrixU();
    V_ = svd.matrixV();
    U_.conservativeResize(n_tau, dim_);
    V_.conservativeResize(2*n_omega, dim_);
    for (int l = 0; l < dim_; ++l) {
      double norm2_U = 0.0;
      for (int i = 0; i < n_tau; ++i) {
        U_(i,l) /= weight_x(i);
        norm2_U += (2*tmin_/n_tau) * (U_(i,l) * U_(i,l) * weight_x(i) * weight_x(i));
      }
      U_.col(l) /= std::sqrt(norm2_U);
      if (U_(n_tau-1, l) < 0.0) {
        U_.col(l) *= -1;
      }

      //FIXME: normalize V correctly
      for (int i = 0; i < 2*n_omega; ++i) {
        V_(i,l) /= weight_w(i);
      }
    }

    //Cubic spline interpolation and orthogonalization
    basis_functions_.resize(0);
    {
      const int n_section = n_tau + 1;
      std::vector<double> x_array(n_tau+2), y_array(n_tau+2);
      boost::multi_array<double,2> coeff(boost::extents[n_tau+1][4]);

      //set up x values
      x_array[0] = -1.0;
      for (int itau = 0; itau < n_tau; ++itau) {
        x_array[itau+1] = xvec_[itau];
      }
      x_array[n_tau+1] = 1.0;

      // symmetrization and spline interpolation
      for (int l = 0; l < dim_; ++l) {
        //set up y values
        y_array[0] = U_(0, l);
        for (int itau = 0; itau < n_tau; ++itau) {
          y_array[itau+1] = U_(itau, l);
        }
        y_array[n_tau+1] = U_(n_tau-1, l);

        //symmetrization
        if (l%2 == 0) {
          for (int itau = 0; itau < n_tau/2+1; ++itau) {
            y_array[itau] = y_array[n_tau+1-itau] = 0.5*(y_array[itau] + y_array[n_tau+1-itau]);
          }
        } else {
          for (int itau = 0; itau < n_tau/2+1; ++itau) {
            y_array[itau] = 0.5*(y_array[itau] - y_array[n_tau+1-itau]);
            y_array[n_tau+1-itau] = -y_array[itau];
          }
        }

        basis_functions_.push_back(construct_piecewise_polynomial_cspline<double>(x_array, y_array));
      }

      orthonormalize(basis_functions_);
    }
  };

  template<typename Scalar, typename Kernel>
  void
  Basis<Scalar,Kernel>::value(double x, std::vector<double> &val) const {
    assert(val.size() >= dim_);
    assert(x >= -1.00001 && x <= 1.00001);

    if (dim_ > val.size()) {
      val.resize(dim_);
    }
    for (int l = 0; l < dim_; l++) {
      val[l] = basis_functions_[l].compute_value(x);
    }
  }
}


