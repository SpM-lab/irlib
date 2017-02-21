#pragma once

#include <iostream>
#include <complex>
#include <cmath>
#include <vector>
#include <assert.h>

#include <boost/multi_array.hpp>

#include <gsl/gsl_errno.h>
#include <gsl/gsl_spline.h>

#include <Eigen/Core>
#include <Eigen/SVD>

namespace ir {

  namespace detail {
    template<typename T>
    piecewise_polynomial<T, 3> construct_piecewise_polynomial_cspline(
        const std::vector<double> &x_array, const std::vector<double> &y_array);
  }

/**
 * Class for representing a piecewise polynomial
 */
  template<typename T, int k>
  class piecewise_polynomial {
   private:
    typedef boost::multi_array<T, 2> coefficient_type;

    template<typename TT, int kk>
    friend piecewise_polynomial<TT, kk>
    operator+(const piecewise_polynomial<TT, kk> &f1, const piecewise_polynomial<TT, kk> &f2);

    template<typename TT, int kk>
    friend piecewise_polynomial<TT, kk>
    operator-(const piecewise_polynomial<TT, kk> &f1, const piecewise_polynomial<TT, kk> &f2);

    template<typename TT, int kk>
    friend const piecewise_polynomial<TT, kk> operator*(TT scalar, const piecewise_polynomial<TT, kk> &pp);

    /// number of sections
    int n_sections_;

    /// edges of sections. The first and last elements should be -1 and 1, respectively.
    std::vector<double> section_edges_;

    /// coefficients of cubic spline of l-th basis function: the index is (l, section, power).
    /// 0: value at left section edges
    /// 1: first derivatives at left section edges
    ///  ...
    /// k: k derivatives at left section edges
    coefficient_type coeff_;

    void check_range(double x) const {
      if (x < section_edges_[0] || x > section_edges_[section_edges_.size() - 1]) {
        throw std::runtime_error("Give x is out of the range.");
      }
    }

   public:
    piecewise_polynomial() : n_sections_(0) {};

    piecewise_polynomial(int n_section,
                         const std::vector<double> &section_edges,
                         const boost::multi_array<T, 2> &coeff) : n_sections_(section_edges.size() - 1),
                                                                  section_edges_(section_edges),
                                                                  coeff_(coeff) {};

    /// Number of sections
    int num_sections() const {
      return n_sections_;
    }

    double section_edge(int i) const { return section_edges_[i]; }

    /// Compute the value at x
    T compute_value(double x) const {
      check_range(x);

      const int sec = find_section(x);

      const double dx = x - section_edges_[sec];
      T r = 0.0, x_pow = 1.0;
      for (int p = 0; p < k + 1; ++p) {
        r += coeff_[sec][p] * x_pow;
        x_pow *= dx;
      }
      return r;
    }

    /// Find the section involving the given x
    int find_section(double x) const {
      //FIXME: make this O(1)
      if (x == section_edges_[0]) {
        return 0;
      } else if (x == section_edges_.back()) {
        return coeff_.size() - 1;
      }
      std::vector<double>::const_iterator it =
          std::lower_bound(section_edges_.begin(), section_edges_.end(), x);
      --it;
      return std::distance(section_edges_.begin(), it);//O(N)
    }

    /// Compute overlap <this | other> with complex conjugate
    template<int k2>
    T overlap(const piecewise_polynomial<T, k2> &other) {
      if (section_edges_ != other.section_edges_) {
        throw std::runtime_error("Not supported");
      }

      T r = 0.0;
      boost::array<double, k + k2 + 2> x_min_power, dx_power;

      for (int s = 0; s < n_sections_; ++s) {
        dx_power[0] = 1.0;
        const double dx = section_edges_[s + 1] - section_edges_[s];
        for (int p = 1; p < dx_power.size(); ++p) {
          dx_power[p] = dx * dx_power[p - 1];
        }

        for (int p = 0; p < k + 1; ++p) {
          for (int p2 = 0; p2 < k2 + 1; ++p2) {
            r += alps::numeric::outer_product(coeff_[s][p], other.coeff_[s][p2])
                * dx_power[p + p2 + 1] / (p + p2 + 1.0);
          }
        }
      }
      return r;
    }
  };

// Add piecewise_polynomial objects
  template<typename T, int k>
  piecewise_polynomial<T, k> operator+(const piecewise_polynomial<T, k> &f1, const piecewise_polynomial<T, k> &f2) {
    if (f1.section_edges_ != f2.section_edges_) {
      throw std::runtime_error("Cannot add two numerical functions with different sections!");
    }
    boost::multi_array<T, 2> coeff_sum(f1.coeff_);
    std::transform(
        f1.coeff_.origin(), f1.coeff_.origin() + f1.coeff_.num_elements(),
        f2.coeff_.origin(), coeff_sum.origin(),
        std::plus<T>()

    );
    return piecewise_polynomial<T, k>(f1.num_sections(), f1.section_edges_, coeff_sum);
  }

// Substract piecewise_polynomial objects
  template<typename T, int k>
  piecewise_polynomial<T, k> operator-(const piecewise_polynomial<T, k> &f1, const piecewise_polynomial<T, k> &f2) {
    if (f1.section_edges_ != f2.section_edges_) {
      throw std::runtime_error("Cannot add two numerical functions with different sections!");
    }
    boost::multi_array<T, 2> coeff_sum(f1.coeff_);
    std::transform(
        f1.coeff_.origin(), f1.coeff_.origin() + f1.coeff_.num_elements(),
        f2.coeff_.origin(), coeff_sum.origin(),
        std::minus<T>()

    );
    return piecewise_polynomial<T, k>(f1.num_sections(), f1.section_edges_, coeff_sum);
  }

// Multiply piecewise_polynomial by a scalar
  template<typename T, int k>
  const piecewise_polynomial<T, k> operator*(T scalar, const piecewise_polynomial<T, k> &pp) {
    piecewise_polynomial<T, k> pp_copy(pp);
    std::transform(
        pp_copy.coeff_.origin(), pp_copy.coeff_.origin() + pp_copy.coeff_.num_elements(),
        pp_copy.coeff_.origin(), std::bind1st(std::multiplies<T>(), scalar)

    );
    return pp_copy;
  }

/// Gram-Schmidt orthonormalization (This should result in Legendre polynomials)
  template<typename T, int k>
  void orthonormalize(std::vector<piecewise_polynomial<T, k> > &pps) {
    typedef piecewise_polynomial<T, k> pp_type;

    for (int l = 0; l < pps.size(); ++l) {
      pp_type pp_new(pps[l]);
      for (int l2 = 0; l2 < l; ++l2) {
        const T overlap = pps[l2].overlap(pps[l]);
        pp_new = pp_new - overlap * pps[l2];
      }
      double norm = pp_new.overlap(pp_new);
      pps[l] = (1.0 / std::sqrt(norm)) * pp_new;
    }
  }

/**
 * Class template for kernel Ir basis
 */
  template<typename Scalar, typename Kernel>
  class Basis {
   public:
    Basis(double Lambda, int n_omega = 501, int n_tau = 501, double cutoff = 1e-10);

   private:
    typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> matrix_t;
    typedef alps::gf::piecewise_polynomial<double, 3> pp_type;

    const double tmin_;
    std::vector<double> tvec_, xvec_;
    matrix_t U_, V_;
    int dim_;

    std::vector<pp_type> basis_functions_;

   public:
    void value(double x, std::vector<double> &val) const;
    const pp_type &operator()(int l) const { return basis_functions_[l]; }
    int dim() const { return dim_; }
  };
}
