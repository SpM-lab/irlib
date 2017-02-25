#pragma once

#include <iostream>
#include <complex>
#include <cmath>
#include <vector>
#include <assert.h>

#include <boost/multi_array.hpp>
#include <boost/type_traits.hpp>
#include <boost/typeof/typeof.hpp>

#include <gsl/gsl_errno.h>
#include <gsl/gsl_spline.h>

//#include <boost/timer/timer.hpp>

#include <Eigen/Core>
#include <Eigen/SVD>

extern "C" void dgesvd_( const char* jobu, const char* jobvt,
                         const int* m, const int* n, double* a, const int* lda,
                         double* s, double* u, const int* ldu,
                         double* vt, const int* ldvt,
                         double* work, const int* lwork, int* info);


namespace ir {
  template<typename T, int k>
  class piecewise_polynomial;

  enum statistics {fermionic, bosonic};

  namespace detail {
    /**
     *
     * @tparam T double or std::complex<double>
     * @param a  scalar
     * @param b  scalar
     * @return   conj(a) * b
     */
    template<class T>
    typename boost::enable_if<boost::is_floating_point<T>, T>::type
    outer_product(T a, T b) {
      return a * b;
    }

    template<class T>
    std::complex<T>
    outer_product(const std::complex<T>& a, const std::complex<T>& b) {
      return std::conj(a) * b;
    }

    template<class T>
    typename boost::enable_if<boost::is_floating_point<T>, T>::type
    conjg(T a) {
      return a;
    }

    template<class T>
    std::complex<T>
    conjg(const std::complex<T>& a) {
      return std::conj(a);
    }

    template<class T, int k>
    void compute_transformation_matrix_to_matsubara(
        int n,
        statistics s,
        const std::vector<piecewise_polynomial<T,k> >& bf_src,
        std::vector<std::complex<double> >& Tnl
    );
  }


/**
 * Class for representing a piecewise polynomial
 *   A function is represented by a polynomial in each section [x_n, x_{n+1}).
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

    template<typename TT, int kk>
    friend class piecewise_polynomial;

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

    void check_validity() const {
      if (n_sections_ < 1) {
        throw std::runtime_error("pieacewise_polynomial object is not properly constructed!");
      }
    }

   public:
    piecewise_polynomial() : n_sections_(0) {};

    piecewise_polynomial(int n_section,
                         const std::vector<double> &section_edges,
                         const boost::multi_array<T, 2> &coeff) : n_sections_(section_edges.size() - 1),
                                                                  section_edges_(section_edges),
                                                                  coeff_(coeff) {
      //FIXME: DO NOT THROW IN A CONSTRUCTOR
      if (n_section < 1) {
        std::runtime_error("n_section cannot be less than 1!");
      }
      if (section_edges.size() != n_section + 1) {
        std::runtime_error("size of section_edges is wrong!");
      }
      if (coeff.shape()[0] != n_section) {
        std::runtime_error("first dimension of coeff is wrong!");
      }
      if (coeff.shape()[1] != k+1) {
        std::runtime_error("second dimension of coeff is wrong!");
      }
      for (int i = 0; i < n_section; ++i) {
        if (section_edges[i] >= section_edges[i+1]) {
          std::runtime_error("section_edges must be in strictly increasing order!");
        }
      }
    };

    /// Number of sections
    int num_sections() const {
      return n_sections_;
    }

    inline double section_edge(int i) const {
      assert(i >= 0 && i < section_edges_.size());
      return section_edges_[i];
    }

    const std::vector<double>& section_edges() const {
      return section_edges_;
    }

    inline T coefficient(int i, int p) const {
      assert(i >= 0 && i < section_edges_.size());
      assert(p >= 0 && p <= k);
      return coeff_[i][p];
    }

    /// Compute the value at x
    inline T compute_value(double x) const {
      return compute_value(x, find_section(x));
    }

    /// Compute the value at x. x must be in the given section.
    inline T compute_value(double x, int section) const {
      if (x < section_edges_[section] || (x != section_edges_.back() && x >= section_edges_[section+1]) ) {
        throw std::runtime_error("The given x is not in the given section.");
      }

      const double dx = x - section_edges_[section];
      T r = 0.0, x_pow = 1.0;
      for (int p = 0; p < k + 1; ++p) {
        r += coeff_[section][p] * x_pow;
        x_pow *= dx;
      }
      return r;
    }

    /// Find the section involving the given x
    int find_section(double x) const {
      if (x == section_edges_[0]) {
        return 0;
      } else if (x == section_edges_.back()) {
        return coeff_.size() - 1;
      }

      std::vector<double>::const_iterator it =
          std::upper_bound(section_edges_.begin(), section_edges_.end(), x);
      --it;
      return (&(*it) - &(section_edges_[0]));
    }

    /// Compute overlap <this | other> with complex conjugate
    template<class T2, int k2>
    T overlap(const piecewise_polynomial<T2, k2> &other) const {
      if (section_edges_ != other.section_edges_) {
        throw std::runtime_error("Not supported");
      }
      typedef BOOST_TYPEOF(T(1.0) * T2(1.0)) Tr;

      Tr r = 0.0;
      boost::array<double, k + k2 + 2> x_min_power, dx_power;

      for (int s = 0; s < n_sections_; ++s) {
        dx_power[0] = 1.0;
        const double dx = section_edges_[s + 1] - section_edges_[s];
        for (int p = 1; p < dx_power.size(); ++p) {
          dx_power[p] = dx * dx_power[p - 1];
        }

        for (int p = 0; p < k + 1; ++p) {
          for (int p2 = 0; p2 < k2 + 1; ++p2) {
            r += detail::outer_product((Tr) coeff_[s][p], (Tr) other.coeff_[s][p2])
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

/// Gram-Schmidt orthonormalization
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

  template<typename T>// we expect T = double
  piecewise_polynomial<T, 3> construct_piecewise_polynomial_cspline(
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

    return piecewise_polynomial<T, 3>(n_section, x_array, coeff);
  };


  /**
   * Fermionic kernel
   */
  class FermionicKernel {
   public:
    FermionicKernel(double Lambda) : Lambda_(Lambda) {}

    double operator()(double x, double y) const {
      const double limit = 100.0;
      if (Lambda_ * y > limit) {
        return std::exp(-0.5*Lambda_*x*y - 0.5*Lambda_*y);
      } else if (Lambda_ * y < -limit) {
        return std::exp(-0.5*Lambda_*x*y + 0.5*Lambda_*y);
      } else {
        return std::exp(-0.5*Lambda_*x*y)/(2*std::cosh(0.5*Lambda_*y));
      }
    }

    static statistics statistics() {
      return fermionic;
    }

   private:
    double Lambda_;
  };

  /**
   * Bosonic kernel
   */
  class BosonicKernel {
   public:
    BosonicKernel(double Lambda) : Lambda_(Lambda) {}

    double operator()(double x, double y) const {
      const double limit = 100.0;
      if (std::abs(Lambda_*y) < 1e-10) {
        return std::exp(-0.5*Lambda_*x*y)/Lambda_;
      } else if (Lambda_ * y > limit) {
        return y*std::exp(-0.5*Lambda_*x*y - 0.5*Lambda_*y);
      } else if (Lambda_ * y < -limit) {
        return -y*std::exp(-0.5*Lambda_*x*y + 0.5*Lambda_*y);
      } else {
        return y*std::exp(-0.5*Lambda_*x*y)/(2*std::sinh(0.5*Lambda_*y));
      }
    }

    static statistics statistics() {
      return bosonic;
    }

   private:
    double Lambda_;
  };

/**
 * Class template for kernel Ir basis
 */
  template<typename Scalar, typename Kernel>
  class Basis {
   public:
    Basis(double Lambda, int max_dim, double cutoff = 1e-10, int N = 501);

   private:
    typedef piecewise_polynomial<double, 3> pp_type;

    std::vector<pp_type> basis_functions_;

   public:
    void value(double x, std::vector<double> &val) const;
    const pp_type &operator()(int l) const { return basis_functions_[l]; }
    int dim() const { return basis_functions_.size(); }

    void compute_Tnl(
        int n,
        std::vector<std::complex<double> >& Tnl
    ) const;

    void compute_Tnl(
        int n_min, int n_max,
        boost::multi_array<std::complex<double>,2> & Tnl
    ) const;
  };

  /**
   * Typedefs for convenience
   */
  typedef Basis<double, FermionicKernel> FermionicBasis;
  typedef Basis<double, BosonicKernel> BosonicBasis;
}

#include "ir_basis.ipp"
