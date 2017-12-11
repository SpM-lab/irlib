#pragma once

#include <iostream>
#include <complex>
#include <cmath>
#include <vector>
#include <set>
#include <assert.h>
#include <memory>

#include <Eigen/CXX11/Tensor>
#include <Eigen/MPRealSupport>

#include "common.hpp"
#include "kernel.hpp"
#include "piecewise_polynomial.hpp"

#include "detail/aux.hpp"

namespace irlib {

/**
 * Class for kernel Ir basis
 */

 template<typename SVDFPType>
    class basis_impl {
    public:
        // Floating-point type used for SVD
        using svd_scalar_type = SVDFPType;

        //The most accurate floating-point type
        using accurate_fp_type = mpfr::mpreal;

        /**
         * Constructor
         * @param knl  kernel
         * @param max_dim  max number of basis functions to be computed.
         * @param cutoff  we drop basis functions corresponding to small singular values  |s_l/s_0~ < cutoff.
         * @param n_local_poly   Number of Legendre polynomials used to expand basis functions in each sector
         */
        basis_impl(statistics::statistics_type s, double Lambda, int max_dim, double cutoff, int n_local_poly) throw(std::runtime_error) {
            // Increase default precision if needed
            {
                auto min_prec = std::max(
                        mpfr::digits2bits(std::log10(1/cutoff))+10,
                        long(64)//At least 19 digits
                );
                if (min_prec > mpfr::mpreal::get_default_prec()) {
                    mpfr::mpreal::set_default_prec(min_prec);
                }
            }

            statistics_ = s;
            if (s == statistics::FERMIONIC) {
                std::tie(sv_, u_basis_, v_basis_) = generate_ir_basis_functions<SVDFPType>(
                        fermionic_kernel<SVDFPType>(Lambda), max_dim, cutoff, n_local_poly);
            } else if (s == statistics::BOSONIC) {
                std::tie(sv_, u_basis_, v_basis_) = generate_ir_basis_functions<SVDFPType>(
                        bosonic_kernel<SVDFPType>(Lambda), max_dim, cutoff, n_local_poly);
            }
            assert(u_basis_.size()>0);
            assert(u_basis_[0].num_sections()>0);
        }

    private:
        statistics::statistics_type statistics_;
        //singular values
        std::vector<double> sv_;
        std::vector< piecewise_polynomial<double,accurate_fp_type>> u_basis_, v_basis_;

    public:
        /**
         * Compute the values of the basis functions for a given x.
         * @param x    x = 2 * tau/beta - 1  (-1 <= x <= 1)
         * @param val  results
         */
        double sl(int l) const throw(std::runtime_error) {
            assert(l >= 0 && l < dim());
            python_runtime_check(l >= 0 && l < dim(), "Index l is out of range.");
            return sv_[l];
        }

        /**
         * @param l  order of basis function
         * @param x  x on [-1,1]
         * @return   The value of u_l(x)
         */
        double ulx(int l, double x) const throw(std::runtime_error) {
            assert(x >= -1 && x <= 1);
            assert(l >= 0 && l < dim());
            return ulx_mp(l, SVDFPType(x));
        }

        /**
         * @param l  order of basis function
         * @param y  y on [-1,1]
         * @return   The value of v_l(y)
         */
        double vly(int l, double y) const throw(std::runtime_error) {
            assert(y >= -1 && y <= 1);
            assert(l >= 0 && l < dim());
            return vly_mp(l, SVDFPType(y));
        }

        /**
         * This function should not be called outside this library
         * @param l  order of basis function
         * @param x  x on [-1,1]
         * @return   The value of u_l(x)
         */
        double ulx_mp(int l, const accurate_fp_type& x) const throw(std::runtime_error) {
            assert(x >= -1 && x <= 1);
            assert(l >= 0 && l < dim());
            python_runtime_check(l >= 0 && l < dim(), "Index l is out of range.");
            python_runtime_check(x >= -1 && x <= 1, "x must be in [-1,1].");

            if (x >= 0) {
                return u_basis_[l].compute_value(x);
            } else {
                return u_basis_[l].compute_value(-x) * (l%2==0 ? 1 : -1);
            }
        }

        /**
         * This function should not be called outside this library
         * @param l  order of basis function
         * @param y  y on [-1,1]
         * @return   The value of v_l(y)
         */
        double vly_mp(int l, const accurate_fp_type& y) const throw(std::runtime_error) {
            assert(y >= -1 && y <= 1);
            assert(l >= 0 && l < dim());
            python_runtime_check(l >= 0 && l < dim(), "Index l is out of range.");
            python_runtime_check(y >= -1 && y <= 1, "y must be in [-1,1].");
            if (y >= 0) {
                return v_basis_[l].compute_value(y);
            } else {
                return v_basis_[l].compute_value(-y) * (l%2==0 ? 1 : -1);
            }
        }

        /**
         * Return a reference to the l-th basis function
         * @param l l-th basis function
         * @return  reference to the l-th basis function
         */
        const piecewise_polynomial<double,accurate_fp_type> &ul(int l) const throw(std::runtime_error) {
            assert(l >= 0 && l < dim());
            python_runtime_check(l >= 0 && l < dim(), "Index l is out of range.");
            return u_basis_[l];
        }

        const piecewise_polynomial<double,accurate_fp_type> &vl(int l) const throw(std::runtime_error) {
            assert(l >= 0 && l < dim());
            python_runtime_check(l >= 0 && l < dim(), "Index l is out of range.");
            return v_basis_[l];
        }

        /**
         * Return number of basis functions
         * @return  number of basis functions
         */
        int dim() const { return u_basis_.size(); }

        /// Return statistics
        irlib::statistics::statistics_type get_statistics() const {
            return statistics_;
        }

#ifndef SWIG //DO NOT EXPOSE TO PYTHON
        /**
         * Compute transformation matrix to Matsubara freq.
         * The computation may take some time. You may store the result somewhere and do not call this routine frequenctly.
         * @param n_vec  This vector must contain indices of Matsubara freqencies
         * @param Tnl    Results
         */
        void compute_Tnl(
                const std::vector<long> &n_vec,
                Eigen::Tensor<std::complex<double>, 2> &Tnl
        ) const {
            auto trans_to_non_negative = [&](long n) {
                if (n >= 0) {
                    return n;
                } else {
                    if (statistics_ == irlib::statistics::FERMIONIC) {
                        return -n-1;
                    } else {
                        return -n;
                    }
                }
            };

            auto nl = dim();

            std::set<long> none_negative_n;
            for (const auto& n : n_vec) {
                none_negative_n.insert(trans_to_non_negative(n));
            }

            Eigen::Tensor<std::complex<double>, 2> Tnl_tmp;
            compute_transformation_matrix_to_matsubara<double>(
                    std::vector<long>(none_negative_n.begin(), none_negative_n.end()),
                    statistics_, u_basis_, Tnl_tmp
            );

            Tnl = Eigen::Tensor<std::complex<double>, 2>(n_vec.size(), nl);
            for (int i=0; i<n_vec.size(); ++i) {
                auto index_data = std::distance(
                        none_negative_n.begin(),
                        none_negative_n.find(trans_to_non_negative(n_vec[i]))
                );
                if (n_vec[i] >= 0) {
                    for (int l=0; l<nl; ++l) {
                        Tnl(i, l) = Tnl_tmp(index_data, l);
                    }
                } else {
                    for (int l=0; l<nl; ++l) {
                        Tnl(i, l) = std::conj(Tnl_tmp(index_data, l));
                    }
                }
            }
        }
#endif

        /**
         * Compute transformation matrix to Matsubara freq.
         * The computation may take some time. You may store the result somewhere and do not call this routine frequenctly.
         * @param n_vec  This vector must contain indices of Matsubara freqencies
         * @return Results
         */
        Eigen::Tensor<std::complex<double>, 2>
        compute_Tnl(const std::vector<long> &n_vec) const {
            Eigen::Tensor<std::complex<double>, 2> Tnl;
            compute_Tnl(n_vec, Tnl);
            return Tnl;
        }

        /**
         * Compute transformation matrix (Lewin's shifted representation)
         * The computation may take some time. You may store the result somewhere and do not call this routine frequenctly.
         * @param o_vec  This vector must contain o >= in ascending order.
         * @return Results
         */
         /*
        Eigen::Tensor<std::complex<double>, 2>
        compute_Tbar_ol(const std::vector<long> &o_vec) const {
            int no = o_vec.size();
            int nl = u_basis_.size();

            std::set<long> none_negative_o;

            for (const auto& o : o_vec) {
                none_negative_o.insert(std::abs(o));
            }

            Eigen::Tensor<std::complex<double>, 2> Tbar_ol_tmp(none_negative_o.size(), nl);
            irlib::compute_Tbar_ol(
                    std::vector<long>(none_negative_o.begin(), none_negative_o.end()),
                    u_basis_, Tbar_ol_tmp
            );

            Eigen::Tensor<std::complex<double>, 2> Tbar_ol(o_vec.size(), nl);
            for (int i=0; i<o_vec.size(); ++i) {
                auto index_data = std::distance(
                        none_negative_o.begin(),
                        none_negative_o.find(std::abs(o_vec[i]))
                );
                if (o_vec[i] >= 0) {
                    for (int l=0; l<nl; ++l) {
                        Tbar_ol(i, l) = Tbar_ol_tmp(index_data, l);
                    }
                } else {
                    for (int l=0; l<nl; ++l) {
                        Tbar_ol(i, l) = std::conj(Tbar_ol_tmp(index_data, l));
                    }
                }
            }
            return Tbar_ol;
        }
          */


    };

#ifdef SWIG
%template(mpreal_basis_impl) basis_impl<mpfr::mpreal>;
%template(mpreal_basis_dp_impl) basis_impl<double>;
%template(mpreal_basis_ldp_impl) basis_impl<long double>;
#endif

}
