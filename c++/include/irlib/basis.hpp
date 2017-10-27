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
    //extern void * enabler ;

/**
 * Class for kernel Ir basis
 */
 template<typename MPREAL>
    class basis_set {
    public:
        using scalar_type = MPREAL;

        /**
         * Constructor
         * @param knl  kernel
         * @param max_dim  max number of basis functions to be computed.
         * @param cutoff  we drop basis functions corresponding to small singular values  |s_l/s_0~ < cutoff.
         * @param n_local_poly   Number of Legendre polynomials used to expand basis functions in each sector
         */
        basis_set(const irlib::kernel<MPREAL> &knl, int max_dim, double cutoff, int n_local_poly) throw(std::runtime_error) {
            statistics_ = knl.get_statistics();
            std::tie(sv_, u_basis_, v_basis_) = generate_ir_basis_functions(knl, max_dim, cutoff, n_local_poly);
            assert(u_basis_.size()>0);
            assert(u_basis_[0].num_sections()>0);
        }

    private:
        statistics::statistics_type statistics_;
        //singular values
        std::vector<double> sv_;
        std::vector< piecewise_polynomial<double,MPREAL> > u_basis_, v_basis_;

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
            return ulx_mp(l, MPREAL(x));
        }

        /**
         * @param l  order of basis function
         * @param y  y on [-1,1]
         * @return   The value of v_l(y)
         */
        double vly(int l, double y) const throw(std::runtime_error) {
            assert(y >= -1 && y <= 1);
            assert(l >= 0 && l < dim());
            return vly_mp(l, MPREAL(y));
        }

        /**
         * This function should not be called outside this library
         * @param l  order of basis function
         * @param x  x on [-1,1]
         * @return   The value of u_l(x)
         */
        //template<typename U = MPREAL>
        //typename std::enable_if<!std::is_floating_point<U>::value, double>::type
        double ulx_mp(int l, const MPREAL& x) const throw(std::runtime_error) {
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
        //template<typename U = MPREAL>
        //typename std::enable_if<!std::is_floating_point<U>::value, double>::type
        double vly_mp(int l, const MPREAL& y) const throw(std::runtime_error) {
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
        const piecewise_polynomial<double,MPREAL> &ul(int l) const throw(std::runtime_error) {
            assert(l >= 0 && l < dim());
            python_runtime_check(l >= 0 && l < dim(), "Index l is out of range.");
            return u_basis_[l];
        }

        const piecewise_polynomial<double,MPREAL> &vl(int l) const throw(std::runtime_error) {
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
         * @param n_vec  This vector must contain indices of non-negative Matsubara freqencies (i.e., n>=0) in ascending order.
         * @param Tnl    Results
         */
        void compute_Tnl(
                const std::vector<long> &n_vec,
                Eigen::Tensor<std::complex<double>, 2> &Tnl
        ) const {
            compute_transformation_matrix_to_matsubara<double>(n_vec,
                                                                                   statistics_,
                                                                                   u_basis_,
                                                                                   Tnl);
        }
#endif

        /**
         * Compute transformation matrix to Matsubara freq.
         * The computation may take some time. You may store the result somewhere and do not call this routine frequenctly.
         * @param n_vec  This vector must contain indices of non-negative Matsubara freqencies (i.e., n>=0) in ascending order.
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
        Eigen::Tensor<std::complex<double>, 2>
        compute_Tbar_ol(const std::vector<long> &o_vec) const {
            int no = o_vec.size();
            int nl = u_basis_.size();

            Eigen::Tensor<std::complex<double>, 2> Tbar_ol(no, nl);
            irlib::compute_Tbar_ol(o_vec, u_basis_, Tbar_ol);

            return Tbar_ol;
        }


    };

#ifdef SWIG
%template(mpreal_basis_set) basis_set<irlib::mpreal>;
%template(double_basis_set) basis_set<double>;
#endif

    /**
     * Fermionic IR basis
     */
    class basis_f : public basis_set<irlib::mpreal> {
    public:
        basis_f(double Lambda, int max_dim = 10000, double cutoff = 1e-12, int n_local_poly=10) throw(std::runtime_error)
                : basis_set(fermionic_kernel<irlib::mpreal>(Lambda), max_dim, cutoff, n_local_poly) {}
    };

    /*
    class basis_f_dp : public basis_set<double> {
    public:
        basis_f_dp(double Lambda, int max_dim = 10000, double cutoff = 1e-8, int n_local_poly=10) throw(std::runtime_error)
                : basis_set(fermionic_kernel<double>(Lambda), max_dim, cutoff, n_local_poly) {}
    };
     */

    /**
     * Bosonic IR basis
     */
    class basis_b : public basis_set<irlib::mpreal> {
    public:
        basis_b(double Lambda, int max_dim = 10000, double cutoff = 1e-12, int n_local_poly=10) throw(std::runtime_error)
                : basis_set(bosonic_kernel<irlib::mpreal>(Lambda), max_dim, cutoff, n_local_poly) {}
    };

    /*
    class basis_b_dp : public basis_set<double> {
    public:
        basis_b_dp(double Lambda, int max_dim = 10000, double cutoff = 1e-8, int n_local_poly=10) throw(std::runtime_error)
                : basis_set(bosonic_kernel<double>(Lambda), max_dim, cutoff, n_local_poly) {}
    };
    */
}
