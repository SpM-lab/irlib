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
    class ir_basis_set {
    public:
        /**
         * Constructor
         * @param knl  kernel
         * @param max_dim  max number of basis functions to be computed.
         * @param cutoff  we drop basis functions corresponding to small singular values  |s_l/s_0~ < cutoff.
         * @param Nl   Number of Legendre polynomials used to expand basis functions in each sector
         */
        ir_basis_set(const irlib::kernel<mpfr::mpreal> &knl, int max_dim, double cutoff, int Nl) throw(std::runtime_error) {
            statistics_ = knl.get_statistics();
            std::tie(sv_, u_basis_, v_basis_) = generate_ir_basis_functions(knl, max_dim, cutoff, Nl);
            assert(u_basis_.size()>0);
            assert(u_basis_[0].num_sections()>0);
        }

    private:
        statistics::statistics_type statistics_;
        //singular values
        std::vector<double> sv_;
        std::vector< irlib::piecewise_polynomial<double> > u_basis_, v_basis_;

    public:
        /**
         * Compute the values of the basis functions for a given x.
         * @param x    x = 2 * tau/beta - 1  (-1 <= x <= 1)
         * @param val  results
         */
//#ifndef SWIG
        //void values(double x, std::vector<double> &val) const throw(std::runtime_error);
//#endif

        double sl(int l) const throw(std::runtime_error) {
            assert(l >= 0 && l < dim());
            python_runtime_check(l >= 0 && l < dim(), "Index l is out of range.");
            return sv_[l];
        }

        double ulx(int l, double x) const throw(std::runtime_error) {
            assert(x >= -1 && x <= 1);
            assert(l >= 0 && l < dim());
            python_runtime_check(l >= 0 && l < dim(), "Index l is out of range.");
            python_runtime_check(x >= -1 && x <= 1, "x must be in [-1,1].");
            if (l < 0 || l >= dim()) {
                throw std::runtime_error("Invalid index of basis function!");
            }
            if (x < -1 || x > 1) {
                throw std::runtime_error("Invalid value of x!");
            }
            return u_basis_[l].compute_value(x);
        }

        double vly(int l, double y) const throw(std::runtime_error) {
            assert(y >= -1 && y <= 1);
            assert(l >= 0 && l < dim());
            python_runtime_check(l >= 0 && l < dim(), "Index l is out of range.");
            python_runtime_check(y >= -1 && y <= 1, "y must be in [-1,1].");
            if (l < 0 || l >= dim()) {
                throw std::runtime_error("Invalid index of basis function!");
            }
            if (y < -1 || y > 1) {
                throw std::runtime_error("Invalid value of y!");
            }
            return v_basis_[l].compute_value(y);
        }

        /*
        std::vector<double> values(double x) const throw(std::runtime_error) {
            if (x < -1 || x > 1) {
                throw std::runtime_error("Invalid value of x!");
            }
            std::vector<double> val;
            values(x, val);
            return val;
        }
        */

        /**
         * Return a reference to the l-th basis function
         * @param l l-th basis function
         * @return  reference to the l-th basis function
         */
        const irlib::piecewise_polynomial<double> &ul(int l) const throw(std::runtime_error) {
            assert(l >= 0 && l < dim());
            python_runtime_check(l >= 0 && l < dim(), "Index l is out of range.");
            return u_basis_[l];
        }

        const irlib::piecewise_polynomial<double> &vl(int l) const throw(std::runtime_error) {
            assert(l >= 0 && l < dim());
            python_runtime_check(l >= 0 && l < dim(), "Index l is out of range.");
            return v_basis_[l];
        }

        /**
         * Return a reference to all basis functions
         */
        //const std::vector<irlib::piecewise_polynomial<double> > all() const { return u_basis_; }

        /**
         * Return number of basis functions
         * @return  number of basis functions
         */
        int dim() const { return u_basis_.size(); }

        /// Return statistics
        irlib::statistics::statistics_type get_statistics() const {
            return statistics_;
        }

        /**
         * Compute transformation matrix to Matsubara freq.
         * The computation may take some time. You may store the result somewhere and do not call this routine frequenctly.
         * @param n_min min Matsubara freq. index
         * @param n_max max Matsubara freq. index
         * @param Tnl max
         */
#ifndef SWIG

        void compute_Tnl(
                const std::vector<long> &n_vec,
                Eigen::Tensor<std::complex<double>, 2> &Tnl
        ) const {
            irlib::compute_transformation_matrix_to_matsubara<double>(n_vec,
                                                                                   statistics_,
                                                                                   u_basis_,
                                                                                   Tnl);
        }
#endif

        Eigen::Tensor<std::complex<double>, 2>
        compute_Tnl(const std::vector<long> &n_vec) const {
            Eigen::Tensor<std::complex<double>, 2> Tnl;
            compute_Tnl(n_vec, Tnl);
            return Tnl;
        }

        Eigen::Tensor<std::complex<double>, 2>
        compute_Tbar_ol(const std::vector<long> &o_vec) const {
            int no = o_vec.size();
            int nl = u_basis_.size();

            Eigen::Tensor<std::complex<double>, 2> Tbar_ol(no, nl);
            irlib::compute_Tbar_ol(o_vec, u_basis_, Tbar_ol);

            return Tbar_ol;
        }


    };

//#ifdef SWIG
    //%template(real_ir_basis_set) ir_basis_set<>;
//#endif

    /**
     * Fermionic IR basis
     */
    class basis_f : public ir_basis_set {
    public:
        basis_f(double Lambda, int max_dim, double cutoff = 1e-12, int Nl=10) throw(std::runtime_error)
                : ir_basis_set(fermionic_kernel(Lambda), max_dim, cutoff, Nl) {}
    };

    /**
     * Bosonic IR basis
     */
    class basis_b : public ir_basis_set {
    public:
        basis_b(double Lambda, int max_dim, double cutoff = 1e-12, int Nl=10) throw(std::runtime_error)
                : ir_basis_set(bosonic_kernel(Lambda), max_dim, cutoff, Nl) {}
    };
}
