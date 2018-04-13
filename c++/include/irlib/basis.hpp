#pragma once

#include <iostream>
#include <complex>
#include <cmath>
#include <vector>
#include <set>
#include <assert.h>
#include <memory>
#include <iomanip>
#include <fstream>

#include <Eigen/CXX11/Tensor>
#include <Eigen/MPRealSupport>

#include "common.hpp"
#include "kernel.hpp"
#include "piecewise_polynomial.hpp"

#include "irlib/detail/basis_impl.ipp"

namespace irlib {

/**
 * Class representing kernel Ir basis
 */
    class basis {
    public:
        /**
         * Constructor
         * @param s  statistics
         * @param Lambda Lambda
         * @param sv singular values
         * @param u_basis piecewise polynomials representing u_l(x)
         * @param v_basis piecewise polynomials representing v_l(y)
         */
        basis(statistics::statistics_type s,
            double Lambda,
            const std::vector<mpfr::mpreal> &sv,
            const std::vector<piecewise_polynomial<mpfr::mpreal, mpfr::mpreal>> &u_basis,
            const std::vector<piecewise_polynomial<mpfr::mpreal, mpfr::mpreal>> &v_basis
        ) throw(std::runtime_error) {
            statistics_ = s;
            Lambda_ = Lambda;
            sv_ = sv;
            u_basis_ = u_basis;
            v_basis_ = v_basis;
        }

    private:
        statistics::statistics_type statistics_;
        double Lambda_;
        std::vector<mpfr::mpreal> sv_;
        std::vector<piecewise_polynomial<mpfr::mpreal, mpfr::mpreal>> u_basis_, v_basis_;

        //mutable mp_prec_t default_prec_bak;

        mp_prec_t save_default_prec() const {
            mp_prec_t default_prec_bak = mpfr::mpreal::get_default_prec();
            mpfr::mpreal::set_default_prec(get_prec());
            return default_prec_bak;
        }

        void restore_default_prec(mp_prec_t prec) const {
            mpfr::mpreal::set_default_prec(prec);
        }

    public:
        /**
         * Compute the values of the basis functions for a given x.
         * @param x    x = 2 * tau/beta - 1  (-1 <= x <= 1)
         * @param val  results
         */
        double sl(int l) const throw(std::runtime_error) {
            assert(l >= 0 && l < dim());
            python_runtime_check(l >= 0 && l < dim(), "Index l is out of range.");
            return static_cast<double>(sv_[l]);
        }

        mpfr::mpreal sl_mp(int l) const throw(std::runtime_error) {
            assert(l >= 0 && l < dim());
            python_runtime_check(l >= 0 && l < dim(), "Index l is out of range.");
            return sv_[l];
        }

        double Lambda() const {
            return Lambda_;
        }

        /**
         * @param l  order of basis function
         * @param x  x on [-1,1]
         * @return   The value of u_l(x)
         */
        double ulx(int l, double x) const throw(std::runtime_error) {
            assert(x >= -1 && x <= 1);
            assert(l >= 0 && l < dim());
            auto bak = save_default_prec();

            //auto val = ulx_mp(l, mpfr::mpreal(x));
            //auto r = static_cast<double>(val);

            auto r = static_cast<double>(ulx_mp(l, mpfr::mpreal(x)));
            restore_default_prec(bak);
            return r;
        }

        double ulx_derivative(int l, double x, int order) const throw(std::runtime_error) {
            assert(x >= -1 && x <= 1);
            assert(l >= 0 && l < dim());
            auto bak = save_default_prec();
            auto r = static_cast<double>(ulx_derivative_mp(l, mpfr::mpreal(x), order));
            restore_default_prec(bak);
            return r;
        }

        /**
         * @param l  order of basis function
         * @param y  y on [-1,1]
         * @return   The value of v_l(y)
         */
        double vly(int l, double y) const throw(std::runtime_error) {
            assert(y >= -1 && y <= 1);
            assert(l >= 0 && l < dim());
            auto bak = save_default_prec();
            auto r = static_cast<double>(vly_mp(l, mpfr::mpreal(y)));
            restore_default_prec(bak);
            return r;
        }

        double vly_derivative(int l, double y, int order) const throw(std::runtime_error) {
            assert(y >= -1 && y <= 1);
            assert(l >= 0 && l < dim());
            auto bak = save_default_prec();
            auto r = static_cast<double>(vly_derivative_mp(l, mpfr::mpreal(y), order));
            restore_default_prec(bak);
            return r;
        }


        /**
         * Direct access to coefficients of u_l(x)
         */
        double coeff_ulx(int l, int section, int p) const {
            assert(l >= 0 && l < dim());
            return static_cast<double>(u_basis_[l].coefficient(section, p));
        }

        /**
         * Direct access to coefficients of v_l(y)
         */
        double coeff_vly(int l, int section, int p) const {
            assert(l >= 0 && l < dim());
            return static_cast<double>(v_basis_[l].coefficient(section, p));
        }

        /**
         * Access to sections
         */
        int num_sections_ulx() const {
            return u_basis_[0].num_sections();
        }

        int num_sections_vly() const {
            return v_basis_[0].num_sections();
        }

        double section_edge_ulx(int i) const {
            return static_cast<double>(u_basis_[0].section_edge(i));
        }

        double section_edge_vly(int i) const {
            return static_cast<double>(v_basis_[0].section_edge(i));
        }

        int num_local_poly_ulx() const {
            return u_basis_[0].order() + 1;
        }

        int num_local_poly_vly() const {
            return v_basis_[0].order() + 1;
        }

        /**
         * This function should not be called outside this library
         * @param l  order of basis function
         * @param x  x on [-1,1]
         * @return   The value of u_l(x)
         */
#ifndef SWIG //DO NOT EXPOSE TO PYTHON
        mpfr::mpreal ulx_mp(int l, const mpfr::mpreal &x) const throw(std::runtime_error) {
            assert(x >= -1 && x <= 1);
            assert(l >= 0 && l < dim());
            python_runtime_check(l >= 0 && l < dim(), "Index l is out of range.");
            python_runtime_check(x >= -1 && x <= 1, "x must be in [-1,1].");

            auto bak = save_default_prec();

            mpfr:mpreal r;
            if (x >= 0) {
                r = u_basis_[l].compute_value(x);
            } else {
                r = u_basis_[l].compute_value(-x) * (l % 2 == 0 ? 1 : -1);
            }

            restore_default_prec(bak);


            return r;
        }

        mpfr::mpreal ulx_derivative_mp(int l, const mpfr::mpreal &x, int order) const throw(std::runtime_error) {
            assert(x >= -1 && x <= 1);
            assert(l >= 0 && l < dim());
            python_runtime_check(l >= 0 && l < dim(), "Index l is out of range.");
            python_runtime_check(x >= -1 && x <= 1, "x must be in [-1,1].");

            auto bak = save_default_prec();

            mpfr::mpreal r;
            if (x >= 0) {
                r = u_basis_[l].derivative(x, order);
            } else {
                r = u_basis_[l].derivative(-x, order) * ((l+order) % 2 == 0 ? 1 : -1);
            }

            restore_default_prec(bak);

            return r;
        }

        /**
         * This function should not be called outside this library
         * @param l  order of basis function
         * @param y  y on [-1,1]
         * @return   The value of v_l(y)
         */
        mpfr::mpreal vly_mp(int l, const mpfr::mpreal &y) const throw(std::runtime_error) {
            assert(y >= -1 && y <= 1);
            assert(l >= 0 && l < dim());
            python_runtime_check(l >= 0 && l < dim(), "Index l is out of range.");
            python_runtime_check(y >= -1 && y <= 1, "y must be in [-1,1].");

            auto bak = save_default_prec();

            mpfr::mpreal r;
            if (y >= 0) {
                r = v_basis_[l].compute_value(y);
            } else {
                r = v_basis_[l].compute_value(-y) * (l % 2 == 0 ? 1 : -1);
            }

            restore_default_prec(bak);

            return r;
        }

        mpfr::mpreal vly_derivative_mp(int l, const mpfr::mpreal &y, int order) const throw(std::runtime_error) {
            assert(y >= -1 && y <= 1);
            assert(l >= 0 && l < dim());
            python_runtime_check(l >= 0 && l < dim(), "Index l is out of range.");
            python_runtime_check(y >= -1 && y <= 1, "y must be in [-1,1].");

            auto bak = save_default_prec();

            mpfr::mpreal r;
            if (y >= 0) {
                r = v_basis_[l].derivative(y, order);
            } else {
                r = v_basis_[l].derivative(-y, order) * ((l+order) % 2 == 0 ? 1 : -1);
            }

            restore_default_prec(bak);

            return r;
        }
#endif

        std::string ulx_str(int l, const std::string& str_x) const throw(std::runtime_error) {
            auto bak = save_default_prec();

            auto prec = u_basis_[l].section_edge(0).get_prec();
            mpfr::mpreal x(str_x, prec);
            auto ulx = ulx_mp(l, x);

            std::ostringstream out;
            out << std::setprecision(mpfr::bits2digits(ulx.get_prec())) << ulx;
            //std::cout << "debug ulx_str " << std::setprecision(20) << str_x << " " << x << " " << ulx << " " << out.str() << std::endl;

            restore_default_prec(bak);

            return out.str();
        }

        std::string vly_str(int l, const std::string& str_y) const throw(std::runtime_error) {
            auto bak = save_default_prec();

            auto prec = v_basis_[l].section_edge(0).get_prec();
            mpfr::mpreal y(str_y, prec);
            auto vly = vly_mp(l, y);

            std::ostringstream out;
            out << std::setprecision(mpfr::bits2digits(vly.get_prec())) << vly;

            restore_default_prec(bak);

            return out.str();
        }

        /**
         * Return a reference to the l-th basis function
         * @param l l-th basis function
         * @return  reference to the l-th basis function
         */
        const piecewise_polynomial<mpfr::mpreal, mpfr::mpreal> &ul(int l) const throw(std::runtime_error) {
            assert(l >= 0 && l < dim());
            python_runtime_check(l >= 0 && l < dim(), "Index l is out of range.");
            return u_basis_[l];
        }

        const piecewise_polynomial<mpfr::mpreal, mpfr::mpreal> &vl(int l) const throw(std::runtime_error) {
            assert(l >= 0 && l < dim());
            python_runtime_check(l >= 0 && l < dim(), "Index l is out of range.");
            return v_basis_[l];
        }

        /**
         * Return number of basis functions
         * @return  number of basis functions
         */
        int dim() const { return u_basis_.size(); }


        mpfr_prec_t get_prec() const {
            return ul(0).section_edge(0).get_prec();
        }

        /// Return statistics
        irlib::statistics::statistics_type get_statistics() const {
            return statistics_;
        }

        std::string get_statistics_str() const {
            return statistics_ == statistics::FERMIONIC ? "F" : "B" ;
        }

        int get_prec_int() const {
            return static_cast<int>(ul(0).section_edge(0).get_prec());
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
            auto bak = save_default_prec();

            auto trans_to_non_negative = [&](long n) {
                if (n >= 0) {
                    return n;
                } else {
                    if (statistics_ == irlib::statistics::FERMIONIC) {
                        return -n - 1;
                    } else {
                        return -n;
                    }
                }
            };

            auto nl = dim();

            std::set<long> none_negative_n;
            for (const auto &n : n_vec) {
                none_negative_n.insert(trans_to_non_negative(n));
            }

            Eigen::Tensor<std::complex<double>, 2> Tnl_tmp;
            compute_transformation_matrix_to_matsubara<mpreal>(
                    std::vector<long>(none_negative_n.begin(), none_negative_n.end()),
                    statistics_, u_basis_, Tnl_tmp
            );

            Tnl = Eigen::Tensor<std::complex<double>, 2>(n_vec.size(), nl);
            for (int i = 0; i < n_vec.size(); ++i) {
                auto index_data = std::distance(
                        none_negative_n.begin(),
                        none_negative_n.find(trans_to_non_negative(n_vec[i]))
                );
                if (n_vec[i] >= 0) {
                    for (int l = 0; l < nl; ++l) {
                        Tnl(i, l) = Tnl_tmp(index_data, l);
                    }
                } else {
                    for (int l = 0; l < nl; ++l) {
                        Tnl(i, l) = std::conj(Tnl_tmp(index_data, l));
                    }
                }
            }

            restore_default_prec(bak);
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
            std::cout << "Warning: compute_Tnl() is not well tested!" << std::endl;
            Eigen::Tensor<std::complex<double>, 2> Tnl;
            compute_Tnl(n_vec, Tnl);
            return Tnl;
        }

        std::complex<double> compute_Tnl_safe(long n, int l) {
            auto bak = save_default_prec();

            auto o = (statistics_ == irlib::statistics::FERMIONIC ? 2*n+1 : 2*n);
            auto r = to_dcomplex(
                    compute_Tnl_impl(u_basis_[l], l%2==0, mpfr::const_pi() * 0.5 * o,
                                    mpfr::digits2bits(get_prec()),
                                    mpfr::digits2bits(get_prec()))
            );

            restore_default_prec(bak);

            return r;
        }

    };

    inline basis compute_basis(statistics::statistics_type s,
                        double Lambda,
                        int max_dim = 1000,
                        double cutoff = 1e-8,
                        const std::string& fp_mode="mp",
                        double r_tol = 1e-6,
                        long prec = 64,
                        int n_local_poly = 10,
                        int num_nodes_gauss_legendre = 24,
                        bool verbose = true
        ) throw(std::runtime_error) {
        std::vector<mpfr::mpreal> sv;
        std::vector<piecewise_polynomial<mpfr::mpreal, mpfr::mpreal>> u_basis;
        std::vector<piecewise_polynomial<mpfr::mpreal, mpfr::mpreal>> v_basis;

        // Increase default precision if needed
        auto min_prec = std::max(
                mpfr::digits2bits(std::log10(1/cutoff)+2*std::log10(1/r_tol)),
                long(64)//At least 19 digits
        );
        //min_prec = std::max(min_prec, mpfr::digits2bits(std::log10(1/r_tol))+10);
        min_prec = std::max(min_prec, prec);
        if (min_prec > mpfr::mpreal::get_default_prec()) {
            mpfr::mpreal::set_default_prec(min_prec);
        }
        if (verbose) {
            std::cout << "Using default precision = " << min_prec << " bits." << std::endl;
        }

        if (fp_mode == "mp") {
            if (s == statistics::FERMIONIC) {
                std::tie(sv, u_basis, v_basis) = generate_ir_basis_functions<mpfr::mpreal>(
                        fermionic_kernel<mpfr::mpreal>(Lambda), max_dim, cutoff, verbose, r_tol, n_local_poly, num_nodes_gauss_legendre);
            } else if (s == statistics::BOSONIC) {
                std::tie(sv, u_basis, v_basis) = generate_ir_basis_functions<mpfr::mpreal>(
                        bosonic_kernel<mpfr::mpreal>(Lambda), max_dim, cutoff, verbose, r_tol, n_local_poly, num_nodes_gauss_legendre);
            }
        } else if (fp_mode == "long double") {
            if (cutoff < 1e-8) {
                std::cout << "Warning : cutoff cannot be smaller than 1e-8 for long-double precision version. Please use fp_mode='mp'!" << std::endl;
            }
            if (s == statistics::FERMIONIC) {
                std::tie(sv, u_basis, v_basis) = generate_ir_basis_functions<long double>(
                        fermionic_kernel<mpfr::mpreal>(Lambda), max_dim, cutoff, verbose, r_tol, n_local_poly, num_nodes_gauss_legendre);
            } else if (s == statistics::BOSONIC) {
                std::tie(sv, u_basis, v_basis) = generate_ir_basis_functions<long double>(
                        bosonic_kernel<mpfr::mpreal>(Lambda), max_dim, cutoff, verbose, r_tol, n_local_poly, num_nodes_gauss_legendre);
            }
        } else {
            throw std::runtime_error("Unknown fp_mode " + fp_mode + ". Only 'mp' is supported.");
        }

        return basis(s, Lambda, sv, u_basis, v_basis);
    }

    inline void savetxt(const std::string& fname, const basis& b) throw(std::runtime_error) {
        std::ofstream ofs(fname);

        int version = 1;
        ofs << version << std::endl;
        ofs << b.get_statistics() << std::endl;
        ofs << b.Lambda() << std::endl;
        ofs << b.dim() << std::endl;

        ofs << b.sl_mp(0).get_prec() << std::endl;
        for (int l=0; l<b.dim(); ++l) {
            auto sl = b.sl_mp(l);
            ofs << std::setprecision(mpfr::bits2digits(sl.get_prec())) << sl << std::endl;
        }
        for (int l=0; l<b.dim(); ++l) {
            ofs << b.ul(l);
        }
        for (int l=0; l<b.dim(); ++l) {
            ofs << b.vl(l);
        }
    }

    inline basis loadtxt(const std::string& fname) throw(std::runtime_error) {
        std::ifstream ifs(fname);

        if (!ifs.is_open()) {
            throw std::runtime_error(fname + " cannot be opened!");
        }

        statistics::statistics_type s;
        double Lambda;
        int dim;

        int version;
        ifs >> version;

        if (version == 1) {
            {
                int itmp;
                ifs >> itmp;
                s = static_cast<statistics::statistics_type>(itmp);
            }
            ifs >> Lambda;
            ifs >> dim;

            mpfr_prec_t prec;
            ifs >> prec;
            std::vector<mpfr::mpreal> sv(dim);
            for (int l=0; l<dim; ++l) {
                sv[l].set_prec(prec);
                ifs >> sv[l];
            }

            std::vector<piecewise_polynomial<mpfr::mpreal, mpfr::mpreal>> u_basis(dim), v_basis(dim);

            for (int l=0; l<dim; ++l) {
                ifs >> u_basis[l];
            }

            for (int l=0; l<dim; ++l) {
                ifs >> v_basis[l];
            }

            return basis(s, Lambda, sv, u_basis, v_basis);
        } else {
            throw std::runtime_error("Version " + std::to_string(version) + " is not supported!");
        }
    }
}

