#pragma once

#include "basis_impl.hpp"
//#include "basis_impl2.hpp"

namespace irlib {
    /**
     * Fermionic IR basis
     */
    class basis_f : public basis_impl<mpfr::mpreal> {
    public:
        basis_f(double Lambda, int max_dim = 10000, double cutoff = 1e-12,
                int n_local_poly = 10) throw(std::runtime_error)
                : basis_impl<mpfr::mpreal>(irlib::statistics::FERMIONIC, Lambda, max_dim, cutoff, n_local_poly) {}
    };

    class basis_f_dp : public basis_impl<double> {
    public:
        basis_f_dp(double Lambda, int max_dim = 10000, double cutoff = 1e-6,
                int n_local_poly = 10) throw(std::runtime_error)
                : basis_impl<double>(irlib::statistics::FERMIONIC, Lambda, max_dim, cutoff, n_local_poly) {}
    };

    class basis_f_ldp : public basis_impl<long double> {
    public:
        basis_f_ldp(double Lambda, int max_dim = 10000, double cutoff = 1e-6,
                   int n_local_poly = 10) throw(std::runtime_error)
                : basis_impl<long double>(irlib::statistics::FERMIONIC, Lambda, max_dim, cutoff, n_local_poly) {}
    };


    /*
    class basis_f_dp2 : public basis_impl2<double> {
    public:
        basis_f_dp2(double Lambda, int max_dim = 10000, double cutoff = 1e-12,
                   int n_local_poly = 10) throw(std::runtime_error)
                : basis_impl2<double>(irlib::statistics::FERMIONIC, Lambda, max_dim, cutoff, n_local_poly) {}
    };
     */

    /**
     * Bosonic IR basis
     */
    class basis_b : public basis_impl<mpfr::mpreal> {
    public:
        basis_b(double Lambda, int max_dim = 10000, double cutoff = 1e-12,
                int n_local_poly = 10) throw(std::runtime_error)
                : basis_impl<mpfr::mpreal>(irlib::statistics::BOSONIC, Lambda, max_dim, cutoff, n_local_poly) {}
    };

    class basis_b_dp : public basis_impl<double> {
    public:
        basis_b_dp(double Lambda, int max_dim = 10000, double cutoff = 1e-6,
                   int n_local_poly = 10) throw(std::runtime_error)
                : basis_impl<double>(irlib::statistics::BOSONIC, Lambda, max_dim, cutoff, n_local_poly) {}
    };

}
