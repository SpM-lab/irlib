#pragma once

#include "basis_impl.hpp"

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

    /**
     * Bosonic IR basis
     */
    class basis_b : public basis_impl<mpfr::mpreal> {
    public:
        basis_b(double Lambda, int max_dim = 10000, double cutoff = 1e-12,
                int n_local_poly = 10) throw(std::runtime_error)
                : basis_impl<mpfr::mpreal>(irlib::statistics::BOSONIC, Lambda, max_dim, cutoff, n_local_poly) {}
    };
}
