#pragma once

#include <mpreal.h>

namespace irlib {

    namespace detail {

        template<typename MPREAL>
        MPREAL
        legendre_next(unsigned l, const MPREAL &x, const MPREAL &Pl, const MPREAL &Plm1) {
            MPREAL mp_l(l);
            return ((2 * mp_l + 1) * x * Pl - mp_l * Plm1) / (mp_l + 1);
        }

        template<typename MPREAL>
        MPREAL
        legendre_p_impl(unsigned int l, const MPREAL &x) {
            if (x < MPREAL(-1) || x > MPREAL(1)) {
                throw std::runtime_error("Legendre polynomials are defined for -1<=x<=1");
            }

            MPREAL p0(1);
            MPREAL p1(x);

            if (l == 0) {
                return p0;
            }

            int n = 1;

            while (n < l) {
                std::swap(p0, p1);
                p1 = legendre_next(n, x, p0, p1);
                ++n;
            }
            return p1;
        }

    }

    template<typename MPREAL>
    MPREAL
    legendre_p(unsigned int l, const MPREAL &x) {
        return detail::legendre_p_impl(l, x);
    }

    template<typename MPREAL>
    MPREAL
    normalized_legendre_p(unsigned int l, const MPREAL &x) {
        return sqrt(MPREAL(l) + MPREAL(0.5)) * legendre_p(l, x);
    }

    /**
     * Compute derivatives of normalized Legendre polynomials at a given x
     * @param Nl The number of polynomials
     * @param x  -1 <= x <= 1
     * @return   Two dimension arrays (l, d). l is the index of polynoamials. d is the order of the derivatives.
     */
    template<typename MPREAL>
    std::vector<std::vector<MPREAL>>
    normalized_legendre_p_derivatives(unsigned int Nl, const MPREAL &x) {
        using matrix_t = Eigen::Matrix<MPREAL, Eigen::Dynamic, Eigen::Dynamic>;
        using vector_t = Eigen::Matrix<MPREAL, Eigen::Dynamic, 1>;

        matrix_t S(Nl, Nl);

        // Overlap matrix for {x^l} in the interval of [-1,1].
        for (int l1=0; l1<Nl; ++l1) {
            for (int l2=0; l2<Nl; ++l2) {
                if ((l1+l2)%2 == 0) {
                    S(l1, l2) = MPREAL(2)/(MPREAL(l1)+MPREAL(l2)+MPREAL(1));
                } else {
                    S(l1, l2) = 0.0;
                }
            }
        }

        // {x^0, x^1, ..., x^{Nl-1}}
        std::vector<vector_t> basis;
        for (int l=0; l<Nl; ++l) {
            vector_t  b(Nl);
            b.setZero();
            b(l) = MPREAL(1);
            basis.push_back(b);
        }

        // Gram–Schmidt orthonormalization
        auto dot_product = [&](const vector_t & b1, const vector_t & b2) {
            return (b1.transpose() * S * b2)(0,0);
        };
        for (int l1=0; l1<Nl; ++l1) {
            for (int l2 = 0; l2 < l1; ++l2) {
                basis[l1] -= (dot_product(basis[l1], basis[l2])/dot_product(basis[l2], basis[l2])) * basis[l2];
            }
            basis[l1] /= sqrt(dot_product(basis[l1], basis[l1]));
        }

        // compute the derived function of a given function
        auto deriv = [](const vector_t& b) {
            vector_t db(b);
            db.setZero();
            for (int i=0; i<b.rows()-1; ++i) {
                db[i] = (i+1) * b[i+1];
            }
            return db;
        };

        // evaluate the value of a given function at a given x
        auto eval = [](const vector_t& b, const MPREAL& x) {
            MPREAL xn(1);
            MPREAL val(0);
            for (int n=0; n<b.rows(); ++n) {
                val += b[n] * xn;
                xn *= x;
            }
            return val;
        };

        std::vector<std::vector<MPREAL> > derivatives;
        for (int l1=0; l1<Nl; ++l1) {
            std::vector<MPREAL> db(Nl, MPREAL(0));
            auto b = basis[l1];
            for (int d=0; d<Nl; ++d) {
                db[d] = eval(b, x);
                b = deriv(b);
            }
            derivatives.push_back(db);
        }

        return derivatives;

    }
}
