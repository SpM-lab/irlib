#pragma once

#include <complex>
#include <memory>
#include <utility>

#include <Eigen/SVD>

#include "common.hpp"
#include "piecewise_polynomial.hpp"
#include "irlib/detail/basis_impl.ipp"

namespace irlib {
    /**
     * Abstract class representing an analytical continuation kernel
     */
    template<typename T>
    class kernel {
    public:
        typedef T mp_type;

        virtual ~kernel() {};

        /// return the value of the kernel for given x and y in the [-1,1] interval.
        virtual T operator()(T x, T y) const = 0;

        /// return statistics
        virtual irlib::statistics::statistics_type get_statistics() const = 0;

        /// return lambda
        virtual double Lambda() const = 0;

#ifndef SWIG

        /// return a reference to a copy
        virtual std::shared_ptr<kernel> clone() const = 0;

#endif
    };

#ifdef SWIG
    %template(real_kernel) kernel<mpreal>;
#endif

    /**
     * Fermionic kernel
     */
    template<typename S>
    class fermionic_kernel : public kernel<S> {
    public:
        fermionic_kernel(S Lambda) : Lambda_(Lambda) {}

        virtual ~fermionic_kernel() {};

        S operator()(S x, S y) const {
            const S limit = 100.0;
            if (Lambda_ * y > limit) {
                return std::exp(-0.5 * Lambda_ * x * y - 0.5 * Lambda_ * y);
            } else if (Lambda_ * y < -limit) {
                return std::exp(-0.5 * Lambda_ * x * y + 0.5 * Lambda_ * y);
            } else {
                return std::exp(-0.5 * Lambda_ * x * y) / (2 * std::cosh(0.5 * Lambda_ * y));
            }
        }

        irlib::statistics::statistics_type get_statistics() const {
            return irlib::statistics::FERMIONIC;
        }

        double Lambda() const {
            return Lambda_;
        }

#ifndef SWIG

        std::shared_ptr <kernel<S> > clone() const {
            return std::shared_ptr<kernel<S> >(new fermionic_kernel<S>(Lambda_));
        }

#endif

    private:
        double Lambda_;
    };

    template<>
    class fermionic_kernel<mpreal> : public kernel<mpreal> {
    public:
        fermionic_kernel(double Lambda) : Lambda_(Lambda) {}

        virtual ~fermionic_kernel() {};

        mpreal operator()(mpreal x, mpreal y) const {
            mpreal half_Lambda = mpreal("0.5") * mpreal(Lambda_);

            const double limit = 200.0;
            if (Lambda_ * y > limit) {
                return mpfr::exp(-half_Lambda * x * y - half_Lambda * y);
            } else if (Lambda_ * y < -limit) {
                return mpfr::exp(-half_Lambda * x * y + half_Lambda * y);
            } else {
                return mpfr::exp(-half_Lambda * x * y) / (2 * mpfr::cosh(half_Lambda * y));
            }
        }

        irlib::statistics::statistics_type get_statistics() const {
            return irlib::statistics::FERMIONIC;
        }

        double Lambda() const {
            return Lambda_;
        }

#ifndef SWIG

        std::shared_ptr<kernel<mpreal>> clone() const {
            return std::shared_ptr<kernel<mpreal>>(new fermionic_kernel(Lambda_));
        }

#endif

    private:
        double Lambda_;
    };


    /**
     * Bosonic kernel
     */
    template<typename S>
    class bosonic_kernel : public kernel<S> {
    public:
        bosonic_kernel(double Lambda) : Lambda_(Lambda) {}

        virtual ~bosonic_kernel() {};

        S operator()(S x, S y) const {
            const S limit = 100.0;
            if (std::abs(Lambda_ * y) < 1e-10) {
                return std::exp(-0.5 * Lambda_ * x * y) / Lambda_;
            } else if (Lambda_ * y > limit) {
                return y * std::exp(-0.5 * Lambda_ * x * y - 0.5 * Lambda_ * y);
            } else if (Lambda_ * y < -limit) {
                return -y * std::exp(-0.5 * Lambda_ * x * y + 0.5 * Lambda_ * y);
            } else {
                return y * std::exp(-0.5 * Lambda_ * x * y) / (2 * std::sinh(0.5 * Lambda_ * y));
            }
        }

        irlib::statistics::statistics_type get_statistics() const {
            return irlib::statistics::BOSONIC;
        }

        double Lambda() const {
            return Lambda_;
        }

#ifndef SWIG

        std::shared_ptr <kernel<S>> clone() const {
            return std::shared_ptr<kernel<S>>(new bosonic_kernel<S>(Lambda_));
        }

#endif

    private:
        double Lambda_;
    };

    template<>
    class bosonic_kernel<mpreal> : public kernel<mpreal> {
    public:
        bosonic_kernel(double Lambda) : Lambda_(Lambda) {}

        virtual ~bosonic_kernel() {};

        mpreal operator()(mpreal x, mpreal y) const {
            const double limit = 200.0;
            mpreal half_Lambda = mpreal("0.5") * mpreal(Lambda_);

            if (mpfr::abs(Lambda_ * y) < 1e-30) {
                return mpfr::exp(-half_Lambda * x * y) / Lambda_;
            } else if (Lambda_ * y > limit) {
                return y * mpfr::exp(-half_Lambda * x * y - half_Lambda * y);
            } else if (Lambda_ * y < -limit) {
                return -y * mpfr::exp(-half_Lambda * x * y + half_Lambda * y);
            } else {
                return y * mpfr::exp(-half_Lambda * x * y) / (2 * mpfr::sinh(half_Lambda * y));
            }
        }

        irlib::statistics::statistics_type get_statistics() const {
            return irlib::statistics::BOSONIC;
        }

        double Lambda() const {
            return Lambda_;
        }

#ifndef SWIG

        std::shared_ptr<kernel<mpreal>> clone() const {
            return std::shared_ptr<kernel<mpreal>>(new bosonic_kernel(Lambda_));
        }

#endif

    private:
        double Lambda_;
    };


}
