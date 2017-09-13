#pragma once

#include <complex>
#include <memory>

#include "common.hpp"

namespace irlib {
    /**
     * Abstract class representing an analytical continuation kernel
     */
    template<typename T>
    class kernel {
    public:
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
    %template(real_kernel) kernel<mpfr::mpreal>;
#endif

    /**
     * Fermionic kernel
     */
    class fermionic_kernel : public kernel<mpfr::mpreal> {
    public:
        fermionic_kernel(double Lambda) : Lambda_(Lambda) {}

        virtual ~fermionic_kernel() {};

        mpfr::mpreal operator()(mpfr::mpreal x, mpfr::mpreal y) const {
            mpfr::mpreal half_Lambda = mpfr::mpreal("0.5") * mpfr::mpreal(Lambda_);

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

        std::shared_ptr<kernel> clone() const {
            return std::shared_ptr<kernel>(new fermionic_kernel(Lambda_));
        }

#endif

    private:
        double Lambda_;
    };

    /**
     * Bosonic kernel
     */
    class bosonic_kernel : public kernel<mpfr::mpreal> {
    public:
        bosonic_kernel(double Lambda) : Lambda_(Lambda) {}

        virtual ~bosonic_kernel() {};

        mpfr::mpreal operator()(mpfr::mpreal x, mpfr::mpreal y) const {
            const double limit = 200.0;
            mpfr::mpreal half_Lambda = mpfr::mpreal("0.5") * mpfr::mpreal(Lambda_);

            if (std::abs(Lambda_ * y) < 1e-30) {
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

        std::shared_ptr<kernel> clone() const {
            return std::shared_ptr<kernel>(new bosonic_kernel(Lambda_));
        }

#endif

    private:
        double Lambda_;
    };
}
